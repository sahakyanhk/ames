"""CANYA aggregation/nucleation scoring -- NumPy-only inference.

A port of the *inference* path of CANYA (https://github.com/lehner-lab/canya,
Thompson et al. 2024, MIT licence) with no TensorFlow, Keras, pandas or torch
dependency.  The network is 17,491 parameters, so a handful of batched matmuls
is all it takes; see README.md for provenance and the validation status.

    from canya import Canya
    scores = Canya().score(["KLSGFKFEVTCDGHEFWGHP", "NFAYCYFFS"])

`score()` returns one float per input sequence, in order, same length as the
input: the median CANYA score over all 20-residue windows of that sequence
(`--mode default --summarize median`, the model whose interpretations are
presented in the paper).  Sequences that cannot be scored yield nan.

Every numbered comment below marks a place where a naive re-implementation
silently diverges from Keras; the notes in README.md explain each one.
"""

import math
import warnings
from pathlib import Path

import numpy as np

__all__ = ["Canya", "score_sequences"]

# --- constants, all from the original CANYA source ------------------------
AAS = "ACDEFGHIKLMNPQRSTVWY"  # modeling.aas
WINDOW = 20                   # utils.windowsize
SEQ_LEN = 24                  # modeling.seqminsize, the model's input length
FLANK = 2                     # modeling.paddings: 2 zero rows at each end
MASK_VALUE = -1.0             # Masking(mask_value=-1)
D_MODEL = 6                   # num_heads(1) * key_len(6)
NUM_FILTERS = 100
FILTER_WIDTH = 3
MIN_HALF_LIFE = 3.0           # f_exponential default
NEG_MASK = -10000.0           # (5) hard-coded in MultiHeadAttention_wpos, not -inf
BN_EPSILON = 1e-3             # (6) Keras BatchNormalization default

DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent
    / "data" / "models" / "model_weights1.npz"
)

# ascii -> amino-acid index, -1 for anything CANYA cannot encode
_LUT = np.full(256, -1, dtype=np.int8)
for _i, _a in enumerate(AAS):
    _LUT[ord(_a)] = _i


class InvalidSequence(ValueError):
    """Raised for a sequence CANYA cannot score (bad residue, or empty)."""


# --- positional-encoding basis, translated from modeling.py ---------------
# Pure functions of seq_len, so they are evaluated once at load time.

def _xlogy(x, y):
    """tf.math.xlogy: x*log(y), and exactly 0 wherever x == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x * np.log(y)
    return np.where(x == 0, 0.0, out)


def _f_exponential(positions, feature_size, seq_length, min_half_life=MIN_HALF_LIFE):
    max_range = np.log(seq_length) / np.log(2.0)
    half_life = 2.0 ** np.linspace(min_half_life, max_range, feature_size)
    positions = np.abs(positions)
    return np.exp(-np.log(2.0) / half_life * positions[..., None])


def _f_central_mask(positions, feature_size):
    center_widths = 2.0 ** np.arange(1, feature_size + 1, dtype=np.float64) - 1
    return (center_widths > np.abs(positions)[..., None]).astype(np.float64)


def _f_gamma(positions, feature_size, seq_length):
    stdv = seq_length / (2 * feature_size)
    start_mean = seq_length / feature_size
    mean = np.linspace(start_mean, seq_length, feature_size)
    concentration = (mean / stdv) ** 2
    rate = mean / stdv**2
    x = np.abs(positions)[..., None]
    log_unnormalized = _xlogy(concentration - 1.0, x) - rate * x
    lgamma = np.array([math.lgamma(c) for c in np.ravel(concentration)])
    log_normalization = lgamma.reshape(concentration.shape) - concentration * np.log(rate)
    probabilities = np.exp(log_unnormalized - log_normalization)
    return probabilities / probabilities.max()


def _positional_encodings(seq_len, embedding_size=D_MODEL):
    """(1, 2*seq_len-1, embedding_size); for CANYA: (1, 47, 6)."""
    # (4) embedding_size // 6 == 1: each of the six bases contributes 1 feature.
    feature_size = embedding_size // 6
    pos = np.arange(-seq_len + 1, seq_len, dtype=np.float64)[None, :]
    sign = np.sign(pos)[..., None]
    seq_length = float(seq_len)
    exp1 = _f_exponential(np.abs(pos), feature_size, seq_length)
    cm1 = _f_central_mask(np.abs(pos), feature_size)
    gam1 = _f_gamma(np.abs(pos), feature_size, seq_length)
    return np.concatenate(
        [exp1, exp1 * sign, cm1, cm1 * sign, gam1, gam1 * sign], axis=-1
    )


def _relative_shift(x):
    """modeling.relative_shift, with the single attention head folded into the
    batch axis: (B, t1, 2*t1-1) -> (B, t1, t1).  (8)"""
    x = np.concatenate([np.zeros_like(x[..., :1]), x], axis=-1)
    t1, t2 = x.shape[-2], x.shape[-1]
    x = x.reshape(-1, t2, t1)
    x = x[:, 1:, :]
    x = x.reshape(-1, t1, t2 - 1)
    return x[..., : (t2 + 1) // 2]


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


class Canya:
    """CANYA scorer.  Load once, reuse; loading parses a 75 KB .npz."""

    def __init__(self, weights=DEFAULT_WEIGHTS, batch_size=2048):
        w = np.load(weights)
        self.source = str(w["_source"]) if "_source" in w else str(weights)
        self.batch_size = int(batch_size)

        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
        # (7) Conv1D kernel (width, in, out) -> im2col matrix (width*in, out),
        #     C-order so the column blocks are [t-1, t, t+1] x channels.
        self._conv = f32(w["seq_conv/kernel"].reshape(FILTER_WIDTH * len(AAS), NUM_FILTERS))
        self._wq, self._wk, self._wv = (f32(w[f"mha/w{c}"]) for c in "qkv")
        # (1,1,1,depth) -> (1,1,depth): broadcast over (batch, position, depth)
        self._r_w = f32(w["mha/r_w"].reshape(1, 1, D_MODEL))
        self._r_r = f32(w["mha/r_r"].reshape(1, 1, D_MODEL))
        self._mha_out_w = f32(w["mha/out_kernel"])
        self._mha_out_b = f32(w["mha/out_bias"])
        self._dense_w = f32(w["dense_mha/kernel"])
        self._dense_b = f32(w["dense_mha/bias"])
        self._out_w = f32(w["output/kernel"])
        self._out_b = f32(w["output/bias"])
        # Keras BN at inference: gamma*(x-mean)/sqrt(var+eps) + beta, fused.
        inv_std = 1.0 / np.sqrt(w["bn/moving_variance"].astype(np.float64) + BN_EPSILON)
        self._bn_scale = f32(w["bn/gamma"] * inv_std)
        self._bn_shift = f32(w["bn/beta"] - w["bn/moving_mean"] * w["bn/gamma"] * inv_std)

        # r_k depends only on seq_len and the weights -> precompute (d_model, 2L-1).
        pe = _positional_encodings(SEQ_LEN)  # (1, 47, 6); (7) Dropout here is
        r_k = pe[0] @ w["mha/r_k_layer"].astype(np.float64)  # inference-inert
        self._r_k_T = f32(r_k.T)
        self._q_scale = np.float32(1.0 / np.sqrt(D_MODEL))

    # --- encoding ---------------------------------------------------------
    def encode(self, codes, lengths):
        """(B, 20) residue indices (-1 padded) + lengths -> (B, 24, 20) input.

        seq_to_vector + tf.pad(paddings) + dynamic_padding, vectorised.
        """
        b = codes.shape[0]
        x = np.zeros((b, SEQ_LEN, len(AAS)), dtype=np.float32)
        rows = np.arange(SEQ_LEN)
        # (2) the FLANK rows added by tf.pad are *zeros* and stay unmasked;
        #     only dynamic_padding's trailing rows carry -1.
        x[rows[None, :] >= (lengths[:, None] + 2 * FLANK)] = MASK_VALUE
        bi, pi = np.nonzero(codes >= 0)
        x[bi, pi + FLANK, codes[bi, pi]] = 1.0
        return x

    # --- forward ----------------------------------------------------------
    def forward(self, x):
        """(B, 24, 20) -> (B,) sigmoid scores."""
        # (1) Keras Masking does not merely attach a mask, it multiplies the
        #     input by it: the -1 rows become all-zero rows before the conv.
        valid = np.any(x != MASK_VALUE, axis=-1)
        vf = valid.astype(np.float32)
        x = x * vf[:, :, None]

        # Conv1D(100, 3, padding="same", use_bias=False) -> exponential -> mask
        padded = np.zeros((x.shape[0], SEQ_LEN + 2, len(AAS)), dtype=np.float32)
        padded[:, 1:-1] = x
        cols = np.concatenate([padded[:, 0:SEQ_LEN], padded[:, 1:SEQ_LEN + 1],
                               padded[:, 2:SEQ_LEN + 2]], axis=2)
        h = cols @ self._conv
        np.exp(h, out=h)
        h *= vf[:, :, None]  # Multiply(name="seq_conv_mask"); exp(0)==1, so this matters

        # MultiHeadAttention_wpos, num_heads=1 so the head axis is dropped
        q = h @ self._wq
        k = h @ self._wk
        v = h @ self._wv
        q *= self._q_scale  # (3) scaled *before* r_w/r_r are added
        logits = (q + self._r_w) @ k.transpose(0, 2, 1)
        logits += _relative_shift((q + self._r_r) @ self._r_k_T)
        # getattnmask masks *keys*: column j is suppressed where position j is padding
        if not valid.all():
            logits = np.where(valid[:, None, :], logits, np.float32(NEG_MASK))
        attended = _softmax(logits) @ v
        attended = attended @ self._mha_out_w + self._mha_out_b

        # Flatten -> Dense(64) -> BN -> relu -> Dense(1, sigmoid)
        d = attended.reshape(attended.shape[0], -1) @ self._dense_w + self._dense_b
        d *= self._bn_scale
        d += self._bn_shift
        np.maximum(d, 0.0, out=d)
        logit = d @ self._out_w + self._out_b
        return (1.0 / (1.0 + np.exp(-logit.astype(np.float64)))).ravel()

    # --- public API -------------------------------------------------------
    def windows(self, sequence):
        """Sliding 20-mers of one sequence as (n, 20) indices + (n,) lengths."""
        if not isinstance(sequence, str):
            raise InvalidSequence(f"expected str, got {type(sequence).__name__}")
        body = sequence.split("*")[0]  # utils: truncate at the first stop
        if not body:
            raise InvalidSequence("empty sequence")
        try:
            raw = np.frombuffer(body.encode("ascii"), dtype=np.uint8)
        except UnicodeEncodeError:
            raise InvalidSequence("non-ascii character") from None
        codes = _LUT[raw]
        bad = np.flatnonzero(codes < 0)
        if bad.size:
            raise InvalidSequence(
                f"{body[bad[0]]!r} at position {int(bad[0]) + 1} is not one of {AAS}"
            )
        if codes.size > WINDOW:
            view = np.lib.stride_tricks.sliding_window_view(codes, WINDOW)
            return np.ascontiguousarray(view), np.full(view.shape[0], WINDOW)
        # shorter than a full window: one window, right-padded to the common
        # table width with -1 ("no residue"), which encode() turns into the
        # MASK_VALUE rows that dynamic_padding would have written.
        row = np.full((1, WINDOW), -1, dtype=codes.dtype)
        row[0, : codes.size] = codes
        return row, np.array([codes.size])

    def score(self, sequences, on_invalid="warn"):
        """One median CANYA score per sequence, in input order.

        on_invalid: "warn" (default) reports the problem and returns nan for
        that sequence; "raise" stops at the first bad sequence; "nan" is silent.
        """
        if on_invalid not in ("warn", "raise", "nan"):
            raise ValueError("on_invalid must be 'warn', 'raise' or 'nan'")
        sequences = list(sequences)
        out = [float("nan")] * len(sequences)
        batch, pending = [], 0
        for i, seq in enumerate(sequences):
            try:
                codes, lengths = self.windows(seq)
            except InvalidSequence as exc:
                if on_invalid == "raise":
                    raise InvalidSequence(f"sequence {i}: {exc}") from None
                if on_invalid == "warn":
                    warnings.warn(f"CANYA: sequence {i} not scored: {exc}",
                                  RuntimeWarning, stacklevel=2)
                continue
            batch.append((i, codes, lengths))
            pending += codes.shape[0]
            if pending >= self.batch_size:
                self._run(batch, out)
                batch, pending = [], 0
        if batch:
            self._run(batch, out)
        return out

    def _run(self, batch, out):
        codes = np.concatenate([c for _, c, _ in batch])
        lengths = np.concatenate([l for _, _, l in batch])
        scores = self.forward(self.encode(codes, lengths))
        off = 0
        for i, c, _ in batch:
            n = c.shape[0]
            # summarize == "median" (CANYA's default) over the sequence's windows
            out[i] = float(scores[off] if n == 1 else np.median(scores[off:off + n]))
            off += n


_DEFAULT = None


def aggregation_score(sequence, on_invalid="warn"):
    """Convenience wrapper around a lazily created module-level Canya."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Canya()
    aggscore = _DEFAULT.score(sequence, on_invalid=on_invalid)
    return round(aggscore[0], 3) if isinstance(sequence, str) else aggscore