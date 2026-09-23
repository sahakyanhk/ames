from __future__ import annotations

import os
import subprocess
import threading
from contextlib import ExitStack, suppress
from dataclasses import dataclass

__all__ = ["Alignment", "tmalign", "USALIGN"]

USALIGN = os.environ.get("USALIGN_BIN", "USalign")


@dataclass(frozen=True)
class Alignment:
    """Result of one US-align run (`-outfmt 2` columns)."""

    tmscore: float        # TM-score under the requested normalization
    seqid: float          # n_identical / n_aligned over the aligned pairs
    alilen: int           # number of aligned residue pairs
    rmsd: float
    tm_query: float       # TM-score normalized by the query length
    tm_target: float      # TM-score normalized by the target length
    len_query: int
    len_target: int

    def __iter__(self):
        """Allow `tmscore, seqid, alilen = tmalign(...)`."""
        return iter((self.tmscore, self.seqid, self.alilen))


def tmalign(query, target, qchain=None, tchain=None, *,
                        normalize="query", fast=False):
    """Align protein `query` onto `target` and return an :class:`Alignment`.

    query, target  path to a PDB/mmCIF file (str, bytes or os.PathLike) or the
                   contents of one as str/bytes.
    qchain, tchain chain to align; the first chain if None, "_" if blank.
    normalize      length the reported `tmscore` is normalized by: "query",
                   "target", "shorter" (highest) or "longer" (lowest).
    fast           use the faster, slightly less accurate search.
    """
    if normalize not in ("query", "target", "shorter", "longer"):
        raise ValueError(f"unknown normalize={normalize!r}")

    with ExitStack() as stack:
        # -mol prot -ter 2: proteins only, a single chain from each structure.
        argv = [USALIGN, None, None, "-outfmt", "2", "-mol", "prot", "-ter", "2"]
        if fast:
            argv.append("-fast")

        pass_fds, writers = [], []
        stack.callback(lambda: [_close(fd) for fd in pass_fds])

        for slot, (obj, chain, name) in enumerate(
                ((query, qchain, "query"), (target, tchain, "target")), 1):
            path, data = _classify(obj, name)
            if chain is not None:
                argv += [f"-chain{slot}", chain]
            if path is not None:
                argv[slot] = path
                continue
            rfd, wfd = os.pipe()
            pass_fds.append(rfd)
            writers.append(_feeder(wfd, data))
            argv[slot] = f"/dev/fd/{rfd}"
            is_cif = data.startswith(b"data_") or b"_atom_site." in data[:65536]
            argv += [f"-infmt{slot}", "3" if is_cif else "0"]

        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, pass_fds=pass_fds,
                                    text=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{USALIGN!r} not found in PATH; set $USALIGN_BIN to its location"
            ) from None

        # The child holds its own copies now; dropping ours lets a feeder see
        # EOF (as BrokenPipeError) instead of blocking if US-align dies early.
        while pass_fds:
            _close(pass_fds.pop())

        stdout, stderr = proc.communicate()
        for t in writers:
            t.join(timeout=5)

    # columns: PDBchain1 PDBchain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali
    row = next((line.split("\t") for line in stdout.splitlines()
                if not line.startswith("#") and line.count("\t") >= 10), None)
    if proc.returncode != 0 or row is None:
        why = (f"exited with {proc.returncode}" if proc.returncode else
               "produced no alignment (unparsable structure, or no such chain?)")
        raise RuntimeError(f"US-align {why}\n{(stderr.strip() or stdout.strip())[:2000]}")

    tm_query, tm_target = float(row[2]), float(row[3])
    return Alignment(
        tmscore={"query": tm_query,
                 "target": tm_target,
                 "shorter": max(tm_query, tm_target),
                 "longer": min(tm_query, tm_target)}[normalize],
        seqid=float(row[7]),
        alilen=int(row[10]),
        rmsd=float(row[4]),
        tm_query=tm_query,
        tm_target=tm_target,
        len_query=int(row[8]),
        len_target=int(row[9]),
    )


def _classify(obj, name):
    """Return (path, None) for a file, or (None, bytes) for structure text."""
    if isinstance(obj, os.PathLike):
        return os.fspath(obj), None
    if isinstance(obj, str):
        data = obj.encode()
    elif isinstance(obj, bytes):
        data = obj
    else:
        raise TypeError(f"{name} must be a path or PDB/mmCIF text, "
                        f"got {type(obj).__name__}")
    if b"\n" not in data:  # only a one-liner can plausibly be a file name
        path = os.fsdecode(obj)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}: no such file: {path!r}")
        return path, None
    return None, data


def _feeder(wfd, data):
    """Start a daemon thread that pushes `data` into `wfd` and closes it."""

    def run():
        try:
            fh = open(wfd, "wb")
        except OSError:
            return _close(wfd)
        with suppress(OSError), fh:  # US-align may exit before reading it all
            fh.write(data)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _close(fd):
    with suppress(OSError):
        os.close(fd)
