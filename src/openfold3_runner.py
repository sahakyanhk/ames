"""Sequential wrapper around native OpenFold3 inference."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from pathlib import Path
import string
import sys
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

logging.disable(logging.WARNING)
warnings.simplefilter("ignore")


DEFAULT_OPENFOLD_CACHE = Path(
    os.environ.get("OPENFOLD_CACHE", "/data/saakyanh2/.openfold3")
).expanduser()
DEFAULT_OF3_PYTHON = Path(
    os.environ.get(
        "OPENFOLD3_PYTHON",
        "/vf/users/saakyanh2/micromamba/envs/of3/bin/python",
    )
).expanduser()
DEFAULT_OUTPUT_ROOT = Path("outputs/openfold3_sequential_raw")
DEFAULT_RUNNER_NAME = "of3_sequential_runner"
DEFAULT_SEEDS = (42,)
_TYPE_ALIASES = {"ion": "ligand"}
_DEFAULT_MODEL_UPDATE = {
    "settings": {
        "memory": {
            "train": {"use_deepspeed_evo_attention": False},
            "eval": {"use_deepspeed_evo_attention": False},
        }
    }
}

if "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = str(
        Path("/tmp") / os.environ.get("USER", "openfold3") / "triton"
    )


@dataclass(frozen=True)
class OF3Imports:
    torch_gpu_setup: Any
    write_structure: Any
    Chain: Any
    Query: Any
    InferenceQuerySet: Any
    InferenceExperimentConfig: Any
    InferenceExperimentRunner: Any


@lru_cache(maxsize=1)
def _get_of3() -> OF3Imports:
    try:
        from openfold3.core.data.io.structure.cif import write_structure
        from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
        from openfold3.entry_points.import_utils import _torch_gpu_setup
        from openfold3.entry_points.validator import InferenceExperimentConfig
        from openfold3.projects.of3_all_atom.config.inference_query_format import (
            Chain,
            InferenceQuerySet,
            Query,
        )
    except Exception as exc:
        raise RuntimeError(
            "of3_sequential_runner must be executed from the OpenFold3 "
            f"environment. Current Python is {sys.executable!s}. "
            f"Use {DEFAULT_OF3_PYTHON!s} or the matching Jupyter kernel."
        ) from exc

    return OF3Imports(
        torch_gpu_setup=_torch_gpu_setup,
        write_structure=write_structure,
        Chain=Chain,
        Query=Query,
        InferenceQuerySet=InferenceQuerySet,
        InferenceExperimentConfig=InferenceExperimentConfig,
        InferenceExperimentRunner=InferenceExperimentRunner,
    )


def _default_chain_id(index: int) -> str:
    if index < len(string.ascii_uppercase):
        return string.ascii_uppercase[index]
    return f"C{index + 1}"


def _normalize_chain(entry: Mapping[str, Any], index: int) -> dict[str, Any]:
    chain = dict(entry)
    molecule_type = str(chain.get("type", chain.get("molecule_type", "protein"))).lower()
    molecule_type = _TYPE_ALIASES.get(molecule_type, molecule_type)

    chain_ids = chain.pop("id", chain.get("chain_ids", _default_chain_id(index)))
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]

    normalized = {
        "molecule_type": molecule_type,
        "chain_ids": chain_ids,
    }

    if "ccd_code" in chain:
        normalized["ccd_codes"] = [chain.pop("ccd_code")]
    elif "ccd_codes" in chain:
        ccd_codes = chain.pop("ccd_codes")
        normalized["ccd_codes"] = ccd_codes if isinstance(ccd_codes, list) else [ccd_codes]

    if "smiles" in chain:
        normalized["smiles"] = chain["smiles"]

    sequence = chain.get("sequence")
    if isinstance(sequence, str):
        sequence = sequence.upper()
        if molecule_type == "rna":
            sequence = sequence.replace("T", "U")
        normalized["sequence"] = sequence

    return normalized


def make_query(
    seq_list: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    name: str,
):
    of3 = _get_of3()
    chain_entries = [seq_list] if isinstance(seq_list, Mapping) else list(seq_list)
    chains = [
        of3.Chain.model_validate(_normalize_chain(entry, index))
        for index, entry in enumerate(chain_entries)
    ]
    return of3.Query(
        query_name=name,
        chains=chains,
        use_msas=False,
        use_paired_msas=False,
        use_main_msas=False,
    )


def _normalize_output_dir(output_dir: str | os.PathLike[str] | None, name: str) -> Path:
    if output_dir is None:
        return DEFAULT_OUTPUT_ROOT / name
    return Path(output_dir)


def _make_runner(
    name: str = DEFAULT_RUNNER_NAME,
    output_dir: str | os.PathLike[str] | None = None,
):
    of3 = _get_of3()
    of3.torch_gpu_setup()

    config = of3.InferenceExperimentConfig(
        experiment_settings={
            "seeds": list(DEFAULT_SEEDS),
            "output_dir": _normalize_output_dir(output_dir, name),
        },
        data_module_args={
            "num_workers": 0,
            "num_workers_validation": 0,
        },
        model_update={"custom": deepcopy(_DEFAULT_MODEL_UPDATE)},
        cache_path=DEFAULT_OPENFOLD_CACHE,
    )
    return of3.InferenceExperimentRunner(
        config,
        num_diffusion_samples=1,
        use_msa_server=False,
        use_templates=False,
        output_dir=_normalize_output_dir(output_dir, name),
    )


def _reset_query_state(runner) -> None:
    runner.__dict__.pop("data_module_config", None)
    runner.__dict__.pop("lightning_data_module", None)


def _first_prediction(predictions: Any) -> Any:
    if not isinstance(predictions, list):
        return predictions
    if not predictions:
        raise RuntimeError("OpenFold3 returned no predictions.")
    first = predictions[0]
    if isinstance(first, list):
        if not first:
            raise RuntimeError("OpenFold3 returned an empty prediction batch.")
        return first[0]
    return first


def run_query(
    seq_list: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    name: str = "query",
    runner=None,
    write_outputs: bool = False,
):
    if runner is None:
        runner = OF3_RUNNER

    of3 = _get_of3()
    runner.inference_query_set = of3.InferenceQuerySet(
        seeds=list(DEFAULT_SEEDS),
        queries={name: make_query(seq_list, name)},
    )
    _reset_query_state(runner)
    if not write_outputs:
        runner.__dict__["callbacks"] = []

    predictions = runner.trainer.predict(
        model=runner.lightning_module,
        datamodule=runner.lightning_data_module,
        return_predictions=True,
    )
    return _first_prediction(predictions)


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    array = _to_numpy(value).reshape(-1)
    if array.size == 0:
        return default
    scalar = float(array[0])
    return scalar if np.isfinite(scalar) else default


def _sample_scores(value: Any, sample_count: int) -> np.ndarray:
    scores = np.atleast_1d(_to_numpy(value)).astype(float, copy=False).reshape(-1)
    if scores.size == 0:
        scores = np.full(sample_count, -np.inf, dtype=float)
    elif scores.size == 1 and sample_count > 1:
        scores = np.repeat(scores, sample_count)
    elif scores.size < sample_count:
        scores = np.pad(scores, (0, sample_count - scores.size), constant_values=-np.inf)
    else:
        scores = scores[:sample_count]
    return np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def _take_batch_item(value: Any, batch_index: int = 0) -> Any:
    if isinstance(value, dict):
        return {key: _take_batch_item(item, batch_index) for key, item in value.items()}
    array = _to_numpy(value)
    if array.ndim == 0:
        return array
    return array[batch_index]


def _take_sample_item(value: Any, sample_index: int, sample_count: int) -> Any:
    if isinstance(value, dict):
        return {
            key: _take_sample_item(item, sample_index, sample_count)
            for key, item in value.items()
        }
    array = _to_numpy(value)
    if array.ndim == 0:
        return array
    if array.shape[0] == sample_count:
        return array[sample_index]
    return array


def _prediction_to_pdb(atom_array: Any, predicted_coords: Any, plddt: Any) -> str:
    atom_array = atom_array.copy()
    atom_array.coord = _to_numpy(predicted_coords)
    atom_array.set_annotation("b_factor", _to_numpy(plddt))
    with tempfile.TemporaryDirectory(prefix="openfold3_pdb_") as tmpdir:
        output_path = Path(tmpdir) / "prediction.pdb"
        _get_of3().write_structure(atom_array, output_path, include_bonds=True)
        return output_path.read_text()


def prediction_to_ames_output(prediction: tuple[Any, Any]) -> tuple[str, float, float, float]:
    batch, outputs = prediction
    atom_array = batch["atom_array"][0]
    confidence_scores = _take_batch_item(outputs["confidence_scores"])
    predicted_coords = _to_numpy(outputs["atom_positions_predicted"][0])

    if predicted_coords.ndim == 2:
        predicted_coords = predicted_coords[None, ...]

    sample_count = predicted_coords.shape[0]
    ranking_scores = _sample_scores(
        confidence_scores.get("sample_ranking_score"),
        sample_count=sample_count,
    )
    best_index = int(np.argmax(ranking_scores))

    best_confidence = _take_sample_item(confidence_scores, best_index, sample_count)
    best_plddt = np.nan_to_num(
        _to_numpy(best_confidence.get("plddt")),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        _prediction_to_pdb(atom_array, predicted_coords[best_index], best_plddt),
        round(float(best_plddt.mean()) * 0.01, 3),
        round(_to_float(best_confidence.get("ptm")), 3),
        round(_to_float(best_confidence.get("iptm")), 3),
    )


def predict_raw(
    seq_list: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    name: str = "query",
):
    return run_query(seq_list, name=name)


def _ames_entry_to_query(seq_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    query = [
        {
            "type": seq_data["seq1"]["type"],
            "sequence": seq_data["seq1"]["sequence"],
            "id": "A",
        }
    ]

    if "seq2" in seq_data:
        query.append(
            {
                "type": seq_data["seq2"]["type"],
                "sequence": seq_data["seq2"]["sequence"],
                "id": "B",
            }
        )

    if "ligand" in seq_data:
        for index, ligand in enumerate(seq_data["ligand"], start=len(query)):
            query.append(
                {
                    "type": "ligand",
                    "ccd_code": ligand,
                    "id": _default_chain_id(index),
                }
            )

    return query


def _normalize_input_fold_list(
    input_fold_list: Sequence[Any] | Mapping[str, Any],
) -> list[list[dict[str, Any]]]:
    entries = [input_fold_list] if isinstance(input_fold_list, Mapping) else list(input_fold_list)
    if not entries:
        return []

    first = entries[0]
    if isinstance(first, Mapping) and "seq1" in first:
        return [_ames_entry_to_query(entry) for entry in entries]
    if isinstance(first, Mapping):
        return [list(entries)]
    return [list(query) for query in entries]


def of3_sequential_runner(
    input_fold_list: Sequence[Any] | Mapping[str, Any],
) -> tuple[list[str], list[float], list[float], list[float]]:
    queries = _normalize_input_fold_list(input_fold_list)
    if not queries:
        return [], [], [], []

    ames_outputs = []
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            for index, query in enumerate(queries, start=1):
                prediction = run_query(query, name=f"id{index}")
                ames_outputs.append(prediction_to_ames_output(prediction))

    return (
        [item[0] for item in ames_outputs],
        [item[1] for item in ames_outputs],
        [item[2] for item in ames_outputs],
        [item[3] for item in ames_outputs],
    )


OF3_RUNNER = _make_runner()
OF3_RUNNER.setup()

of3_runner = of3_sequential_runner


__all__ = [
    "OF3_RUNNER",
    "make_query",
    "run_query",
    "prediction_to_ames_output",
    "predict_raw",
    "of3_runner",
    "of3_sequential_runner",
]
