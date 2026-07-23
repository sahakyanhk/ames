#!/usr/bin/env python3
"""Standalone structure prediction for AMES.

Reads protein/nucleic/ligand queries from FASTA or JSON, predicts a structure
with a selectable engine (esmfold2 by default, alphafold3 alternative) and
writes the top structure plus confidence metrics for each query.

This script only *reuses* the AMES engine code (the engine runner modules,
``pdbutils`` and ``pdb_contacts``); it does not modify any AMES module.

Input formats
-------------
FASTA (-if/--ifasta): a single file or a directory. In a directory every file
ending in .fa/.fas/.fasta is read. Each file becomes one query; if it holds
several records they are predicted together as a complex (chains A, B, C ...).
The molecule type of each record is auto-detected from its alphabet.

JSON (-ij/--ijson): a single file or a directory of .json files. The
"convenient" chain-list format is used:

    # one query per file (output dir = file stem)
    [{"type": "protein", "sequence": "MKL...", "id": "A"},
     {"type": "rna",     "sequence": "AUGC",   "id": "B"}]

    # several named queries in one file (output dir = each key)
    {"complexA": [{"type": "protein", "sequence": "MKL..."}],
     "complexB": [{"type": "protein", "sequence": "..."},
                  {"type": "ligand",  "ccd_code": "ATP"}]}

Output
------
For every query a directory ``<output>/<name>/`` is created containing the
top predicted structure ``<name>.pdb`` and ``metrics.json`` (plddt, ptm, iptm,
iplddt, per-chain info). A combined ``<output>/summary.tsv`` is also written.
"""

import os
import sys
import json
import string
import argparse
from pathlib import Path

# Ensure src/ is on sys.path so sibling AMES modules are importable from anywhere
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from seqtools import Seqstat
import pdb_contacts as pc


FASTA_EXTS = (".fa", ".fas", ".fasta")
INTERFACE_PLDDT_CUTOFF = 8.0  # matches simparam.json default
_NUCLEIC_ALPHABET = set("ACGTUN")

# accepted --engine values mapped to the canonical runner key used in ames.py
ENGINE_ALIASES = {
    "esmfold2": "esmfold2",
    "alphafold3": "af3", "af3": "af3",
    "openfold3": "of3", "of3": "of3",
}


#==============================# INPUT PARSING #==============================#

def chain_id(index: int) -> str:
    """A, B, ... Z, then C27, C28, ... for very large complexes."""
    return string.ascii_uppercase[index] if index < 26 else f"C{index + 1}"


def guess_type(sequence: str) -> str:
    """Auto-detect molecule type from the sequence alphabet."""
    letters = set(sequence.upper())
    if letters and letters <= _NUCLEIC_ALPHABET:
        return "rna" if "U" in letters else "dna"
    return "protein"


def normalize_chains(raw_chains: list, query_name: str) -> list:
    """Validate/complete a chain-list query (assign ids and infer types)."""
    if not isinstance(raw_chains, list) or not raw_chains:
        raise ValueError(f"query '{query_name}' must be a non-empty list of chains")

    chains = []
    for i, raw in enumerate(raw_chains):
        chain = dict(raw)
        chain.setdefault("id", chain_id(i))

        if "type" not in chain:
            if "sequence" in chain:
                chain["type"] = guess_type(chain["sequence"])
            elif "ccd_code" in chain or "smiles" in chain:
                chain["type"] = "ligand"
            else:
                raise ValueError(
                    f"chain {chain['id']} in '{query_name}' needs a 'type', "
                    "'sequence', 'ccd_code' or 'smiles'")
        chain["type"] = chain["type"].lower()

        if chain["type"] != "ligand" and "sequence" not in chain:
            raise ValueError(f"chain {chain['id']} in '{query_name}' is missing 'sequence'")

        chains.append(chain)
    return chains


def fasta_to_query(path: Path) -> dict:
    """Convert a FASTA file into a chain-list query (a complex if multi-record)."""
    records = Seqstat.read_fasta_to_dict(str(path))
    if not records:
        raise ValueError(f"no sequences found in {path}")

    chains = [
        {"type": guess_type(seq), "sequence": seq, "id": chain_id(i)}
        for i, seq in enumerate(records.values())
    ]
    return {"name": path.stem, "chains": chains}


def json_to_queries(path: Path) -> list:
    """Convert a JSON file into one or more chain-list queries."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):  # a single query; name it after the file
        return [{"name": path.stem, "chains": normalize_chains(data, path.stem)}]

    if isinstance(data, dict):  # several named queries keyed by output name
        return [{"name": name, "chains": normalize_chains(chains, name)}
                for name, chains in data.items()]

    raise ValueError(f"{path}: top-level JSON must be a list or an object")


def collect_queries(ifasta: str | None, ijson: str | None) -> list:
    queries = []

    if ifasta:
        path = Path(ifasta)
        files = (sorted(p for p in path.iterdir() if p.suffix.lower() in FASTA_EXTS)
                 if path.is_dir() else [path])
        if not files:
            print(f"WARNING: no FASTA files ({'/'.join(FASTA_EXTS)}) found in {path}")
        for f in files:
            queries.append(fasta_to_query(f))

    if ijson:
        path = Path(ijson)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        if not files:
            print(f"WARNING: no JSON files found in {path}")
        for f in files:
            queries.extend(json_to_queries(f))

    # guard against colliding output directory names
    seen = {}
    for q in queries:
        name = q["name"]
        if name in seen:
            seen[name] += 1
            new_name = f"{name}_{seen[name]}"
            print(f"WARNING: duplicate query name '{name}', renaming to '{new_name}'")
            q["name"] = new_name
        else:
            seen[name] = 1

    return queries


#==================================# ENGINES #==================================#

def load_predictor(engine: str, num_diffusion: int):
    """Import the selected engine (loads the model) and return a predictor.

    The predictor takes (chains, name) and returns (pdb_text, plddt, ptm, iptm),
    keeping the single top-ranked structure. The N-best feature is reserved for
    a future version.
    """
    if engine == "esmfold2":
        from esmfold2_runner import (model, inp_type,
                                      ESMFold2InputBuilder, StructurePredictionInput)
        from pdbutils import cif2pdb

        def predict(chains, name):
            sequences = []
            for c in chains:
                if c["type"] == "ligand":
                    sequences.append(inp_type["ligand"](id=c["id"], ccd=[c["ccd_code"]]))
                else:
                    sequences.append(inp_type[c["type"]](id=c["id"], sequence=c["sequence"]))

            spi = StructurePredictionInput(sequences=sequences)
            results = ESMFold2InputBuilder().fold(model, spi,
                                                  num_loops=3,
                                                  num_sampling_steps=50,
                                                  num_diffusion_samples=num_diffusion,
                                                  seed=0)
            results = results if isinstance(results, list) else [results]
            best = max(results, key=lambda r: 0.8 * r.iptm + 0.2 * r.ptm)

            pdb = cif2pdb(best.complex.to_mmcif())
            return (pdb, round(float(best.plddt.mean()), 3),
                    round(float(best.ptm), 3), round(float(best.iptm), 3))

        return predict

    if engine == "af3":
        import numpy as np
        from alphafold3_runner import create_fold_input, run_inference, model_runner
        from pdbutils import cif2pdb

        def predict(chains, name):
            fold_input = create_fold_input(seq_list=chains, name=name)
            results_for_seeds = run_inference(fold_input, model_runner)

            best, best_score = None, None
            for results_for_seed in results_for_seeds:
                for result in results_for_seed.inference_results:
                    score = float(result.metadata["ranking_score"])
                    if best_score is None or score > best_score:
                        best_score, best = score, result

            plddt = float(best.predicted_structure.atom_b_factor.mean()) * 0.01
            ptm = float(best.metadata["ptm"])
            iptm = float(best.metadata["iptm"])
            iptm = 0.0 if np.isnan(iptm) else iptm

            pdb = cif2pdb(best.predicted_structure.to_mmcif())
            return pdb, round(plddt, 3), round(ptm, 3), round(iptm, 3)

        return predict

    if engine == "of3":
        from openfold3_runner import of3_runner

        def predict(chains, name):
            structures, plddts, ptms, iptms = of3_runner(chains)
            return structures[0], plddts[0], ptms[0], iptms[0]

        return predict

    raise ValueError(f"unknown engine '{engine}'")


def compute_iplddt(pdb_text: str, chains: list) -> float:
    """Interface pLDDT of the first chain vs. all remaining chains (0 if monomer)."""
    ids = [c["id"] for c in chains]
    if len(ids) < 2:
        return 0.0
    iplddt = pc.interface_plddt(pdb_text,
                                chain1=ids[0],
                                chain2=",".join(ids[1:]),
                                cutoff=INTERFACE_PLDDT_CUTOFF)
    return round(iplddt * 0.01, 3)


#==================================# OUTPUT #==================================#

SUMMARY_COLUMNS = ["name", "engine", "n_chains", "plddt", "ptm", "iptm", "iplddt"]


def write_query_output(outroot: Path, query: dict, pdb: str, metrics: dict) -> None:
    qdir = outroot / query["name"]
    qdir.mkdir(parents=True, exist_ok=True)

    (qdir / f"{query['name']}.pdb").write_text(pdb)
    with open(qdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def write_summary(outroot: Path, rows: list) -> None:
    lines = ["\t".join(SUMMARY_COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row[c]) for c in SUMMARY_COLUMNS))
    (outroot / "summary.tsv").write_text("\n".join(lines) + "\n")


#===================================# MAIN #===================================#

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMES standalone structure prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-if", "--ifasta",
                        help="input FASTA file or directory (.fa/.fas/.fasta)")
    parser.add_argument("-ij", "--ijson",
                        help="input JSON file or directory (.json), chain-list format")
    parser.add_argument("-o", "--output", required=True,
                        help="output directory for predicted structures and metrics")
    parser.add_argument("--engine", default="esmfold2", choices=list(ENGINE_ALIASES),
                        help="structure prediction engine")
    parser.add_argument("-nd", "--number_diffusion", type=int, default=1,
                        help="number of diffusion samples (only the top structure is saved)")

    args = parser.parse_args()
    if not args.ifasta and not args.ijson:
        parser.error("provide at least one input: -if/--ifasta and/or -ij/--ijson")
    return args


def main() -> None:
    args = parse_args()
    engine = ENGINE_ALIASES[args.engine]

    queries = collect_queries(args.ifasta, args.ijson)
    if not queries:
        sys.exit("ERROR: no valid queries found in the provided input(s)")

    outroot = Path(args.output)
    outroot.mkdir(parents=True, exist_ok=True)

    print(f"#engine={engine}  queries={len(queries)}  output={outroot}")
    predict = load_predictor(engine, args.number_diffusion)

    summary_rows = []
    for i, query in enumerate(queries, start=1):
        chains = query["chains"]
        pdb, plddt, ptm, iptm = predict(chains, query["name"])
        iplddt = compute_iplddt(pdb, chains)

        metrics = {
            "name": query["name"],
            "engine": engine,
            "num_diffusion": args.number_diffusion,
            "n_chains": len(chains),
            "plddt": plddt,
            "ptm": ptm,
            "iptm": iptm,
            "iplddt": iplddt,
            "chains": [{"id": c["id"], "type": c["type"],
                        "len": len(c.get("sequence", ""))} for c in chains],
        }
        write_query_output(outroot, query, pdb, metrics)
        summary_rows.append({c: metrics[c] for c in SUMMARY_COLUMNS})

        print(f"[{i}/{len(queries)}] {query['name']:<24} "
              f"chains={len(chains)} plddt={plddt} ptm={ptm} iptm={iptm} iplddt={iplddt}")

    write_summary(outroot, summary_rows)
    print(f"#done, summary written to {outroot / 'summary.tsv'}")


if __name__ == "__main__":
    main()


