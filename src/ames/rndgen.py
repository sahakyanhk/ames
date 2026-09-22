#!/usr/bin/env python3
"""Generate random sequences / sequence pairs as input for fold.py.

Each of the two chains is either the keyword ``random`` (a sequence is drawn
from the alphabet of its type) or an explicit list of sequences given on the
command line (comma- and/or space-separated). ``-N`` is the number of queries
generated *per combination* of the listed sequences: with two lists every
seq1 is paired with every seq2 (cross product) and each pair yields N queries.

Output is always a directory:
  * --otype json  : one file <output>/<prefix>.json, the chain-list format
                    read by ``fold.py -ij``
  * --otype fasta : a single <output>/<prefix>.fasta when only seq1 is used,
                    otherwise one <output>/<name>.fasta per complex
                    (both readable by ``fold.py -if``)
Without -o everything is printed to stdout instead.

Examples:
    rndgen.py -N 10 -s1 random --seq1_type protein --seq1_len 50 -o rnd/
    rndgen.py -N 10 -s1 random --seq1_len 40-120 -s2 GGGAUCCUUAAGG -o cplx/
    rndgen.py -N 5 -s1 random --seq1_len 50 -s2 "MKLLVV,GSHMQRK" --otype fasta -o cplx/
    rndgen.py -s1 "MKLLVV GSHMQRK" -o seqs/          # just repackage a list
"""

import re
import sys
import json
import random
import argparse
from pathlib import Path

ALPHABETS = {
    "protein": "ACDEFGHIKLMNPQRSTVWY",
    "rna": "ACGU",
    "dna": "ACGT",
}

RANDOM_KEYWORD = "random"
_NUCLEIC_ALPHABET = set("ACGTUN")


#=================================# SEQUENCES #=================================#

def guess_type(sequence: str) -> str:
    """Auto-detect molecule type from the sequence alphabet (as in fold.py)."""
    letters = set(sequence.upper())
    if letters and letters <= _NUCLEIC_ALPHABET:
        return "rna" if "U" in letters else "dna"
    return "protein"


def parse_length(spec: str, flag: str) -> tuple:
    """'50' -> (50, 50); '40-120' -> (40, 120). Returns an inclusive range."""
    match = re.fullmatch(r"\s*(\d+)\s*(?:[-:]\s*(\d+)\s*)?", spec)
    if not match:
        raise ValueError(f"{flag}: expected an integer or a range like 40-120, got '{spec}'")

    low = int(match.group(1))
    high = int(match.group(2)) if match.group(2) else low
    if low < 1:
        raise ValueError(f"{flag}: length must be >= 1")
    if high < low:
        raise ValueError(f"{flag}: range '{spec}' has max < min")
    return low, high


def random_sequence(seq_type: str, length_range: tuple) -> str:
    length = random.randint(*length_range)
    return "".join(random.choices(ALPHABETS[seq_type], k=length))


def split_sequences(values: list) -> list:
    """Accept sequences separated by spaces and/or commas, in any mix."""
    return [s.upper() for s in re.split(r"[,\s]+", " ".join(values)) if s]


class ChainSpec:
    """One side of a query: either a random generator or a fixed sequence list."""

    def __init__(self, values, seq_type, length, chain_id, flag):
        self.id = chain_id
        self.is_random = len(values) == 1 and values[0].lower() == RANDOM_KEYWORD

        if self.is_random:
            if length is None:
                raise ValueError(f"{flag} is random: {flag}_len is required")
            self.type = seq_type or "protein"
            self.length_range = parse_length(length, f"{flag}_len")
            self.sequences = [None]  # a single "slot", filled per query
        else:
            if length is not None:
                raise ValueError(f"{flag}_len only applies when {flag} is 'random'")
            self.sequences = split_sequences(values)
            if not self.sequences:
                raise ValueError(f"{flag}: no sequences given")
            self.type = seq_type or guess_type(self.sequences[0])
            self._check_alphabet(flag, seq_type is not None)

    def _check_alphabet(self, flag, explicit):
        allowed = set(ALPHABETS[self.type])
        for seq in self.sequences:
            bad = sorted(set(seq) - allowed)
            if bad:
                origin = "given" if explicit else "detected"
                print(f"WARNING: {flag} sequence has letters {''.join(bad)} outside the "
                      f"{self.type} alphabet ({origin} type)", file=sys.stderr)

    def n_slots(self) -> int:
        return len(self.sequences)

    def chain(self, slot: int) -> dict:
        seq = random_sequence(self.type, self.length_range) if self.is_random \
            else self.sequences[slot]
        return {"type": self.type, "sequence": seq, "id": self.id}


def build_queries(specs: list, n: int, prefix: str) -> dict:
    """Cross product over the fixed-sequence slots, N queries per combination."""
    slot_counts = [s.n_slots() for s in specs]
    combos = [()]
    for count in slot_counts:
        combos = [combo + (i,) for combo in combos for i in range(count)]

    queries = {}
    for c_idx, combo in enumerate(combos, start=1):
        for i in range(1, n + 1):
            if len(combos) == 1:      # one combination: complex1, complex2, ...
                name = f"{prefix}{i}"
            elif n == 1:              # one query each: complex1, complex2, ...
                name = f"{prefix}{c_idx}"
            else:                     # both: complex1_1, complex1_2, complex2_1, ...
                name = f"{prefix}{c_idx}_{i}"
            queries[name] = [spec.chain(slot) for spec, slot in zip(specs, combo)]
    return queries


#==================================# OUTPUT #==================================#

def fasta_records(name: str, chains: list, tag_chains: bool) -> str:
    lines = []
    for chain in chains:
        header = f"{name}_{chain['id']}" if tag_chains else name
        lines.append(f">{header}\n{chain['sequence']}")
    return "\n".join(lines)


def write_output(queries: dict, outdir, otype: str, prefix: str) -> None:
    n_chains = max(len(chains) for chains in queries.values())
    outroot = Path(outdir) if outdir else None
    if outroot:
        outroot.mkdir(parents=True, exist_ok=True)

    if otype == "json":
        text = json.dumps(queries, indent=2)
        if not outroot:
            print(text)
            return
        path = outroot / f"{prefix}.json"
        path.write_text(text + "\n")
        print(f"#wrote {len(queries)} queries to {path}", file=sys.stderr)
        return

    # fasta: single file for single-chain queries, one file per complex otherwise
    if not outroot:
        print("\n".join(fasta_records(name, chains, n_chains > 1)
                        for name, chains in queries.items()))
        return

    if n_chains == 1:
        path = outroot / f"{prefix}.fasta"
        path.write_text("\n".join(fasta_records(name, chains, False)
                                  for name, chains in queries.items()) + "\n")
        print(f"#wrote {len(queries)} sequences to {path}", file=sys.stderr)
    else:
        for name, chains in queries.items():
            (outroot / f"{name}.fasta").write_text(fasta_records(name, chains, True) + "\n")
        print(f"#wrote {len(queries)} complexes to {outroot}/", file=sys.stderr)


#===================================# MAIN #===================================#

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random sequences or sequence pairs as fold.py input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", type=int, default=10,
                        help="number of queries per combination of listed sequences")
    parser.add_argument("-s1", "--seq1", nargs="+", required=True,
                        help="'random', or sequence(s) separated by commas and/or spaces")
    parser.add_argument("-s2", "--seq2", nargs="+",
                        help="second chain: 'random', or sequence(s); omit for single-chain queries")
    parser.add_argument("--seq1_type", choices=list(ALPHABETS),
                        help="molecule type of seq1 (default: protein if random, else auto-detected)")
    parser.add_argument("--seq2_type", choices=list(ALPHABETS),
                        help="molecule type of seq2 (default: protein if random, else auto-detected)")
    parser.add_argument("--seq1_len", help="length of seq1, e.g. 50 or 40-120; only if seq1 is random")
    parser.add_argument("--seq2_len", help="length of seq2, e.g. 50 or 40-120; only if seq2 is random")
    parser.add_argument("-o", "--output", default=None,
                        help="output directory (printed to stdout if omitted)")
    parser.add_argument("--otype", default="json", choices=("json", "fasta"),
                        help="output format: fold.py chain-list JSON, or FASTA")
    parser.add_argument("--prefix", default="input",
                        help="query name prefix (complex1, complex2, ...)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible output")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.N < 1:
        parser.error("-N must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    try:
        specs = [ChainSpec(args.seq1, args.seq1_type, args.seq1_len, "A", "--seq1")]
        if args.seq2:
            specs.append(ChainSpec(args.seq2, args.seq2_type, args.seq2_len, "B", "--seq2"))
        elif args.seq2_len or args.seq2_type:
            raise ValueError("--seq2_len/--seq2_type given without --seq2")
    except ValueError as err:
        sys.exit(f"ERROR: {err}")

    n = args.N
    if n > 1 and not any(spec.is_random for spec in specs):
        print(f"WARNING: no random chain, -N {n} would only duplicate the listed "
              "sequences; generating one query per combination", file=sys.stderr)
        n = 1

    queries = build_queries(specs, n, args.prefix)
    write_output(queries, args.output, args.otype, args.prefix)


if __name__ == "__main__":
    main()
