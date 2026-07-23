#!/usr/bin/env python3
"""Generate an input JSON of N queries for predict.py.

Modes (output is the chain-list format read by predict.py -ij, keyed by name):
  * -s with -rt/-rl : per static sequence, N complexes of random chain A +
                      static chain B (e.g. random protein + static RNA).
  * -s without -rt/-rl : one single-chain query per static sequence (-N ignored)
                      -- just prepares input files from a list of sequences.
  * no -s : N single random-sequence queries (chain A only).

Examples:
    rndgen.py -N 10 -s GGGAUCCUUAAGG -st rna -rt protein -rl 50 -o complexes.json
    rndgen.py -s "MKLLVV,GSHMQRK" -st protein -o queries.json   # one query/seq
    rndgen.py -N 10 -rt protein -rl 50 -o randoms.json          # random only
"""

import re
import sys
import json
import random
import argparse

ALPHABETS = {
    "protein": "ACDEFGHIKLMNPQRSTVWY",
    "rna": "ACGU",
    "dna": "ACGT",
}


def random_sequence(seq_type: str, length: int) -> str:
    return "".join(random.choices(ALPHABETS[seq_type], k=length))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a predict.py input JSON: complexes, plain sequences, or random sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", type=int, default=10, help="number of complexes")
    parser.add_argument("-s", "--static_seq", nargs="+",
                        help="fixed sequence(s), space- or comma-separated; N complexes are generated per sequence. "
                             "If omitted, only random sequences are generated")
    parser.add_argument("-st", "--static_type", default="rna", choices=list(ALPHABETS),
                        help="molecule type of the static sequence")
    parser.add_argument("-rt", "--random_type", default=None, choices=list(ALPHABETS),
                        help="molecule type of the random sequence (default: protein)")
    parser.add_argument("-rl", type=int, default=None, help="length of the random sequence (default: 50)")
    parser.add_argument("--prefix", default="complex", help="prefix for names JSON file (default: complex1, complex2, ...)")
    parser.add_argument("-o", "--output", help="output JSON file (printed to stdout if omitted)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # a random chain is only added when -rt or -rl is given (or no static seq at all)
    random_requested = args.random_type is not None or args.rl is not None
    random_type = args.random_type or "protein"
    random_len = args.rl if args.rl is not None else 50

    def random_chain():
        return {"type": random_type, "sequence": random_sequence(random_type, random_len), "id": "A"}

    complexes = {}

    if args.static_seq:
        # accept one or many static sequences, separated by spaces and/or commas
        static_seqs = [s.upper() for s in re.split(r"[,\s]+", " ".join(args.static_seq)) if s]

        if random_requested:
            # complex per static sequence: random chain A + static chain B
            multi = len(static_seqs) > 1
            for s_idx, static_seq in enumerate(static_seqs, start=1):
                for i in range(1, args.N + 1):
                    name = f"{args.prefix}{s_idx}_{i}" if multi else f"{args.prefix}{i}"
                    complexes[name] = [
                        random_chain(),
                        {"type": args.static_type, "sequence": static_seq, "id": "B"},
                    ]
        else:
            # only static sequences given: one single-chain query per sequence (-N ignored)
            for s_idx, static_seq in enumerate(static_seqs, start=1):
                complexes[f"{args.prefix}{s_idx}"] = [
                    {"type": args.static_type, "sequence": static_seq, "id": "A"},
                ]
    else:
        # no static sequence: emit single random-sequence queries
        for i in range(1, args.N + 1):
            complexes[f"{args.prefix}{i}"] = [random_chain()]

    text = json.dumps(complexes, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        if args.static_seq and random_requested:
            kind = "complexes"
        elif args.static_seq:
            kind = "sequences"
        else:
            kind = "random sequences"
        print(f"#wrote {len(complexes)} {kind} to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
