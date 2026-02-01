import json
import typing as T
from pathlib import Path
from collections import Counter
from itertools import product
import numpy as np


class Seqstat:

    def __init__(self, stat_from = None, seqtype = 'protein'):
        
        self.kmer_stat = {}
        self.seqtype = seqtype

        assert self.seqtype in ['protein', 'rna', 'dna'], "Wrong setype, must be 'protein', 'rna' or 'dna'"

        # Priority: Load from FASTA if provided
        if stat_from == None:
            print("No statistics file provided. \nPlease run calculate_background_distribution to generate statistics from a FASTA file or load statistics for a json.")
            
        elif Path(stat_from).exists():  
            if stat_from.split('.')[-1] == "json":
                try:
                    print(f"Loading statistics from {stat_from}")
                    with open(stat_from, 'r') as f:
                        self.kmer_stat = json.load(f)
                except Exception as e:
                    print(f"ERROR: Could not process {stat_from}.\
                          \nCalculate dictribution from a fasta file with \
                          calculate_background_distribution or provide valid JSON\n{e}")
                    pass

            elif stat_from.split('.')[-1] in ["fasta", "fa", "fas"]:
                try:
                    print(f"Calculating statistics from {stat_from}")
                    self.calculate_background_distribution(stat_from)

                except Exception as e:
                    print(f"ERROR: Could not process {stat_from}.\
                          \nCalculate dictribution from a fasta file with \
                          calculate_background_distribution or provide valid JSON\n{e}")
                    pass

        else:
            print(f"{stat_from} does not exist. Run calculate_background_distribution or provide valid JSON.")


    def _get_alphabet(self) -> str:
        """Return valid characters for the sequence type."""
        if self.seqtype == "protein":
            return "ACDEFGHIKLMNPQRSTVWY"
        elif self.seqtype == "rna":
            return "AUGC"
        elif self.seqtype == "dna":
            return "ATGC"

    def _all_kmers(self, k: int) -> list:
        """Generate all possible k-mers for the sequence type."""
        alphabet = self._get_alphabet()
        return [''.join(p) for p in product(alphabet, repeat=k)]

    @staticmethod
    def read_fasta_to_dict(fasta_path: str) -> dict:
            
            seq_dict = {}
            writedata = False
            seq = ""
            seq_id = None

            with open(fasta_path) as fh:
                for oline in fh:
                    if oline.startswith(">"):
                        if writedata and seq_id is not None:
                            seq_dict[seq_id] = seq.upper()
                            seq = ""
                        sline1 = oline.split()
                        seq_id = sline1[0].lstrip(">")
                        writedata = True
                    else:
                        seq += oline.strip()

            if seq_id is not None:
                seq_dict[seq_id] = seq.upper()

            return seq_dict


    def read_fasta_generator(self, fasta_path: str):
        """
        Generator to read FASTA file one sequence at a time.
        Saves memory compared to loading a full dict.
        """
        seq_id = None
        seq = []
        
        with open(fasta_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                
                if line.startswith(">"):
                    if seq_id:
                        yield "".join(seq).upper()
                    seq_id = line[1:].split()[0] # ID extraction
                    seq = []
                else:
                    seq.append(line)
            
            if seq_id and seq:
                yield "".join(seq).upper()


    def split2kmers(self, seq: str|list, k=3) -> list:
        seqlen = len(seq)
        assert 0 < k <= seqlen, f"Invalid k={k} for sequence of length {seqlen}"
        return [seq[i:i+k] for i in range(seqlen-k+1)]

    def calculate_probabilities(self, kmers) -> dict:
        num_kmers = len(kmers)
        return {i: j/num_kmers for i, j in Counter(kmers).items()}    

    def calculate_background_distribution(self, fasta_path: str) -> None:
        
        self.kmer_stat = {}
        counts = {
            1: Counter(),
            2: Counter(),
            3: Counter()
        }

        print("Parsing sequences...")
        count = 0
        
        for seq in self.read_fasta_generator(fasta_path):

            if len(seq) < 10:
                continue

            counts[1].update(self.split2kmers(seq, 1))
            counts[2].update(self.split2kmers(seq, 2))
            counts[3].update(self.split2kmers(seq, 3))
            count += 1
            
            if count % 100000 == 0:
                print(f"Processed {count} sequences...", end='\r')
            
        print(f"\nProcessed {count} sequences.")
        
        print("Filtering invalid k-mers...")

        # Prune invalid k-mers
        valid_chars = set(self._get_alphabet())

        for n in [1, 2, 3]:
            unique_kmers = list(counts[n].keys())
            for kmer in unique_kmers:
                # Check if this specific k-mer contains any bad char
                if set(kmer) - valid_chars:
                    del counts[n][kmer]

        # Convert counts to probabilities with Laplace smoothing
        def counts_to_probs_smoothed(counter, k, pseudocount=1):
            all_possible = self._all_kmers(k)
            total = sum(counter.values()) + pseudocount * len(all_possible)
            return {kmer: (counter.get(kmer, 0) + pseudocount) / total for kmer in all_possible}

        self.kmer_stat = {
            'monomers': counts_to_probs_smoothed(counts[1], 1),
            'dimers':   counts_to_probs_smoothed(counts[2], 2),
            'trimers':  counts_to_probs_smoothed(counts[3], 3)
        }

    def save_background_distribution(self, output_path):

        if self.kmer_stat: 
            with open(output_path, 'w') as f:
                json.dump(self.kmer_stat, f)


    def kullback_leibler(self, p, q):
        D_KL = sum([p_val * np.log(p_val / q[k]) for k, p_val in p.items() if k in q])
        return D_KL


    def n_gram_prior(self, sequence):

        P1_seq = self.calculate_probabilities(self.split2kmers(sequence, 1))
        P2_seq = self.calculate_probabilities(self.split2kmers(sequence, 2))
        P3_seq = self.calculate_probabilities(self.split2kmers(sequence, 3))

        # Calculate D_KL for each n-gram size
        energy_uni = self.kullback_leibler(P1_seq, self.kmer_stat['monomers'])
        energy_bi  = self.kullback_leibler(P2_seq, self.kmer_stat['dimers'])
        energy_tri = self.kullback_leibler(P3_seq, self.kmer_stat['trimers'])

        # Total N-gram Energy
        ngram_energy = energy_uni + energy_bi + energy_tri
        return  round(float(ngram_energy), 3)
        
