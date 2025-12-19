import os
import json
import typing as T
from pathlib import Path
from collections import Counter
import numpy as np

class Seqstat:

    DEFAULT_KMER_STAT = 'data/seqstat.json'

    def __init__(self, stat_from_fasta: T.Optional[str] = None):
        self.kmer_stat = {}
        
        # Priority: Load from FASTA if provided
        if stat_from_fasta:  
            try:
                print(f"Calculating new statistics from {stat_from_fasta}")
                # Optimization: Do not load all sequences into memory at once
                self.kmer_stat = self.calculate_background_distribution(stat_from_fasta)
                
                with open(f"seq_stat_from_{os.path.basename(stat_from_fasta)}", 'w') as f:
                    json.dump(self.kmer_stat, f)

            except Exception as e:
                print(f"ERROR: Could not process {stat_from_fasta}.\n{e}")
                raise e # Re-raise to stop execution if init fails
        
        # Fallback: Load from JSON
        elif Path(self.DEFAULT_KMER_STAT).exists():  
            print(f"Loading statistics from {self.DEFAULT_KMER_STAT}")
            with open(self.DEFAULT_KMER_STAT, 'r') as f:
                self.kmer_stat = json.load(f)
        else:
            print("Warning: No stats loaded. Run background_distribution or provide valid JSON.")


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
        assert 0 < k <= seqlen // 2 # make sure k-mer size is 2x smaller than initital string
        return [seq[i:i+k] for i in range(seqlen-k+1)]

    def calcule_probabilities(self, kmers) -> dict:
        num_kmers = len(kmers)
        return {i: j/num_kmers for i, j in Counter(kmers).items()}    
    

    def calculate_background_distribution(self, fasta_path: str) -> dict:

        counts = {
            1: Counter(),
            2: Counter(),
            3: Counter()
        }
        

        print("Parsing sequences...")
        count = 0
        
        for seq in self.read_fasta_generator(fasta_path):
            
            counts[1].update(self.split2kmers(seq, 1))
            counts[2].update(self.split2kmers(seq, 2))
            counts[3].update(self.split2kmers(seq, 3))
            count += 1
            
            if count % 100000 == 0:
                print(f"Processed {count} sequences...", end='\r')
            
        print(f"\nRaw processing complete. Processed {count} sequences.")
        
        print("Filtering invalid k-mers...")

        #Prune invalid k-mers
        invalid_chars = set(['X', 'Z', 'B', 'J', 'O', 'U'])
        for n in [1, 2, 3]:

            unique_kmers = list(counts[n].keys())
            
            for kmer in unique_kmers:
                # Check if this specific k-mer contains any bad char
                if set(kmer) & invalid_chars:
                    del counts[n][kmer]

        # Calculate Probabilities on the clean data
        return {
            'monomers': self.calcule_probabilities(counts[1]), 
            'dimers':   self.calcule_probabilities(counts[2]),
            'trimers':  self.calcule_probabilities(counts[3])
        }

    def kullback_leibler(self, p,q):

        D_KL = sum([p_val * np.log(p_val / q[k]) for k, p_val in p.items()])
        
        return D_KL


    def n_gram_prior(self, sequence):

        P1_seq = self.calcule_probabilities(self.split2kmers(sequence, 1))
        P2_seq = self.calcule_probabilities(self.split2kmers(sequence, 2))
        P3_seq = self.calcule_probabilities(self.split2kmers(sequence, 3))

        # Calculate D_KL for each n-gram size
        energy_uni = self.kullback_leibler(P1_seq, self.kmer_stat['monomers'])
        energy_bi  = self.kullback_leibler(P2_seq, self.kmer_stat['dimers'])
        energy_tri = self.kullback_leibler(P3_seq, self.kmer_stat['trimers'])

        # Total N-gram Energy
        return energy_uni + energy_bi + energy_tri
        
