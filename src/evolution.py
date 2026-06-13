import random
import numpy as np
import pandas as pd
import typing as T 
import json


class Evolver:

    #==== protein dictionaries
    protein = {"uniform": {'A':1, 'C':1, 'D':1, 'E':1, 'F':1, 
                          'G':1, 'H':1, 'I':1, 'K':1, 'L':1, 
                          'M':1, 'N':1, 'P':1, 'Q':1, 'R':1, 
                          'S':1, 'T':1, 'V':1, 'W':1, 'Y':1},

                #by number of codons (nr_of_codos_for_i/sum(all codons)) * 20
                "codonrates": {'A': 1.311475, #4 
                               'C': 0.655738, #2
                               'D': 0.655738, #2
                               'E': 0.655738, #2
                               'F': 0.655738, #2
                               'G': 1.311475, #4
                               'H': 0.655738, #2
                               'I': 0.983607, #3
                               'K': 0.655738, #2
                               'L': 1.967213, #6
                               'M': 0.327869, #1
                               'N': 0.655738, #2
                               'P': 1.311475, #4
                               'Q': 0.655738, #2
                               'R': 1.967213, #6
                               'S': 1.967213, #6
                               'T': 1.311475, #4
                               'V': 1.311475, #4
                               'W': 0.327869, #1
                               'Y': 0.655738, #2
                                },

                #https://www.uniprot.org/uniprotkb/statistics#amino-acid-composition
                # normalized by % * 20
                "uniprot": {'A': 1.7380, 'C': 0.2840, 'D': 1.0880, 'E': 1.2560, 'F': 0.7680, 
                            'G': 1.4140, 'H': 0.4600, 'I': 1.0660, 'K': 1.0100, 'L': 1.9520, 
                            'M': 0.4640, 'N': 0.7740, 'P': 1.0300, 'Q': 0.7860, 'R': 1.1620, 
                            'S': 1.4400, 'T': 1.1200, 'V': 1.3500, 'W': 0.2600, 'Y': 0.5700,},

                "npm": {'+': 1.0, '-': 1.0, '*': 0.4, '/':0.4, '%': 0.9, 'p': 0.1, 'd': 0.05, 'r': 0.05}, # non-point mutations
                "pmo": {'+': 1.0, '-': 1.0}, #point mutations only
                "rnd": {'r': 1.0}, #radom sequence every round
                "rso": None # no mutations, protein length is fixed
              }

    
    #===RNA dictionaries
    rna = {"uniform": {'A': 1, 'U': 1, 'G': 1, 'C': 1,},
            "npm": {'+': 1.0, '-': 1.0, '*': 0.4, '/':0.4, '%': 0.9, 'p': 0.1, 'd': 0.05},
            "pmo": {'+': 1.0, '-': 1.0},
            "rso": None
              }

    #===DNA dictionaries
    dna = {"uniform": {'A': 1, 'T': 1, 'G': 1, 'C': 1,},
           "npm": {'+': 1.0, '-': 1.0, '*': 0.4, '/':0.4, '%': 0.9, 'p': 0.1, 'd': 0.05},
           "pmo": {'+': 1.0, '-': 1.0},
           "rso": None
            }

    
    evoldict = {"protein": protein, "rna": rna, "dna":dna}


    def __init__(self, 
                 evoldict = None,
                 protein_alphabet = "uniform",
                 rna_alphabet = "uniform",
                 dna_alphabet = "uniform",
                 protein_mutations = "npm",
                 rna_mutations = "npm",
                 dna_mutations = "npm",
                 custom_evoldict: T.Optional[str] = None):

        if custom_evoldict:
            with open(custom_evoldict, 'r') as f:
                self.evoldict = json.load(f)
        else:
            self.evoldict = evoldict if evoldict else Evolver.evoldict


        def unpack_evoldict(self, mol_type, alphabet_type, mutations_type):
                alphabet = list(self.evoldict[mol_type][alphabet_type].keys())
                if  mutations_type in ["npm", "pmo"]:
                    mutations = [*alphabet, *list(self.evoldict[mol_type][mutations_type].keys())]
                    weigths_raw = [*list(self.evoldict[mol_type][alphabet_type].values()),
                                  *list(self.evoldict[mol_type][mutations_type].values())]
                elif mutations_type == "rso":
                    mutations = alphabet
                    weigths_raw = list(self.evoldict[mol_type][alphabet_type].values())
                elif mutations_type == "rnd":
                    mutations = [*alphabet, 'r']
                    weigths_raw = len(alphabet) * [1e-100] + [1e+100] 
                else:
                    raise ValueError(f"unknown mutations_type: {mutations_type!r}; expected 'npm', 'pmo', 'rso' or 'rnd'")
                    
                weigths_sum = sum(weigths_raw)
                weigths = [i/weigths_sum for i in weigths_raw] # normalize weights

                return alphabet, mutations, weigths
  

        self.protein_alphabet, self.protein_mutations, self.protein_weigths = unpack_evoldict(self, "protein", protein_alphabet, protein_mutations)
        self.rna_alphabet, self.rna_mutations, self.rna_weigths = unpack_evoldict(self, "rna", rna_alphabet, rna_mutations)
        self.dna_alphabet, self.dna_mutations, self.dna_weigths = unpack_evoldict(self, "dna", dna_alphabet, dna_mutations)
        
        self.protein_alphabet_size = len(self.protein_alphabet)
        self.rna_alphabet_size = len(self.rna_alphabet)
        self.dna_alphabet_size = len(self.dna_alphabet)

    #random sequence generator
    def randomseq(self, sequence_type: str = "protein", nres: int = 24, weights = None ) -> str:
        """
        random sequence generator
        
        randomseq("protein", 100) generates random aa sequence of lenght 100 

        """
        if sequence_type == "protein":
            return ''.join(random.choices(self.protein_alphabet, k=nres, weights=self.protein_weigths[:self.protein_alphabet_size])) #alphabet size can be different if some AA are excluded

        elif sequence_type == "rna":
            return ''.join(random.choices(self.rna_alphabet, k=nres, weights=self.rna_weigths[:self.rna_alphabet_size]))

        elif sequence_type == "dna":
            return ''.join(random.choices(self.dna_alphabet, k=nres, weights=self.dna_weigths[:self.dna_alphabet_size]))
        else:
            raise ValueError(f"unknown sequence_type: {sequence_type!r}; expected 'protein', 'rna' or 'dna'")



    def mutate(self, sequence_type: str, sequence: str) -> T.Tuple[str, str]:  

            """
            randomly mutate a sequence 

            mutate("rna", "AAUGAU") returns mutated sequence, mutation info

            """

            seq_len = len(sequence)
            
            if sequence_type == "protein":
                mutation_types = self.protein_mutations
                alphabet = self.protein_alphabet
                p = self.protein_weigths

            elif sequence_type == "rna":
                mutation_types = self.rna_mutations
                alphabet = self.rna_alphabet
                p = self.rna_weigths

            elif sequence_type == "dna":
                mutation_types = self.dna_mutations
                alphabet = self.dna_alphabet
                p = self.dna_weigths

            else: 
                raise ValueError(f"unknown sequence_type: {sequence_type!r}; expected 'protein', 'rna' or 'dna'")


            min_seq_len = 2
            if seq_len < min_seq_len:
                mutation = 'd'
                mutation_position = 0
            else:
                mutation_position = random.choice(range(seq_len))
                mutation =  random.choices(mutation_types, weights=p)[0]

            if mutation in alphabet:
                sequence_mutated = sequence[:mutation_position] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}.{mutation}'

            elif mutation =='+':
                mutation = random.choices(alphabet)[0]
                sequence_mutated = sequence[:mutation_position + 1] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}+{mutation}'

            elif mutation == '-' and seq_len > min_seq_len:
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}-'

            elif mutation =='*' and seq_len >= min_seq_len: #partial duplication
                max_chunk = min(seq_len - mutation_position, max(1, int(seq_len/2)))
                insertion_len = random.choice(range(1, max_chunk + 1)) #TODO insertion length probabability
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position:][:insertion_len] + sequence[mutation_position:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}*{sequence[mutation_position:][:insertion_len]}'

            elif mutation =='/': #random insertion
                max_insertion = max(1, int(seq_len/2))
                mutation = self.randomseq(sequence_type=sequence_type, nres=random.choice(range(1, max_insertion + 1)))
                sequence_mutated = sequence[:mutation_position + 1] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}/{mutation}'

            elif mutation =='%' and seq_len > min_seq_len: #partial deletion, keep at least min_seq_len residues
                max_deletion = seq_len - min_seq_len
                deletion_len = random.choice(range(1, max_deletion + 1))
                mutation_position = random.choice(range(seq_len - deletion_len + 1))
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position + deletion_len:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}%{deletion_len}'

            elif mutation =='p' and seq_len >= min_seq_len: #permutation
                sequence_mutated =  sequence[mutation_position:] + sequence[:mutation_position]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}p{mutation}'

            elif mutation =='d': #full duplication #TODO reduce the duplication probability with sequence growth
                linker = self.randomseq(sequence_type=sequence_type, nres=2)
                sequence_mutated = sequence + linker + sequence
                mutation_info = f'd{linker}'

            elif mutation =='r': #all residues are ramdomly changed
                sequence_mutated = self.randomseq(sequence_type=sequence_type, nres=seq_len)
                mutation_info = 'r'

            else: #fallback to point mutation when chosen op can't apply (e.g. short sequence)
                new_residue = self.randomseq(sequence_type=sequence_type, nres=1)
                sequence_mutated = sequence[:mutation_position] + new_residue + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}.{new_residue}'

            #TODO random change for a chunk of the sequence. (imitation of a frameshift)

            return sequence_mutated, mutation_info



    def select(self, input_new_gen, input_init_gen, pop_size:int, selection_mode:str = 'weak', norepeat:bool = False, beta = 1): 

        mixed_pop = pd.concat([input_new_gen, input_init_gen], axis=0, ignore_index=True) 

        if norepeat and len(mixed_pop['sequence'].unique()) >= pop_size:
            mixed_pop = mixed_pop.drop_duplicates(subset=['sequence'])

        elif selection_mode == "strong":
            new_init_gen = mixed_pop.sort_values('score', ascending=False).head(pop_size)

        elif selection_mode == "weak":
            weights = np.array(np.exp(beta * mixed_pop.score) / np.array(np.exp(beta * mixed_pop.score)).sum())
            new_init_gen = mixed_pop.sample(n=pop_size, weights=weights, replace=(not norepeat)).sort_values('score', ascending=False)

        elif selection_mode == "weak_nt": #weak selection without temperature, i.e. weights are proportional to scores
            weights = np.array((mixed_pop.score) / ((mixed_pop.score).sum()))
            new_init_gen = mixed_pop.sample(n=pop_size, weights=weights, replace=(not norepeat)).sort_values('score', ascending=False)
        else:
            raise ValueError(f"unknown selection_mode: {selection_mode!r}; expected 'strong', 'weak' or 'weak_nt'")
        return new_init_gen
    


