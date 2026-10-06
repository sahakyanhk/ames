import random
import numpy as np
import pandas as pd
import typing as T 
import json


class Evolver:

    evoldict = {"alphabets":
                    {"protein":
                        {"A":1,"C":1,"D":1,"E":1,"F":1,"G":1,"H":1,"I":1,"K":1,"L":1,
                         "M":1,"N":1,"P":1,"Q":1,"R":1,"S":1,"T":1,"V":1,"W":1,"Y":1},
                    "protein_codonrates":
                        {"A":1.311475,"C":0.655738,"D":0.655738,"E":0.655738,"F":0.655738,
                         "G":1.311475,"H":0.655738,"I":0.983607,"K":0.655738,"L":1.967213,
                         "M":0.327869,"N":0.655738,"P":1.311475,"Q":0.655738,"R":1.967213,
                         "S":1.967213,"T":1.311475,"V":1.311475,"W":0.327869,"Y":0.655738},
                    "protein_uniprot":
                        {"A":1.738,"C":0.284,"D":1.088,"E":1.256,"F":0.768,
                         "G":1.414,"H":0.46,"I":1.066,"K":1.01,"L":1.952,
                         "M":0.464,"N":0.774,"P":1.03,"Q":0.786,"R":1.162,
                         "S":1.44,"T":1.12,"V":1.35,"W":0.26,"Y":0.57},
                    "rna":
                        {"A":1,"U":1,"G":1,"C":1},
                    "dna":
                        {"A":1,"T":1,"G":1,"C":1}
                        },

                "mutations":
                    {
                        "npm":{"+":1,"-":1,"*":0.4,"/":0.4,"%":0.9,"p":0.1,"d":0.05,"r":0.05},
                        "pmo":{"+":1,"-":1},
                        "rnd":{"r":1},
                        "rso":None
                    }
                }

    def __init__(self, 
                 alphabet = "protein",
                 mutations = "npm",
                 evoldict = None,
                 ):

        if evoldict and isinstance(evoldict, str):
            with open(evoldict, 'r') as f:
                evoldict_untested = json.load(f)
                if alphabet not in evoldict_untested["alphabets"] or mutations not in evoldict_untested["mutations"]:
                    raise ValueError(f"Invalid evoldict: {evoldict!r}. \
                        \nSee https://github.com/sahakyanhk/ames/blob/dev/src/ames/data/evoldict.json as an example.")
                else:
                    self.evoldict = evoldict_untested
        else:
            self.evoldict = evoldict if evoldict else Evolver.evoldict


        def unpack_evoldict(self, alphabet_type, mutations_type):
                alphabet = list(self.evoldict["alphabets"][alphabet_type].keys())
                if  mutations_type in ["npm", "pmo"]:
                    mutations = [*alphabet, *list(self.evoldict["mutations"][mutations_type].keys())]
                    weigths_raw = [*list(self.evoldict["alphabets"][alphabet_type].values()), 
                                   *list(self.evoldict["mutations"][mutations_type].values())]
                elif mutations_type == "rso": # residues substitution only, no insertions or deletions
                    mutations = alphabet
                    weigths_raw = list(self.evoldict["alphabets"][alphabet_type].values())
                elif mutations_type == "rnd":
                    mutations = [*alphabet, 'r']
                    weigths_raw = len(alphabet) * [1e-100] + [1e+100] 
                else:
                    raise ValueError(f"unknown mutations_type: {mutations_type!r}; expected {self.evoldict['mutations'].keys()}")

                weigths_sum = sum(weigths_raw)
                weigths = [i/weigths_sum for i in weigths_raw] # normalize weights

                return alphabet, mutations, weigths
  

        self.alphabet, self.mutations, self.weigths = unpack_evoldict(self, alphabet, mutations)
        
        self.alphabet_size = len(self.alphabet)

    #random sequence generator
    def randomseq(self,  nres: int = 24, weights = None ) -> str:
        """
        random sequence generator
        
        randomseq(100) generates random aa sequence of lenght 100 

        """
        return ''.join(random.choices(self.alphabet, k=nres, weights=self.weigths[:self.alphabet_size])) #alphabet size can be different if some AA are excluded



    def mutate(self, sequence: str) -> T.Tuple[str, str]:  

        """
        randomly mutate a sequence 

        mutate("AAUGAU") returns mutated sequence, mutation info

        """

        seq_len = len(sequence)
        min_seq_len = 2
        mutation_types = self.mutations
        alphabet = self.alphabet
        p = self.weigths

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
            mutation = self.randomseq(nres=random.choice(range(1, max_insertion + 1)))
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
            linker = self.randomseq(nres=2)
            sequence_mutated = sequence + linker + sequence
            mutation_info = f'd{linker}'

        elif mutation =='r': #all residues are ramdomly changed
            random_seq_len = random.choice(range(3, 6))
            #random_seq_len = seq_len
            sequence_mutated = self.randomseq(nres=random_seq_len)
            mutation_info = 'r'

        else: #fallback to point mutation when chosen op can't apply (e.g. short sequence)
            new_residue = self.randomseq(nres=1)
            sequence_mutated = sequence[:mutation_position] + new_residue + sequence[mutation_position + 1:]
            mutation_info = f'{sequence[mutation_position]}{mutation_position+1}.{new_residue}'

        #TODO random change for a chunk of the sequence. (imitation of a frameshift)

        return sequence_mutated, mutation_info


    def select(input_new_gen, input_init_gen, pop_size:int, selection_mode:str = 'weak', norepeat:bool = False, beta = 1): 

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
    


