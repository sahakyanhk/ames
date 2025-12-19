import random
import numpy as np
import pandas as pd
import typing as T 
import json


class Evolver:

    protein = {"molecule_type": "protein",
               "alphabet": ['A', 'C', 'D', 'E', 'F', 
                            'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 
                            'S', 'T', 'V', 'W', 'Y'],
               "weights": [1, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1,],
               "npm": ['+', #single residue insertion
                       '-', #single residue deletion
                       '*', #partial duplication
                       '/', #random insertion 
                       '%', #partial deletion
                       'p', #Circular permutation
                       'd'  #full duplication    
                       ],
               "npm_weigths": [1.0, 1.0, 0.4, 0.4, 0.9, 0.1, 0.05]
               }
    
    rna = {"molecule_type": "rna",
           "alphabet": ['A', 'U', 'G', 'C'],
           "weights": [1, 1, 1, 1],
           "npm": ['+', '-', '*', '/', '%', 'p', 'd'],
           "npm_weigths": [1.0, 1.0, 0.4, 0.4, 0.9, 0.1, 0.05]
           }

    dna = {"molecule_type": "dna",
           "alphabet": ['A', 'T', 'G', 'C'],
           "weights": [1, 1, 1, 1],
           "npm": ['+', '-', '*', '/', '%', 'p', 'd'],
           "npm_weigths": [1.0, 1.0, 0.4, 0.4, 0.9, 0.1, 0.05]
           }

    
    evoldict = {"protein": protein, "rna": rna, "dna":dna}

    #by number of codons. 
    # Calculated as (Codon_i/sum(all codons)) * 20
    # so the mean probability is 1. 

    codonrates = {'A' : 1.311475, #4 
                  'C' : 0.655738, #2
                  'D' : 0.655738, #2
                  'E' : 0.655738, #2
                  'F' : 0.655738, #2
                  'G' : 1.311475, #4
                  'H' : 0.655738, #2
                  'I' : 0.983607, #3
                  'K' : 0.655738, #2
                  'L' : 1.967213, #6
                  'M' : 0.327869, #1
                  'N' : 0.655738, #2
                  'P' : 1.311475, #4
                  'Q' : 0.655738, #2
                  'R' : 1.967213, #6
                  'S' : 1.967213, #6
                  'T' : 1.311475, #4
                  'V' : 1.311475, #4
                  'W' : 0.327869, #1
                  'Y' : 0.655738 #2
                  }

    #https://www.uniprot.org/uniprotkb/statistics#amino-acid-composition
    uniprotrates = {'A' : 0.0826, 'C' : 0.0139, 'D' : 0.0546, 'E' : 0.0672, 
                    'F' : 0.0387, 'G' : 0.0707, 'H' : 0.0228, 'I' : 0.0591, 
                    'K' : 0.0580, 'L' : 0.0965, 'M' : 0.0241, 'N' : 0.0406, 
                    'P' : 0.0475, 'Q' : 0.0393, 'R' : 0.0553, 'S' : 0.0665, 
                    'T' : 0.0536, 'V' : 0.0686, 'W' : 0.0110, 'Y' : 0.0292}
    


    def __init__(self, evoldict = None, include_npm: bool = False, custom_evoldict: T.Optional[str] = None):

        if custom_evoldict:
            with open(custom_evoldict, 'r') as f:
                self.evoldict = json.load(f)
        else:
            self.evoldict = evoldict if evoldict else Evolver.evoldict

        
        # Protein
        self.protein_alphabet = self.evoldict["protein"]["alphabet"]
        self.protein_mutation_types = [*self.evoldict["protein"]["alphabet"], *self.evoldict["protein"]["npm"]] if include_npm else self.evoldict["protein"]["alphabet"]
        self.protein_weigths_raw = [*self.evoldict["protein"]["weights"], *self.evoldict["protein"]["npm_weigths"]] if include_npm else self.evoldict["protein"]["weights"]
        self.protein_weigths_raw_sum = sum(self.protein_weigths_raw)
        self.protein_weigths = [i/self.protein_weigths_raw_sum for i in self.protein_weigths_raw] # normalize weights

        # RNA
        self.rna_alphabet = self.evoldict["rna"]["alphabet"]
        self.rna_mutation_types = [*self.evoldict["rna"]["alphabet"], *self.evoldict["rna"]["npm"]] if include_npm else self.evoldict["rna"]["alphabet"]
        self.rna_weigths_raw = [*self.evoldict["rna"]["weights"], *self.evoldict["rna"]["npm_weigths"]] if include_npm else self.evoldict["rna"]["weights"]
        self.rna_weigths_raw_sum = sum(self.rna_weigths_raw)
        self.rna_weigths = [i/self.rna_weigths_raw_sum for i in self.rna_weigths_raw] # normalyze weigths

        # DNA
        self.dna_alphabet = self.evoldict["dna"]["alphabet"]
        self.dna_mutation_types = [*self.evoldict["dna"]["alphabet"], *self.evoldict["dna"]["npm"]] if include_npm else self.evoldict["dna"]["alphabet"]
        self.dna_weigths_raw = [*self.evoldict["dna"]["weights"], *self.evoldict["dna"]["npm_weigths"]] if include_npm else self.evoldict["dna"]["weights"]
        self.dna_weigths_raw_sum = sum(self.dna_weigths_raw)
        self.dna_weigths = [i/self.dna_weigths_raw_sum for i in self.dna_weigths_raw] # normalyze weigths




    #random sequence generator
    def randomseq(self, sequence_type: str = "protein", nres: int = 24, weights = None ) -> str:
        """
        random sequence generator
        
        randomseq("protein", 100) generates random aa sequence of lenght 100 

        """
        if sequence_type == "protein":
            return ''.join(random.choices(self.protein_alphabet, k=nres, weights=self.protein_weigths[:20])) #use only first 20 weights to correspond aa alphabet

        elif sequence_type == "rna":
            return ''.join(random.choices(self.rna_alphabet, k=nres, weights=self.rna_weigths[:4]))

        elif sequence_type == "dna":
            return ''.join(random.choices(self.dna_alphabet, k=nres, weights=self.dna_weigths[:4]))
        else:
            raise ValueError(f"unknown sequence_type: {sequence_type!r}; expected 'protein', 'rna' or 'dna'")



    def mutate(self, sequence_type: str, sequence: str) -> T.Tuple[str, str]:  

            """
            randomly mutate a sequence 

            mutate("rna", "AAUGAU") returns mutated sequence, mutation info

            """



            seq_len = len(sequence)
            
            if sequence_type == "protein":
                mutation_types = self.protein_mutation_types
                alphabet = self.protein_alphabet
                p = self.protein_weigths

            elif sequence_type == "rna":
                mutation_types = self.rna_mutation_types
                alphabet = self.rna_alphabet
                p = self.rna_weigths

            elif sequence_type == "dna":
                mutation_types = self.dna_mutation_types
                alphabet = self.dna_alphabet
                p = self.dna_weigths

            else: 
                raise ValueError(f"unknown sequence_type: {sequence_type!r}; expected 'protein', 'rna' or 'dna'")


            # if seq_len < 6:
            #     mutation = 'd'


            # else:
            
            mutation_position = random.choice(range(seq_len))
            mutation =  random.choices(mutation_types, weights=p)[0]
            
            if mutation in alphabet:
                sequence_mutated = sequence[:mutation_position] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}.{mutation}'

            elif mutation =='+':
                mutation = random.choices(alphabet)[0]
                sequence_mutated = sequence[:mutation_position + 1] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}+{mutation}'

            elif mutation == '-':
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}-'

            elif mutation =='*' and seq_len > 5: #partial duplication
                insertion_len = random.choice(range(2, int(seq_len/2))) #TODO insertion length probabability
            #   insertion_len = round(np.random.normal(loc=round(seq_len/2), scale=1.0, size=None))  to use normal distribution. TODO try also exp decline          
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position:][:insertion_len] + sequence[mutation_position:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}*{sequence[mutation_position:][:insertion_len]}'

            elif mutation =='/': #random insertion
                mutation = self.randomseq(sequence_type=sequence_type, nres=random.choice(range(2, int(seq_len/2)))) 
                sequence_mutated = sequence[:mutation_position + 1] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}/{mutation}'

            elif mutation =='%' and seq_len > 5: #partial deletion
                deletion_len = random.choice(range(2, int(seq_len/2))) #what is the probable deletion lenght?
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position + deletion_len:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}%{deletion_len}'

            elif mutation =='p' and seq_len > 5: #permutation 
                sequence_mutated =  sequence[mutation_position:] + sequence[:mutation_position]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}p{mutation}'

            elif mutation =='d': #full duplication #TODO reduce the duplication probability with sequence growth
                linker = self.randomseq(sequence_type=sequence_type, nres=2)
                sequence_mutated = sequence + linker + sequence     
                mutation_info = f'd{linker}'
                
            elif mutation =='r' and seq_len > 5: #TODO recombination 
                sequence_mutated = sequence
                mutation_info = f'{mutation_position+1}'

            #TODO random change for a chanck of the sequence. (imitation of a frameshift)
            return sequence_mutated, mutation_info



    def select(self, input_new_gen, input_init_gen, pop_size:int, selection_mode:str = 'week', norepeat:bool = False, beta = 1): 

        mixed_pop = pd.concat([input_new_gen, input_init_gen], axis=0, ignore_index=True) 

        if norepeat and len(mixed_pop['sequence'].unique()) == pop_size:
            mixed_pop = mixed_pop.drop_duplicates(subset=['sequence'])

        if selection_mode == "strong":
            new_init_gen = mixed_pop.sort_values('score', ascending=False).head(pop_size)

        if selection_mode == "weak":
            weights = np.array(np.exp(beta * mixed_pop.score) / np.array(np.exp(beta * mixed_pop.score)).sum())
            new_init_gen = mixed_pop.sample(n=pop_size, weights=weights, replace=(not norepeat)).sort_values('score', ascending=False)
        
        if selection_mode == "weak2":
            weights = np.array((mixed_pop.score) / ((mixed_pop.score).sum()))
            new_init_gen = mixed_pop.sample(n=pop_size, weights=weights, replace=(not norepeat)).sort_values('score', ascending=False)
            print(weights.sum())

        return new_init_gen
    


