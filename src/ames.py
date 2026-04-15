import os
import sys
import shutil
import random
import copy
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure src/ is on sys.path so sibling modules are importable from any directory
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

#AMES modules
from evolution import Evolver
from seqtools import Seqstat
import pdb_contacts as pc
from psique import pypsique


from amestools import (parse_args,
                       generate_loghead,
                       save_checkpoint,
                       sequence_signature,
                       build_sequence_lookup,
                       update_beta,
                       gzip_str, 
                       sigmoid,
                       backup_output, 
                       batch_sequence_dataset, 
                       create_init_gen,
                       print_genlog,
                       ScoringFunction,
                       DATA_DIR
                      )


write_lock = threading.Lock()


#========================================================================================#
#======================================# EVOLVER #=======================================#

def fold_evolution_simulator() -> None: 

    global new_gen #this will be modified in the extract_results() 

    logpath = os.path.join(args.outpath, args.log)
    checkpoint_path = os.path.join(args.outpath, args.ckp)

    loghead = generate_loghead(args)

    print(loghead)


    with open(logpath, 'w') as f:
        f.write(loghead)

    init_gen = create_init_gen(evolver, args)
    init_gen.to_csv(logpath, mode='a', index=False, header=True, sep='\t')

    sequence_lookup = build_sequence_lookup(init_gen)
    print_genlog(init_gen, args)
    
    #mutate seqs from init_gen and select the best N seqs for the next generation    
    for gen_i in range(1, args.num_generations):

        n = 0 # seq n
        new_gen = pd.DataFrame()
        now = datetime.now()
        threads = []
        generated_sequences = []
        repeat_rows = []
        
        # dynamic temperature control for annealing
        if args.annealing and args.annealing_start <= gen_i <= args.annealing_end:
            print("#selection temperatura was updated")
            update_beta(args)

        for prev_id, sequence_data in zip(init_gen.id, init_gen.sequence_data):
            
            seq_data = copy.deepcopy(sequence_data)

            # if coevolution chose mutation rates for each sequece
            if args.seq2_evol:
                mutate_seq1 = False
                mutate_seq2 = False
                while not (mutate_seq1 or mutate_seq2):
                    mutate_seq1 = random.choices([True, False], weights=[args.seq1_rate, 1-args.seq1_rate])[0]
                    mutate_seq2 = random.choices([True, False], weights=[args.seq2_rate, 1-args.seq2_rate])[0]
                    
            else: 
                mutate_seq1 = True #either mutate_seq1 or mutate_seq2 must be true
            

            #  mutate each sequence separately
            if mutate_seq1:
                seq1, mutation_data1 = evolver.mutate(args.seq1_type, sequence_data['seq1']['sequence'])
                seq_data["seq1"]["sequence"] = seq1 
                seq_data["seq1"]["len"] = len(seq1) 
            else: 
                mutation_data1 = "none"
                
            if args.seq2_evol and mutate_seq2:            
                seq2, mutation_data2 = evolver.mutate(args.seq2_type, sequence_data['seq2']['sequence'])
                seq_data["seq2"]["sequence"] = seq2
                seq_data["seq2"]["len"] = len(seq2) 
            else: 
                mutation_data2 = "none"


            mutation_data = mutation_data1 + ':' + mutation_data2


            #check if the mutated seqeuece was already predicted
            seq_key = sequence_signature(seq_data)
            repeat_row = sequence_lookup.get(seq_key)
            
            #if --norepeat and seq is in the ancestral_memory mutate it again
            if args.norepeat and repeat_row is not None:
                while repeat_row is not None:
                    if mutate_seq1:
                        seq1, mutation_data1 = evolver.mutate(args.seq1_type, sequence_data['seq1']['sequence'])
                        seq_data["seq1"]["sequence"] = seq1
                        seq_data["seq1"]["len"] = len(seq1)
                    else:
                        mutation_data1 = "none"

                    if args.seq2_evol and mutate_seq2:
                        seq2, mutation_data2 = evolver.mutate(args.seq2_type, sequence_data['seq2']['sequence'])
                        seq_data["seq2"]["sequence"] = seq2
                        seq_data["seq2"]["len"] = len(seq2)
                    else:
                        mutation_data2 = "none"

                    mutation_data = mutation_data1 + ':' + mutation_data2
                    seq_key = sequence_signature(seq_data)
                    repeat_row = sequence_lookup.get(seq_key)

            uid = f"g{gen_i}s{n}_{prev_id}_{mutation_data}"; n+=1 # gives an unique id even if the same sequence already exists            

            if repeat_row is not None: #if sequence already exits do not predict a structure again
                repeat_rows.append(repeat_row.copy())
            else:
                generated_sequences.append((uid, seq_data))

        if repeat_rows:
            new_gen = pd.concat([new_gen, pd.DataFrame(repeat_rows)], axis=0, ignore_index=True)

        batched_sequence_data = batch_sequence_dataset(sequences = generated_sequences, 
                                                    pop_size = args.pop_size, 
                                                    max_seq_per_batch = args.max_seq_per_batch
                                                    )
        #predict data for the new batch        
        for headers, sequence_data_batch in batched_sequence_data:
            
            if args.engine in ["af3", "of3", "esmfold"]:
                structure_predictor_ouptut = structure_predictor(sequence_data_batch)  # type: ignore

            elif args.engine == "simulacrum":
                structure_predictor_ouptut = fold_evolution_simulacrum(sequence_data_batch, args) #imitate empty of3/af3 engine output



            #run extract_results() in beckground and imediately start next the round of model.infer()
            trd = threading.Thread(target=extract_results, \
                        args=(gen_i, headers, sequence_data_batch, structure_predictor_ouptut, args))
            trd.start()
            threads.append(trd)


        for t in threads:
                t.join()
        
        seconds_per_generation = (datetime.now() - now).total_seconds()
        print(f"""
#{seconds_per_generation:.1f}s per generation
#{86400 / seconds_per_generation:.0f} generations per day
#{86400 / seconds_per_generation * args.pop_size:.0f} mutations per day""")
        
        # print each new generation in terminal 
        print_genlog(new_gen, args)  

        sequence_lookup.update(build_sequence_lookup(init_gen))

        #select the next generation 
        init_gen = evolver.select(new_gen, init_gen, args.pop_size, args.selection_mode, args.norepeat, args.beta)
        init_gen.gndx = gen_i #assign a new gen index
        init_gen.to_csv(logpath, mode='a', index=False, header=False, sep='\t')

        if gen_i % args.checkpoint_interval == 0:
            save_checkpoint(init_gen, args)
 
#==================================== EVOLVER =====================================#
#==================================================================================#



#==================================================================================#
#================================ RESULT PROCESSING ===============================#

def extract_results(gen_i: int, 
                    headers: list[str], 
                    sequence_data_batch: list[str], 
                    structure_predictor_ouptut: tuple, 
                    args,
                    ) -> None:
    
    global new_gen # Access the global dataframe

    structures, plddts, ptms, iptms = structure_predictor_ouptut

    batch_rows = [] 

    for meta_id, seq_data, pdb_txt, ptm, plddt, iptm in \
        zip(headers, sequence_data_batch, structures, ptms, plddts, iptms):
        
        uid_data = meta_id.split('_')

        uid = uid_data[0]
        prev_id = uid_data[1]
        mutation = uid_data[2]
        
        # calculate seq_stat 
        if args.seq1_type == 'protein':
            seq_data["seq1"]["seqstat"] = protein_seqstat.n_gram_prior(seq_data["seq1"]["sequence"])
        else:
            seq_data["seq1"]["seqstat"] = rna_seqstat.n_gram_prior(seq_data["seq1"]["sequence"])

        if args.seq2:
            if args.seq2_type == 'protein':
                seq_data["seq2"]["seqstat"] = protein_seqstat.n_gram_prior(seq_data["seq2"]["sequence"])
            else:
                seq_data["seq2"]["seqstat"] = rna_seqstat.n_gram_prior(seq_data["seq2"]["sequence"])


        # imitate simulation without real structure prediction
        if args.engine == "simulacrum" or pdb_txt == "STRUCTURESIMULACRUM":
            score = (15-seq_data["seq1"]["seqstat"]) / 15
            row_data = {
                        'gndx': gen_i,
                        'id': uid, 
                        'beta': args.beta,
                        'plddt': 0.0,
                        'ptm': 0.0, 
                        'iplddt': 0.0,
                        'iptm': 0.0,
                        "n_atoms": 0.0,
                        'cd': 0.0,
                        'lcd': 0.0,
                        "n_clashes": 0.0,
                        "clashscore": 0.0,
                        'score': score,
                        'sequence_data': seq_data, 
                        'mutation': mutation,
                        'prev_id': prev_id,
                        'structure': pdb_txt,
                    }


            # Append to local list instead of the global DataFrame immediately
            batch_rows.append(row_data)

            continue

        #=======================================================================# 
        #=============================== SCORING ===============================# 


        if args.seq1_type == 'protein':
            protein_ss, _, _ = pypsique(pdb_txt, chain=args.protein_chain)
            seq_data["seq1"]["ss"] = protein_ss
        else:
            nucleic_ss = "NASECONDARYSTRUCTURES"
            seq_data["seq1"]["ss"] = nucleic_ss
    
        if args.seq2:

            iplddt = round(pc.interface_plddt(pdb_txt, chain1 = "A", chain2 = "B", cutoff = args.interface_plddt_cutoff) * 0.01, 3)

            if args.seq2_type == 'protein':
                protein_ss, _, _ = pypsique(pdb_txt, chain=args.protein_chain)
                seq_data["seq2"]["ss"] = protein_ss
            else:
                nucleic_ss = "NASECONDARYSTRUCTURES"
                seq_data["seq2"]["ss"] = nucleic_ss

        else:
            iplddt = 0.0


        if args.protein_chain == "A" or args.protein_chain == "B":
            contact_density = pc.contact_density(pdb_txt, 
                                                cutoff=args.contact_cutoff,
                                                chain=args.protein_chain, 
                                                min_plddt=args.contact_min_plddt, 
                                                min_seq_dist=args.contact_min_seq_dist) # for a 30aa polyA helix: min_seq_dist=3 => 51 contacts, 4 => 25, 5 => 0

        elif args.protein_chain == "A" and args.protein_chain == "B":

            chainA_density = pc.contact_density(pdb_txt, chain="A",
                                                         min_plddt=args.contact_min_plddt,
                                                         min_seq_dist=args.contact_min_seq_dist)
            chainB_density = pc.contact_density(pdb_txt, chain="B",
                                                         min_plddt=args.contact_min_plddt,
                                                         min_seq_dist=args.contact_min_seq_dist)

            contact_density = (chainA_density + chainB_density) / 2

        else: 
            contact_density = 0.0

        clashscore_row = pc.clashscore(pdb_txt, 
                                        overlap_threshold=args.clash_overlap_threshold, 
                                        exclude_hydrogen=True, 
                                        chain="", 
                                        min_seq_dist=args.clash_min_seq_dist)

        num_clashes = clashscore_row['num_clashes']
        num_atoms = clashscore_row['num_atoms']
        clashscore = clashscore_row['clashscore']
        clash_penalty = 1 - np.clip(clashscore_row['clashscore'], 0, 1)
        

        if args.ligand:
            ligand_contact_density = pc.ligand_contact_density(pdb_txt, 
                                                               cutoff = args.lig_contact_cutoff,
                                                               polymer_chain=args.polymer_chains,
                                                               ligand_chain=args.ligand_chains,
                                                               min_plddt=args.lig_contact_min_plddt)

            ligand_iplddt = pc.interface_plddt(pdb_txt, chain1 = args.polymer_chains, chain2 = args.ligand_chains, cutoff = args.interface_plddt_cutoff) * 0.01 

            if args.seq2:
                iplddt = (iplddt + ligand_iplddt) / 2
            else:
                iplddt = ligand_iplddt
        
        else:
            ligand_contact_density = 0.0


        contact_density = round(contact_density, 3)  
        ligand_contact_density = round(ligand_contact_density, 3)
        iplddt = round(iplddt, 3)
        clashscore = round(clashscore, 3)

        # calculate penalties
        seq1_len_penalty =  1 - sigmoid(seq_data["seq1"]["len"], args.seq1_len_constr, 0.2)
        
        if args.seq2:
            seq2_len_penalty =  1 - sigmoid(seq_data["seq2"]["len"], args.seq2_len_constr, 0.2)
        else: 
            seq2_len_penalty = 1

        penalty = seq1_len_penalty * seq2_len_penalty  * clash_penalty#* max_alpha_penalty * max_beta_penalty
        
        #=============================== SCORING ===============================#

        score = scoring.score(ptm, 
                              plddt, 
                              iptm, 
                              iplddt, 
                              contact_density, 
                              ligand_contact_density,
                              penalty
                              ) 

        row_data = {
            'gndx': gen_i,
            'id': uid, 
            'beta': args.beta,
            'plddt': plddt,
            'ptm': ptm, 
            'iplddt': iplddt,
            'iptm': iptm,
            "n_atoms": num_atoms,
            'cd': contact_density,
            'lcd': ligand_contact_density,
            "n_clashes": num_clashes,
            "clashscore": clashscore,
            'score': score,
            'sequence_data': seq_data, 
            'mutation': mutation,
            'prev_id': prev_id,
            'structure': gzip_str(pdb_txt),
        }


        # Append to local list instead of the global DataFrame immediately
        batch_rows.append(row_data)

    batch_df = pd.DataFrame(batch_rows)
    
    # lock ONCE per batch
    with write_lock:
        if new_gen.empty:
            new_gen = batch_df
        else:
            new_gen = pd.concat([new_gen, batch_df], axis=0, ignore_index=True) 



#================================= RESULT PROCESSING ================================#
#====================================================================================#


args = parse_args()

evolver = Evolver(protein_alphabet = args.protein_alphabet,
                  rna_alphabet = args.rna_alphabet,
                  dna_alphabet = args.dna_alphabet,
                  protein_mutations = args.protein_mutations,
                  rna_mutations = args.rna_mutations,
                  dna_mutations = args.dna_mutations
                  ) 


protein_seqstat = Seqstat(str(DATA_DIR / 'pfam80_stat.json'))
rna_seqstat = Seqstat(str(DATA_DIR / 'rnacentral90_stat.json'))

scoring = ScoringFunction(args.evolution_type)


#backup if output directory exists
if args.nobackup:
    if os.path.isdir(args.outpath):
        print(f'\nWARNING! Directory {args.outpath} exists, it will be replaced!\n')
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath)
else:
    backup_output(args.outpath)



if args.engine == "af3":
    from adapters import ames_to_af3 as adapter
    from af3_runner import af3_runner as structure_predictor

elif args.engine == "of3":
    from adapters import ames_to_of3 as adapter
    from openfold3_runner import openfold3_runner as structure_predictor

elif args.engine == "esmfold":
    from adapters import ames_to_esmfold as adapter
    from esmfold_runner import esmfold_runner as structure_predictor

elif args.engine == "simulacrum":
    from simulacra import fold_evolution_simulacrum

else:
    raise ValueError("Unknown engine, available options are: af3, of3, esmfold, simulacrum")

if __name__ == '__main__':
    fold_evolution_simulator()

