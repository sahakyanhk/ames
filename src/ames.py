import os
import sys
import shutil
import random
import copy
import threading
import pandas as pd
from datetime import datetime

#AMES modules
from evolution import Evolver
import pdb_contacts
from psique import pypsique


from amestools import (parse_args,
                       update_beta,
                       gzip_str, 
                       sigmoid,
                       backup_output, 
                       batch_sequence_dataset, 
                       create_init_gen,
                       prepare_af3_input,
                       print_genlog,
                      )

from af3_runner import af3_runner as structure_predictor


write_lock = threading.Lock()


#========================================================================================#
#======================================# EVOLVER #=======================================#

def fold_evolution_simulator(args, evolver) -> None: 

    global new_gen #this will be modified in the extract_results() 
    
    logpath = os.path.join(args.outpath, args.log)

    now = datetime.now() # current date and time
    date_now = now.strftime("%d-%b-%Y")
    time_now = now.strftime("%H:%M:%S")


    param_lines = [f"# --{param:<18} = {value}\n" for param, value in vars(args).items()]
    


    loghead = f'''
#======================== AMESv0.1 ========================#
#====================== {date_now} =======================#
#======================== {time_now} ========================#
#WD: {os.getcwd()}
#${' '.join(sys.argv)}
#
#======================= input params =====================#
#
''' + ''.join(param_lines) + '''
#
#==========================================================#
'''
    print(loghead)

    init_gen = create_init_gen(evolver, args)

    with open(logpath, 'w') as f:
        f.write(loghead)

    #    f.write('\t'.join(init_gen.keys()) + '\n')
    
    init_gen.to_csv(logpath, mode='a', index=False, header=True, sep='\t')

    #ancestral_memory = set() #-> !!! TODO make ancestral memory a set of unique sequences !!! <-# 
    ancestral_memory = init_gen  
    print_genlog(init_gen, args)
    
    #mutate seqs from init_gen and select the best N seqs for the next generation    
    for gen_i in range(args.num_generations):
        
        n = 0
        new_gen = pd.DataFrame()
        now = datetime.now()
        threads = []
        generated_sequences = []
        mutation_collection = []
        
        # dynamic temperature control for annealing
        if args.annealing & args.annealing_start < gen_i < args.annealing_end:
            print("#selection temperatura was updated")
            update_beta(args)

        for prev_id, sequence_data in zip(init_gen.id, init_gen.sequence_data):
            
            seq_data = copy.deepcopy(sequence_data)

            
            # if coevolution chose mutation rates for each sequece
            if args.seq2_evol:
                mutate_seq1 = random.choices([True, False], weights=[0.25, 0.75])[0]
                mutate_seq2 = random.choices([True, False], weights=[0.75, 0.25])[0]
                
                if not (mutate_seq1 or mutate_seq2):
                    mutate_seq1 = True

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
            seqmask = ancestral_memory["sequence_data"] == seq_data 
            
            #if --norepeat and seq is in the ancestral_memory mutate it again
            if args.norepeat and seqmask.any():  
                while seqmask.any():
                    seq1, mutation_data = evolver.mutate(args.seq1_type, sequence_data['seq1']['sequence'])
                    seq_data["seq1"]["sequence"] = seq1 
                    if args.seq2_evol:            
                        seq2, mutation_data = evolver.mutate(args.seq2_type, sequence_data['seq2']['sequence'])
                        seq_data["seq2"]["sequence"] = seq2
                    seqmask = ancestral_memory["sequence_data"] == seq_data 

            uid = f"g{gen_i}s{n}_{prev_id}_{mutation_data}"; n+=1 # gives an unique id even if the same sequence already exists            

            if seqmask.any(): #if sequence already exits do not predict a structure again 
                repeat = ancestral_memory[seqmask].drop_duplicates(subset=['sequence_data'], keep='last') 
                new_gen = pd.concat([new_gen, repeat])
            else:
                generated_sequences.append((uid, seq_data)) 
                mutation_collection.append(mutation_data) 

        batched_sequence_data = batch_sequence_dataset(sequences = generated_sequences, 
                                                    pop_size = args.pop_size, 
                                                    max_seq_per_batch = args.max_seq_per_batch
                                                    )
        #predict data for the new batch        
        for headers, sequence_data_batch in batched_sequence_data:
            

            fold_input = prepare_af3_input(sequence_data_batch, args)

            structure_predictor_ouptut  = structure_predictor(fold_input)  # type: ignore

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

        ancestral_memory =  pd.concat([ancestral_memory, init_gen])

        #select the next generation 
        init_gen = evolver.select(new_gen, init_gen, args.pop_size, args.selection_mode, args.norepeat, args.beta)
        init_gen.gndx = gen_i #assign a new gen index
        init_gen.to_csv(logpath, mode='a', index=False, header=False, sep='\t')

 
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

    structures, plddts, ptms, iptms, _ = structure_predictor_ouptut

    batch_rows = [] 

    for meta_id, seq_data, structure, ptm, plddt, iptm in \
        zip(headers, sequence_data_batch, structures, ptms, plddts, iptms):
        
        uid_data = meta_id.split('_')

        uid = uid_data[0]
        prev_id = uid_data[1]
        mutation = uid_data[2]
        

        #=======================================================================# 
        #=============================== SCORING ===============================# 

        if args.seq1_type == 'protein':
            prot_chain = "A"
            na_chain = None
            ss, max_helix, max_beta = pypsique(structure, chain=prot_chain)
            seq_data["seq1"]["ss"] = ss
        else: 
            prot_chain = None
            na_chain = "A"
            ss = "NASS"

        if args.seq2_type == 'protein':
            prot_chain = "B"
            na_chain = None
            ss, max_helix, max_beta = pypsique(structure, chain=prot_chain)
            seq_data["seq2"]["ss"] = ss
        else: 
            prot_chain = None
            na_chain = "B"
            ss = "NASS"
        

        min_plddt = 60
        min_seq_dist=4 #min_seq_dist=4 counts ~1 contact for each aa in a helix,
                        #min_seq_dist=5 will not count contacts in helices
                                                                                    
        if prot_chain in ["A", "B"]:
            contact_density = pdb_contacts.contact_density(structure, 
                                                        chain=prot_chain, 
                                                        min_plddt=min_plddt, 
                                                        min_seq_dist=min_seq_dist) 
        else: 
            contact_density = 0.0

        seq1_len_penalty =  1 - sigmoid(seq_data["seq1"]["len"], args.seq1_len_constr, 0.2)
        
        if args.seq2:
            seq2_len_penalty =  1 - sigmoid(seq_data["seq2"]["len"], args.seq2_len_constr, 0.2)
        else: 
            seq2_len_penalty = 1

        # max_alpha_penalty = 1 - sigmoid(max_helix, args.helix_len_penalty, 0.5)
        # max_beta_penalty = 1 - sigmoid(max_beta, args.beta_len_penalty, 0.6)

        penalty = seq1_len_penalty * seq2_len_penalty #* max_alpha_penalty * max_beta_penalty

        if args.evolution_type == 'PROTEIN_FOLD_EVOLUTION':
            score = (0.4*ptm + 0.2*plddt + 0.4*contact_density) * penalty

        elif args.evolution_type == 'NA_FOLD_EVOLUTION':                
            score = (0.8*ptm + 0.2*plddt) * penalty

        elif args.evolution_type in ['PROTEIN_NA_COEVOLUTION', 'PROTEIN_NA_EVOLUTION']:
            score =  (0.5*iptm + 0.2*ptm + 0.15*plddt + 0.15*contact_density) * penalty

        elif args.evolution_type == 'NA_COMPLEX_COEVOLUTION':                
            score = (0.6*iptm + 0.2*plddt + 2*plddt) * penalty

        elif args.evolution_type in ['PROTEIN_COMPLEX_COEVOLUTION', 'PROTEIN_COMPLEX_EVOLUTION']:
            
            chainA_density = pdb_contacts.contact_density(structure, chain="A", min_plddt=min_plddt, min_seq_dist=min_seq_dist)
            chainB_density = pdb_contacts.contact_density(structure, chain="B", min_plddt=min_plddt, min_seq_dist=min_seq_dist)
            
            contact_density = (chainA_density + chainB_density) / 2

            score =  (0.4*iptm + 0.2*ptm + 0.1*plddt + 0.3*contact_density) * penalty
            

        score = round(score, 3)

        #=============================== SCORING ===============================#
        #=======================================================================# 
        
        
        structure = gzip_str(structure)

        # Create the dictionary for this specific row
        row_data = {
            'gndx': gen_i,
            'id': uid, 
            'beta': args.beta,
            'plddt': plddt,
            'ptm': ptm, 
            'iptm': iptm,
            'cd': contact_density,
            'score': score,
            'sequence_data': seq_data, 
            'mutation': mutation,
            'prev_id': prev_id,
            'structure': structure,
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

evolver = Evolver()


#backup if output directory exists
if args.nobackup:
    if os.path.isdir(args.outpath):
        print(f'\nWARNING! Directory {args.outpath} exists, it will be replaced!\n')
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath)
else:
    backup_output(args.outpath)
    

if __name__ == '__main__':

    fold_evolution_simulator(args, evolver)


