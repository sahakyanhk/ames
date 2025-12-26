import os
import sys
import gzip
import json
import uuid
import shutil
import argparse
import numpy as np
import pandas as pd
import typing as T
from pathlib import Path
from datetime import datetime

from af3_runner import af3_runner as structure_predictor



def parse_args():
    # First parser: just to get config path
    parser = argparse.ArgumentParser(description='Evolution simulation', add_help=False)
    parser.add_argument('--config', type=str, default='data/simparam.json',
                        help='Path to JSON config file')

    args, remaining = parser.parse_known_args()

    with open(args.config, 'r') as f:
        defaults = json.load(f)

    parser = argparse.ArgumentParser(description='Evolution simulation')
    parser.set_defaults(**defaults)

    #selection mode and simulations parameters
    parser.add_argument('--config', type=str, default='../data/simparam.json', help='default configs')
    parser.add_argument('-sm', '--selection_mode', type=str, help='selection mode\n options: strong, weak, weak2')
    parser.add_argument('-ed', '--evoldict', type=str, help='a dictionary with parameters for simulation')
    #pop_size and num generations
    parser.add_argument('-ng', '--num_generations', type=int, help='number of generations')
    parser.add_argument('-ps', '--pop_size', type=int, help='population size')
    #temperature control
    parser.add_argument('-b0', '--beta', type=float, help='selection strength, the higher beta the lower temperature and stronger selection')
    parser.add_argument('-ann','--annealing', action='store_true', help='')
    parser.add_argument('-bt', '--beta_target', type=float, help='')
    parser.add_argument('-ann_s', '--annealing_start', type=int, help='generation when annealing starts')
    parser.add_argument('-ann_e','--annealing_end', type=int, help='generation when annealing reaches target beta')
    parser.add_argument('-ann_step','--annealing_step', type=int, help='annealing step')
    #seq1 setup
    parser.add_argument('--iseq1', type=str, help='a sequence info to initiate with "protein:random:25:evolves"')
    parser.add_argument('--seq1_init', type=str, help='a sequence to initiate with [random, randoms, or sequence]')
    parser.add_argument('--seq1_type', type=str, help='a sequence to initiate with, if "random" pop_size random sequences will ')
    parser.add_argument('--seq1_len', type=int, help='seq len')
    parser.add_argument('--seq1_evol', help='does this sequence evolve [True/False]', action='store_true')
    #seq2 setup
    parser.add_argument('--iseq2', type=str, help='a sequence info to initiate with "protein:random:25:evolves"')
    parser.add_argument('--seq2_init', type=str, help='the 2nd sequence to initiate with [random, randoms, or sequence]')
    parser.add_argument('--seq2_type', type=str, help='seq type [proten, rna, dna] ')
    parser.add_argument('--seq2_len', type=int, help='seq len')
    parser.add_argument('--seq2_evol', help='does this sequence evolve [True/False]', action='store_false')
    parser.add_argument('--ligand', help="ligand(s) provided in slmiles of ccd format")
    #constraints
    parser.add_argument('--seq1_len_constr', type=int, help='constain seq1 length')
    parser.add_argument('--seq2_len_constr', type=int, help='constain seq2 length')
    #outputs
    parser.add_argument('-o','--outpath', type=str, help='output filepath for saving sampled sequences')
    parser.add_argument('-l', '--log', type=str, help='output log file name')
    parser.add_argument('-c', '--ckp', type=str, help='checkpoint file name')    
    parser.add_argument('-ckpi', '--checkpoint_interval', type=int, help='chechpoint saving frequency (generations)')
    parser.add_argument('--nobackup', action='store_true', help='overwrite files if exists')
    #contact calculatsion
    parser.add_argument('--contact_min_seq_dist', type=int, help='annealing step')
    parser.add_argument('--contact_cutoff', type=float, help='annealing step')
    parser.add_argument('--contact_min_plddt', type=float, help='annealing step')
    #other
    parser.add_argument('--norepeat', action='store_true', help='do not generate and/or select the same sequences more than once')
    parser.add_argument('--max_seq_per_batch', type=int, help='max_seq_per_batch, by defaulf it is half or population size')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(remaining)

    if args.iseq1:
        try:
            iseqinfo = args.iseq1.split(':')
            if len(iseqinfo) != 4:
                raise ValueError("Expected 4 colon-separated values")
            
            if iseqinfo[0] not in ["protein", "rna", "dna"]:
                raise ValueError(f"First field must be 'protein', 'rna', or 'dna', got '{iseqinfo[0]}'")
            args.seq1_type = iseqinfo[0]
            args.seq1_init = iseqinfo[1]
            args.seq1_len = int(iseqinfo[2])  
            
            if iseqinfo[3] not in ["evolv", "static"]:
                raise ValueError(f"Fourth field must be 'evolv' or 'static', got '{iseqinfo[3]}'")
            
            args.seq1_evol = (iseqinfo[3] == "evolv")
            
        except ValueError as e:
            print(f"ERROR: Invalid -iseq1 input: {e}")
            print("Valid input example: 'protein:random:50:evolv'")
            sys.exit(1)

    if args.iseq2:
        try:
            iseqinfo = args.iseq2.split(':')
            if len(iseqinfo) != 4:
                raise ValueError("Expected 4 colon-separated values")
            
            if iseqinfo[0] not in ["protein", "rna", "dna"]:
                raise ValueError(f"First field must be 'protein', 'rna', or 'dna', got '{iseqinfo[0]}'")
            args.seq2_type = iseqinfo[0]
            args.seq2_init = iseqinfo[1]
            args.seq2_len = int(iseqinfo[2])  
            
            if iseqinfo[3] not in ["evolv", "static"]:
                raise ValueError(f"Fourth field must be 'evolv' or 'static', got '{iseqinfo[3]}'")
            
            args.seq2_evol = (iseqinfo[3] == "evolv")
            
        except ValueError as e:
            print(f"ERROR: Invalid -iseq2 input: {e}")
            print("Valid input example: 'protein:random:50:evolv'")
            sys.exit(1)

    assert args.seq1_evol == True or args.seq2_evol == True, "either seq1_evol or seq1_evol must be True"

    args.prediction_engine = "AF3"


    if args.max_seq_per_batch is None:
        args.max_seq_per_batch = args.pop_size // 2
    
    args.beta = np.clip(args.beta, 0, 709)
    #annealing setup
    
    if args.annealing: 

        if args.beta_target is None:
            print("a target beta must be provided for temperature annealing")
            sys.exit(1)
        else:
            args.beta_target = np.clip(args.beta_target, 0, 709)

        if args.annealing_start is None:
            args.annealing_start = int(args.num_generations * 0.2)
            print(f"WARNING: annealing_start is not provided Annealing will start generation {args.annealing_start} by default")
    
        if args.annealing_end is None:
            args.annealing_end = int(args.num_generations * 0.8)
            print(f"WARNING: annealing_end is not provided Annealing will end generation {args.annealing_end} by default")
        
        if args.annealing_step is None:
            args.annealing_step = round((args.beta_target - args.beta) / (args.annealing_end - args.annealing_start), 3) #linearly increas beta from ann_start to ann_end
        
    args.protein_chain = None
    args.nucleic_chain = None

    # determine evoltion type from input params
    if args.seq1_type == 'protein' and args.seq2_type is None:
        args.evolution_type = 'PROTEIN_FOLD_EVOLUTION'
        args.protein_chain = 'A'

    elif args.seq1_type in ['rna', 'dna'] and args.seq2_type is None:
        args.evolution_type = 'NA_FOLD_EVOLUTION'
        args.nucleic_chain = 'A'

    elif args.seq1_type == 'protein' and args.seq2_type == 'protein':
        if args.seq2_evol:
            args.evolution_type = 'PROTEIN_COMPLEX_COEVOLUTION'
        else:
            args.evolution_type = 'PROTEIN_COMPLEX_EVOLUTION'

        args.protein_chain = ["A", "B"]

    elif (args.seq1_type == 'protein' and args.seq2_type in ['rna', 'dna']) \
        or (args.seq1_type in ['rna', 'dna'] and args.seq2_type == 'protein'):

        if args.seq2_evol:
            args.evolution_type = 'PROTEIN_NA_COEVOLUTION'
        else:
            args.evolution_type = 'PROTEIN_NA_EVOLUTION'
        
        if args.seq1_type == 'protein':
            args.protein_chain = 'A'
            args.nucleic_chain = 'B'
        else:
            args.protein_chain = 'B'
            args.nucleic_chain = 'A'

    elif args.seq1_type in ['rna', 'dna'] and args.seq2_type in ['rna', 'dna']:
        if args.seq2_evol:
            args.evolution_type = 'NA_COMPLEX_COEVOLUTION'
        else:
            args.evolution_type = 'NA_COMPLEX_EVOLUTION'

        args.nucleic_chain = ['A','B']


    if args.evolution_type in ['PROTEIN_FOLD_EVOLUTION', 'NA_FOLD_EVOLUTION']:
        args.seq2 = False
    else:
        args.seq2 = True


    return args

def save_checkpoint(generation, args):
    ckekpoint_path = os.path.join(args.outpath, args.ckp)
    loghead = generate_loghead(args)
    
    with open(ckekpoint_path, "w") as f:
        f.write(loghead)
    
    generation.to_csv(ckekpoint_path, mode='a', index=True, header=True, sep='\t')





# def load_checkpoint():

def generate_loghead(args) -> str:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())
    params = [f"#--{param:<24} = {value}\n" for param, value in vars(args).items()]

    loghead = f'''
#>======================= AMESv0.1 =======================<#
#>=================== {timestamp} ====================<#
#>======== {uid} ==========<#
#WD: {os.getcwd()}
#${' '.join(sys.argv)}
#
#======================= input params =====================#
#
''' + ''.join(params).strip() + '''
#
#==========================================================#
'''
    return loghead

def gzip_str(cif_str):
    return gzip.compress(cif_str.encode('utf-8'))


def sigmoid(x:float|int, L0=0.0, c=0.1) -> float:
    z = c * (L0 - x)
    # Clip to prevent overflow
    z = np.clip(z, -709, 709)  # e^500 is near max float, e^-500 is near 0
    return 1 / (1 + np.exp(z))

def update_beta(args):
    args.beta += args.annealing_step



def gc_content(seq:str) -> float:
    return round(((seq.count('C') + seq.count('G')) / len(seq)), 3)


def backup_output(directory_path, backup_suffix=None, max_backups=None) -> T.Optional[str]:

    """Backup a directory if it exists by renaming it with a timestamp"""
    
    def cleanup_old_backups(parent_dir, base_name, max_backups):
        backup_pattern = f"{base_name}_backup_*"
        backups = sorted(
            parent_dir.glob(backup_pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True)

        for old_backup in backups[max_backups:]:
            try:
                shutil.rmtree(old_backup)
                print(f"Removed old backup: {old_backup}")
            except OSError as e:
                print(f"Warning: Could not remove old backup {old_backup}: {e}")

    directory_path = Path(directory_path)    

    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        return None
    
    if not directory_path.is_dir():
        raise ValueError(f"Path exists but is not a directory: {directory_path}")
    
    if backup_suffix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_suffix = f"backup_{timestamp}"
    
    backup_path = directory_path.parent / f"{directory_path.name}_{backup_suffix}"
    
    # if backup path already exists
    counter = 1
    original_backup_path = backup_path
    while backup_path.exists():
        backup_path = Path(f"{original_backup_path}_{counter}")
        counter += 1
    
    try:
        shutil.move(str(directory_path), str(backup_path))
        print(f"{directory_path} exists, creating backup: {backup_path}")
        
        if max_backups is not None:
            cleanup_old_backups(directory_path.parent, directory_path.name, max_backups)
        
        directory_path.mkdir(parents=True, exist_ok=True)
                
        return str(backup_path)
    
    except (OSError, shutil.Error) as e:
        raise OSError(f"Failed to backup directory: {e}")



def batch_sequence_dataset(sequences: T.List[T.Tuple[str, dict]], 
                           pop_size: int = 50, 
                           max_seq_per_batch: int = 25
                        ) -> T.Generator[T.Tuple[T.List[str], T.List[dict]], None, None]:

    batch_headers, batch_sequences, num_sequences= [], [], 0 

    for header, seq in sequences:

        if num_sequences > max_seq_per_batch:
            yield batch_headers, batch_sequences

            batch_headers, batch_sequences, num_sequences= [], [], 0

        batch_headers.append(header)
        batch_sequences.append(seq)
        num_sequences += 1

        if num_sequences > pop_size: #TODO test this with args.pop_size / 4 and lartge pop size
           yield batch_headers, batch_sequences

           batch_headers, batch_sequences, num_sequences= [], [], 0

    yield batch_headers, batch_sequences


def prepare_af3_input(seq_data_list, args) -> list[list[dict]]:
    
    """prepares sequences in generation dataframe for af3 input"""

    if isinstance(seq_data_list, dict):
        seq_data_list = [seq_data_list]

    if args.seq2:
        inputs = [[{"type": args.seq1_type, "sequence": data["seq1"]["sequence"], "id": "A"},
                   {"type": args.seq2_type, "sequence": data["seq2"]["sequence"], "id": "B"}] for data in seq_data_list]
    else:
        inputs = [[{"type": args.seq1_type, "sequence": data["seq1"]["sequence"], "id": "A"}] for data in seq_data_list]

    return inputs


def create_init_gen(evolver, args) -> pd.DataFrame:
    '''Create initial generation to start simulation'''
    
    init_gen = pd.DataFrame(columns=["gndx", 
                                     "id", 
                                     "beta", 
                                     "plddt", 
                                     "ptm", 
                                     "iptm", 
                                     "cd",
                                     "score",  
                                     "sequence_data", 
                                     "mutation", 
                                     "prev_id", 
                                     "structure"])

    if args.seq1_init == 'random':
        randomsequence1 = evolver.randomseq(args.seq1_type, args.seq1_len)
        
        seq_data = [{
            "seq1": {
                "type": args.seq1_type, 
                "sequence": randomsequence1,
                "ss": "SECONDARYSTRUCTURES", 
                "len": args.seq1_len,
                "evolve": args.seq1_evol,
            }
        } for _ in range(args.pop_size)]
        
        if args.seq2_init == 'randoms':
             init_gen['id'] = [f'initseq{i}' for i in range(args.pop_size)]
        else:
             init_gen['id'] = ['initseq'] * args.pop_size

    elif args.seq1_init == 'randoms':
        init_gen['id'] = [f'initseq{i}' for i in range(args.pop_size)]
        seq_data = [{
            "seq1": {
                "type": args.seq1_type, 
                "sequence": evolver.randomseq(args.seq1_type, args.seq1_len), 
                "ss": "SECONDARYSTRUCTURES",
                "len": args.seq1_len,
                "evolve": args.seq1_evol,
            }
        } for i in range(args.pop_size)]

    else: # predefined sequence
        seq_data = [{
            "seq1": {
                "type": args.seq1_type,
                "sequence": args.seq1_init,
                "len": args.seq1_len,
                "ss": "SECONDARYSTRUCTURES",
                "evolve": args.seq1_evol,
            }
        } for _ in range(args.pop_size)]
        
        if args.seq2_init == 'randoms':
             init_gen['id'] = [f'initseq{i}' for i in range(args.pop_size)]
        else:
             init_gen['id'] = ['initseq'] * args.pop_size


    if args.seq2:
        if args.seq2_init == 'random':
            randomsequence2 = evolver.randomseq(args.seq2_type, args.seq2_len)
            for i in range(args.pop_size):
                seq_data[i]["seq2"] = {
                    "type": args.seq2_type, 
                    "sequence": randomsequence2, 
                    "ss": "SECONDARYSTRUCTURES",
                    "len": args.seq2_len,
                    "evolve": args.seq2_evol,
                }
        
        elif args.seq2_init == 'randoms':
            for i in range(args.pop_size):
                randomsequence2 = evolver.randomseq(args.seq2_type, args.seq2_len)
                seq_data[i]["seq2"] = {
                    "type": args.seq2_type, 
                    "sequence": randomsequence2, 
                    "ss": "SECONDARYSTRUCTURES",
                    "len": args.seq2_len,
                    "evolve": args.seq2_evol,
                }
        
        else: # if predefined seq2
            for i in range(args.pop_size):
                seq_data[i]["seq2"] = {
                    "type": args.seq2_type,
                    "sequence": args.seq2_init,
                    "ss": "SECONDARYSTRUCTURES",
                    "len": args.seq2_init,
                    "evolve": args.seq2_evol,
                }

    init_gen['sequence_data'] = seq_data
    
    fold_input = prepare_af3_input(init_gen['sequence_data'], args)

    if args.seq1_init == 'random' and (args.seq2_init == 'random' or args.seq2_init is None):
        #do not predict the same structure multiple times
        tmp_pdbs, init_gen_plddt, init_gen_ptm, init_gen_iptm, ranking_scores = structure_predictor(fold_input[0]) # type: ignore
        tmp_pdbs, init_gen["plddt"], init_gen["ptm"], init_gen["iptm"], _ = tmp_pdbs * args.pop_size, \
                                                                                                    init_gen_plddt * args.pop_size, \
                                                                                                    init_gen_ptm * args.pop_size, \
                                                                                                    init_gen_iptm * args.pop_size, \
                                                                                                    ranking_scores * args.pop_size
    else:
        tmp_pdbs, init_gen["plddt"], init_gen["ptm"], init_gen["iptm"], _ = structure_predictor(fold_input) # type: ignore

    init_gen["structure"] = [gzip_str(pdb) for pdb in tmp_pdbs]
    init_gen["cd"] = 0.0
    init_gen["score"] = 0.01 # 0.5 * init_gen["ptm"] + 0.5 * init_gen["plddt"] # TODO automatically adjust score based on sequene types
    init_gen["beta"] = args.beta
    init_gen['mutation'] = 'init_gen'
    init_gen['prev_id'] = 'init_gen'
    init_gen['gndx'] = 0

    init_gen.round(3)

    return init_gen


# def single_prot_score():
#     return score


def extract_sequence(seq_data: dict) -> str:
    seq1 = seq_data['seq1']['sequence']
    ss1 = seq_data['seq1']['ss']
    print_srting = f"{seq1}[{ss1}]"

    if 'seq2' in seq_data:
        seq2 = seq_data['seq2']['sequence']
        ss2 = seq_data['seq2']['ss']
        print_srting += f":{seq2}[{ss2}]"

    return print_srting


def print_genlog(genlog:pd.DataFrame, args) -> None:

    genlog = genlog.tail(args.pop_size).drop(columns=['gndx', 'structure'], 
                                             axis=1).round(3)
    
    genlog['sequence_data'] = genlog['sequence_data'].apply(extract_sequence)
    print(genlog.to_string(index=False, header=True))

