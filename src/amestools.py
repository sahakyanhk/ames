import os
import sys
import gzip
import base64
import json
import uuid
import shutil
import argparse
import numpy as np
import pandas as pd
import typing as T
from pathlib import Path
from datetime import datetime

from seqtools import Seqstat

# Resolve repo root from this file's location so paths work from any directory
DATA_DIR = Path(__file__).resolve().parent / "data/"

protein_seqstat = Seqstat(str(DATA_DIR / 'pfam80_stat.json'))
rna_seqstat = Seqstat(str(DATA_DIR / 'rnacentral90_stat.json'))
seqstat = {"protein": protein_seqstat, "rna": rna_seqstat}


def parse_args() -> argparse.Namespace:
    # First parser: just to get config path
    parser = argparse.ArgumentParser(description='Evolution simulation', add_help=False)
    parser.add_argument('--config', type=str, default=str(DATA_DIR / 'simparam.json'),
                        help='Path to JSON config file')

    args, remaining = parser.parse_known_args()

    with open(args.config, 'r') as f:
        defaults = json.load(f)

    parser = argparse.ArgumentParser(description='Evolution simulation')
    parser.set_defaults(**defaults)

    #selection mode and simulations parameters
    parser.add_argument('--config', type=str, default=str(DATA_DIR / 'simparam.json'), help='default configs')
    parser.add_argument('-sm', '--selection_mode', type=str, help='selection mode\n options: strong, weak, weak2')
    parser.add_argument('-ed', '--evoldict', type=str, help='a dictionary with parameters for simulation')

    parser.add_argument('-pa', '--protein_alphabet', type=str, help='protein_alphabet [uniform, uniprot, codonrates]')
    parser.add_argument('-ra', '--rna_alphabet', type=str, help='rna_alphabet')
    parser.add_argument('-da', '--dna_alphabet', type=str, help='dna_alphabet')
    parser.add_argument('-pm', '--protein_mutations', type=str, help='protein_mutations [npm, pmo, srs]')
    parser.add_argument('-rm', '--rna_mutations', type=str, help='rna_mutations [npm, pmo, srs]')
    parser.add_argument('-dm', '--dna_mutations', type=str, help='dna_mutations [npm, pmo, srs]')

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
    parser.add_argument('--iseq1', type=str, help='a sequence info to initiate with "protein:random:25:evolv"')
    parser.add_argument('--seq1_init', type=str, help='a sequence to initiate with [random, randoms, or sequence]')
    parser.add_argument('--seq1_type', type=str, help='a sequence to initiate with, if "random" pop_size random sequences will ')
    parser.add_argument('--seq1_len', type=int, help='seq len')
    parser.add_argument('--seq1_evol', help='does this sequence evolve [True/False]', action='store_true')
    parser.add_argument('--seq1_rate', type=float, help='seq1 mutation rate')
    #seq2 setup
    parser.add_argument('--iseq2', type=str, help='a sequence info to initiate with "rna:AGUCGAUCA:25:static"')
    parser.add_argument('--seq2_init', type=str, help='the 2nd sequence to initiate with [random, randoms, or sequence]')
    parser.add_argument('--seq2_type', type=str, help='seq type [proten, rna, dna] ')
    parser.add_argument('--seq2_len', type=int, help='seq len')
    parser.add_argument('--seq2_evol', help='does this sequence evolve [True/False]', action='store_false')
    parser.add_argument('--seq2_rate', type=float, help='seq2 mutation rate')
    parser.add_argument('--ligand', help="ligand(s) provided in slmiles of ccd format separated with commas")
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
    parser.add_argument('--interface_plddt_cutoff', type=float, help='cutoff for interface plddt')
    parser.add_argument('--lig_contact_cutoff', type=float, help='annealing step')
    parser.add_argument('--lig_contact_min_plddt', type=float, help='annealing step')

    #other
    parser.add_argument('--engine', type=str, help="structure prediction engine")
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

    elif not args.iseq2 and args.seq2_init:
        assert args.seq2_type is not None, "if seq2_init is provided seq2_type must also be provided"
        args.seq2_len = len(args.seq2_init)
        if args.seq2_evol is None:
            args.seq2_evol = False
            args.seq1_rate = 1.0
           
    assert args.seq1_evol == True or args.seq2_evol == True, "either seq1_evol or seq2_evol must be True"

    if args.ligand != None:
        args.ligand = args.ligand.split(",")

    if args.max_seq_per_batch is None:
        args.max_seq_per_batch = args.pop_size // 2
    
    #annealing setup
    args.beta = np.clip(args.beta, 0, 709)
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
        

    # determine evoltion type from input params
    args.protein_chain = None
    args.nucleic_chain = None

    if args.seq1_type == 'protein' and args.seq2_type is None:
        args.evolution_type = 'PROTEIN_EVOLUTION'
        args.protein_chain = 'A'
        
    elif args.seq1_type in ['rna', 'dna'] and args.seq2_type is None:
        args.evolution_type = 'NUCLEIC_EVOLUTION'
        args.nucleic_chain = 'A'


    elif args.seq1_type == 'protein' and args.seq2_type == 'protein':
        if args.seq2_evol:
            args.evolution_type = 'PROTEIN_PROTEIN_COEVOLUTION'
        else:
            args.evolution_type = 'PROTEIN_PROTEIN_EVOLUTION'

        args.protein_chain = ["A", "B"]

    elif (args.seq1_type == 'protein' and args.seq2_type in ['rna', 'dna']) \
        or (args.seq1_type in ['rna', 'dna'] and args.seq2_type == 'protein'):

        if args.seq2_evol:
            args.evolution_type = 'PROTEIN_NUCLEIC_COEVOLUTION'
        else:
            args.evolution_type = 'PROTEIN_NUCLEIC_EVOLUTION'
        
        if args.seq1_type == 'protein':
            args.protein_chain = 'A'
            args.nucleic_chain = 'B'
        else:
            args.protein_chain = 'B'
            args.nucleic_chain = 'A'

    elif args.seq1_type in ['rna', 'dna'] and args.seq2_type in ['rna', 'dna']:
        if args.seq2_evol:
            args.evolution_type = 'NUCLEIC_NUCLEIC_COEVOLUTION'
        else:
            args.evolution_type = 'NUCLEIC_NUCLEIC_EVOLUTION'

        args.nucleic_chain = ['A','B']


    if args.ligand:
        if args.evolution_type.endswith('_COEVOLUTION'):
            args.evolution_type = args.evolution_type.replace('_COEVOLUTION', '_LIGAND_COEVOLUTION')
        else:
            args.evolution_type = args.evolution_type.replace('_EVOLUTION', '_LIGAND_EVOLUTION')
    

    if args.evolution_type in ['PROTEIN_EVOLUTION', 'PROTEIN_LIGAND_EVOLUTION', 
                               'NUCLEIC_EVOLUTION', 'NUCLEIC_LIGAND_EVOLUTION']:
        args.seq2 = False 
    else:
        args.seq2 = True #second chain exists (regardless of whether it evolves or not)

    #assign prot/rna chains
    if args.seq2:
        args.polymer_chains = 'A,B' 
    else:
        args.polymer_chains = 'A'

    #assign ligand chains
    if args.ligand:
        chain_id = args.polymer_chains.split(",")[-1]
        args.ligand_chains = ','.join([chr(ord(chain_id) + 1 + i) for i in range(len(args.ligand))]) #asign lig chain ids sequenctially after prot/nuc chains


    #normalize mutation rates so the largest is 1.0
    rate_max = max(args.seq1_rate, args.seq2_rate)
    if args.seq2:
        args.seq1_rate /= rate_max
        args.seq2_rate /= rate_max
    else: 
        print("Only 1 sequence is provided, setting seq_rate = 1.0")
        args.seq1_rate = 1.0
        args.seq2_rate = 0.0

    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.uid = str(uuid.uuid4())
    return args

def save_checkpoint(ckp_gen, args):
    ckeckpoint_path = os.path.join(args.outpath, args.ckp)
    loghead = generate_loghead(args)
    
    with open(ckeckpoint_path, "w") as f:
        f.write(loghead)
    
    ckp_gen.to_csv(ckeckpoint_path, mode='a', index=False, header=True, sep='\t')

def read_header(log_path:str) -> dict:
    
    assert os.path.isfile(log_path), print(log_path, " does not exist")

    def parse_value(value): #Convert string value to appropriate Python type
        value = value.strip()
        # None
        if value == "None":
            return None
        
        # Boolean
        if value == "True":
            return True
        if value == "False":
            return False
        
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Default to string
        return value
    
    loghead_lines = []
    arg_dict = {}

    for line in open(log_path):
        
        if line == "\n":
            continue

        if line.startswith("#"):
            loghead_lines.append(line)            
        else:
            break


    # Parse the config
    for line in loghead_lines:
        if line.startswith("#--"):
            key, value = line.split("=", 1)  # split only on first '='
            key = key.strip().lstrip("#--")
            arg_dict[key] = parse_value(value)

    return arg_dict

def load_checkpoint(checkpoint_path):

    arg_dict = read_header(checkpoint_path)    
    ckp_gen  = pd.read_csv(checkpoint_path, 
                           sep='\t', comment = '#', index_col = None)
    gndx = ckp_gen.gndx[0]

    return  ckp_gen, arg_dict

def generate_loghead(args) -> str:

    params = [f"#--{param:<24} = {value}\n" for param, value in vars(args).items()]

    loghead = f'''#======================== AMESv0.1 ========================#
#WD: {os.getcwd()}
#${' '.join(sys.argv)}
#
#========================= params =========================#
#
''' + ''.join(params).strip() + '''
#
#==========================================================#
'''
    return loghead

# compress and decompres strings:
def gzip_str(cif_str: str) -> str:
    compressed = gzip.compress(cif_str.encode('utf-8'))
    return base64.b64encode(compressed).decode('ascii')

def ungzip_str(b64_str: str) -> str:
    compressed = base64.b64decode(b64_str)
    return gzip.decompress(compressed).decode('utf-8')


def sigmoid(x:T.Union[float, int], L0=0.0, c=0.1) -> float:
    z = c * (L0 - x)
    # Clip to prevent overflow
    z = np.clip(z, -709, 709)  # e^500 is near max float, e^-500 is near 0
    return 1 / (1 + np.exp(z))

def update_beta(args):
    args.beta += args.annealing_step
    args.beta = round(args.beta, 3)

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


def create_init_gen(evolver, args) -> pd.DataFrame:
    '''Create initial generation to start simulation'''
    
    init_gen = pd.DataFrame(columns=["gndx", 
                                     "id", 
                                     "beta", 
                                     "plddt", 
                                     "ptm", 
                                     "iplddt",
                                     "iptm", 
                                     "cd",
                                     "lcd",
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
                "seqstat": seqstat[args.seq1_type].n_gram_prior(randomsequence1)
            }
        } for _ in range(args.pop_size)]
        
        if args.seq2_init == 'randoms':
             init_gen['id'] = [f'initseq{i}' for i in range(args.pop_size)]
        else:
             init_gen['id'] = ['initseq'] * args.pop_size

    elif args.seq1_init == 'randoms':
        init_gen['id'] = [f'initseq{i}' for i in range(args.pop_size)]
        seq_data = []
        for _ in range(args.pop_size):
            randomsequence1 = evolver.randomseq(args.seq1_type, args.seq1_len)
            seq_data.append(
                {"seq1": 
                    {
                    "type": args.seq1_type, 
                    "sequence": randomsequence1,
                    "ss": "SECONDARYSTRUCTURES",
                    "len": args.seq1_len,
                    "evolve": args.seq1_evol,
                    "seqstat": seqstat[args.seq1_type].n_gram_prior(randomsequence1)
                    }
                })

    else: # predefined sequence
        seq_data = [{
            "seq1": {
                "type": args.seq1_type,
                "sequence": args.seq1_init,
                "len": args.seq1_len,
                "ss": "SECONDARYSTRUCTURES",
                "evolve": args.seq1_evol,
                "seqstat": seqstat[args.seq1_type].n_gram_prior(args.seq1_init)
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
                    "seqstat": seqstat[args.seq2_type].n_gram_prior(randomsequence2)
                }
        
        elif args.seq2_init == 'randoms':
            for seq_data_i in seq_data:
                randomsequence2 = evolver.randomseq(args.seq2_type, args.seq2_len)
                seq_data_i["seq2"] = {
                    "type": args.seq2_type, 
                    "sequence": randomsequence2, 
                    "ss": "SECONDARYSTRUCTURES",
                    "len": args.seq2_len,
                    "evolve": args.seq2_evol,
                    "seqstat": seqstat[args.seq2_type].n_gram_prior(randomsequence2)
                }

        else: # if predefined seq2
            for i in range(args.pop_size):
                seq_data[i]["seq2"] = {
                    "type": args.seq2_type,
                    "sequence": args.seq2_init,
                    "ss": "SECONDARYSTRUCTURES",
                    "len": args.seq2_len,
                    "evolve": args.seq2_evol,
                    "seqstat": seqstat[args.seq1_type].n_gram_prior(args.seq2_init)
                }


    init_gen["gndx"] = 0
    init_gen["beta"] = args.beta
    init_gen["plddt"] = 0.0
    init_gen["ptm"] = 0.0
    init_gen["iplddt"] = 0.0
    init_gen["iptm"] = 0.0
    init_gen["cd"] = 0.0
    init_gen["lcd"] = 0.0
    init_gen["score"] = 0.001 
    init_gen['sequence_data'] = seq_data        
    init_gen["mutation"] = "init_gen"
    init_gen["prev_id"] = "init_gen"
    init_gen["structure"] = "init_gen" 

    init_gen.round(3)

    return init_gen


class ScoringFunction:
    
    def __init__(self, evolution_type):

        self.scoring_weights = {
        'PROTEIN_EVOLUTION':                    {"ptm": 0.4, "plddt": 0.2, "iptm": 0.0,  "iplddt": 0.0,  "cd": 0.4, "lcd": 0.0},
        'NUCLEIC_EVOLUTION':                    {"ptm": 0.5, "plddt": 0.5, "iptm": 0.0,  "iplddt": 0.0,  "cd": 0.0,  "lcd": 0.0},
        'PROTEIN_PROTEIN_EVOLUTION':            {"ptm": 0.2, "plddt": 0.1, "iptm": 0.25, "iplddt": 0.25, "cd": 0.2, "lcd": 0.0},
        'PROTEIN_PROTEIN_COEVOLUTION':          {"ptm": 0.2, "plddt": 0.1, "iptm": 0.25, "iplddt": 0.25, "cd": 0.2, "lcd": 0.0},
        'PROTEIN_NUCLEIC_EVOLUTION':            {"ptm": 0.2, "plddt": 0.1, "iptm": 0.25, "iplddt": 0.25, "cd": 0.2, "lcd": 0.0},
        'PROTEIN_NUCLEIC_COEVOLUTION':          {"ptm": 0.2, "plddt": 0.1, "iptm": 0.25, "iplddt": 0.25, "cd": 0.2, "lcd": 0.0},
        'NUCLEIC_NUCLEIC_EVOLUTION':            {"ptm": 0.2, "plddt": 0.2, "iptm": 0.3, "iplddt": 0.3, "cd": 0.0, "lcd": 0.0},
        'NUCLEIC_NUCLEIC_COEVOLUTION':          {"ptm": 0.2, "plddt": 0.2, "iptm": 0.3, "iplddt": 0.3, "cd": 0.0, "lcd": 0.0},
        'PROTEIN_LIGAND_EVOLUTION':             {"ptm": 0.1, "plddt": 0.1, "iptm": 0.2, "iplddt": 0.2, "cd": 0.2, "lcd": 0.2},
        'NUCLEIC_LIGAND_EVOLUTION':             {"ptm": 0.1, "plddt": 0.1, "iptm": 0.25, "iplddt": 0.3, "cd": 0.0, "lcd": 0.25},
        'PROTEIN_PROTEIN_LIGAND_EVOLUTION':     {"ptm": 0.1, "plddt": 0.1, "iptm": 0.2, "iplddt": 0.2, "cd": 0.2, "lcd": 0.2},
        'PROTEIN_PROTEIN_LIGAND_COEVOLUTION':   {"ptm": 0.1, "plddt": 0.1, "iptm": 0.2, "iplddt": 0.2, "cd": 0.2, "lcd": 0.2},
        'PROTEIN_NUCLEIC_LIGAND_EVOLUTION':     {"ptm": 0.1, "plddt": 0.1, "iptm": 0.2, "iplddt": 0.2, "cd": 0.2, "lcd": 0.2},
        'PROTEIN_NUCLEIC_LIGAND_COEVOLUTION':   {"ptm": 0.1, "plddt": 0.1, "iptm": 0.2, "iplddt": 0.2, "cd": 0.2, "lcd": 0.2},
        'NUCLEIC_NUCLEIC_LIGAND_EVOLUTION':     {"ptm": 0.1, "plddt": 0.1, "iptm": 0.25, "iplddt": 0.3, "cd": 0.0, "lcd": 0.25},
        'NUCLEIC_NUCLEIC_LIGAND_COEVOLUTION':   {"ptm": 0.1, "plddt": 0.1, "iptm": 0.25, "iplddt": 0.3, "cd": 0.0, "lcd": 0.25}
                   }       
        self.weigths = self.scoring_weights[evolution_type]

    def score(self, ptm, plddt, iptm, iplddt, contact_density, ligand_contact_density,  penalty):
        
        s = (self.weigths["iptm"]*iptm + 
             self.weigths["iplddt"]*iplddt + 
             self.weigths["ptm"]*ptm + 
             self.weigths["plddt"]*plddt + 
             self.weigths["cd"]*contact_density + 
             self.weigths["lcd"]*ligand_contact_density) * penalty
        
        return round(s, 3)

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

    genlog = genlog.tail(args.pop_size).drop(columns=['gndx', 'structure']).round(3)
    
    genlog['sequence_data'] = genlog['sequence_data'].apply(extract_sequence)
    print(genlog.to_string(index=False, header=True))

