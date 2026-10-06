import os
import sys
import json
import uuid
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from pdbutils import parsepdb
from seqtools import fasta2dict

DATA_DIR = Path(__file__).resolve().parent / "data/"


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
    parser.add_argument('-ed', '--evoldict', type=str, help='alphabets and mutations')
    parser.add_argument('-sm', '--selection_mode', type=str, help='selection mode\n options: strong, weak, weak2')

    parser.add_argument('-a1', '--seq1_alphabet', type=str, help='protein_alphabet [uniform, uniprot, codonrates]')
    parser.add_argument('-m1', '--seq1_mutations', type=str, help='protein_mutations [npm, pmo, rso]')
    parser.add_argument('-a2', '--seq2_alphabet', type=str, help='protein_alphabet [uniform, uniprot, codonrates]')
    parser.add_argument('-m2', '--seq2_mutations', type=str, help='protein_mutations [npm, pmo, rso]')

    #pop_size and num generations
    parser.add_argument('-ng', '--num_generations', type=int, help='number of generations')
    parser.add_argument('-ps', '--pop_size', type=int, help='population size')
    #temperature control
    parser.add_argument('-b0', '--beta', type=float, help='selection strength, the higher beta the lower temperature and stronger selection')
    parser.add_argument('-ann','--annealing', action='store_true', help='use temperature annealying, see -b0, -bt, -ann_s, -ann_e, or -ann_step for annealing setup')
    parser.add_argument('-bt', '--beta_target', type=float, help='')
    parser.add_argument('-ann_s', '--annealing_start', type=int, help='generation when annealing starts')
    parser.add_argument('-ann_e','--annealing_end', type=int, help='generation when annealing reaches target beta')
    parser.add_argument('-ann_step','--annealing_step', type=int, help='annealing step')
    #seq1 setup
    parser.add_argument('--iseq1', type=str, help='1st seq info to initiate simulation "protein:random:25:evolv"')
    parser.add_argument('--seq1_init', type=str, help='1st seqence "sequence" - actual sequence, "random" - the same random sequence for entire populations, \
                                                        "randoms" - each sequence in the population is random')
    parser.add_argument('--seq1_type', type=str, help='1st seq type [proten, rna, dna] ')
    parser.add_argument('--seq1_len', type=int, help='if random(s) selected provide random sequence length, ignore if real sequence is provided')
    parser.add_argument('--seq1_evol', help='does seq1 evolve [True/False]', action='store_true')
    parser.add_argument('--seq1_rate', type=float, help='probability of 1st sequence to mutate [0,1]')
    #seq2 setup
    parser.add_argument('--iseq2', type=str, help='2nd seq info to initiate simulation "rna:AGUCGAUCA:25:static"')
    parser.add_argument('--seq2_init', type=str, help='2nd seqence "sequence" - actual sequence, "random" - the same random sequence for entire populations, \
                                                        "randoms" - each sequence in the population is random')
    parser.add_argument('--seq2_type', type=str, help='2nd seq type [proten, rna, dna] ')
    parser.add_argument('--seq2_len', type=int, help='if random(s) selected provide random sequence length, ignore if real sequence is provided')
    parser.add_argument('--seq2_evol', help='does seq2 evolve [True/False]', action='store_false')
    parser.add_argument('--seq2_rate', type=float, help='probability of 2nd sequence to mutate [0,1]')
    #ligand setup
    parser.add_argument('--ligand', help="ligand(s) provided in slmiles of ccd format separated with commas")
    #constraints
    parser.add_argument('--seq1_min_len', type=int, help='seq1 minimal length constraint')
    parser.add_argument('--seq1_max_len', type=int, help='seq1 maximal length constraint')
    parser.add_argument('--seq2_min_len', type=int, help='seq2 minimal length constraint')
    parser.add_argument('--seq2_max_len', type=int, help='seq2 maximal length constraint')
    parser.add_argument('--helix_len_penalty', type=int, help='alpha-helix maximal length constraint')
    parser.add_argument('--strand_len_penalty', type=int, help='beta-strand maximal length constraint')
    #outputs
    parser.add_argument('-o','--outpath', type=str, help='output dir name where log and checkpoint files are saved, "ames_output/output" by default')
    parser.add_argument('-l', '--log', type=str, help='output log file name, "progress.log" by default')
    parser.add_argument('-c', '--ckp', type=str, help='checkpoint file name, "progress.ckp" by default')    
    parser.add_argument('-ckpi', '--checkpoint_interval', type=int, help='chechpoint saving frequency (generations)')
    parser.add_argument('--nobackup', action='store_true', help='overwrite output if exists')
    #contact calculatsion
    parser.add_argument('--contact_min_seq_dist', type=int, help='minimum sequence distance for contact calculation')
    parser.add_argument('--contact_cutoff', type=float, help='cutoff distance for contact calculation')
    parser.add_argument('--contact_min_plddt', type=float, help='minimum plddt for contact calculation')
    parser.add_argument('--interface_plddt_cutoff', type=float, help='cutoff distance for interface plddt')
    parser.add_argument('--lig_contact_cutoff', type=float, help='cutoff distance for ligand contact calculation')
    parser.add_argument('--lig_contact_min_plddt', type=float, help='minimum plddt for ligand contact calculation')
    parser.add_argument('--clash_overlap_threshold', type=float, help='cutoff distance for clash calculation in angstroms')
    parser.add_argument('--clash_min_seq_dist', type=int, help='minimum sequence distance for clash calculation')
    #other
    parser.add_argument('--rfam_scoring', type=str, help="reward RNA function based on RFAM search")
    parser.add_argument('--engine', type=str, help="structure prediction engine [esmfold2, alphafold3, openfold3, esmfold, simulacrum]")
    parser.add_argument('--norepeat', action='store_true', help='do not generate and/or select the same sequences more than once, off by default')
    parser.add_argument('--max_seq_per_batch', type=int, help='max_seq_per_batch, half or population size by default ')
    parser.add_argument('--structure_template1', '-strtmpl1', type=str, help="path to the first structure template (pdb/cif)")
    parser.add_argument('--structure_template2', '-strtmpl2', type=str, help="path to the second structure template (pdb/cif)")
    parser.add_argument('--sequence_template', '-seqtmpl', type=str, help="path to the first sequence template (PDB/FASTA/sequence string)")


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

    #prepare templates
    if args.structure_template1:
        if os.path.isfile(args.structure_template1):
            args.structure_template1 = Path(args.structure_template1).resolve()
        else:
            raise FileNotFoundError(f"Structure template 1 file not found: {args.structure_template1}")

    # if args.structure_template2:
    #     if os.path.isfile(args.structure_template2):
    #         with open(args.structure_template2, 'r') as f:
    #             args.structure_template2 = f.read().strip()
    #     else:
    #         raise FileNotFoundError(f"Structure template 2 file not found: {args.structure_template2}")

    if args.sequence_template:
        if os.path.isfile(args.sequence_template):

            if args.sequence_template.endswith(('.fasta', '.fas', '.fa')):
                fasta_dict = fasta2dict(args.sequence_template)
                args.sequence_template = next(iter(fasta_dict.values())).strip() 

            elif args.sequence_template.endswith(('.pdb')):
                args.sequence_template = parsepdb(args.sequence_template).sequence[0]

            elif set(args.sequence_template).issubset(set('ACDEFGHIKLMNPQRSTVWY')): 
                args.sequence_template = args.sequence_template.upper()

            else:
                raise ValueError(f"sequence template should be either a FASTA file, \
                                 PDB file, or a amino acid sequence. Got: {args.sequence_template}")
    return args



