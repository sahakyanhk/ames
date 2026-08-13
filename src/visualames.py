#!/usr/bin/env python3

import argparse
import os, re, sys
import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from amestools import DATA_DIR

# Ensure src/ is on sys.path so sibling modules are importable from any directory
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from amestools import read_header, decompress_str
from pdbutils import extract_backbone
from seqtools import Seqstat


parser = argparse.ArgumentParser(description="Analyse PFES")
parser.add_argument('-l', '--log', type=str, help='log file name', default='progress.log') 
parser.add_argument('-o', '--outdir', type=str, help='output directory name')
parser.add_argument('-b', '--start', type=int, help='first point to read from trajectory', default=0)
parser.add_argument('-e', '--end', type=int, help='last point to read from trajectory', default=99999999)

parser.add_argument('--reseqstat', action='store_true', )
parser.add_argument('--noplots', action='store_false', )
parser.add_argument('--nostr', action='store_false', )


args = parser.parse_args()

if args.outdir is None:
    args.outdir = os.path.dirname(args.log)


protein_seqstat = Seqstat(str(DATA_DIR / 'pfam80_stat.json'))
rna_seqstat = Seqstat(str(DATA_DIR / 'rnacentral90_stat.json'))
seqstat = {"protein": protein_seqstat, "rna": rna_seqstat}


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def extract_lineage(log) -> pd.DataFrame:
    traj_len = len(log)
    #pop_size = len(log[log.gndx == 'gndx0'])

    print(f'Processing a trajectory from "{args.log}" \nwith {traj_len} mutations')
    lineage = log.drop_duplicates('gndx').tail(1)
    df = lineage
    ndx = df.id.to_string(index=False)
        
    def return_ancestor(log, node):
        parent = log[log.id == node]
        parent = parent.drop_duplicates('sequence_data')
        return parent

    
    pbar = tqdm(desc='Extracting lineage')
    i=0
    while not df.empty:
        ndx = return_ancestor(log, ndx)
        df = ndx
        lineage = pd.concat([lineage, df], axis=0)
        ndx = ndx.prev_id.to_string(index=False)
        i+=1
        pbar.update(i)
    pbar.close()
    lineage = lineage.sort_index()
    ltail = lineage.tail(1)

    print(f"""
{ltail[['gndx',
        'id',
        'ptm',
        'plddt',
        'score'
        ]].iloc[-1]}
{json.dumps(ltail.sequence_data.iloc[-1], indent=4)}
""")
    lineage["lndx"] = lineage.reset_index().index
    #lineage["evolrate"] = 1 - lineage["homogen"] #future update
    lineage["evolrate"] = 0

    lineage_cols = ["gndx", "lndx", "id", "mutation", "beta", "evolrate",
                    "plddt", "ptm", "iplddt", "iptm", "cd", "lcd",
                    "n_atoms", "n_clashes", "clashscore", "rfam_score", "rfam_hit", "score", 
                    "seq1", "seq1_ss", "seq1_len", "seq1_stat"]
    if simparam["seq2"]:
        lineage_cols += ["seq2", "seq2_ss", "seq2_len", "seq2_stat"]
    lineage_cols += ["structure"]
    return lineage[lineage_cols]


def extract_sequences(log):
    fasta  = "fasta"
    return fasta

#======================= extract structures and make traj =======================#

def extract_structures(log, outdir) -> None:
    
    structures_path = os.path.join(outdir, 'structures')

    os.makedirs(structures_path, exist_ok=True)
    
    decoded_frames = []

    for id, compressed_structure in zip(log.gndx[1:], log.structure[1:]):

        try:

            structure_txt = decompress_str(compressed_structure)
            with open(f"{structures_path}/{id:04d}.pdb", "w") as f:
                f.write(structure_txt)

            decoded_frames.append(structure_txt)

        except Exception as e:
            print(f"{id} failed\n{e}")


#======================= make separate plots =======================#

labels = {
    "gndx": "Generation index",
    "lndx": "Lineage index",
    "evolrate": "Evolution rate",
    "score": "Score",
    "ptm": "pTM", 
    "plddt": "pLDDT", 
    "iptm": "ipTM",  
    "iplddt": "ipLDDT",  
    "cd": "Contact Density", 
    "lcd": "Ligand Contact Density",
    "beta": "Selection strength",
    "penalty": "Penalty",
    "seq1_len": "Seq1 len",
    "seq2_len": "Seq2 len",
    "seq1_stat": "Seq1 ngram loss",
    "seq2_stat": "Seq2 ngram loss",
    "n_atoms": "Number of atoms",
    "n_clashes": "Number of clashes",
    "clashscore": "Clashscore",
    "rfam_score": "RFAM score",
    "rfam_hit": "RFAM",
        }


def make_plots(log, bestlog, lineage):

    ms=0.1
    lw=1.4
    dpi=500

    os.makedirs(plotdir, exist_ok=True)
    for colname in log.keys(): 
        if colname in ['beta', 'plddt', 'ptm', 'iplddt', 'iptm', 
                       'cd', 'lcd', 'score', 'evolrate', 
                       'n_atoms', 'n_clashes', 'clashscore', 
                       'rfam_score', "rfam_hit",
                       'seq1_len', 'seq1_stat',
                       'seq2_len', 'seq2_stat']:
                
                fig, ax1 = plt.subplots(figsize=(9, 3))
                ax1.plot(log[colname],'.', markersize=ms,    color='silver', label='all mutations')
                ax1.plot(bestlog[colname],'-', linewidth=lw, label='best of the generation')
                ax1.plot(lineage[colname],'-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={len(lineage[colname])})')
                ax1.legend(loc ="lower right")
                ax1.grid(True, which="both",linestyle='--', linewidth=0.3)
                ax1.set(xlabel="Total number of mutations", ylabel=colname.capitalize())
                #ax2 = ax1.twiny()
                #ax2.plot(lineage[colname].tolist(),'-', linewidth=lw, color='mediumslateblue')
                #ax2.set(xlabel="Lineage lenght")
                fig.tight_layout()
                fig.savefig(plotdir + colname + '.png', dpi=dpi)
                fig.clf()

#======================= Summary plot =======================#
def make_summary_plot(log, bestlog, lineage, simparam):
    fig, axs = plt.subplots(3,2, figsize=(10, 8))
    fig.suptitle(None) # type: ignore

    ms=0.1
    lw=1.0
    dpi=500
    markerscale=25
    
    def summ_plot(axs, colname, last_row = False):
        colnames = [colname] if isinstance(colname, str) else colname
        multi = len(colnames) > 1
        for cn in colnames:
            if multi:
                line, = axs.plot(lineage[cn], '-', linewidth=lw, label=f'lineage {labels[cn]}')
                axs.plot(log[cn], '.', markersize=ms, color=line.get_color(), alpha=0.25)
            else:
                axs.plot(log[cn], '.', markersize=ms,    color='silver', label='all mutations')
                axs.plot(bestlog[cn], '-', linewidth=lw, label='best of the generation')
                axs.plot(lineage[cn], '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={len(lineage)})')
        axs.set(xlabel=None, ylabel='Sequence length' if multi else labels[colnames[0]])
        axs.grid(True, which="both",linestyle='--', linewidth=0.5)
        if multi:
            axs.legend()
        if last_row:
            axs.set(xlabel='Total number of mutations')
        else:
            axs.set_xticklabels([])


    summ_plot(axs[0,0], 'ptm')
    summ_plot(axs[1,0], 'plddt')
    summ_plot(axs[2,0], 'score', last_row = True)
    axs[2,0].legend(loc ="lower right", markerscale=markerscale)

    if simparam["seq2"]:
        summ_plot(axs[0,1], 'iptm')
        summ_plot(axs[1,1], 'iplddt')
        summ_plot(axs[2,1], ['seq1_len', 'seq2_len'], last_row = True)

    else:
        summ_plot(axs[0,1], 'cd')
        summ_plot(axs[1,1], 'lcd')
        summ_plot(axs[2,1], 'seq1_len', last_row = True)


    fig.tight_layout()
    fig.savefig(os.path.join(outdir,'Summary.png'), dpi=dpi)



def make_lineage_summary(lineage, simparam):

    lineage = lineage.copy()
    lineage.index = lineage.index / simparam['pop_size']
    
    lw=1.0
    dpi=500
        
    def lin_summ_plot(axs, colnames, last_row = False):
        for colname in colnames:
            axs.plot(lineage["gndx"], lineage[colname], '-', linewidth=lw, label=labels[colname])
        axs.grid(True, which="both",linestyle='--', linewidth=0.5)
        axs.set(xlabel=None, ylabel=None)
        axs.legend()
        if last_row:
            axs.set(xlabel='Generations')
        else:
            axs.set_xticklabels([])


    fig, axs = plt.subplots(3,2, figsize=(10, 6))
    fig.suptitle(None) # type: ignore

    if simparam['ligand']:
        lin_summ_plot(axs[0,0], ['ptm','plddt'])
        lin_summ_plot(axs[1,0], ['iptm', 'iplddt'])
        lin_summ_plot(axs[2,0], ['evolrate', 'score'], last_row=True)
        lin_summ_plot(axs[0,1], ['cd', 'lcd'])
        if simparam["seq2"]:
            lin_summ_plot(axs[1,1], ['seq1_len', 'seq2_len'])
            lin_summ_plot(axs[2,1], ['seq1_stat', 'seq2_stat', 'beta'], last_row=True)
        else: 
            lin_summ_plot(axs[1,1], ['seq1_len'])
            lin_summ_plot(axs[2,1], ['seq1_stat', 'beta'], last_row=True)

    elif simparam['seq2'] and not simparam['ligand']:
        fig, axs = plt.subplots(3,2, figsize=(10, 6))
        fig.suptitle(None) # type: ignore

        lin_summ_plot(axs[0,0], ['ptm','plddt'])
        lin_summ_plot(axs[1,0], ['iptm', 'iplddt'])
        lin_summ_plot(axs[2,0], ['evolrate', 'score'], last_row=True)
        lin_summ_plot(axs[0,1], ['cd'])
        lin_summ_plot(axs[1,1], ['seq1_len', 'seq2_len'])
        lin_summ_plot(axs[2,1], ['seq1_stat', 'seq2_stat', 'beta'], last_row=True)


    elif not simparam['seq2'] and simparam['ligand']:
        lin_summ_plot(axs[0,0], ['ptm','plddt'])
        lin_summ_plot(axs[1,0], ['iptm', 'iplddt'])
        lin_summ_plot(axs[2,0], ['evolrate', 'score'], last_row=True)
        lin_summ_plot(axs[0,1], ['cd', 'lcd'])
        lin_summ_plot(axs[1,1], ['seq1_len'])
        lin_summ_plot(axs[2,1], ['seq1_stat', 'beta'], last_row=True)

    elif not simparam['seq2'] and not simparam['ligand']:
        lin_summ_plot(axs[0,0], ['ptm'])
        lin_summ_plot(axs[1,0], ['plddt'])
        lin_summ_plot(axs[2,0], ['evolrate', 'score'], last_row=True)
        lin_summ_plot(axs[0,1], ['cd'])
        lin_summ_plot(axs[1,1], ['seq1_len'])
        lin_summ_plot(axs[2,1], ['seq1_stat', 'beta'], last_row=True)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir,'Lineage_summary.png'), dpi=dpi)


#======================= functions end here =======================#
#==================================================================#


if os.path.isfile(args.log) is False:
    raise FileNotFoundError(f'Log file "{args.log}" not found')

outdir = args.outdir 
os.makedirs(outdir, exist_ok=True)
plotdir = os.path.join(outdir, 'plots/')
tmp = os.path.join(outdir, 'tmp/')
tmp_svg = os.path.join(outdir, 'tmp/svg/')
tmp_png = os.path.join(outdir, 'tmp/png/')


simparam = read_header(args.log)
print(''.join([f"#--{param:<24} = {value}\n" for param, value in simparam.items()]))

if not args.nostr:
    print("#============= dropping structures =============#", end="\r")
    with open("/tmp/tmp_progress_nopdb.log", "w") as f:
        for line in open(args.log):
            if line.startswith("#--"):
                f.write(line)
            else:
                break
  
    os.system(f"grep -v '^#' {args.log} | cut -f1-17 >> /tmp/tmp_progress_nopdb.log")

    args.log = "/tmp/tmp_progress_nopdb.log"

print('#============= reading trajectory ==============#', end="\r")
log = pd.read_csv(args.log, sep='\t', comment='#', on_bad_lines='skip')
log = log.iloc[args.start:args.end]

log["sequence_data"] = log.sequence_data.apply(ast.literal_eval)

log["seq1"] = log["sequence_data"].apply(lambda x: x["seq1"]["sequence"])
log["seq1_ss"] = log["sequence_data"].apply(lambda x: x["seq1"]["ss"])
log["seq1_len"] = log["sequence_data"].apply(lambda x: x["seq1"]["len"])
log["seq1_stat"] = log["sequence_data"].apply(lambda x: x["seq1"].get("seqstat", 0))

if simparam["seq2"]:
    log["seq2"] = log["sequence_data"].apply(lambda x: x["seq2"]["sequence"])
    log["seq2_ss"] = log["sequence_data"].apply(lambda x: x["seq2"]["ss"])
    log["seq2_len"] = log["sequence_data"].apply(lambda x: x["seq2"]["len"])
    log["seq2_stat"] = log["sequence_data"].apply(lambda x: x["seq2"].get("seqstat", 0))


if args.reseqstat:
    print('#=========== recalculating statistics ===========#', end="\r")
    log["seq1_stat"] = log["sequence_data"].apply(lambda x: seqstat[x["seq1"]["type"]](x["seq1"]))
    if simparam["seq2"]:
        log["seq2_stat"] = log["sequence_data"].apply(lambda x: seqstat[x["seq2"]["type"]](x["seq2"]))

bestlog = log.groupby('gndx').head(1)
bestlog_cols = ["gndx", "id", "prev_id", "mutation", "beta", 
                    "plddt", "ptm", "iplddt", "iptm", "cd", "lcd", 
                    "n_atoms", "n_clashes", "clashscore", "score", 
                    "seq1", "seq1_ss", "seq1_len", "seq1_stat"]
if simparam["seq2"]:
    bestlog_cols += ["seq2", "seq2_ss", "seq2_len", "seq2_stat"]
bestlog_cols += ["structure"]
bestlog = bestlog[bestlog_cols]
bestlog.to_csv(os.path.join(outdir, 'bestlog.tsv'), sep='\t', index=False, header=True)


print('#================================================#')
lineage = extract_lineage(log)
lineage.to_csv(os.path.join(outdir, 'lineage.tsv'), sep='\t', index=False, header=True)


if args.nostr:
    print("#============ extracting structures ==+=========#", end="\r")
    extract_structures(lineage, outdir)


if args.noplots:
    print("#=========== preparing summary plots ============#", end="\r")
    make_summary_plot(log, bestlog, lineage, simparam)
    make_lineage_summary(lineage, simparam)

#    print("#============ preparing other plots =============#", end="\r")
#    make_plots(log, bestlog, lineage)


if not args.nostr:
    os.remove("/tmp/tmp_progress_nopdb.log")


print('#==================== done ======================#')
print('#================================================#\n')


