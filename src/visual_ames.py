import argparse
import os, re
import pandas as pd
import numpy as np
import json
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt


#import cairosvg 
#from moviepy import ImageClip, concatenate_videoclips

from amestools import read_header, ungzip_str
from pdbutils import extract_backbone
from seqtools import Seqstat

### old ungzip remove when done
# import gzip
# def ungzip_str(string):
#     compressed = gzip.decompress(ast.literal_eval(string))
#     return compressed.decode('utf-8')


parser = argparse.ArgumentParser(description="Analyse PFES")
parser.add_argument('-l', '--log', type=str, help='log file name', default='progress.log') 
parser.add_argument('-o', '--outdir', type=str, help='output directory name')
parser.add_argument('-b', '--start', type=int, help='first point to read from trajectory', default=0)
parser.add_argument('-e', '--end', type=int, help='last point to read from trajectory', default=99999999)
parser.add_argument('--seqstat', action='store_true', )

parser.add_argument('--summary', action='store_true', )
parser.add_argument('--plots', action='store_true', )
parser.add_argument('--traj', action='store_true', )

parser.add_argument('--noplots', action='store_false', )
parser.add_argument('--notraj', action='store_false', )


args = parser.parse_args()

if args.outdir is None:
    args.outdir = os.path.dirname(args.log)


protein_seqstat = Seqstat('data/pfam80_stat.json')
rna_seqstat = Seqstat('data/rnacentral90_stat.json') #test! using prot stat for RNA
seqstat = {"protein": protein_seqstat, "rna": rna_seqstat}

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def extract_lineage(log: pd.DataFrame) -> pd.DataFrame:
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
        i=+1
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
    return lineage

def extract_sequences(log):
    fasta  = "fasta"
    return fasta

#======================= extract structures and make traj =======================#

def make_backbone_traj(frames: list[str], trajout: str = "backbone_traj.pdb"):
        
        if os.path.isfile(trajout):
            os.remove(trajout)
    
        for i, frame in enumerate(frames):
            
            backbone_frame = extract_backbone(frame)

            with open(trajout, 'a') as f:
                f.write(f'MODEL        {i}\n' + backbone_frame + '\nTER\nENDMDL\n')


def extract_structures(log, outdir):
    
    structures_path = os.path.join(outdir, 'structures')
    trajectory_path = os.path.join(outdir, 'backbone_traj.pdb')

    os.makedirs(structures_path, exist_ok=True)
    
    decoded_frames = []

    for id, gziped_structure in zip(log.gndx[1:], log.structure[1:]):

        try:

            structure_txt = ungzip_str(gziped_structure)
            with open(f"{structures_path}/{id}.pdb", "w") as f:
                f.write(structure_txt)

            decoded_frames.append(structure_txt)

        except Exception as e:
            print(id, "failed\n", e)

    print("#=== preparing backbone traj")
    make_backbone_traj(decoded_frames, trajectory_path)


#======================= make separate plots =======================#
def make_plots(log, bestlog, lineage):

    ms=0.1
    lw=1.4
    dpi=500

    os.makedirs(plotdir, exist_ok=True)
    for colname in log.keys(): 
        if colname in ['beta', 'plddt', 'ptm', 'iplddt', 'iptm', 'cd', 'score',
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
                #ax2.set(xlabel="Lineage length")
                fig.tight_layout()
                fig.savefig(plotdir + colname + '.png', dpi=dpi)
                fig.clf()

#======================= Summary plot =======================#
def make_summary_plot(log, bestlog, lineage, sim_param):
    
    ms=0.1
    lw=1.0
    dpi=500

    fig, axs = plt.subplots(3,2, figsize=(10, 8))

    fig.suptitle(None) # type: ignore

    L = len(lineage)

    axs[0,0].plot(log.ptm, '.', markersize=ms,    color='silver', label='all mutations')
    axs[0,0].plot(bestlog.ptm, '-', linewidth=lw, label='best of the generation')
    axs[0,0].plot(lineage.ptm, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[0,0].set(xlabel=None, ylabel='pTM')
    axs[0,0].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[0,0].set_xticklabels([])

    axs[1,0].plot(log.plddt, '.', markersize=ms,    color='silver', label='all mutations')
    axs[1,0].plot(bestlog.plddt, '-', linewidth=lw, label='best of the generation')
    axs[1,0].plot(lineage.plddt, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[1,0].set(xlabel=None, ylabel='pLDDT')
    axs[1,0].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[1,0].set_xticklabels([])

    axs[2,0].plot(log.score,  '.', markersize=ms,    color='silver', label='all mutations')
    axs[2,0].plot(bestlog.score, '-', linewidth=lw,  label='best of the generation')
    axs[2,0].plot(lineage.score,  '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[2,0].set(xlabel='Total number of mutations', ylabel='Score')
    axs[2,0].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[2,0].legend(loc ="lower right")

    if sim_param["seq2"]:
        axs[0,1].plot(log.iptm, '.', markersize=ms,    color='silver', label='all mutations')
        axs[0,1].plot(bestlog.iptm, '-', linewidth=lw, label='best of the generation')
        axs[0,1].plot(lineage.iptm, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
        axs[0,1].set(xlabel=None, ylabel='ipTM')
        axs[0,1].grid(True, which="both",linestyle='--', linewidth=0.5)
        axs[0,1].set_xticklabels([])

        axs[1,1].plot(log.plddt, '.', markersize=ms,    color='silver', label='all mutations')
        axs[1,1].plot(bestlog.plddt, '-', linewidth=lw, label='best of the generation')
        axs[1,1].plot(lineage.plddt, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
        axs[1,1].set(xlabel=None, ylabel='iPLDDT')
        axs[1,1].grid(True, which="both",linestyle='--', linewidth=0.5)
        axs[1,1].set_xticklabels([])

    else:
        axs[0,1].plot(log.cd, '.', markersize=ms,    color='silver', label='all mutations')
        axs[0,1].plot(bestlog.cd, '-', linewidth=lw, label='best of the generation')
        axs[0,1].plot(lineage.cd, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
        axs[0,1].set(xlabel=None, ylabel='Contact Density')
        axs[0,1].grid(True, which="both",linestyle='--', linewidth=0.5)
        axs[0,1].set_xticklabels([])

        axs[1,1].plot(log.seq1_len, '.', markersize=ms,    color='silver', label='all mutations')
        axs[1,1].plot(bestlog.seq1_len, '-', linewidth=lw, label='best of the generation')
        axs[1,1].plot(lineage.seq1_len, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
        axs[1,1].set(xlabel=None, ylabel='Sequence length')
        axs[1,1].grid(True, which="both",linestyle='--', linewidth=0.5)
        axs[1,1].set_xticklabels([])


    axs[2,1].plot(log.beta, '.', markersize=ms,  color='silver', label='all mutations')
    axs[2,1].plot(bestlog.beta, '-', linewidth=lw, label='best of the generation')
    axs[2,1].plot(lineage.beta, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[2,1].set(xlabel='Total number of mutations', ylabel='Beta')
    axs[2,1].grid(True, which="both",linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir,'Summary.png'), dpi=dpi)



#======================= functions end here =======================#

if os.path.isfile(args.log) is False:
    raise FileNotFoundError(f'Log file "{args.log}" not found')
    
outdir = args.outdir 
os.makedirs(outdir, exist_ok=True)
plotdir = os.path.join(outdir, 'plots/')
tmp = os.path.join(outdir, 'tmp/')
tmp_svg = os.path.join(outdir, 'tmp/svg/')
tmp_png = os.path.join(outdir, 'tmp/png/')


print('#============= reading progress log =============#')
sim_param = read_header(args.log)
print(''.join([f"#--{param:<24} = {value}\n" for param, value in sim_param.items()]))

log = pd.read_csv(args.log, sep='\t', comment='#', on_bad_lines='skip')
log = log.iloc[args.start:args.end]

log.sequence_data = log.sequence_data.apply(ast.literal_eval)

log["seq1"] = log["sequence_data"].apply(lambda x: x["seq1"]["sequence"])
log["seq1_ss"] = log["sequence_data"].apply(lambda x: x["seq1"]["ss"])
log["seq1_len"] = log["sequence_data"].apply(lambda x: x["seq1"]["len"])
log["seq1_stat"] = log["seq1"].apply(lambda x: seqstat[sim_param["seq1_type"]].n_gram_prior(x))
if sim_param["seq2"]:
    log["seq2"] = log["sequence_data"].apply(lambda x: x["seq2"]["sequence"])
    log["seq2_ss"] = log["sequence_data"].apply(lambda x: x["seq2"]["ss"])
    log["seq2_len"] = log["sequence_data"].apply(lambda x: x["seq2"]["len"])
    log["seq2_stat"] = log["seq2"].apply(lambda x: seqstat[sim_param["seq2_type"]].n_gram_prior(x))


print(f'#processing trajectory with {len(log)} records')

bestlog = log.groupby('gndx').head(1)
bestlog.to_csv(os.path.join(outdir, 'bestlog.tsv'), sep='\t', index=False, header=True)


print('#================================================#')
lineage = extract_lineage(log)
lineage.to_csv(os.path.join(outdir, 'lineage.tsv'), sep='\t', index=False, header=True)



if args.noplots:

    print('#=== making summary plot')
    make_summary_plot(log, bestlog, lineage, sim_param)

    print('#=== making plots')
    make_plots(log, bestlog, lineage)

if args.notraj:
    print("#=== extracting structures")
    extract_structures(lineage, outdir)

# a future update
# if args.summary:
#     print('#=== making summary plot')
#     make_summary_plot(log, bestlog, lineage, sim_param)

# if args.plots:
#     print('#=== making plots')
#     make_plots(log, bestlog, lineage)

# if args.traj:
#     print("#=== extracting structures")
#     extract_structures(lineage, outdir)

print('#================================================#\n')

