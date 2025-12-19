import argparse
import os, re
import pandas as pd
import numpy as np
import shutil
import ast
import gzip
from tqdm import tqdm
import matplotlib.pyplot as plt

#import cairosvg 
#from moviepy import ImageClip, concatenate_videoclips


from pdbutils import extract_backbone

parser = argparse.ArgumentParser(description="Analyse PFES")
parser.add_argument('-l', '--log', type=str, help='log file name', default='progress.log') 
parser.add_argument('-o', '--outdir', type=str, help='output directory name')
parser.add_argument('-b', '--start', type=int, help='first point to read from trajectory', default=0)
parser.add_argument('-e', '--end', type=int, help='last point to read from trajectory', default=99999999)
parser.add_argument('--noplots', action='store_false', )


args = parser.parse_args()

if args.outdir is None:
    args.outdir = os.path.dirname(args.log)


#class VisualPFES():


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def extract_lineage(log):
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
{ltail.sequence_data.iloc[-1]}
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

    for id, gziped_structure in zip(log.gndx, log.structure):

        try:
            
            structure_encoded = gzip.decompress(ast.literal_eval(gziped_structure))
            structure_txt = structure_encoded.decode('utf-8')
    
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
        if not colname in ['gndx', 'id', 'seq_len', 'num_conts', 'sel_mode', 
                           'sequence_data', 'mutation', 'prev_id', 'ss', 'structure']:
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
def make_summary_plot(log, bestlog, lineage):
    
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

    axs[0,1].plot(log.iptm, '.', markersize=ms,    color='silver', label='all mutations')
    axs[0,1].plot(bestlog.iptm, '-', linewidth=lw, label='best of the generation')
    axs[0,1].plot(lineage.iptm, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[0,1].set(xlabel=None, ylabel='ipTM')
    axs[0,1].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[0,1].set_xticklabels([])

    axs[1,1].plot(log.gc_cont, '.', markersize=ms,    color='silver', label='all mutations')
    axs[1,1].plot(bestlog.gc_cont, '-', linewidth=lw, label='best of the generation')
    axs[1,1].plot(lineage.gc_cont, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[1,1].set(xlabel=None, ylabel='GC content')
    axs[1,1].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[1,1].set_xticklabels([])

    axs[2,1].plot(log.seq_len, '.', markersize=ms,  color='silver', label='all mutations')
    axs[2,1].plot(bestlog.seq_len, '-', linewidth=lw, label='best of the generation')
    axs[2,1].plot(lineage.seq_len, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[2,1].set(xlabel='Total number of mutations', ylabel='Sequence length')
    axs[2,1].grid(True, which="both",linestyle='--', linewidth=0.5)

    #plt.xticks(rotation=45)

    #for ax in axs.flat:
    #   ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #   ax.label_outer()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,'Summary.png'), dpi=dpi)



#======================= functions end here =======================#

outdir = args.outdir 
os.makedirs(outdir, exist_ok=True)
plotdir = os.path.join(outdir, 'plots/')
tmp = os.path.join(outdir, 'tmp/')
tmp_svg = os.path.join(outdir, 'tmp/svg/')
tmp_png = os.path.join(outdir, 'tmp/png/')


print('#============= reading progress log =============#')
log = pd.read_csv(args.log, sep='\t', comment='#', on_bad_lines='skip')
log = log.iloc[args.start:args.end]

print(f'#processing trajectory with {len(log)} records')

bestlog = log.groupby('gndx').head(1)
bestlog.to_csv(os.path.join(outdir, 'bestlog.tsv'), sep='\t', index=False, header=True)


print('#================================================#')
lineage = extract_lineage(log)
lineage.to_csv(os.path.join(outdir, 'lineage.tsv'), sep='\t', index=False, header=True)



if args.noplots:

    print('#=== making summary plot')
    #make_summary_plot(log, bestlog, lineage)

    print('#=== making plots')
    make_plots(log, bestlog, lineage)

    print("#=== extracting structures")
    extract_structures(lineage, outdir)

print('#================================================#')

