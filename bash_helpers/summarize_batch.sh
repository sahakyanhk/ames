#!/bin/bash
#set -e

runs="${1:?Usage: $0 <runs_dir> [visualames]}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd $runs

if [ -d "summary" ]; then
    echo "summary directory alredy exists, will be backuped to summary_backup"
    if [ -d "summary_backup" ]; then
        rm -rf summary_backup
    fi
    mv summary summary_backup
fi


if [[ "$2" == "visualames" || "$3" == "visualames" ]]; then
    for dir in `ls -d run*/`; do 
        visualames.py -l  $dir/progress.log  ; 
    done
fi

echo "summarizing $(ls -d run*/ | wc -l) simulations..."

get_last_seq() {
    seq1_col=$(head -1 "$1" | tr '\t' '\n' | grep -n '^seq1$' | cut -d: -f1)
    seq2_col=$(head -1 "$1" | tr '\t' '\n' | grep -n '^seq2$' | cut -d: -f1)
    if [ "$seq2_col" ]; then
        tail -n 1 "$1" | awk -F '\t' -v id="$(basename ${1%_*})" -v seq1_col="$seq1_col" -v seq2_col="$seq2_col" '{print ">"id "_seq1" "\n" $seq1_col "\n" ">"id "_seq2" "\n"$seq2_col}'
    else
        tail -n 1 "$1" | awk -F '\t' -v id="$(basename ${1%_*})" -v seq1_col="$seq1_col" '{print ">"id "\n" $seq1_col}'
    fi
    }

summary_pdb="summary/pdb"
summary_seq="summary/sequence"
summary_log="summary/lineage"
#summary_plot="summary/allplot"
summary_summary="summary/summary_general"
summary_lineage_summary="summary/summary_lineage"
summarytsv="summary/summary_final.tsv"
batch_summary_lineage="summary/summary_batch_lineage"
batch_summary_bestlog="summary/summary_batch_bestlog"

mkdir -p "$summary_pdb" "$summary_seq" "$summary_log" "$summary_summary" "$summary_lineage_summary" "$batch_summary_lineage" "$batch_summary_bestlog"

for run in run*/; do 
    base=${run::-1}
    echo -ne "$base\r"

    cp "${run}/Summary.png" "$summary_summary/${base}_summary.png"
    cp "${run}/Lineage_summary.png" "$summary_lineage_summary/${base}_lineage_summary.png"
    
    final_pdb=$(ls "${run}/structures/" | sort -V | tail -n 1)
    cp "${run}/structures/$final_pdb" "$summary_pdb/${base}_final.pdb"
#    cp -r "${run}/plots/" "$summary_plot/${base}/"

    cp "${run}/bestlog.tsv" "$summary_log/${base}_bestlog.tsv"
    cp "${run}/lineage.tsv" "$summary_log/${base}_lineage.tsv"
    echo -e "$base\t$(tail -n 1 "${run}/lineage.tsv")" >> "${summarytsv}.tmp"
    
    get_last_seq "$summary_log/${base}_lineage.tsv" > "$summary_seq/${base}_final.fasta"
    
done
    
echo -e "run\t$(head -n 1 "${run}/lineage.tsv")" > "$summarytsv"
cat "${summarytsv}.tmp" >> "$summarytsv" && rm "${summarytsv}.tmp"


echo "generating summary plots..."

python - <<EOF
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def summary_plot_with_violin(basedir, outdir, suffix='_lineage.tsv', param='seq1_stat'):
    param_list = []      # param value indexed by generation (gndx), per run
    lineage_lengths = [] # number of ancestors in the lineage chain, per run
    final_values = []

    for fname in sorted(os.listdir(basedir)):
        if fname.endswith(suffix):
            df = pd.read_csv(f"{basedir}/{fname}", sep='\t')
            if param not in df.columns:
                return
            df = df.drop_duplicates(subset='gndx', keep='last')
            param_list.append(pd.Series(df[param].values, index=df['gndx'].values))
            lineage_lengths.append(df.shape[0])
            final_values.append(df[param].iloc[-1])

    if not param_list:
        return

    # Each run advances through generations sparsely (a lineage skips generations
    # where it produced no surviving ancestor). Reindex every run onto a common
    # generation grid and forward-fill, since a lineage value persists until the
    # next ancestor appears. Positions past a run's last generation stay NaN so
    # short runs do not bias the aggregate at high generations.
    max_gen = max(int(s.index.max()) for s in param_list)
    grid = np.arange(0, max_gen + 1)

    def align(s):
        g = s.reindex(grid).ffill()
        g[grid > s.index.max()] = np.nan
        return g

    combined = pd.concat([align(s) for s in param_list], axis=1)

    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})

    for col in combined.columns:
        ax1.plot(grid, combined[col], linewidth=0.5, alpha=0.5, color='silver')
    n_runs = combined.shape[1]
    mean_series = combined.mean(axis=1, skipna=True)
    median_series = combined.median(axis=1, skipna=True)
    ax1.plot(grid, median_series.values, color='#800080', linewidth=2, alpha=0.7, label='median')
    ax1.plot(grid, mean_series.values, color='#6495ED', linewidth=2, label='mean')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel(param)
    avg_lineage_length = np.mean(lineage_lengths)
    ax1.legend(title=f'number of runs = {n_runs}\navg lineage length = {avg_lineage_length:.0f}', loc='lower right')

    parts = ax2.violinplot(final_values, positions=[0], showmeans=True, showmedians=True)
    parts['cmedians'].set_color('#800080')
    parts['cmeans'].set_color('#6495ED')
    ax2.set_xticks([0])
    ax2.set_xticklabels([param])
    ax2.set_ylabel(param)

    plt.tight_layout()
    plt.savefig(f"{outdir}/{param}.png")
    plt.clf()
    plt.close()

params = ['plddt', 'ptm', 'iplddt', 'iptm', 'score', 'evolrate',
          'seq1_stat', 'seq2_stat', 'n_atoms', 'cd', 'lcd',
          'seq1_len', 'seq2_len', 'clashscore', 'num_clashes', 'rfam_score', 'beta']

for label, outdir, suffix in [("lineage", "$batch_summary_lineage", "_lineage.tsv"),
                              ("bestlog", "$batch_summary_bestlog", "_bestlog.tsv")]:
    for param in params:
        print(f"generating {label} summary plot for {param}...", end='\x1b[1K\r')
        summary_plot_with_violin("$summary_log", outdir, suffix=suffix, param=param)



print("done", end='\x1b[1K\r')

EOF

if [[ "$2" == "avaclust" || "$3" == "avaclust" ]]; then
    echo "clustering complexes with avaclust"
    avaclust -i "summary/pdb" -o "summary/clust" -c 0.4 --chains A
    avaclust -i "summary/pdb" -o "summary/allclust" -c 0.1 --chains A
fi

