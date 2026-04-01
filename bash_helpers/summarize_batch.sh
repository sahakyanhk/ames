#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

runs=$1

cd $runs

if [ -d "summary" ]; then
    echo "summary directory alredy exists, will be backuped to summary_backup"
    if [ -d "summary_backup" ]; then
        rm -rf summary_backup
    fi
    mv summary summary_backup
fi


if [ "$2" == "run_va" ]; then
    for dir in `ls -d run*/`; do 
        python $REPO_ROOT/src/visual_ames.py -l  $dir/progress.log  ; 
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
summary_plot="summary/allplot"
summary_summary="summary/summary_general"
summary_lineage_summary="summary/summary_lineage"
summarytsv="summary/summary_final.tsv"
batch_summary="summary/summary_batch"

mkdir -p "$summary_pdb" "$summary_seq" "$summary_log" "$summary_plot" "$summary_summary" "$summary_lineage_summary" "$batch_summary"

for run in $(ls -d run*/); do 
    base=${run::-1}
    echo -ne "$base\r"

    cp "${run}/Summary.png" "$summary_summary/${base}_summary.png"
    cp "${run}/Lineage_summary.png" "$summary_lineage_summary/${base}_lineage_summary.png"
    
    final_pdb=$(ls "${run}/structures/" | sort -V | tail -n 1)
    cp "${run}/structures/$final_pdb" "$summary_pdb/${base}_final.pdb"
    cp -r "${run}/plots/" "$summary_plot/${base}/"

    cp "${run}/bestlog.tsv" "$summary_log/${base}_bestlog.tsv"
    cp "${run}/lineage.tsv" "$summary_log/${base}_lineage.tsv"
    echo -e "$base\t$(tail -n 1 "${run}/lineage.tsv")" >> summarytsv
    
    get_last_seq "$summary_log/${base}_lineage.tsv" > "$summary_seq/${base}_final.fasta"
    
done
    
echo -e "run\\t$(head -n 1 "${run}/lineage.tsv")" > $summarytsv
cat summarytsv >> $summarytsv && rm summarytsv 


echo "generating summary plots..."

python - <<EOF
import os
import pandas as pd
import matplotlib.pyplot as plt

def summary_plot_with_violin(basedir, param='seq1_stat'):
    series_list = []
    final_values = []

    for lineage in os.listdir(basedir):
        if lineage.endswith("_lineage.tsv"):
            df = pd.read_csv(f"{basedir}/{lineage}", sep='\t')
            if param not in df.columns:
                return
            series_list.append(df[param])
            final_values.append(df[param].iloc[-1])

    combined = pd.concat(series_list, axis=1)

    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})

    for col in combined:
        ax1.plot(combined.index, combined[col], linewidth=0.5, alpha=0.08)
    n_runs = combined.shape[1]
    mean_series = combined.mean(axis=1, skipna=True)
    median_series = combined.median(axis=1, skipna=True)
    ax1.plot(median_series.index, median_series.values, color='red', linewidth=2, alpha=0.7, label='median')
    ax1.plot(mean_series.index, mean_series.values, color='black', linewidth=2, label='mean')
    ax1.set_xlabel('lineage length')
    ax1.set_ylabel(param)
    ax1.legend(title=f'n={n_runs}', loc='lower right')

    parts = ax2.violinplot(final_values, positions=[0], showmeans=True, showmedians=True)
    parts['cmedians'].set_color('black')
    parts['cmeans'].set_color('red')
    ax2.set_xticks([0])
    ax2.set_xticklabels([param])
    ax2.set_ylabel(param)

    plt.tight_layout()
    plt.savefig(f"$batch_summary/{param}.png")
    plt.clf()
    plt.close()

for param in ['plddt', 'ptm', 'iplddt', 'iptm', 'score', 'evolrate',
            'seq1_stat', 'seq2_stat', 'n_atoms', 'cd', 'lcd', 
            'seq1_len', 'seq2_len', 'clashscore', 'num_clashes']:

    print(f"generating summary plot for {param}...", end='\x1b[1K\r')
    summary_plot_with_violin("$summary_log", param=param)



print("done", end='\x1b[1K\r')

EOF

