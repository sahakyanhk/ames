#!/bin/bash

#set -e

runs=$1

cd $runs

#for dir in `ls -d run*/`; do python ../../src/visual_ames.py -l  $dir/progress.log  ; done

summary_pdb="summary/PDB"
summary_traj="summary/TRAJ"
summary_seq="summary/SEQ"
summary_log="summary/LOG"
summary_plot="summary/PLOT"
summary_summary="summary/SUMMARY"
summary_lineage_summary="summary/LSUMMARY"

mkdir -p "$summary_pdb" "$summary_traj" "$summary_seq" "$summary_log" "$summary_plot" "$summary_summary" "$summary_lineage_summary"

for run in $(ls -d run*/); do 
    base=${run::-1}
    
    cp "${run}/Summary.png" "$summary_summary/${base}_summary.png"
    cp "${run}/Lineage_summary.png" "$summary_lineage_summary/${base}_lineage_summary.png"

    #cp "${run}/backbone_traj.pdb" "$summary_traj/${base}_traj.pdb"
    
    last_pdb=$(ls "${run}/structures/" | sort -V | tail -n 1)
    cp "${run}/structures/$last_pdb" "$summary_pdb/${base}_lastpdb.pdb"
    cp -r "${run}/structures/" "$summary_traj/${base}/"
    cp -r "${run}/plots/" "$summary_plot/${base}/"

    cp "${run}/bestlog.tsv" "$summary_log/${base}_bestlog.tsv"
    cp "${run}/lineage.tsv" "$summary_log/${base}_lineage.tsv"
    
    grep -v "#" "${run}/lineage.tsv" | tail -n +2 | sort -Vk1 | awk '{print ">" $1, "\n" $14}' > "${run}/lineage.fasta"
    tail -n 2 "${run}/lineage.fasta" > "$summary_seq/${base}_lastseq.fasta"

done

