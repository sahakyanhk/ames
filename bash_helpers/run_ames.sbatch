#!/bin/bash
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

sbatch <<EOF
#!/bin/sh
#SBATCH --job-name=$(basename $1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=50g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --output="${1}.out"

source ~/.bashrc
micromamba activate ames


ames --iseq1 'protein:randoms:65:evolv' --seq1_rate 1 \
        --seq1_len_constr 160 \
        --ligand GTP,MG \
        -pm1 npm \
        -ps 100 \
        -ng 1000 \
        -b0 0.8 \
        -ann \
        -bt 8.0 \
        -ann_s 150 \
        -ann_e 999 \
        -o "$1"

visualames -l "$1"/progress.log

EOF

