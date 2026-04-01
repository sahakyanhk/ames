#!/bin/bash
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

sbatch <<EOF
#!/bin/sh
#SBATCH --job-name=$(basename $1)
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpunlm2025.2
#SBATCH --mem=50g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10-00:00:00
#SBATCH --output="${1}.out"

source ~/.bashrc
micromamba activate af3


python "$REPO_ROOT/src/ames.py" --iseq1 'protein:randoms:65:evolv' --seq1_rate 1 \
                                    --seq1_len_constr 160 \
                                    --seq2_init CGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA \
                                    --seq2_type rna \
                                    --seq2_rate 0 \
                                    --ligand GTP,MG \
                                    -pm npm \
                                    -ps 100 \
                                    -ng 1000 \
                                    -b0 0.8 \
                                    -ann \
                                    -bt 8.0 \
                                    -ann_s 150 \
                                    -ann_e 999 \
                                    -o "$1"

python "$REPO_ROOT/src/visual_ames.py" -l "$1"/progress.log

EOF

