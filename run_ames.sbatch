#!/bin/bash
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


python src/ames.py --iseq1 'rna:randoms:65:evolv' --seq1_rate 0.8 \
                   --iseq2 'rna:randoms:65:evolv' --seq2_rate 0.8 \
                                            -ps 100 \
                                            -ng 1000 \
                                            -b0 0.8 \
                                            -ann \
                                            -bt 15.0 \
                                            -ann_s 500 \
                                            -ann_e 999 \
                                            --nobackup \
                                            -o "$1"

python src/visual_ames.py -l "$1"/progress.log 


EOF
