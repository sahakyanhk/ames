#!/bin/bash
sbatch <<EOF
#!/bin/sh
#SBATCH --job-name=$(basename $1)
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpunlm2025.2
#SBATCH --mem=50g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00
#SBATCH --output="${1}.out"

source ~/.bashrc
micromamba activate af3


python src/ames.py --iseq1 'protein:randoms:50:evolv' \
                   --iseq2 'rna:randoms:50:evolv' \
                                            -ann \
                                            -b0 0.8 \
                                            -bt 8 \
                                            -ann_s 150 \
                                            -ann_e 400 \
                                            -ps 100 \
                                            -ng 500 \
                                            --nobackup \
                                            -o "$1"

python src/vial_ames.py -l "$1"/progress.log -o "$1"


EOF
