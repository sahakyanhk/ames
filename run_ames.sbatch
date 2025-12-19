#!/bin/bash
sbatch <<EOF
#!/bin/sh
#SBATCH --job-name=$(basename $1)
#SBATCH --cpus-per-task=24
#SBATCH --qos=gpunlm2025.2
#SBATCH --mem=50g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00
#SBATCH --output="${1}.out"

source ~/.bashrc
micromamba activate af3


python src/ames.py --iseq1 'protein:random:50:evolv' \
                   --iseq2 'rna:random:50:evolv' \
                   -ps 100 \
                   -ng 1000 \
                   -b 1 \
                   nobackup \
                   -o "$1"

python src/vial_ames.py -l "$1"/progress.log -o "$1"


EOF

