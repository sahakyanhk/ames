# AMES: atomistic molecular evolution simulator


### Installation

Install dependencies:

[openfold3](https://github.com/aqlaboratory/openfold-3) 

or

[alphafold3](https://github.com/google-deepmind/alphafold3)


### Usage

```
python src/ames.py --iseq1 protein:random:50:evolv --iseq2 rna:random:24:evolv -ps 100 -ng 1000 -b 1 -o outputs/test


python src/visual_ames.py -l test/run1/progress.log
```
or submit a sbatch job with SLURM:

```
bash run_ames.sbatch test/run1
```