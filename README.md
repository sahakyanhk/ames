# AMES: atomistic molecular evolution simulator

### Experimental code for protein-RNA evolution sumulation

### Installation

```
pip install numpy pandas pybind11 setuptools
pip install . # install pdb_contacts
```
Install [alphafold3](https://github.com/google-deepmind/alphafold3) or [openfold3](https://github.com/aqlaboratory/openfold-3)

### Usage

```
python src/ames.py --iseq1 protein:random:50:evolv --iseq2 rna:random:24:evolv -ps 100 -ng 1000 -b 1 -o outputs/test

python src/visual_ames.py -l outputs/test/progress.log
```
or submit a batch job with SLURM:

```
for i in {01..10};do run_ames.sbatch outputs/run$i; done
```