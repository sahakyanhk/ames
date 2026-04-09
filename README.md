# AMES: atomistic molecular evolution simulator

### Experimental code for simulating structural evolution of protein and RNA fold and and their complexes.

## Installation

```
git clone https://github.com/sahakyanhk/ames.git && cd ames
pip install numpy pandas pybind11 setuptools matplotlib 
pip install . # install pdb_contacts
```
Install [alphafold3](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md), [openfold3](https://github.com/aqlaboratory/openfold-3?tab=readme-ov-file#quick-start-for-inference), or [transformers](https://github.com/huggingface/transformers?tab=readme-ov-file#installation) to use ESMFold

## Quick Start

**Single protein fold evolution with ESMFold**
```
python src/ames.py --iseq1 'protein:randoms:65:evolv' --seq1_rate 1 \
                    -pm pmo -ps 100 -ng 1000 \
                    --engine esmfold \
                    -o outputs/esmfold_test
```


Use `src/visualames.py` to process trajectory and run basic analyses.\
This will extract main lineage, structures in pdb format and generate summary plots
```
python src/visualames.py -l outputs/esmfold_test/progress.log
```

Use [AMESViewer](https://github.com/sahakyanhk/ames_viewer) to visualize and analyse the simulation trajectories in [ChimeraX](https://www.cgl.ucsf.edu/chimerax/)

**Evolution of a protein interacting with tRNA**
```
python src/ames.py --iseq1 'protein:randoms:65:evolv' --seq1_rate 0.5 \
                    --iseq2 'rna:randoms:24:evolv' --seq2_rate 1 \
                    -pm npm -rm pmo \
                    -ps 100 -ng 1000 \
                    -ann -ann_s 150 -ann_e 999 \
                    -b0 0.8 -bt 8.0 \
                    --engine af3 \
                    -o outputs/protein_rna_test

python src/visualames.py -l outputs/protein_rna_test/progress.log
``` 

Submit a batch job with SLURM:
```
for i in {01..09};do bash_helpers/run_ames.sbatch outputs/batch1/run$i; done
```

## Settings

**simulation control** \
`-ng`, `--num_generations` Total number of generations in the simulation \
`-ps`, `--pop_size` Population size in each generation \

**sequence setup** \
use the same settings for seq2 with `--iseq2`, `--seq2_init`, `--seq2_type`, `--seq2_len`, `--seq2_evol`, `--seq2_rate` \
`--iseq1` 1st seq info to initiate simulation [protein, rna, dna]:[random, randoms]:[sequence_length]:[evolv, static]. e.g., "protein:random:25:evolv" \
`--seq1_init` 1st seqence, can be actual sequence, `random` - the same random sequence for entire populations, `randoms` - each sequence in the population is random \
`--seq1_type` 1st seq type [proten, rna, dna] \
`--seq1_len` if `random`(s) is used, provide random sequence length, ignore if sequence is provided \
`--seq1_evol` does seq1 evolve [True/False], store_true \
`--seq1_rate` probability of 1st sequence to mutate [0,1] \

**temperature control** \
`-ann`,`--annealing` Use temperature annealing, see `-b0`, `-bt`, `-ann_s`, `-ann_e`, or `-ann_step` for annealing setup \
`-b0`, `--beta` Selection strength, the higher beta the lower temperature and stronger selection \
`-bt`, `--beta_target` Target temerature if annealing is used \
`-ann_s`, `--annealing_start`, Generation when annealing starts \
`-ann_e`,`--annealing_end`, Generation when annealing reaches target temerature \
`-ann_step`,`--annealing_step`, Annealing step, calculated automatically if `-ann_s ` and `-ann_e` are provided \

**mutation setup** \
`-pm`, `-rm`, `-dm` protein, RNA and DNA mutations types \
&ensp;&ensp;&ensp;&ensp;`npm` substitutions, insertions, deletions, permutations and duplications \
&ensp;&ensp;&ensp;&ensp;`pmo` substitutions and single residues indels \
&ensp;&ensp;&ensp;&ensp;`rso` residue substitutions only \

**ligand setup** \
`--ligand` ligand(s) provided in ccd or smiles format separated with commas e.g., "--ligand ATP,MG" \


**outputs control** \
`-o`,`--outpath` output dir name where log and checkpoint files are saved, "ames_output/output" by default` \
`-l`, `--log` output log file name, "progress.log" by default` \
`-c`, `--ckp` checkpoint file name, "progress.ckp" by default` \    
`-ckpi`, `--checkpoint_interval` chechpoint saving frequency generations \
`--nobackup`, action=`store_true` overwrite output if exists` \

**constraints** \
`--seq1_len_constr` constain seq1 length \
`--seq2_len_constr` constain seq2 length \

**other** \
`--engine` structure prediction engine [alphafold3, openfold3, esmfold, simulacrum]" \
`--contact_min_seq_dist` minimum sequence distance for contact calculation \
`--contact_cutoff` cutoff distance for contact calculation \
`--contact_min_plddt` minimum plddt for contact calculation \
`--interface_plddt_cutoff` cutoff distance for interface plddt \
`--lig_contact_cutoff` cutoff distance for ligand contact calculation \
`--lig_contact_min_plddt` minimum plddt for ligand contact calculation \
`--clash_overlap_threshold` cutoff distance for clash calculation in angstroms \
`--clash_min_seq_dist` minimum sequence distance for clash calculation \
`--norepeat` do not generate and/or select the same sequences more than once, off by default \
`--max_seq_per_batch` max_seq_per_batch, half or population size by default \

see [simparam.json](https://github.com/sahakyanhk/ames/blob/dev/src/data/simparam.json) for full list of settings and default values
