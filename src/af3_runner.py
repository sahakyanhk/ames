"""
AlphaFold 3 - Modular Functions Based on run_alphafold.py

"""

import sys
from pathlib import Path
import numpy as np
import json
import datetime
from typing import Optional

import os
from contextlib import redirect_stdout


# Setup paths
ALPHAFOLD3_SRC = Path("/data/saakyanh2/WD/af3/alphafold3/src").resolve()
ALPHAFOLD3_ROOT = Path("/data/saakyanh2/WD/af3/alphafold3").resolve()

# Add both to path
if str(ALPHAFOLD3_SRC) not in sys.path:
    sys.path.insert(0, str(ALPHAFOLD3_SRC))
if str(ALPHAFOLD3_ROOT) not in sys.path:
    sys.path.insert(0, str(ALPHAFOLD3_ROOT))

import jax # type: ignore
import run_alphafold # type: ignore

import absl.flags
absl.flags.FLAGS.mark_as_parsed()


from alphafold3.common import folding_input # type: ignore
from alphafold3.common import resources # type: ignore
from alphafold3.constants import chemical_components # type: ignore
from alphafold3.data import featurisation # type: ignore


from pdbutils import cif2pdb
# ============================================================================
# STEP 1: Create Fold Input (supports protein, RNA, DNA)
# ============================================================================

def create_fold_input(
    seq_list: list[dict],
    name: str = "prediction",
    seed = [1], #NUMBER_OF_SEEDS
) -> folding_input.Input:
    """
    Universal function to create fold input for ANY case:
    - Single protein/RNA/DNA
    - Multi-chain complexes
    - Complexes with ligands and ions
    
    Args:
        sequences: Can be:
            3. A list of dicts: [
                {'type': 'protein', 'sequence': 'MKLLVV...', 'id': 'A'},
                {'type': 'rna', 'sequence': 'AUGC...', 'id': 'B'},
                {'type': 'ligand', 'ccd_code': 'ATP', 'id': 'C'},
                {'type': 'ion', 'ccd_code': 'MG', 'id': 'D'}
            ]
        name: Job name
        seed: Random seed
        
    Returns:
        folding_input.Input object ready for inference
        
    Examples:
        # Single chain
        fold_input = create_fold_input({'type': 'rna', 'sequence': 'AUGC', 'id': 'A'})
        
        # Complex
        fold_input = create_fold_input([
            {'type': 'protein', 'sequence': 'MKLLVV...', 'id': 'A'},
            {'type': 'rna', 'sequence': 'AUGC', 'id': 'B'}
        ])
        
        # Complex with ligands/ions
        fold_input = create_fold_input([
            {'type': 'rna', 'sequence': 'AUGC', 'id': 'A'},
            {'type': 'ion', 'ccd_code': 'MG', 'id': 'B'}
        ])
    """
        
    # Build JSON sequences
    json_sequences = []
    
    for seq_info in seq_list:
        seq_type = seq_info['type'].lower()
        
        # Handle protein
        if seq_type == 'protein':
            sequence = seq_info['sequence']
            chain_id = seq_info.get('id', 'A')
            
            json_seq = {
                "protein": {
                    "id": chain_id if isinstance(chain_id, list) else [chain_id],
                    "sequence": sequence,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": []
                }
            }
        
        # Handle RNA
        elif seq_type == 'rna':
            sequence = seq_info['sequence']
            chain_id = seq_info.get('id', 'A')
            
            json_seq = {
                "rna": {
                    "id": chain_id if isinstance(chain_id, list) else [chain_id],
                    "sequence": sequence.upper().replace('T', 'U'),
                    "unpairedMsa": "",
                }
            }
        
        # Handle DNA
        elif seq_type == 'dna':
            sequence = seq_info['sequence']
            chain_id = seq_info.get('id', 'A')
            
            json_seq = {
                "dna": {
                    "id": chain_id if isinstance(chain_id, list) else [chain_id],
                    "sequence": sequence.upper(),
                    "unpairedMsa": "",
                }
            }
        
        # Handle ligands
        elif seq_type == 'ligand':
            ligand_id = seq_info.get('id', 'L1')
            
            if 'ccd_code' in seq_info:
                json_seq = {
                    "ligand": {
                        "id": [ligand_id] if isinstance(ligand_id, str) else ligand_id,
                        "ccdCodes": [seq_info['ccd_code']]
                    }
                }
            elif 'smiles' in seq_info:
                json_seq = {
                    "ligand": {
                        "id": [ligand_id] if isinstance(ligand_id, str) else ligand_id,
                        "smiles": seq_info['smiles']
                    }
                }
            else:
                raise ValueError("Ligand must have 'ccd_code' or 'smiles'")
        
        # Handle ions
        elif seq_type == 'ligand': # ion -> ligand, if AF3 all ions are considered ligands
            ion_id = seq_info.get('id', 'I1')
            
            json_seq = {
                "ligand": {  # Ions use 'ligand' type in JSON
                    "id": [ion_id] if isinstance(ion_id, str) else ion_id,
                    "ccdCodes": [seq_info['ccd_code']]
                }
            }
        
        else:
            raise ValueError(f"Unknown type: {seq_type}. Use 'protein', 'rna', 'dna', 'ligand', or 'ion'")
        
        json_sequences.append(json_seq)
    
    # Create input dict
    input_dict = {
        "name": name,
        "sequences": json_sequences,
        "modelSeeds": seed,
        "dialect": "alphafold3",
        "version": 1,
    }
    
    # Parse to Input object
    fold_input_obj = folding_input.Input.from_json(json.dumps(input_dict))
    
    return fold_input_obj

# ============================================================================
# STEP 2: Featurize Input (Official Way)
# ============================================================================

def featurize_fold_input(
    fold_input_obj: folding_input.Input,
    buckets: Optional[list[int]] = None,
    ccd: chemical_components.Ccd = None,
) -> list:
    """
    Featurize fold input using featurisation function.
    
    This is what run_alphafold.py does.
    
    Args:
        fold_input_obj: Input from create_fold_input()
        buckets: Bucket sizes for padding (None = use defaults)
        ccd: Chemical component dictionary (None = load automatically)
        
    Returns:
        List of feature dictionaries (one per seed)
    """
    # Load CCD if not provided
    if ccd is None:
        ccd_path = resources.filename(resources.ROOT / 'constants/converters/ccd.pickle')
        ccd = chemical_components.Ccd(ccd_pickle_path=ccd_path)
        print(f"Loaded CCD from {ccd_path}")
    
    # Use official bucket sizes if not specified
    if buckets is None:

        buckets = [24, 32, 64, 128, 256, 384, 512, 768, 1024]
        
    # Call featurisation function
    feature_list = featurisation.featurise_input(
        fold_input=fold_input_obj,
        buckets=buckets,
        ccd=ccd,
        ref_max_modified_date=datetime.date(2021, 9, 30),  # Default from run_alphafold.py
        conformer_max_iterations=None,
        resolve_msa_overlaps=True,
        verbose=True,
    )
        
    return feature_list


# ============================================================================
# STEP 3: Create ModelRunner (Official Way)
# ============================================================================

def create_model_runner(
    model_dir: str,
    num_diffusion_samples: int = 5,
    return_embeddings: bool = False,
    return_distogram: bool = False,
    flash_attention: str = "triton",
) -> run_alphafold.ModelRunner:
    
    """
    Create a ModelRunner using run_alphafold.ModelRunner class.
        
    Args:
        model_dir: Path to model parameters
        num_diffusion_samples: Number of diffusion samples
        return_embeddings: Whether to return embeddings
        return_distogram: Whether to return distogram
        flash_attention: "triton" or "xla"
        
    Returns:
        Official ModelRunner instance
    """
    # Create model config using official function
    config = run_alphafold.make_model_config(
        num_diffusion_samples=num_diffusion_samples,
        return_embeddings=return_embeddings,
        return_distogram=return_distogram,
    )
    
    # Set flash attention
    config.global_config.flash_attention_implementation = flash_attention
    
    # Get GPU device
    device = jax.local_devices()[0]
    print(f"Using device: {device}")
    
    # Create ModelRunner using official class
    model_runner = run_alphafold.ModelRunner(
        config=config,
        device=device,
        model_dir=Path(model_dir),
    )
        
    return model_runner


# ============================================================================
# STEP 4: Run Inference (Official Way)
# ============================================================================

def run_inference(
    fold_input_obj: folding_input.Input,
    model_runner: run_alphafold.ModelRunner,
    buckets: list[int] = [40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160],
) -> list:
    """
    Run inference 
        
    Args:
        fold_input_obj: Fold input from create_fold_input()
        model_runner: ModelRunner from create_model_runner()
        buckets: Bucket sizes (None = use defaults)
        
    Returns:
        List of ResultsForSeed objects (one per seed)
    """

    results_for_seeds = run_alphafold.predict_structure(
        fold_input=fold_input_obj,
        model_runner=model_runner,
        buckets=buckets,
        ref_max_modified_date=datetime.date(2021, 9, 30),
        conformer_max_iterations=None,
        resolve_msa_overlaps=True,
    )
        
    return results_for_seeds




model_runner = create_model_runner(model_dir="/data/saakyanh2/af3_model",
                                       num_diffusion_samples=3) #NUMBER_OF_DIFFUSIONS


def af3_runner(input_fold_list: list[dict] | list[list[dict]]) -> tuple[list[str], list[float], list[float], list[float]]:

    results = {}
    i = 0 

    for input_dict in input_fold_list:

        i+=1
        ndx = f"id{i}"

        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):

                fold_input = create_fold_input(
                                seq_list=input_dict, #type: ignore
                                name=ndx)     

                results[ndx] = run_inference(fold_input, model_runner)

    ptms = []
    iptms = []
    plddts = []
    structures = []
    ranking_scores = []
    
    for key in results.keys(): #for each input find the prediction with the highest ranking score
        
        max_ranking_score = None
        max_ranking_result = None

        for results_for_seed in results[key]:

            for result in results_for_seed.inference_results:

                ranking_score = float(result.metadata['ranking_score'])

                if max_ranking_score is None or ranking_score > max_ranking_score:

                    max_ranking_score = ranking_score

                    max_ranking_result = result


        
        plddt  = float(max_ranking_result.predicted_structure.atom_b_factor.mean()) * 0.01
        ptm = float(max_ranking_result.metadata['ptm'])
        iptm = float(max_ranking_result.metadata['iptm'])
        iptm = 0.0 if np.isnan(iptm) else iptm

        cif = max_ranking_result.predicted_structure.to_mmcif()
        pdb = cif2pdb(cif)

        structures.append(pdb)
        plddts.append(round(plddt, 3))
        ptms.append(round(ptm, 3))
        iptms.append(round(iptm, 3))

    return (structures, plddts, ptms, iptms)  # type: ignore
