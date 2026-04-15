import torch
import numpy as np
from typing import Optional

from transformers import EsmForProteinFolding


model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
model = model.eval().to('cuda')

model.trunk.set_chunk_size(2048) # 5120 works fine with A100


def ames_to_esmfold(seq_data_list: list[dict]) -> list[str]:
    
    """prepares sequences in generation dataframe for esmfold input"""

    if isinstance(seq_data_list, dict):
        seq_data_list = [seq_data_list]

    inputs = [data["seq1"]["sequence"] for data in seq_data_list]
    
    return inputs


#https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py#L610
def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers

#https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py#L670
def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    n = residue_weights.shape[-1]
    pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
    if interface and (asym_id is not None):
        if len(asym_id.shape) > 1:
            assert len(asym_id.shape) <= 2
            batch_size = asym_id.shape[0]
            pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
        pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[..., None, :] * residue_weights[..., :, None]
    )
    denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
    normed_residue_mask = pair_residue_weights / denom
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def esm2data(esm_out):
    output = {key: value.cpu() for key, value in esm_out.items()} 
    output['plddt'] = output['plddt'] * 100
    pdbs = model.output_to_pdb(output) 
    mask = output["atom37_atom_exists"][:,:,1] == 1 
    seq_len = np.sum(mask.numpy(), 1) 
    num_seq = len(seq_len)
    pmts = [compute_tm(output["distogram_logits"][i]).item() for i in range(num_seq)]
    plddt =  [output["plddt"][:,:,1][i][mask[i]]/100 for i in range(num_seq)] 
    mean_plddt = [plddt[i].mean().item() for i in range(len(seq_len))]
    return(pdbs, pmts, mean_plddt) 


def esmfold_runner(sequence_list: str | list[str]) -> tuple[list[str], list[float], list[float], list[float]]:

    sequence_list = ames_to_esmfold(sequence_list) 

    output = model.infer(sequence_list)
    pdbs, ptms, mean_plddts = esm2data(output)
    iptms = [0.0] * len(sequence_list)
 
    return (pdbs, mean_plddts, ptms, iptms)



