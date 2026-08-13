from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

from esm.models.esmfold2 import (
    ProteinInput,
    RNAInput,
    DNAInput,
    LigandInput,
    ESMFold2InputBuilder,
    StructurePredictionInput,
)

from pdbutils import cif2pdb
 

inp_type = {'protein': ProteinInput, 'dna': DNAInput, 'rna': RNAInput, 'ligand': LigandInput}

model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()

def ames_to_esmfold2(seq_data_list) -> list[list[dict]]:
    
    """prepares sequences in generation dataframe for af3 input"""

    if isinstance(seq_data_list, dict):
        seq_data_list = [seq_data_list]

    if "seq2" in seq_data_list[0]:
        inputs = [[inp_type[data["seq1"]["type"]](id = "A", sequence = data["seq1"]["sequence"]),
                   inp_type[data["seq2"]["type"]](id = "B", sequence = data["seq2"]["sequence"])] for data in seq_data_list]

    else:
        inputs = [[inp_type[data["seq1"]["type"]](id = "A", sequence = data["seq1"]["sequence"])] for data in seq_data_list]


    if "ligand" in seq_data_list[0]:

        for inp in inputs:
            chain_id = inp[-1]["id"]
            for lig in seq_data_list[0]['ligand']:
                chain_id = chr(ord(chain_id) + 1)

                #if lig in ccd_list:
                inp.append(LigandInput(id=chain_id, ccd=[lig]))  

    return inputs


def esmfold2_runner(input_fold_list: list[dict] | list[list[dict]]) -> tuple[list[str], list[float], list[float], list[float]]:

    results = {}
    i = 0 

    input_fold_list = ames_to_esmfold2(input_fold_list) 


    for fold_input in input_fold_list:
        
        i+=1
        ndx = f"id{i}"
        spi = StructurePredictionInput(sequences=fold_input)
        results[ndx] = ESMFold2InputBuilder().fold(model, 
                                                    spi, 
                                                    num_loops=10, 
                                                    num_sampling_steps=50, 
                                                    num_diffusion_samples=3, 
                                                    seed=0
                                                    )


    ptms = []
    iptms = []
    plddts = []
    structures = []
    
    for key in results.keys(): #for each input find the prediction with the highest ranking score

        max_ranking_score = None
        max_ranking_result = None


        if len(results[key]) == 1:
            results_list = [results[key]]
        else:
            results_list = results[key]
        
        for result in results_list:

            ranking_score = 0.8 * result.iptm + 0.2 * result.ptm 

            if max_ranking_score is None or ranking_score > max_ranking_score:

                max_ranking_score = ranking_score

                max_ranking_result = result

        pdb = cif2pdb(max_ranking_result.complex.to_mmcif())
        plddt  = float(max_ranking_result.plddt.mean()) 
        ptm = float(max_ranking_result.ptm)
        iptm = float(max_ranking_result.iptm)


        structures.append(pdb)
        plddts.append(round(plddt, 3))
        ptms.append(round(ptm, 3))
        iptms.append(round(iptm, 3))

    return (structures, plddts, ptms, iptms)  # type: ignore



