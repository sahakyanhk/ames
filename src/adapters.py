def ames_to_af3(seq_data_list, args) -> list[list[dict]]:

    """prepares sequences in generation dataframe for af3 input"""

    if isinstance(seq_data_list, dict):
        seq_data_list = [seq_data_list]

    if args.seq2:
        inputs = [[{"type": args.seq1_type, "sequence": data["seq1"]["sequence"], "id": "A"},
                   {"type": args.seq2_type, "sequence": data["seq2"]["sequence"], "id": "B"}] for data in seq_data_list]

    else:
        inputs = [[{"type": args.seq1_type, "sequence": data["seq1"]["sequence"], "id": "A"}] for data in seq_data_list]


    if args.ligand:

        for inp in inputs:
            chain_id = inp[-1]["id"]
            for lig in args.ligand:
                chain_id = chr(ord(chain_id) + 1)

                #if lig in ccd_list:
                inp.append({"type":"ligand", 'ccd_code': lig, "id": chain_id}) 
        
    return inputs



def ames_to_esmfold(seq_data_list: list[dict], args) -> list[str]:

    """prepares sequences in generation dataframe for esmfold input"""

    if isinstance(seq_data_list, dict):
        seq_data_list = [seq_data_list]

    inputs = [data["seq1"]["sequence"] for data in seq_data_list]
    
    return inputs
