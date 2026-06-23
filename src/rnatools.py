import os
import re
import io
import biotite.structure.io.pdb as pdb
import biotite.structure as struc

# import RNA

def rna_secondary_structure(pdb_text, chain="A"):
    """PDB string or file -> (sequence, dot-bracket) for one chain."""
    if os.path.isfile(pdb_text):
        with open(pdb_text, 'r') as f:
            pdb_text = f.read()
    arr = pdb.PDBFile.read(io.StringIO(pdb_text)).get_structure(model=1)
    sub = arr[struc.filter_nucleotides(arr) & (arr.chain_id == chain)]
    n = struc.get_residue_count(sub)
    seq = "".join(struc.get_residues(sub)[1])
    dbr = struc.dot_bracket_from_structure(sub)[0]
    if not dbr:                          # no base pairs -> biotite returns ''
        dbr = "." * n                    # unfolded: all dots
    return seq, dbr, max_stem_len(dbr)


def max_stem_len(rna_ss) -> int:
    pattern = re.compile(rf"[{'\\(+'}{'\\)+'}]+")
    matches = pattern.findall(rna_ss)
    # Returns the length of the longest sequence found
    return len(max(matches, key=len)) if matches else 0

def rna_ss_penalty(rna_ss: str) -> float:
    # penalize wrong rna loop 
    forbidden_motifs = ['(.)', '(..)', '[.]', '(.]', '[.)']
    for motif in forbidden_motifs:
        if motif in rna_ss:
            return 0.5
    else:
        return 1
    
# def rnafold(rnaseq):
#     if isinstance(rnaseq, str):
#         fc  = RNA.fold_compound(rnaseq)
#         return [fc.mfe()]       
#     elif isinstance(rnaseq, list):
#         secondary_structures, energies = [], []
#     for seq in rnaseq:
#         fc  = RNA.fold_compound(seq)
#         ss, e = fc.mfe()
#         secondary_structures.append(ss)
#         energies.append(e)
#         return secondary_structures, energies
#     else:
#         print("The input should be eather string or list of strings")

