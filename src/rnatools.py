import io
import biotite.structure.io.pdb as pdb
import biotite.structure as struc



import io
import biotite.structure.io.pdb as pdb
import biotite.structure as struc


def rna_secondary_structure(pdb_text, chain="A"):
    """PDB string -> (sequence, dot-bracket) for one chain."""
    arr = pdb.PDBFile.read(io.StringIO(pdb_text)).get_structure(model=1)
    sub = arr[struc.filter_nucleotides(arr) & (arr.chain_id == chain)]

    n = struc.get_residue_count(sub)
    seq = "".join(struc.get_residues(sub)[1])
    dbr = struc.dot_bracket_from_structure(sub)[0]
    if not dbr:                          # no base pairs -> biotite returns ''
        dbr = "." * n                    # unfolded: all dots
    return seq, dbr
# import RNA

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

