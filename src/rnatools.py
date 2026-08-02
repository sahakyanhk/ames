import os
import re
import io
import shutil
import uuid
import subprocess
import numpy as np
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
# import RNA
from amestools import sigmoid

from pathlib import Path
_PKG_ROOT = Path(__file__).resolve().parent.parent   # ames/
RFAM_DB = _PKG_ROOT / "rfam" / "Rfam100.cm"


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

# def sigmoid(x, L0=0.0, c=0.1) -> float:
#     z = c * (L0 - x)
#     # Clip to prevent overflow
#     z = np.clip(z, -709, 709)  # e^500 is near max float, e^-500 is near 0
#     return 1 / (1 + np.exp(z))


def norm_evalues(evalues, L0=10.0, c=0.5):
    e = np.atleast_1d(np.asarray(evalues, dtype=float))
    nle = -np.log10(np.clip(e, 1e-300, None))   
    out = np.where(nle > 0, sigmoid(nle, L0=L0, c=c), 0.0)  # E>=1 -> 0
    return out.tolist()


def rna_seq_search(headers, sequences, tmp="/tmp/", keep_tmp=False):
    
	if not RFAM_DB.exists():
		raise FileNotFoundError(f"Rfam.cm not found at {RFAM_DB}")

	# run cmscan and read search results
	tmp += str(uuid.uuid4())
	os.makedirs(tmp, exist_ok=True)


	with open(f"{tmp}/query.fa", "w") as f:
		for header, sequence in zip(headers, sequences):
			f.write(f">{header}\n{sequence}\n")
	
	infernal_command = [
		'cmscan',
		'--noali',
		'--rfam',  
		'-E', '10',
		#'--cut_ga', #not compatible with --E
		#'--hmmonly', #not incompatible with option --rfam
		#'--cpu', '24',
		'--tblout', f'{tmp}/results.tbl',
		RFAM_DB,
		f'{tmp}/query.fa'
	]

	try:
		cmscan_result = subprocess.run(infernal_command, check=True,
										capture_output=True, text=True)
	except subprocess.CalledProcessError as e:
		raise RuntimeError(f"cmscan failed:\n{e.stderr}") from e


    # can skip this
	with open(f"{tmp}/results.cmscan", 'w') as f:
		f.write(cmscan_result.stdout)


	#read results into a dict  
	scores = {}
	rfams = {}
	target_names = {}
	evalues = {}

	for line in open(f'{tmp}/results.tbl'): 
		if line.startswith("#"):
			continue
		splited_line = line.split()
		score = float(splited_line[14])
		evalue = float(splited_line[15])
		qid = splited_line[2]
		rfam = splited_line [1]
		target_name = splited_line[0]

		if qid in evalues:
			if evalue < evalues[qid]:
				scores[qid] = score  
				rfams[qid] = rfam
				target_names[qid] = target_name
				evalues[qid] = evalue
		else:
			scores[qid] = score  
			rfams[qid] = rfam
			target_names[qid] = target_name
			evalues[qid] = evalue


	# create a sorted list and add 0/None if a query does not have a hit
	score_list = []
	rfam_list = []
	target_name_list = []
	eval_list = []

	for header in headers:
		if header in scores:
			score_list.append(scores[header])
			eval_list.append(evalues[header])
			rfam_list.append(rfams[header])
			target_name_list.append(target_names[header])
		else:
			score_list.append(0)
			eval_list.append(None)
			rfam_list.append("-")
			target_name_list.append("-")

	normalized_evalues = norm_evalues(eval_list)

	if not keep_tmp:
		shutil.rmtree(tmp)

	return {"headers":headers,
			"scores":score_list, 
			"evalues": eval_list, 
			"rfams":rfam_list, 
			"target_names": target_name_list, 
			"normalized_evalues": normalized_evalues}


