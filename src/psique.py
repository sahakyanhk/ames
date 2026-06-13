from subprocess import Popen, PIPE
import os
from pathlib import Path

""" psique is availabel here https://github.com/franciscoadasme/psique """
psique_binary_path = "https://github.com/franciscoadasme/psique/releases/latest/download/psique-linux.gz"

# Get the directory of THIS module file
module_dir = Path(__file__).parent.resolve()
bin_path = module_dir / 'bin' 
psique_path = module_dir / 'bin' / 'psique'

if psique_path.exists():
    os.chmod(psique_path, 0o755)
else: 
    bin_path.mkdir(parents=True, exist_ok=True)
    os.system(f"wget {psique_binary_path} -O {bin_path}/psique.gz && \
    gunzip  {bin_path}/psique.gz && \
    chmod u+x {bin_path}/psique")
    os.chmod(psique_path, 0o755)
    


def pypsique(pdb_txt, chain='A'):
    cmd = f"{psique_path} --format stride /dev/stdin" \
        + f" | awk 'BEGIN {{ ORS = \"\" }} $1==\"ASG\" && $3==\"{chain}\" {{print $6}}'"
 
    process = Popen(cmd,
                    stdin=PIPE,
                    stdout=PIPE, 
                    stderr=PIPE, 
                    shell=True)
    
    stdout, stderr = process.communicate(input=pdb_txt.encode('utf-8'))
    
    if process.returncode != 0:
        raise RuntimeError(f"psique failed: {stderr.decode('utf-8')}")
    
    SSstring = stdout.decode('ascii')
    simplestring = SSstring.replace('G', 'C').replace('F', 'C').replace('T', 'C').replace('P', 'C')
    maxhelix = len(max(simplestring.replace('E', 'C').split('C')))
    maxbeta = len(max(simplestring.replace('H', 'C').split('C')))
    
    return SSstring, int(maxhelix), int(maxbeta)