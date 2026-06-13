def cif2pdb(cif_text):
    """
    Convert CIF format to PDB format.
    
    Args:
        cif_text (str): CIF format text
        
    Returns:
        str: PDB format text
    """
    lines = cif_text.strip().split('\n')
    pdb_lines = []
    
    # Parse atom site data
    atom_data = []
    in_atom_section = False
    atom_headers = []
    header_indices = {}
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect start of atom_site loop
        if line.startswith('loop_'):
            # Look ahead to see if this is atom_site loop
            j = i + 1
            temp_headers = []
            while j < len(lines) and lines[j].strip().startswith('_atom_site.'):
                header_line = lines[j].strip()
                header_name = header_line.split('.', 1)[1] if '.' in header_line else ''
                temp_headers.append(header_name)
                j += 1
            
            if temp_headers:
                in_atom_section = True
                atom_headers = temp_headers
                header_indices = {h: idx for idx, h in enumerate(atom_headers)}
                i = j
                continue
        
        # Process atom data lines
        if in_atom_section and not line.startswith('_') and not line.startswith('#') and line:
            # Check if we've reached the end of atom section
            if line.startswith('loop_') or (line.startswith('_') and not line.startswith('_atom_site')):
                in_atom_section = False
                i += 1
                continue
            
            # Parse the line - handle quoted strings
            parts = []
            current = ''
            in_quotes = False
            quote_char = None
            
            for char in line:
                if char in ('"', "'") and not in_quotes:
                    in_quotes = True
                    quote_char = char
                    current = ''  # Start collecting quoted string
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    parts.append(current)
                    current = ''
                    quote_char = None
                elif in_quotes:
                    current += char
                elif char in (' ', '\t'):
                    if current:
                        parts.append(current)
                        current = ''
                else:
                    current += char
            
            if current:
                parts.append(current)
            
            # Create atom dictionary
            if len(parts) >= len(atom_headers):
                atom_dict = {}
                for idx, header in enumerate(atom_headers):
                    if idx < len(parts):
                        atom_dict[header] = parts[idx]
                
                # Only add ATOM/HETATM records
                if atom_dict.get('group_PDB', '').upper() in ('ATOM', 'HETATM'):
                    atom_data.append(atom_dict)
        
        i += 1
    
    # Convert atom data to PDB format
    for atom in atom_data:
        try:
            # Extract fields with defaults
            record_type = atom.get('group_PDB', 'ATOM').upper()
            atom_id = int(atom.get('id', 0))
            
            # Get atom name and remove quotes if present
            atom_name = atom.get('label_atom_id', atom.get('type_symbol', ''))
            # Remove any remaining quotes
            atom_name = atom_name.replace('"', '') 
            
            alt_loc = atom.get('label_alt_id', '.')
            res_name = atom.get('label_comp_id', '')
            chain_id = atom.get('auth_asym_id', atom.get('label_asym_id', ''))
            res_seq = int(atom.get('auth_seq_id', atom.get('label_seq_id', 0)))
            ins_code = atom.get('pdbx_PDB_ins_code', '?')
            x = float(atom.get('Cartn_x', 0.0))
            y = float(atom.get('Cartn_y', 0.0))
            z = float(atom.get('Cartn_z', 0.0))
            occupancy = float(atom.get('occupancy', 1.0))
            temp_factor = float(atom.get('B_iso_or_equiv', 0.0))
            element = atom.get('type_symbol', '')
            
            # Handle special characters for alt_loc and ins_code
            if alt_loc == '.' or alt_loc == '?' or alt_loc == '':
                alt_loc = ' '
            else:
                alt_loc = alt_loc[0]  # Take only first character
            
            if ins_code == '.' or ins_code == '?' or ins_code == '':
                ins_code = ' '
            else:
                ins_code = ins_code[0]  # Take only first character
            
            # Handle chain ID
            if chain_id == '.' or chain_id == '?':
                chain_id = ' '
            else:
                chain_id = chain_id[0] if chain_id else ' '  # Take only first character
            
            # Format atom name according to PDB standard
            # For 1-3 character names, they should be left-aligned with leading space
            # For 4 character names, use all 4 positions
            atom_name_len = len(atom_name)
            if atom_name_len == 1:
                # Single character: space + name + two spaces (handled by format)
                atom_name_formatted = f" {atom_name:<3s}"
            elif atom_name_len == 2:
                # Two characters: space + name + one space (handled by format)
                atom_name_formatted = f" {atom_name:<3s}"
            elif atom_name_len == 3:
                # Three characters: space + name (handled by format)
                atom_name_formatted = f" {atom_name:<3s}"
            else:
                # Four characters: use all 4 positions
                atom_name_formatted = f"{atom_name:<4s}"

            
            pdb_line = (
                f"{record_type:<6s}"           # 1-6: Record name
                f"{atom_id:>5d} "               # 7-11: Atom serial number, 12: space
                f"{atom_name_formatted}"        # 13-16: Atom name
                f"{alt_loc:1s}"                 # 17: Alternate location
                f"{res_name:>3s} "              # 18-20: Residue name, 21: space
                f"{chain_id:1s}"                # 22: Chain ID
                f"{res_seq:>4d}"                # 23-26: Residue sequence number
                f"{ins_code:1s}   "             # 27: Insertion code, 28-30: spaces
                f"{x:>8.3f}"                    # 31-38: X coordinate
                f"{y:>8.3f}"                    # 39-46: Y coordinate
                f"{z:>8.3f}"                    # 47-54: Z coordinate
                f"{occupancy:>6.2f}"            # 55-60: Occupancy
                f"{temp_factor:>6.2f}"          # 61-66: Temperature factor
                f"          "                    # 67-76: spaces (segment ID, etc.)
                f"{element:>2s}"                # 77-78: Element symbol
            )
            pdb_lines.append(pdb_line)
            
        except (ValueError, KeyError, IndexError) as e:
            # Skip malformed entries
            continue
    
    # Add END record
    if pdb_lines:
        pdb_lines.append("END")
    
    return '\n'.join(pdb_lines)

def extract_backbone(pdb_txt: str): 
    pdb_traj = []
    current_residue = []

    prev_resid = "1"

    for line in pdb_txt.splitlines():
        
        if line.startswith("ATOM") or line.startswith("HETATM"):
            
            atom_type = line[11:16].strip()
            resname = line[17:20].strip()
            chain = line[20:22].strip()
            resid = line[22:26].strip()
            if resname in ("A", "T", "U", "G", "C"):
                
                if resid != prev_resid:
                    
                    if "OP3" in current_residue[0]:
                        current_residue = current_residue[0:20]
                    else:
                        current_residue = current_residue[0:19]
                    
                    #print(len(current_residue))
                    #print("\n".join(current_residue))

                    pdb_traj.extend(current_residue)
                    
                    current_residue = []

                current_residue.append(line)
            
            elif line.startswith("HETATM"):
            
                pdb_traj.append(line)
            
            elif resname in ("GLY", "ALA", "VAL", "LEU", "ILE", 
                             "THR", "SER", "MET", "CYS", "PRO", 
                             "PHE", "TYR", "TRP", "HIS", "LYS", 
                             "ARG", "ASP", "GLU", "ASN", "GLN"
                              ):
                
                

                if atom_type in ('N', 'CA', 'C', 'O'):
    
                    pdb_traj.append(line)

            prev_resid = resid
    
    return "\n".join(pdb_traj)




