##################################################################################
######################### Summary generated data #################################
##################################################################################
#   data/parsed/1abc.json  -> for each pdb: residues, coords and ligands (with atoms)
#   data/parse_errors.txt  -> failed PDB IDs
# Extract strucutral information of PDB
# Parse: PDB file -> PDBParser -> Structure -> Model -> Chain -> Residue -> Atom (methods)

import json                           # each PDB as dict 
import numpy as np                    # array (conda install -c conda-forge numpy)
from pathlib import Path              # manage routes
from tqdm import tqdm                 # progress bar (conda install -c conda-forge tqdm)
from Bio.PDB import PDBParser         # parse pdb (conda install -c conda-forge biopython)
from Bio.PDB.Polypeptide import is_aa # determinate if a residue is a standard aa -> separate protein chain from ligands and water

RAW_DIR = Path("data/raw_pdbs")   #protein pdbs from BioLip
OUT_DIR = Path("data/parsed")    #store protein strucutres parsed
ERRORS_PATH = Path("data/parse_errors.txt")     #store failed parse pdbs
INDEX_PATH = Path("data/ligand_index.json")     #dictionaries -> pdb:ligand 

OUT_DIR.mkdir(parents=True, exist_ok=True)

SOLVENTS = {
    "HOH","DOD","WAT","GOL","EDO","PEG","DMS","ACT",
    "FMT","BME","DTT","TRS","MES","EOH","PGE","SO4",     #not relevant ligands 
    "PO4","NAG","MAN","GLC","MPD","ABA","UNK"
}

################################################################################
#################### Extract residues ##########################################
################################################################################

#Result: Store min data for calculate features in feature_engineering.py -> list of dicts (1 dict per residue)
#       - Residue name
#       - Residue ID: chain and position number
#       - C-alpha coords
#       - B-factor of C-alpha
#       - coords of each atom

#Definition of the function for extract residue info
def extract_residues(model) -> list[dict]:
    """
    Extract residue info: pbd_id, chain, resnum, icode, resname, ca_coords, atom_coords, atom_names
    """
    residues = []
    for chain in model:
        for res in chain:
            #filter non standar aa
            if not is_aa(res, standard=True):  #res.id[0] -> AA standard (" "), HETATM ("H_XXX"), water ("W") 
                continue
            # get c-alpha
            if "CA" not in res:   # for non c-alpha modeled
                continue
            ca = res["CA"]

            all_atoms = list(res.get_atoms())

            residues.append({
                "pdb_id": None,          # fill outside the function
                "chain": chain.id,      #residue chain
                "resnum": res.id[1],    #residue number
                "icode": res.id[2].strip(),  #insertion code -> 100A
                "resname": res.get_resname().strip(),   #residue name
                "ca_coord": ca.get_coord().tolist(),   # numpy array [x, y, z] of c-alpha coords-> python list -> json
                # All atoms of resiude coords for calculate the min distance to ligand -> (0/1 label)
                "atom_coords":[a.get_coord().tolist() for a in all_atoms],
                "atom_names": [a.get_name().strip()   for a in all_atoms]
            })
    return residues

##############################################################################
############################# Ligand extraction ##############################
##############################################################################

#Result: list of dicts with ligands

#Definition of the function for extract ligand info
def extract_ligands(model) -> list[dict]:
    """
    Extract ligand information: resname, chain, resnum, atom_coords, num_atoms, center
    """
    ligands = []
    for chain in model:
        for res in chain:
            res_name = res.get_resname().strip()  #get ligand name

            if res.id[0].startswith("H_") and res_name not in SOLVENTS:
                atoms = list(res.get_atoms())
                atom_coords = [a.get_coord().tolist() for a in atoms]   #if ligand is not an irrelevant ligand, then make a python list(numpy array) of its atom coords

                if len(atoms) < 1:    #avoid empty ligands
                    continue

                ligands.append({
                    "resname": res_name,  #ligand name
                    "chain": chain.id,    #chain of the protein that bind the ligand
                    "resnum": res.id[1],  #position number of the ligand
                    "atom_coords": atom_coords,  #coord of all the atoms of the ligand
                    "num_atoms": len(atoms),    #number of atoms of the ligands
                    # Center of ligand for pocket detection
                    "center": np.array(atom_coords).mean(axis=0).tolist() # [nº atoms x 3] -> [x_mean, y_mean, z_mean] for each ligand
                })
    return ligands

############################################################################
############################ Parse PDB #####################################
############################################################################

#Result: dict with all structural info of PDB

#Definition of the function for parse pdb
def parse_pdb(pdb_path:Path) -> dict | None:
    """
    Parse pbd to obtain important info: pdb_id, n_residues, n_ligands, residues, ligands
    """
    parser = PDBParser(QUIET=True)  #eliminate unuseful warnings 
    
    # Parse PDB file -> Structure or return an error if the file is corrupted
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as e:
        return None
    
    # Define model (Structure -> model)
    try:
        model = structure[0]
    except (KeyError, IndexError):
        return None
    
    residues = extract_residues(model)  # Extract residue info
    ligands = extract_ligands(model)    # Extract ligand info

    if not residues:
        return None     #Avoid PDB without protein chain 
    
    pdb_id = pdb_path.stem.lower()  #define pdb ids as 1abc
    for r in residues:
        r["pdb_id"] = pdb_id  #fill the pdb_id (extract_residue dict)
    
    return{
        "pdb_id": pdb_id,
        "n_residues": len(residues),
        "n_ligands": len(ligands),
        "residues": residues,
        "ligands": ligands
    }

##########################################################################
########################## JSON per PDB parsed ###########################
##########################################################################

#Read all the .pdb files from data/raw_pdbs -> parse all pdbs  -> store the resutl as JSON in data/parsed

pdb_files = sorted(RAW_DIR.glob("*.pdb"))
print(f"PDB for parse: {len(pdb_files)}")  #process all the pdb files in alphabetic order

errors = []
ok_count = 0

for pdb_path in tqdm(pdb_files, desc= "Parsing"):
    out_path = OUT_DIR /f"{pdb_path.stem}.json"  # each pdb parse as 1abc.json

    #if out_path.exists():
        #ok_count += 1       # if .json exist jump to the next one
        #continue
    
    result = parse_pdb(pdb_path)  #store json on its folder
    if result is None:
        errors.append(pdb_path.stem)
        continue

    out_path.write_text(json.dumps(result, indent=2)) #legible json
    ok_count += 1

ERRORS_PATH.write_text("\n".join(errors))

print(f"\nCorrect Parsed: {ok_count}")
print(f"Parse errors: {len(errors)}")
print(f"JSON in : {OUT_DIR}")
if errors:
    print(f"IDs with errors in:{ERRORS_PATH}")