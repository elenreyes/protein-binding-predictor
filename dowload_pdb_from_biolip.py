#########################################################################
#################### Summary generated data #############################
#########################################################################
#All generated data is store inside /data folder

#    data/pdb_ids.txt   -> unique pdb ids of BioLip
#    data/ligand_index.json -> {pdb_id: [{ligand, chain, resnum},...]}
#    data/raw_pdbs    -> clean pdb files



import os
import gzip                #unzip dataset
import json                #create json file -> dictionary of pdb_id: ligand
import time                #measure execution time
import requests            #dowload files and data from server -> download pdb from RCSB (conda install -c conda-forge request)
import random              #choose 500 PDBs random -> representative sample -> there are 34000 PDB in BioLip dataset
from pathlib import Path   #manage routes, folders, files
from tqdm import tqdm      #show progress bar   (conda install -c conda-forge tqdm)



BIOLIP_FILE = "BioLip_nr.txt.gz"  #curated dataset with ligands
OUTPUT_DIR = Path("data/raw_pdbs")    #directory for store data (PDBs files)
IDS_PATH = Path("data/pdb_ids.txt")   #extract al the pdb ids of BioLip database
INDEX_PATH = Path("data/ligand_index.json")   #extract the ligand of each protein (pdb_id + ligand)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)    #create directory for store all data (if it is created, then do not create it again) -> data  and raw_data folder
IDS_PATH.parent.mkdir(parents=True, exist_ok=True)     #create a directory to store all the data (folder for raw pdbs, txt of pdb id and json file for ligands) -> data folder

##########################################################################
############## Read BioLip and extract PDB IDs ###########################
##########################################################################

#Result: pdb_ids.txt and ligand_index.json

# Relevant columns of BioLiP_nr.txt (separator: tab):
#   0  -> PDB ID (4 chars) + chain (1 char), ej: "1ABC(pdbid)A(chain)"
#   1  -> receptor chain
#   2  -> ligand ID (name HET, ej: "ATP")
#   3  -> ligand chain
#   4  -> ligand serial number
#   5  -> binding site residues

pdb_ids = set()     #store unique pdb ids of biolip
ligand_index = {}   #store {pdb_id: [ {ligand, chain, resnum}]}

with gzip.open(BIOLIP_FILE, "rt") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) < 5:
            continue

        pdb_id = columns[0].strip()[:4].lower()   # "1abc"
        chain = columns[1].strip() # protein chain where the ligand bind
        lig_id = columns[4].strip() #ligand name
        lig_chain = columns[5].strip()  #ligand chain
        lig_resnum = columns[6].strip() #position number of the ligand

        pdb_ids.add(pdb_id)

        if pdb_id not in ligand_index:
            ligand_index[pdb_id] = [] 
        
        ligand_index[pdb_id].append({
            "ligand": lig_id,
            "chain": chain,
            "lig_chain": lig_chain,
            "lig_resnum": lig_resnum,
        })
# Dataset = 500 random PDB from BioLip
MAX_PDBS = 500
pdb_ids = list(pdb_ids)  # convertir el set a lista
pdb_sample = random.sample(pdb_ids, min(MAX_PDBS, len(pdb_ids)))
print(f"Selected {len(pdb_sample)} PDBs randomly for download")

IDS_PATH.write_text("\n".join(sorted(pdb_sample)))
INDEX_PATH.write_text(json.dumps(ligand_index, indent=2))
print(f"Unique PDBs in BioLip: {len(pdb_sample)}")
print(f"Ligand index store in: {INDEX_PATH}")

##########################################################################
################## Download PDB ##########################################
##########################################################################

#Result: raw_pdbs folder -> 1abc.pdb
#We use this function for obtain .pdb files instead of .ent.gz files

def download_pdb(pdb_id: str) -> bool:
    """
    Download pdb_id.pdb from RCSB in data/raw_data folder
    """
    out_path = OUTPUT_DIR / f"{pdb_id}.pdb"
    if out_path.exists():      #avoid repeted downloads
        return True   

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                out_path.write_text(r.text)
                return True
            elif r.status_code == 404:
                return False   
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return False

failed = [] #store failed pdbs downloads
for pdb_id in tqdm(sorted(pdb_sample), desc="Dowload PDBs"):
    ok = download_pdb(pdb_id)
    if not ok:
        failed.append(pdb_id)
    time.sleep(0.05)  

print(f"\nDownload: {len(pdb_ids) - len(failed)}/{len(pdb_ids)}")
if failed:
    print(f"Failed ({len(failed)}): {failed[:10]}{'...' if len(failed)>10 else ''}")
    Path("data/failed_ids.txt").write_text("\n".join(failed))