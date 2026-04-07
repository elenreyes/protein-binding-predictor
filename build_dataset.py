##########################################################################
#################### Summary generated data ##############################
##########################################################################

#Result: data/labels.csv -> one row per residue and columns are the distance threshold of any atom of residues - any atom of ligand and the last one the label (1 = binging site, 0 = no binding site)
#Example: pbd_id, chain, resnum, icode, resname, ca_x, ca_y, ca_z, label
#         1abc,   A,     78,     A,     GLY,     12.3, 4.8,  9.5,  1

#If dist_threshold is >= 4 A -> label: 1; else -> label: 0

import json                        #build json file for dicts
import numpy as np                 #numpy array  (conda install -c conda-forge numpy)
import pandas as pd                #DataFrame (conda install -c conda-forge pandas)
from pathlib import Path           #manage routes
from scipy.spatial import cKDTree  #calculate distances atom_res - atom_lig faster (conda install -c conda-forge scipy)
from tqdm import tqdm              #process bar (conda install -c conda-forge tqdm)

PARSED_DIR = Path("data/parsed")
INDEX_PATH = Path("data/ligand_index.json")
OUTPUT_CSV = Path("data/labels.csv")

DIST_THRESHOLD = 4.0 #Angstrom
MIN_LIG_ATOMS = 5 #ligands must have at least 5 atoms -> eliminate small ions

############################################################################
###################### Calculate distance residue-ligand ###################
############################################################################

#Result: list of 0/1 per residue of each PDB
#Use ckdtree to avoid double loop 

#Definition of the function to obatain the labels according to the distance
def compute_labels(residues: list[dict],
                   ligands: list[dict],
                   biolip_ligands: set[str]) -> list[int]:
    """
    Obtain labels according the distance between atom_residue - atom_ligand
    """
    #relevant residues filter by BioLip and at least 5 atoms
    relevant = []           # list relevant = [lig1, lig2, lig3, ...]
    for lig in ligands:
        if lig["resname"] in biolip_ligands and lig["num_atoms"] >= MIN_LIG_ATOMS:
            relevant.append(lig)
    if not relevant:               #if there is no relevant ligands, the there is no binding site -> list of 0 per residues of PDB
        return [0] * len(residues)
    
    #all atoms of relevant ligands -> coords array
    all_lig_coords = []
    for lig in relevant:
        for coord in lig["atom_coords"]:       # lig1_coord: (1,2,3); lig2_coord:(3,4,5) -> all_lig_coord: [(1,2,3), (3,4,5)]
            all_lig_coords.append(coord)
    all_lig_coords = np.array(all_lig_coords)   #for calculate distance

    #kdtree of the ligand atoms for calculate distances faster
    tree = cKDTree(all_lig_coords)

    labels = []
    for res in residues:
        res_coords = np.array(res["atom_coords"])  #get residue coords of each atom of the residues
        #search if there is any atom of the residue which its distance with any atom of the ligand is inside dist_threshold <= 4 in kdtree
        #return a list of atoms_residue list of indexes which corresponds to the index of the atom ligand in all_lig_coords
        hits = tree.query_ball_point(res_coords, r=DIST_THRESHOLD) 
        #if there any atom of the residue which make contact -> label = 1; else -> label: 0
        label = 0
        for h in hits:
            if len(h) > 0:
                label = 1
                break
        labels.append(label)
    return labels

############################################################################
#################### Create rows of .csv ###################################
############################################################################

#Result: list of dicts with pdb_id, chain, resnum, icode, resname, ca_x, ca_y, ca_z, label

#Definition of the function for build rows
def build_rows(parsed: dict,   #protein info parsed
               biolip_ligands: set[str]) -> list[dict]:  #biolip ligands
    """
    Build rows with information of the residue and labels: 
    pbd_id, chain, resnum, icode, resname, ca_x, ca_y, ca_z, label
    """
    residues = parsed["residues"]
    ligands = parsed["ligands"]      #extract residues, ligands and pdb_id info from parsed strucutres
    pdb_id = parsed["pdb_id"]

    labels = compute_labels(residues,ligands, biolip_ligands)  #call function compute_labels for calculate labels 0/1 per residue -> labels = [0,0,0,0,1,0,1,0, ...]
    #build rows per residue
    rows = []
    for res, label in zip(residues, labels):
        ca = res["ca_coord"]   
        rows.append({
            "pdb_id":  pdb_id,         
            "chain":   res["chain"],
            "resnum":  res["resnum"],
            "icode":   res["icode"],
            "resname": res["resname"],
            "ca_x": round(ca[0], 3),
            "ca_y": round(ca[1], 3),       #C-alpha coords for feature_engineering -> not included as features at the end
            "ca_z": round(ca[2], 3),
            "label": label    #0/1
        })
    return rows

############################################################################
################## Process parsed PDB to generate a .csv ###################
############################################################################

#Result: .csv file -> pdb_id, ligand, chain, resnum, icode, resname, ca_x, ca_y, ca_z, label
ligand_index = json.loads(INDEX_PATH.read_text())  #python dict -> json python dict: {"1abc": [{"ligand": "liga_name", "chain": "A", "resnum": 50, ...}]

biolip_map = {} #map for unique ligands -> { "1abc": {"ATP", "SH3"}, "2xzy": {"HEM"}, ...}
for pid, entries in ligand_index.items():
    ligands = {e["ligand"] for e in entries}
    biolip_map[pid] = ligands

json_files = sorted(PARSED_DIR.glob("*.json"))  #sort alphabetically the pdb_id.json files
print(f"JSONs to process: {len(json_files)}")

all_rows = []
skipped = 0  #pdb count of pdb whithout any residue as binding site

for json_path in tqdm(json_files, desc= "Calculating labels"):
    parsed = json.loads(json_path.read_text())
    pdb_id = parsed["pdb_id"]
    biolip_ligands = biolip_map.get(pdb_id, set())
    rows = build_rows(parsed, biolip_ligands)     #build rows for each pdb
    
    rows = build_rows(parsed, biolip_ligands)

    if any(r["label"] == 1 for r in rows):
        all_rows.extend(rows)           #include in .csv pdb with at leas a residue as binding site
    else:
        skipped += 1                    #else skip the pdb

df = pd.DataFrame(all_rows) # pdb_id | chain | resnum | resname | ca_x | ca_y | ca_z | label
#Count how many residues are bindng site or not 
n_pos = (df["label"] == 1).sum()
n_neg = (df["label"] == 0).sum()

print(f"\nIncluded PDBs: {df['pdb_id'].nunique()}")
print(f"Eliminated PDBs: {skipped}  (cero binding sites)")
print(f"Total residues: {len(df)}")
print(f"Binding sites (1): {n_pos}  ({100*n_pos/len(df):.1f}%)")
print(f"No binding sites (0): {n_neg}  ({100*n_neg/len(df):.1f}%)")
print(f"Ratio neg/pos:{n_neg/n_pos:.1f}x")

df.to_csv(OUTPUT_CSV, index=False)         #save data in .csv
print(f"\nLabels stored in: {OUTPUT_CSV}")
