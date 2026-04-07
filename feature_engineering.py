################################################################################
########################### Summary generated data #############################
################################################################################

#Result: data/features.csv -> one row per residue and colums of labels.csv + 39 features
#Workflow: 
#   1. Load parsed jason
#   2. Calculate features
#   3. Store info in features.csv
# Source: paper P2Rank

import json                             # json python dicts
import numpy as np                      # Arrays for distance calculation
import pandas as pd                     # DataFrames
import freesasa                         # (conda install -c conda-forge freesasa)
from pathlib import Path                # manage of routes
from scipy.spatial import cKDTree       # calculate distances faster
from tqdm import tqdm                   # process bar
from Bio.PDB import PDBParser, DSSP     # (conda install -c conda-forge dssp)

PARSED_DIR = Path("data/parsed")
RAW_DIR = Path("data/raw_pdbs")
OUTPUT_CSV = Path("data/features.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

################################################################################
################### Static features per aa (17 features) #######################
################################################################################

# HYDROPHOBIC AA
HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "TYR"}

# HYDROPHILIC AA
HYDROPHILIC = {"SER", "THR", "CYS", "ASN", "GLN", "TYR"}

# HYDROPATHY INDEX of Kyte-Doolittle -> + :hydrophobic, -: hydrophilic
HYDROPATHY_INDEX = {
    "ALA":  1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5,
    "CYS":  2.5, "GLN": -3.5, "GLU": -3.5, "GLY": -0.4,
    "HIS": -3.2, "ILE":  4.5, "LEU":  3.8, "LYS": -3.9,
    "MET":  1.9, "PHE":  2.8, "PRO": -1.6, "SER": -0.8,
    "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL":  4.2,
}

# ALIPHATIC AA
ALIPHATIC = {"ALA", "VAL", "ILE", "LEU"}

# AROMATIC AA
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}

# CONTAIN SULFUR ON ITS R-CHAIN
SULFUR = {"CYS", "MET"}

# CONTAIN OH GROUP ON ITS R-CHAIN
HYDROXYL = {"SER", "THR", "TYR"}

# BASIC AA -> positive charge at pH = 7.4
BASIC = {"ARG", "LYS", "HIS"}

# ACID AA -> negative charge at pH = 7.4
ACIDIC = {"ASP", "GLU"}

# CONTAIN AMIDA (-CONH2) ON ITS R-CHAIN
AMIDE = {"ASN", "GLN"}

#  POSITIVE CHARGE NETA AT pH = 7.4
POS_CHARGE = {"ARG", "LYS"}

# NEGATIVE CHARGE NETA AT pH = 7.4
NEG_CHARGE = {"ASP", "GLU"}

# H-BOND DONOR ( NH or OH available)
HBOND_DONOR = {"ARG", "LYS", "HIS", "SER", "THR", "TYR", "TRP", "ASN", "GLN"}

# H-BOND ACCEPTOR ( O o N pair free)
HBOND_ACCEPTOR = {"ASP", "GLU", "SER", "THR", "TYR", "ASN", "GLN", "HIS"}

# H-BOND ACCEPTOR AND DONOR
HBOND_DONOR_ACCEPTOR = HBOND_DONOR & HBOND_ACCEPTOR

# POLAR AA -> partial charge (no neta)
POLAR = {"SER", "THR", "TYR", "ASN", "GLN", "CYS", "HIS"}

# IONIZABLE AA: charge changes due to pH
IONIZABLE = {"ARG", "LYS", "HIS", "ASP", "GLU", "CYS", "TYR"}


def get_aa_features(resname: str) -> dict:
    """
    Return dict of the 17 static features -> All features are binary (YES: 1, NO: 0) less hydropathyIndex which is a floar
    """
    return{
       "hydrophobic":          int(resname in HYDROPHOBIC),
        "hydrophilic":          int(resname in HYDROPHILIC),
        "hydrophatyIndex":      HYDROPATHY_INDEX.get(resname, 0.0),
        "aliphatic":            int(resname in ALIPHATIC),
        "aromatic":             int(resname in AROMATIC),
        "sulfur":               int(resname in SULFUR),
        "hydroxyl":             int(resname in HYDROXYL),
        "basic":                int(resname in BASIC),
        "acidic":               int(resname in ACIDIC),
        "amide":                int(resname in AMIDE),
        "posCharge":            int(resname in POS_CHARGE),
        "negCharge":            int(resname in NEG_CHARGE),
        "hBondDonor":           int(resname in HBOND_DONOR),
        "hBondAcceptor":        int(resname in HBOND_ACCEPTOR),
        "hBondDonorAcceptor":   int(resname in HBOND_DONOR_ACCEPTOR),
        "polar":                int(resname in POLAR),
        "ionizable":            int(resname in IONIZABLE), 
    }

###################################################################################
######################## Geometric features (8 features) ##########################
###################################################################################

# Build a KDTree with all exposed atoms to calculate point-based features

# ATOM NAMES OF CARBON, OXYGEN AND NITROGEN
CARBON_NAMES   = {"C", "CA", "CB", "CG", "CG1", "CG2", "CD", "CD1",
                  "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2"}
OXYGEN_NAMES   = {"O", "OG", "OG1", "OD1", "OD2", "OE1", "OE2",
                  "OH", "OXT"}
NITROGEN_NAMES = {"N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1",
                  "NH2", "NZ"}

# H-DONOR ATOMS (has H)
HDONOR_ATOMS  = {"N", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
                 "ND1", "OG", "OG1", "OH"}

# H-ACCEPTOR ATOMS (free electronic pair)
HACCEP_ATOMS  = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1",
                 "OH", "ND1", "NE2"}

def compute_geometric_features(residues: list[dict]) -> list[dict]:
    """
    Calculate 8 geometric features -> list of dicts for each residue

    Radio: 6 Angstrom for atoms, atomDensity, atomC/o/n, hDonor/hAcceptor and
    10 Angstrom for protrusion
    """
    all_coords = []  # coords of each atom
    all_names = []   # name of each atom

    for res in residues:
        for coord, name in zip(res["atom_coords"], res["atom_names"]):
            all_coords.append(coord)          #fill lists
            all_names.append(name)  

    all_coords = np.array(all_coords)      #conver to numpy array for vector calculations
    all_names = np.array(all_names)           

    # Filter atoms by its type
    filter_C = np.array([n in CARBON_NAMES   for n in all_names])
    filter_O = np.array([n in OXYGEN_NAMES   for n in all_names])
    filter_N = np.array([n in NITROGEN_NAMES for n in all_names])
    filter_donor = np.array([n in HDONOR_ATOMS   for n in all_names])
    filter_accep = np.array([n in HACCEP_ATOMS   for n in all_names])

    # Build KDTree with all atom coords -> fast calculation of distance between neighbors
    tree = cKDTree(all_coords)

    geometric_features = []

    for res in residues:
        ca = np.array(res["ca_coord"]).reshape(1,3) #C-alpha reference for calculate geometric features
        
        # ATOM INDEX INSIDE 6 ANGSTROM RADIO OF C-ALPHA
        index_6 = tree.query_ball_point(ca, r=6.0)[0] 
        n_atoms = len(index_6)  #absolute number of neighbor atoms of radio = 6 Angstrom

        # Calculate ATOM DENSITY around c-alpha
        if n_atoms > 0:
            dists = np.linalg.norm(all_coords[index_6] - ca, axis=1) #Euclideans distances for each neighbor atom of c-alpha
            atom_density = float(np.sum(1.0 / (1.0 + dists))) #near atoms more weight
        else:
            atom_density = 0.0 

        # Count per ATOM TYPE INSIDE RADIO = 6 ANGSTROM        
        index_set = set(index_6)
        atom_C = int(filter_C[index_6].sum()) if n_atoms > 0 else 0
        atom_O = int(filter_O[index_6].sum()) if n_atoms > 0 else 0
        atom_N = int(filter_N[index_6].sum()) if n_atoms > 0 else 0
        h_donor_atoms = int(filter_donor[index_6].sum()) if n_atoms > 0 else 0
        h_accep_atoms = int(filter_accep[index_6].sum()) if n_atoms > 0 else 0

        # PROTRUSION INDEX INSIDE RADIO 10 ANGSTROM -> concave/convex
        index_10 = tree.query_ball_point(ca, r=10.0)[0]
        protrusion = len(index_10) # protusion is the total number of protein atoms in radio 10 Angstrom

        # save geometric features
        geometric_features.append({
            "atoms":          n_atoms,
            "atomDensity":    round(atom_density, 4),
            "atomC":          atom_C,
            "atomO":          atom_O,
            "atomN":          atom_N,
            "hDonorAtoms":    h_donor_atoms,
            "hAcceptorAtoms": h_accep_atoms,
            "protrusion":     protrusion,
        })
    return geometric_features

#################################################################################
####### DSSP (5 features: 2º structure (3), flexibility(b-factor) and RSA) #######
#################################################################################

# TEORICAL MAX RESIDUE ACCESIBLE SURFACE -> normalizate absolute SASA
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
    "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, "GLY": 104.0,
    "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
    "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

def compute_dssp_features(pdb_path: Path,
                          residues: list[dict]) -> list[dict]:
    """
    Calculate secondary structure, B-factor and RAS for all residues 
    Return a list of dicts 
    """
    parser = PDBParser(QUIET=True)  #avoid unuseful warnings

    # DSSP FOR OBTAIN SECONDARY STRUCTRUE
    #Default parameters of DSSP (if there is an unexpected error)
    default = {"ss_helix": 0, "ss_sheet": 0, "ss_coil": 1,
               "bfactor": 0.0, "rsa": 0.5}
    #Get DSSP information
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        model = structure[0]
        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
    except Exception:
        return [default.copy() for _ in residues]
    
    #Build a dict of DSSP result
    dssp_dict = {}
    for key in dssp.property_keys:   #key = (chain_id, (hetflag, resnum, icode))
        data = dssp[key]
        dssp_dict[key] = {
            "ss" : data[2],   #secondary structure
            "asa" : data[3]  #absolute accesible surface
        }
    
    dssp_features = []
    for res in residues:
        key = (res["chain"], (" ", res["resnum"], res["icode"] or " "))

        if key not in dssp_dict:
            dssp_features.append(default.copy())   #if there is no dssp info for the residue, use default values
            continue
        
        #extract secondary structure and asa
        ss = dssp_dict[key]["ss"]
        asa = dssp_dict[key]["asa"]

        #Secondary structure as 3 binary columns -> 1: belongs to the category; 0: does not belong
        ss_helix = int(ss in ("H", "G", "I")) # H: alpha-helix, G: 3-10 helix, I = pi-helix
        ss_sheet = int(ss == "E") #E: beta strand
        ss_coil = int(ss not in ("H", "G", "I", "E")) #T: turn, S: bend, B: bridge, C: coil

        #B-FACTOR OF C-ALPHA FOR OBTAIN RESIDUE FLEXIBILITY -> more flexible, BS
        bfactor = 0.0
        try:
            icode = res["icode"] if res["icode"] else " "
            resiue = model[res["chain"]][(" ", res["resnum"], icode)]
            bfactor = float(resiue["CA"].get_bfactor())
        except (KeyError, Exception):
            pass

        # RSA = ASA/MAX_ASA -> normalizated (0 = burried - 1 = exposed)
        max_asa = MAX_ASA.get(res["resname"], 200.0)
        rsa = min(float(asa) / max_asa, 1.0) if max_asa > 0 else 0.5

        # save dssp features
        dssp_features.append({
            "ss_helix": ss_helix,
            "ss_sheet": ss_sheet,
            "ss_coil": ss_coil,
            "bfactor": round(bfactor, 3),
            "rsa": round(rsa, 4),
        })
    return dssp_features

#################################################################################
################ FREESASA (2 features: absolute and relative SASA) ##############
#################################################################################

# We implement both RSA and SASA as separated features to capture some quite different aspects

def compute_sasa_features(pdb_path:Path,
                          residues: list[dict]) -> list[dict]:
    """
    Calculate SASA with FreeSASA for all residues
    Return a list of dicts with sasa_absolute and sasa_relative
    """
    default = {"sasa_absolute": 0.0, "sasa_relative": 0.0}

    # Calculate SASA
    try:
        structure = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(structure)
        areas = result.residueAreas()
    except Exception:
        return [default.copy() for _ in residues]
    
    sasa_features = []
    for res in residues:
        chain = res["chain"]
        resnum = str(res["resnum"]) + res["icode"].strip()
        #Obtain absolute and relative SASA
        try:
            area = areas[chain][resnum]
            sasa_absolute = float(area.total)
            max_asa = MAX_ASA.get(res["resname"], 200.0)
            sasa_relative = min(sasa_absolute /max_asa, 1.0)
        except (KeyError, Exception):
            sasa_absolute = 0.0
            sasa_relative = 0.0
        #save SASA features 
        sasa_features.append({
            "sasa_absolute": round(sasa_absolute, 3),
            "sasa_relative": round(sasa_relative, 4)
        })
    return sasa_features

#####################################################################################
################### VOLSITE + ATOMIC HYDROPHOBICITY (7 features) ####################
#####################################################################################

# VolSIte is an cavity annotation -> each atom -> 6 types of pharmacophores: vsAromatic, vsCation, vsAnion, vsHydrophobic, vsAcceptor, vsDonor
# atomicHydrophobicity according to Kapcha-Rossky

VOLSITE_TYPES = {
    # Aromatics
    ("PHE", "CG"): "aromatic", ("PHE", "CD1"): "aromatic",
    ("PHE", "CD2"): "aromatic", ("PHE", "CE1"): "aromatic",
    ("PHE", "CE2"): "aromatic", ("PHE", "CZ"): "aromatic",
    ("TYR", "CG"): "aromatic", ("TYR", "CD1"): "aromatic",
    ("TYR", "CD2"): "aromatic", ("TYR", "CE1"): "aromatic",
    ("TYR", "CE2"): "aromatic", ("TYR", "CZ"): "aromatic",
    ("TRP", "CG"): "aromatic", ("TRP", "CD1"): "aromatic",
    ("TRP", "CD2"): "aromatic", ("TRP", "CE2"): "aromatic",
    ("TRP", "CE3"): "aromatic", ("TRP", "CZ2"): "aromatic",
    ("TRP", "CZ3"): "aromatic", ("TRP", "CH2"): "aromatic",
    ("HIS", "CG"): "aromatic", ("HIS", "CD2"): "aromatic",
    ("HIS", "CE1"): "aromatic",
    # Cationic (positive charge)
    ("ARG", "NE"): "cation",  ("ARG", "NH1"): "cation",
    ("ARG", "NH2"): "cation", ("LYS", "NZ"): "cation",
    ("HIS", "ND1"): "cation", ("HIS", "NE2"): "cation",
    # Anionic (negative charge)
    ("ASP", "OD1"): "anion", ("ASP", "OD2"): "anion",
    ("GLU", "OE1"): "anion", ("GLU", "OE2"): "anion",
    # Hydrophobics (non-polar alyphatic carbons)
    ("ALA", "CB"): "hydrophobic", ("VAL", "CB"): "hydrophobic",
    ("VAL", "CG1"): "hydrophobic", ("VAL", "CG2"): "hydrophobic",
    ("ILE", "CB"): "hydrophobic", ("ILE", "CG1"): "hydrophobic",
    ("ILE", "CG2"): "hydrophobic", ("ILE", "CD1"): "hydrophobic",
    ("LEU", "CB"): "hydrophobic", ("LEU", "CG"): "hydrophobic",
    ("LEU", "CD1"): "hydrophobic", ("LEU", "CD2"): "hydrophobic",
    ("MET", "SD"): "hydrophobic", ("MET", "CE"): "hydrophobic",
    ("PRO", "CB"): "hydrophobic", ("PRO", "CG"): "hydrophobic",
    ("PRO", "CD"): "hydrophobic",
    # H-Acceptors (O or N with free pair)
    ("SER", "OG"): "acceptor",  ("THR", "OG1"): "acceptor",
    ("TYR", "OH"): "acceptor",  ("ASN", "OD1"): "acceptor",
    ("GLN", "OE1"): "acceptor", ("HIS", "ND1"): "acceptor",
    ("MET", "SD"): "acceptor",
    # H-Donors (non-charge NH or OH)
    ("SER", "OG"): "donor",  ("THR", "OG1"): "donor",
    ("TYR", "OH"): "donor",  ("ASN", "ND2"): "donor",
    ("GLN", "NE2"): "donor", ("TRP", "NE1"): "donor",
    ("CYS", "SG"): "donor",
}

# Hydrophobicity atomic scale ofKapcha & Rossky (2014)
ATOMIC_HYDROPHOBICITY = {
    "C": 0.72,  "CA": 0.72, "CB": 0.72, "CG": 0.72,
    "CG1": 0.72, "CG2": 0.72, "CD": 0.72, "CD1": 0.72,
    "CD2": 0.72, "CE": 0.72, "CE1": 0.72, "CE2": 0.72,
    "CE3": 0.72, "CZ": 0.72, "CZ2": 0.72, "CZ3": 0.72,
    "CH2": 0.72,
    "N": -0.20,  "NE": -0.20, "NE1": -0.20, "NE2": -0.20,
    "NH1": -0.20, "NH2": -0.20, "ND1": -0.20, "ND2": -0.20,
    "NZ": -0.20,
    "O": -0.20,  "OD1": -0.20, "OD2": -0.20, "OE1": -0.20,
    "OE2": -0.20, "OG": -0.20, "OG1": -0.20, "OH": -0.20,
    "OXT": -0.20,
    "S": 0.11,  "SD": 0.11, "SG": 0.11,
}

def compute_volsite_features(residues: list[dict]) -> list[dict]:
    """
    Calculate VolSite and atomicHydrophobicity features for all residues
    """
    vs_features = []

    for res in residues:
        resname = res["resname"]
        counts = {
            "vsAromatic": 0, "vsCation": 0, "vsAnion": 0,
            "vsHydrophobic": 0, "vsAcceptor": 0, "vsDonor": 0,
        }
        atomicHydrophobicity_sum = 0.0
        #Obtain pocket properties
        for aname in res["atom_names"]:
            vtype = VOLSITE_TYPES.get((resname, aname))
            if vtype == "aromatic": counts["vsAromatic"] += 1
            elif vtype == "cation": counts["vsCation"] += 1
            elif vtype == "anion": counts["vsAnion"] += 1
            elif vtype == "hydrophobic": counts["vsHydrophobic"] += 1
            elif vtype == "acceptor": counts["vsAcceptor"] += 1
            elif vtype == "donor": counts["vsDonor"] += 1
            atomicHydrophobicity_sum += ATOMIC_HYDROPHOBICITY.get(aname, 0.0)
        #save volsite features
        vs_features.append({
            "vsAromatic": counts["vsAromatic"],
            "vsCation": counts["vsCation"],
            "vsAnion": counts["vsAnion"],
            "vsHydrophobic": counts["vsHydrophobic"],
            "vsAcceptor": counts["vsAcceptor"],
            "vsDonor": counts["vsDonor"],
            "atomicHydrophobicity": round(atomicHydrophobicity_sum, 4),
        })
    return vs_features

##################################################################################
############################# Calculate features of dataset ######################
##################################################################################

json_files = sorted(PARSED_DIR.glob("*.json"))
print(f"PDBs to process: {len(json_files)}")

all_rows = []
errors = []

for json_path in tqdm(json_files, desc = "Calculating features..."):
    pdb_id = json_path.stem
    pdb_path = RAW_DIR / f"{pdb_id}.pdb"

    if not pdb_path.exists():
        errors.append(pdb_id)
        continue

    parsed = json.loads(json_path.read_text())
    residues = parsed["residues"]

    if not residues:
        continue

    # Calculate all the features
    aa_feats = [get_aa_features(r["resname"]) for r in residues]
    geometric_feats = compute_geometric_features(residues)
    dssp_feats = compute_dssp_features(pdb_path, residues)
    sasa_feats = compute_sasa_features(pdb_path, residues)
    vs_feats = compute_volsite_features(residues)

    # Join features dicts in one per pdb and residue
    for i, res in enumerate(residues):
        row = {
            "pdb_id": pdb_id,
            "chain": res["chain"],
            "resnum": res["resnum"],
            "icode": res["icode"]
        }
        row.update(aa_feats[i])
        row.update(geometric_feats[i])
        row.update(dssp_feats[i])
        row.update(sasa_feats[i])
        row.update(vs_feats[i])
        all_rows.append(row)

df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_CSV, index= False)

print(f"\nCalculated features: {len(df.columns) - 4} columns")
print(f"Processed residues: {len(df)}")
print(f"PDBs with errors: {len(errors)}")
print(f"Save in: {OUTPUT_CSV}")
