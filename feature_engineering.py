################################################################################
########################### Summary generated data #############################
################################################################################

# Result: data/features.csv -> one row per residue and columns of labels.csv + 39 features
# Workflow:
#   1. Load parsed json
#   2. Calculate features (parallelized + cached)
#   3. Store info in features.csv
# Source: paper P2Rank

import json
import pickle
import hashlib
import os
import numpy as np
import pandas as pd
import freesasa
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm
from Bio.PDB import PDBParser, DSSP
from concurrent.futures import ProcessPoolExecutor, as_completed

PARSED_DIR = Path("data/parsed")
RAW_DIR    = Path("data/raw_pdbs")
OUTPUT_CSV = Path("data/features.csv")
CACHE_DIR  = Path("data/cache")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

#-----------------------Eliminate warnings----------------------
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#-----------------------Eliminate warnings----------------------

################################################################################
################### Static features per aa (17 features) ######################
################################################################################

HYDROPHOBIC          = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","TYR"}
HYDROPHILIC          = {"SER","THR","CYS","ASN","GLN","TYR"}
HYDROPATHY_INDEX     = {
    "ALA": 1.8,"ARG":-4.5,"ASN":-3.5,"ASP":-3.5,"CYS": 2.5,
    "GLN":-3.5,"GLU":-3.5,"GLY":-0.4,"HIS":-3.2,"ILE": 4.5,
    "LEU": 3.8,"LYS":-3.9,"MET": 1.9,"PHE": 2.8,"PRO":-1.6,
    "SER":-0.8,"THR":-0.7,"TRP":-0.9,"TYR":-1.3,"VAL": 4.2,
}
ALIPHATIC            = {"ALA","VAL","ILE","LEU"}
AROMATIC             = {"PHE","TYR","TRP","HIS"}
SULFUR               = {"CYS","MET"}
HYDROXYL             = {"SER","THR","TYR"}
BASIC                = {"ARG","LYS","HIS"}
ACIDIC               = {"ASP","GLU"}
AMIDE                = {"ASN","GLN"}
POS_CHARGE           = {"ARG","LYS"}
NEG_CHARGE           = {"ASP","GLU"}
HBOND_DONOR          = {"ARG","LYS","HIS","SER","THR","TYR","TRP","ASN","GLN"}
HBOND_ACCEPTOR       = {"ASP","GLU","SER","THR","TYR","ASN","GLN","HIS"}
HBOND_DONOR_ACCEPTOR = HBOND_DONOR & HBOND_ACCEPTOR
POLAR                = {"SER","THR","TYR","ASN","GLN","CYS","HIS"}
IONIZABLE            = {"ARG","LYS","HIS","ASP","GLU","CYS","TYR"}

def get_aa_features(resname: str) -> dict:
    return {
        "hydrophobic":        int(resname in HYDROPHOBIC),
        "hydrophilic":        int(resname in HYDROPHILIC),
        "hydrophatyIndex":    HYDROPATHY_INDEX.get(resname, 0.0),
        "aliphatic":          int(resname in ALIPHATIC),
        "aromatic":           int(resname in AROMATIC),
        "sulfur":             int(resname in SULFUR),
        "hydroxyl":           int(resname in HYDROXYL),
        "basic":              int(resname in BASIC),
        "acidic":             int(resname in ACIDIC),
        "amide":              int(resname in AMIDE),
        "posCharge":          int(resname in POS_CHARGE),
        "negCharge":          int(resname in NEG_CHARGE),
        "hBondDonor":         int(resname in HBOND_DONOR),
        "hBondAcceptor":      int(resname in HBOND_ACCEPTOR),
        "hBondDonorAcceptor": int(resname in HBOND_DONOR_ACCEPTOR),
        "polar":              int(resname in POLAR),
        "ionizable":          int(resname in IONIZABLE),
    }

################################################################################
#################### Geometric features (8 features) ##########################
################################################################################

CARBON_NAMES   = {"C","CA","CB","CG","CG1","CG2","CD","CD1","CD2","CE",
                  "CE1","CE2","CE3","CZ","CZ2","CZ3","CH2"}
OXYGEN_NAMES   = {"O","OG","OG1","OD1","OD2","OE1","OE2","OH","OXT"}
NITROGEN_NAMES = {"N","ND1","ND2","NE","NE1","NE2","NH1","NH2","NZ"}
HDONOR_ATOMS   = {"N","NE","NE1","NE2","NH1","NH2","NZ","ND1","OG","OG1","OH"}
HACCEP_ATOMS   = {"O","OD1","OD2","OE1","OE2","OG","OG1","OH","ND1","NE2"}

def compute_geometric_features(residues: list[dict]) -> list[dict]:
    """
    OPTIMIZED: query_ball_point vectorized — all Cα in a single call.
    numpy boolean filters computed in one pass with np.isin.
    """
    all_coords, all_names = [], []
    for res in residues:
        all_coords.extend(res["atom_coords"])
        all_names.extend(res["atom_names"])

    all_coords = np.array(all_coords, dtype=np.float32)
    all_names  = np.array(all_names)

    filter_C     = np.isin(all_names, list(CARBON_NAMES))
    filter_O     = np.isin(all_names, list(OXYGEN_NAMES))
    filter_N     = np.isin(all_names, list(NITROGEN_NAMES))
    filter_donor = np.isin(all_names, list(HDONOR_ATOMS))
    filter_accep = np.isin(all_names, list(HACCEP_ATOMS))

    tree = cKDTree(all_coords)

    # Single vectorized call for all Cα — the key speedup vs the original loop
    ca_coords  = np.array([res["ca_coord"] for res in residues], dtype=np.float32)
    indices_6  = tree.query_ball_point(ca_coords, r=6.0,  workers=-1)
    indices_10 = tree.query_ball_point(ca_coords, r=10.0, workers=-1)

    geometric_features = []
    for i in range(len(residues)):
        idx6    = indices_6[i]
        n_atoms = len(idx6)

        if n_atoms > 0:
            idx6_arr     = np.array(idx6)
            dists        = np.linalg.norm(all_coords[idx6_arr] - ca_coords[i], axis=1)
            atom_density = float(np.sum(1.0 / (1.0 + dists)))
            atom_C       = int(filter_C[idx6_arr].sum())
            atom_O       = int(filter_O[idx6_arr].sum())
            atom_N       = int(filter_N[idx6_arr].sum())
            h_donor      = int(filter_donor[idx6_arr].sum())
            h_accep      = int(filter_accep[idx6_arr].sum())
        else:
            atom_density = atom_C = atom_O = atom_N = h_donor = h_accep = 0

        geometric_features.append({
            "atoms":          n_atoms,
            "atomDensity":    round(atom_density, 4),
            "atomC":          atom_C,
            "atomO":          atom_O,
            "atomN":          atom_N,
            "hDonorAtoms":    h_donor,
            "hAcceptorAtoms": h_accep,
            "protrusion":     len(indices_10[i]),
        })
    return geometric_features

################################################################################
############ DSSP (5 features) + SASA (2 features) — with cache ###############
################################################################################

MAX_ASA = {
    "ALA":129.0,"ARG":274.0,"ASN":195.0,"ASP":193.0,"CYS":167.0,
    "GLN":225.0,"GLU":223.0,"GLY":104.0,"HIS":224.0,"ILE":197.0,
    "LEU":201.0,"LYS":236.0,"MET":224.0,"PHE":240.0,"PRO":159.0,
    "SER":155.0,"THR":172.0,"TRP":285.0,"TYR":263.0,"VAL":174.0,
}

def _cache_path(pdb_path: Path, suffix: str) -> Path:
    """Unique cache file per PDB based on file hash (detects changes)."""
    file_hash = hashlib.md5(pdb_path.read_bytes()).hexdigest()[:10]
    return CACHE_DIR / f"{pdb_path.stem}_{file_hash}_{suffix}.pkl"

def compute_dssp_features(pdb_path: Path,
                           residues: list[dict],
                           model=None) -> list[dict]:
    """
    OPTIMIZED:
      - Accepts pre-loaded BioPython model to avoid re-reading PDB.
      - B-factors pre-built in a single model sweep (no per-residue try/except).
      - Result cached to disk: re-runs are near-instant.
    """
    cache_file = _cache_path(pdb_path, "dssp")
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())

    default = {"ss_helix":0,"ss_sheet":0,"ss_coil":1,"bfactor":0.0,"rsa":0.5}

    try:
        if model is None:
            parser    = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_path.stem, str(pdb_path))
            model     = structure[0]
        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
    except Exception:
        result = [default.copy() for _ in residues]
        cache_file.write_bytes(pickle.dumps(result))
        return result

    # Build DSSP lookup
    dssp_dict = {}
    for key in dssp.property_keys:
        data = dssp[key]
        dssp_dict[key] = {"ss": data[2], "asa": data[3]}

    # Build B-factor lookup in a single model sweep
    bfactor_dict = {}
    for chain in model:
        for residue in chain:
            hetflag, resnum, icode = residue.id
            try:
                bfactor_dict[(chain.id, (hetflag, resnum, icode))] = \
                    float(residue["CA"].get_bfactor())
            except KeyError:
                pass

    dssp_features = []
    for res in residues:
        icode = res["icode"] or " "
        key   = (res["chain"], (" ", res["resnum"], icode))

        if key not in dssp_dict:
            dssp_features.append(default.copy())
            continue

        ss  = dssp_dict[key]["ss"]
        asa = dssp_dict[key]["asa"]

        ss_helix = int(ss in ("H","G","I"))
        ss_sheet = int(ss == "E")
        ss_coil  = int(ss not in ("H","G","I","E"))
        bfactor  = bfactor_dict.get(key, 0.0)
        max_asa  = MAX_ASA.get(res["resname"], 200.0)
        rsa      = min(float(asa) / max_asa, 1.0) if max_asa > 0 else 0.5

        dssp_features.append({
            "ss_helix": ss_helix,
            "ss_sheet": ss_sheet,
            "ss_coil":  ss_coil,
            "bfactor":  round(bfactor, 3),
            "rsa":      round(rsa, 4),
        })

    cache_file.write_bytes(pickle.dumps(dssp_features))
    return dssp_features


def compute_sasa_features(pdb_path: Path, residues: list[dict]) -> list[dict]:
    """FreeSASA with disk cache."""
    cache_file = _cache_path(pdb_path, "sasa")
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())

    default = {"sasa_absolute": 0.0, "sasa_relative": 0.0}

    try:
        structure = freesasa.Structure(str(pdb_path))
        result    = freesasa.calc(structure)
        areas     = result.residueAreas()
    except Exception:
        out = [default.copy() for _ in residues]
        cache_file.write_bytes(pickle.dumps(out))
        return out

    sasa_features = []
    for res in residues:
        chain  = res["chain"]
        resnum = str(res["resnum"]) + res["icode"].strip()
        try:
            area          = areas[chain][resnum]
            sasa_absolute = float(area.total)
            max_asa       = MAX_ASA.get(res["resname"], 200.0)
            sasa_relative = min(sasa_absolute / max_asa, 1.0)
        except (KeyError, Exception):
            sasa_absolute = sasa_relative = 0.0
        sasa_features.append({
            "sasa_absolute": round(sasa_absolute, 3),
            "sasa_relative": round(sasa_relative, 4),
        })

    cache_file.write_bytes(pickle.dumps(sasa_features))
    return sasa_features

################################################################################
################## VolSite + Atomic Hydrophobicity (7 features) ################
################################################################################

VOLSITE_TYPES = {
    ("PHE","CG"):"aromatic",  ("PHE","CD1"):"aromatic", ("PHE","CD2"):"aromatic",
    ("PHE","CE1"):"aromatic", ("PHE","CE2"):"aromatic", ("PHE","CZ"):"aromatic",
    ("TYR","CG"):"aromatic",  ("TYR","CD1"):"aromatic", ("TYR","CD2"):"aromatic",
    ("TYR","CE1"):"aromatic", ("TYR","CE2"):"aromatic", ("TYR","CZ"):"aromatic",
    ("TRP","CG"):"aromatic",  ("TRP","CD1"):"aromatic", ("TRP","CD2"):"aromatic",
    ("TRP","CE2"):"aromatic", ("TRP","CE3"):"aromatic", ("TRP","CZ2"):"aromatic",
    ("TRP","CZ3"):"aromatic", ("TRP","CH2"):"aromatic",
    ("HIS","CG"):"aromatic",  ("HIS","CD2"):"aromatic", ("HIS","CE1"):"aromatic",
    ("ARG","NE"):"cation",  ("ARG","NH1"):"cation", ("ARG","NH2"):"cation",
    ("LYS","NZ"):"cation",  ("HIS","ND1"):"cation", ("HIS","NE2"):"cation",
    ("ASP","OD1"):"anion",  ("ASP","OD2"):"anion",
    ("GLU","OE1"):"anion",  ("GLU","OE2"):"anion",
    ("ALA","CB"):"hydrophobic",  ("VAL","CB"):"hydrophobic",
    ("VAL","CG1"):"hydrophobic", ("VAL","CG2"):"hydrophobic",
    ("ILE","CB"):"hydrophobic",  ("ILE","CG1"):"hydrophobic",
    ("ILE","CG2"):"hydrophobic", ("ILE","CD1"):"hydrophobic",
    ("LEU","CB"):"hydrophobic",  ("LEU","CG"):"hydrophobic",
    ("LEU","CD1"):"hydrophobic", ("LEU","CD2"):"hydrophobic",
    ("MET","SD"):"hydrophobic",  ("MET","CE"):"hydrophobic",
    ("PRO","CB"):"hydrophobic",  ("PRO","CG"):"hydrophobic", ("PRO","CD"):"hydrophobic",
    ("SER","OG"):"donor",   ("THR","OG1"):"donor",  ("TYR","OH"):"donor",
    ("ASN","ND2"):"donor",  ("GLN","NE2"):"donor",  ("TRP","NE1"):"donor",
    ("CYS","SG"):"donor",
    ("SER","OG"):"acceptor",  ("THR","OG1"):"acceptor", ("TYR","OH"):"acceptor",
    ("ASN","OD1"):"acceptor", ("GLN","OE1"):"acceptor",
    ("HIS","ND1"):"acceptor", ("MET","SD"):"acceptor",
}

_VTYPE_TO_KEY = {
    "aromatic":"vsAromatic","cation":"vsCation","anion":"vsAnion",
    "hydrophobic":"vsHydrophobic","acceptor":"vsAcceptor","donor":"vsDonor",
}

ATOMIC_HYDROPHOBICITY = {
    "C":0.72,"CA":0.72,"CB":0.72,"CG":0.72,"CG1":0.72,"CG2":0.72,
    "CD":0.72,"CD1":0.72,"CD2":0.72,"CE":0.72,"CE1":0.72,"CE2":0.72,
    "CE3":0.72,"CZ":0.72,"CZ2":0.72,"CZ3":0.72,"CH2":0.72,
    "N":-0.20,"NE":-0.20,"NE1":-0.20,"NE2":-0.20,"NH1":-0.20,
    "NH2":-0.20,"ND1":-0.20,"ND2":-0.20,"NZ":-0.20,
    "O":-0.20,"OD1":-0.20,"OD2":-0.20,"OE1":-0.20,"OE2":-0.20,
    "OG":-0.20,"OG1":-0.20,"OH":-0.20,"OXT":-0.20,
    "S":0.11,"SD":0.11,"SG":0.11,
}

def compute_volsite_features(residues: list[dict]) -> list[dict]:
    vs_features = []
    for res in residues:
        resname   = res["resname"]
        counts    = {"vsAromatic":0,"vsCation":0,"vsAnion":0,
                     "vsHydrophobic":0,"vsAcceptor":0,"vsDonor":0}
        hydro_sum = 0.0
        for aname in res["atom_names"]:
            vtype = VOLSITE_TYPES.get((resname, aname))
            if vtype:
                key = _VTYPE_TO_KEY.get(vtype)
                if key:
                    counts[key] += 1
            hydro_sum += ATOMIC_HYDROPHOBICITY.get(aname, 0.0)
        vs_features.append({**counts,
                             "atomicHydrophobicity": round(hydro_sum, 4)})
    return vs_features

################################################################################
# Worker — standalone function required by multiprocessing
################################################################################

def process_pdb(args):
    """Process a single PDB, return (error_id_or_None, list_of_rows)."""
    json_path_str, raw_dir_str, cache_dir_str = args
    json_path = Path(json_path_str)
    pdb_id    = json_path.stem
    pdb_path  = Path(raw_dir_str) / f"{pdb_id}.pdb"

    # Each worker needs its own CACHE_DIR reference
    global CACHE_DIR
    CACHE_DIR = Path(cache_dir_str)

    if not pdb_path.exists():
        return pdb_id, []

    try:
        parsed   = json.loads(json_path.read_text())
        residues = parsed["residues"]
        if not residues:
            return None, []

        # Load BioPython model ONCE, reused by DSSP (avoids double PDB read)
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(pdb_path))
        model     = structure[0]

        aa_feats        = [get_aa_features(r["resname"]) for r in residues]
        geometric_feats = compute_geometric_features(residues)
        dssp_feats      = compute_dssp_features(pdb_path, residues, model=model)
        sasa_feats      = compute_sasa_features(pdb_path, residues)
        vs_feats        = compute_volsite_features(residues)

        rows = []
        for i, res in enumerate(residues):
            row = {
                "pdb_id": pdb_id,
                "chain":  res["chain"],
                "resnum": res["resnum"],
                "icode":  res["icode"],
            }
            row.update(aa_feats[i])
            row.update(geometric_feats[i])
            row.update(dssp_feats[i])
            row.update(sasa_feats[i])
            row.update(vs_feats[i])
            rows.append(row)

        return None, rows

    except Exception as e:
        return pdb_id, []

################################################################################
# Main loop — parallel across PDBs
################################################################################

if __name__ == "__main__":
    json_files = sorted(PARSED_DIR.glob("*.json"))
    print(f"PDBs to process: {len(json_files)}")

    # Colab free: 2 CPUs | Colab Pro: up to 8
    N_WORKERS = min(os.cpu_count() or 2, 8)
    print(f"Workers: {N_WORKERS}")
    print(f"Cache dir: {CACHE_DIR}")

    all_rows = []
    errors   = []

    args_list = [
        (str(p), str(RAW_DIR), str(CACHE_DIR))
        for p in json_files
    ]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_pdb, a): a[0] for a in args_list}
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Calculating features..."):
            err, rows = future.result()
            if err:
                errors.append(err)
            else:
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nFeatures calculated : {len(df.columns) - 4} columns")
    print(f"Residues processed  : {len(df)}")
    print(f"PDBs with errors    : {len(errors)}")
    if errors:
        print(f"  -> {errors}")
    print(f"Saved to            : {OUTPUT_CSV}")
