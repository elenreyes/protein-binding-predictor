############################################################################
#################### Summary generated data ################################
############################################################################

# Result:
#   results/predictions_PDBID.csv -> predicted residues as binding sites
#   results/predictions_PDBID.pml -> PyMOL script for visualization

# USE:
#   Direct: python predict.py --pdb data/raw_pdbs/1abc.pdb --threshold 0.4
#   Imported from train.py

# Workflow:
# 1. Parse new PDB -> import parse_structures.py
# 2. Calculate features -> import feature_engineering
# 3. Load prediction model and optimum threshold -> from models/ folder
# 4. Predct residue probabilities for being binding site
# 5. Save ranking csv (predictions_PDBID.csv) and PyMOL script (predictions_PDBID.pml)

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import freesasa
from pathlib import Path
from scipy.spatial import cKDTree 
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.Polypeptide import is_aa

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#############################################################################
################# Parse new PDB #############################################
#############################################################################

def parse_pdb_for_prediction(pdb_path:Path) -> list[dict]:
    """
    Parse a new PDB and return a list of residues dicts with its data (atoms)
    """

    parser = PDBParser(QUIET=True)   #avoid unuseful warnings

    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        model = structure[0]
    except Exception as e:
        raise ValueError(f"Can not parse {pdb_path}: {e}")
    
    residues = []
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            if "CA" not in res:
                continue

            ca        = res["CA"]
            all_atoms = list(res.get_atoms())

            residues.append({
                "chain": chain.id,
                "resnum": res.id[1],
                "icode": res.id[2].strip(),
                "resname": res.get_resname().strip(),
                "ca_coord": ca.get_coord().tolist(),
                "atom_coords": [a.get_coord().tolist() for a in all_atoms],
                "atom_names": [a.get_name().strip()   for a in all_atoms],
            })

    if not residues:
        raise ValueError (f"Standard aminoacids not found in {pdb_path}")
    print(f"Found residues: {len(residues)}")
    return residues, model

#################################################################################
################## Calculate features ###########################################
#################################################################################

def compute_features_for_prediction(residues: list[dict],
                                    model,
                                    pdb_path: Path) -> pd.DataFrame:
    """
    Calculate 39 features for the new PDB reusing functions from feature_engineering.py.
    """
    # Import all the functions from feature_engineering
    from feature_engineering import (
        get_aa_features,
        compute_geometric_features,
        compute_dssp_features,
        compute_sasa_features,
        compute_volsite_features,
    )

    aa_feats   = [get_aa_features(r["resname"]) for r in residues]
    geo_feats  = compute_geometric_features(residues)
    dssp_feats = compute_dssp_features(pdb_path, residues)
    sasa_feats = compute_sasa_features(pdb_path, residues)
    vs_feats   = compute_volsite_features(residues)

    rows = []
    for i, res in enumerate(residues):
        row = {}
        row.update(aa_feats[i])
        row.update(geo_feats[i])
        row.update(dssp_feats[i])
        row.update(sasa_feats[i])
        row.update(vs_feats[i])
        rows.append(row)

    return pd.DataFrame(rows)

##########################################################################
################# Export PyMOL script ####################################
##########################################################################

# Result: .pml file : pymol predictions_PDBID.pml

# Color pattern:
#   Red: probability >= 0.7 (high confidence)
#   Naranja: probability >= 0.5  (medium confidence)
#   Amarillo: probability >= 0.3  (low confidence)
#   Gris: non binding

def export_pymol_script(result_df: pd.DataFrame,
                        pdb_path: Path,
                        output_path: Path):
    """
    Generate a PyMOL script that color residues according its probability of being binding site.
    """

    pdb_abs = pdb_path.resolve() #absolute route that allows PyMOL for find the file

    with open(output_path, "w") as f:

        # Script pymol
        f.write(f"load {pdb_abs}\n")
        f.write("hide everything\n")
        f.write("show cartoon\n")
        f.write("color grey80, all\n")    # grey for all the protein (NBS)
        f.write("set cartoon_transparency, 0.3\n\n")
        # Color residues according to its probability of being BS
        for _, row in result_df.sort_values("binding_prob", ascending=False).iterrows():
            prob   = row["binding_prob"]
            chain  = row["chain"]
            resnum = row["resnum"]
            icode  = str(row["icode"]).strip()

            resi = f"{resnum}{icode}" if icode else str(resnum)

            if prob >= 0.7:
                color = "red"
            elif prob >= 0.5:
                color = "orange"
            elif prob >= 0.3:
                color = "yellow"
            else:
                continue 
            # Pymol script: color residues and show them as sticks
            f.write(f"color {color}, chain {chain} and resi {resi}\n")
        f.write("\n")    
        f.write(f"select predicted_binding, ")

        # Select residues with prob >= model_threshold
        binding_residues = result_df[result_df["predicted"] == 1]
        if len(binding_residues) > 0:
            sel_parts = [
                f"(chain {r['chain']} and resi {r['resnum']})"
                for _, r in binding_residues.iterrows()
            ]
            f.write(" or ".join(sel_parts) + "\n")
            f.write("show sticks, predicted_binding\n")
            f.write("show surface, predicted_binding\n")
            f.write("set surface_transparency, 0.5, predicted_binding\n")
        else:
            f.write("none\n")   # empty selection if there is no predictions

        # Show ligands of PDB if there are some
        f.write("\nshow sticks, organic\n")
        f.write("color green, organic\n\n")

    print(f" Script PyMOL save in: {output_path}")
    print(f" Execute this to see binding sites for you protein: pymol {output_path.name}")

###############################################################################
###################### Prediction #############################################
###############################################################################

def predict_binding_sites (pdb_path: Path | str,
                           pipeline=None,
                           feature_names: list[str] = None,
                           threshold: float = None) -> pd.DataFrame:
    """
    Predict BS for new PDB structure. If pipeline is None -> load binding_site_model.pkl;
    if threshold is None -> use optimum threshold: model_meta.json
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Not found PDB: {pdb_path}")
    
    #Load model and metadata if the script is not use from train.py
    if pipeline is None:
        model_pkl = MODELS_DIR / "binding_site_model.pkl"
        meta_path = MODELS_DIR / "model_meta.json"

        if not model_pkl.exists():
            raise FileNotFoundError(
                f"Not found model in {model_pkl}"
                "Execute train.py first"
            )
        pipeline = joblib.load(model_pkl)
        meta = json.loads(meta_path.read_text())
        feature_names = meta["feature_names"]
        threshold = threshold or meta["threshold"]

    #1.Parse PDB
    print("\n[1\4] Parsing structure ...")
    residues, model = parse_pdb_for_prediction(pdb_path)

    #2. Calculate features
    print("[2/4] Calculate features ...")
    features_df = compute_features_for_prediction(residues, model, pdb_path)
    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0.0
    X = features_df[feature_names].values

    #3. Predict probabilities
    print("[3/4] Predicting ...")
    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs <= threshold).astype(int)
    #Build DataFrame: id.results + pred
    results = pd.DataFrame({
        "chain": [r["chain"] for r in residues],
        "resnum": [r["resnum"] for r in residues],
        "icode": [r["icode"] for r in residues],
        "resname": [r["resname"] for r in residues],
        "binding_prob": probs.round(4),
        "predicted": preds,
    }).sort_values("binding_prob", ascending=False).reset_index(drop=True)
    n_predicted = preds.sum()
    n_total = len(residues)
    print(f"Predict residues as binding site: {n_predicted}/{n_total}")
    print(f"\n Top 10 resiudes:")
    print(results[["chain", "resnum", "resname", "binding_prob"]].head(10)
          .to_string(index=False))
    
    #4. Save results
    print("\n[4/4] Saving results...")
    pdb_stem = pdb_path.stem
    csv_out  = RESULTS_DIR / f"predictions_{pdb_stem}.csv"
    pml_out  = RESULTS_DIR / f"predictions_{pdb_stem}.pml"
    results.to_csv(csv_out, index=False)
    print(f"Predictions in: {csv_out}")
    export_pymol_script(results, pdb_path, pml_out)
    return results

################################################################################
########## Use predict.py  as command line in terminal console #################
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict binding sites in new PDB"
    )
    parser.add_argument("--pdb", required=True,
                        help="route_pdb_file.pdb")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Probability threshold (model_threshold as default)")
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    results = predict_binding_sites(
        pdb_path  = args.pdb,
        threshold = args.threshold,
    )