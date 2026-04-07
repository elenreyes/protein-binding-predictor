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
# 4. Predict residue probabilities for being binding site
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

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Top-N% of residues to predict as binding site (rank-based, threshold-independent)
TOP_PERCENT = 0.12   # 12% → ~24 residues for a 198-residue protein

#-----------------------Eliminate warnings----------------------
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#-----------------------Eliminate warnings----------------------

#############################################################################
################# Parse new PDB #############################################
#############################################################################

def parse_pdb_for_prediction(pdb_path: Path) -> list[dict]:
    """
    Parse a new PDB and return a list of residue dicts with atom data.
    """
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        model = structure[0]
    except Exception as e:
        raise ValueError(f"Cannot parse {pdb_path}: {e}")

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
                "chain":       chain.id,
                "resnum":      res.id[1],
                "icode":       res.id[2].strip(),
                "resname":     res.get_resname().strip(),
                "ca_coord":    ca.get_coord().tolist(),
                "atom_coords": [a.get_coord().tolist() for a in all_atoms],
                "atom_names":  [a.get_name().strip()   for a in all_atoms],
            })

    if not residues:
        raise ValueError(f"No standard amino acids found in {pdb_path}")
    print(f"  Found residues: {len(residues)}")
    return residues, model


#################################################################################
################## Calculate features ###########################################
#################################################################################

def compute_features_for_prediction(residues: list[dict],
                                    model,
                                    pdb_path: Path) -> pd.DataFrame:
    """
    Calculate features for the new PDB reusing functions from feature_engineering.py.
    """
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

# Color pattern (applied only to predicted residues, ranked by probability):
#   Red    → top third of predicted residues    (highest confidence)
#   Orange → middle third of predicted residues (medium confidence)
#   Yellow → bottom third of predicted residues (lowest confidence)
#   Grey   → not predicted (rest of protein)

def export_pymol_script(result_df: pd.DataFrame,
                        pdb_path: Path,
                        output_path: Path):

    """
    Generate a PyMOL script that colors ONLY the predicted binding-site
    residues (predicted == 1), splitting them into three confidence tiers
    based on their rank within the predicted set.
    All other residues remain grey.
    """
    pdb_abs = pdb_path.resolve()

    # Work only with predicted residues
    predicted_df = result_df[result_df["predicted"] == 1].copy()

    def tier_color(prob: float) -> str:
        if prob >= 0.7:
            return "red"
        elif prob >= 0.5:
            return "orange"
        elif prob >= 0.3:
            return "yellow"

    with open(output_path, "w") as f:

        f.write(f"load {pdb_abs}\n")
        f.write("hide everything\n")
        f.write("show cartoon\n")
        f.write("color grey80, all\n")
        f.write("set cartoon_transparency, 0.3\n\n")

        # Color only predicted residues based on probability
        for _, row in predicted_df.iterrows():
            chain  = row["chain"]
            resnum = row["resnum"]
            icode  = str(row["icode"]).strip()
            resi   = f"{resnum}{icode}" if icode else str(resnum)
            color  = tier_color(row["binding_prob"])

            f.write(f"color {color}, chain {chain} and resi {resi}\n")

        f.write("\n")

        # Select all predicted residues for sticks + surface
        f.write("select predicted_binding, ")
        if len(predicted_df) > 0:
            sel_parts = [
                f"(chain {r['chain']} and resi {r['resnum']})"
                for _, r in predicted_df.iterrows()
            ]
            f.write(" or ".join(sel_parts) + "\n")
            f.write("show sticks, predicted_binding\n")
            f.write("show surface, predicted_binding\n")
            f.write("show surface\n")
        else:
            f.write("none\n")

        # Show any ligands already present in the PDB
        f.write("\nshow sticks, organic\n")
        f.write("color green, organic\n\n")

    print(f"  PyMOL script saved: {output_path}")
    print(f"  Visualize with   : pymol {output_path.name}")

###############################################################################
###################### Prediction #############################################
###############################################################################

def predict_binding_sites(pdb_path: Path | str,
                          pipeline=None,
                          feature_names: list[str] = None,
                          threshold: float = None,
                          top_percent: float = TOP_PERCENT) -> pd.DataFrame:
    """
    Predict binding-site residues for a new PDB structure.

    Selection strategy: rank all residues by predicted probability and label
    the top `top_percent` fraction as binding site (rank-based, not threshold-
    based).  The stored model threshold is still used to report how many
    residues exceed it, but does NOT control the final `predicted` column.

    Parameters
    ----------
    pdb_path      : path to .pdb file
    pipeline      : fitted sklearn Pipeline (loaded from disk if None)
    feature_names : list of feature column names (loaded from meta if None)
    threshold     : kept for compatibility / reporting only
    top_percent   : fraction of residues to label as binding site (default 0.12)
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    # Load model and metadata if called standalone
    if pipeline is None:
        model_pkl = MODELS_DIR / "binding_site_model_v1.pkl"
        meta_path = MODELS_DIR / "model_meta_v1.json"

        if not model_pkl.exists():
            raise FileNotFoundError(
                f"Model not found at {model_pkl}. Run train.py first."
            )
        pipeline      = joblib.load(model_pkl)
        meta          = json.loads(meta_path.read_text())
        feature_names = meta["feature_names"]
        threshold     = threshold or meta["threshold"]

    # ── 1. Parse PDB ─────────────────────────────────────────────────────
    print("\n[1/4] Parsing structure ...")
    residues, model = parse_pdb_for_prediction(pdb_path)

    # ── 2. Calculate features ─────────────────────────────────────────────
    print("[2/4] Calculating features ...")
    features_df = compute_features_for_prediction(residues, model, pdb_path)

    # Ensure all expected feature columns exist (fill missing with 0)
    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0.0
    X = features_df[feature_names].values

    # ── 3. Predict probabilities ──────────────────────────────────────────
    print("[3/4] Predicting ...")
    probs = pipeline.predict_proba(X)[:, 1]

    print(f"  Min prob  : {probs.min():.4f}")
    print(f"  Max prob  : {probs.max():.4f}")
    print(f"  Mean prob : {probs.mean():.4f}")
    print(f"  Median    : {np.median(probs):.4f}")
    if threshold is not None:
        print(f"  > threshold ({threshold:.3f}): {(probs >= threshold).sum()} residues")

    # Build results DataFrame sorted by probability (highest first)
    results = pd.DataFrame({
        "chain":        [r["chain"]   for r in residues],
        "resnum":       [r["resnum"]  for r in residues],
        "icode":        [r["icode"]   for r in residues],
        "resname":      [r["resname"] for r in residues],
        "binding_prob": probs.round(4),
        "predicted":    0,              # filled below
    }).sort_values("binding_prob", ascending=False).reset_index(drop=True)

    # Label top N% as predicted binding site (rank-based)
    n_predicted = max(10, int(len(results) * top_percent))
    results.loc[results.index[:n_predicted], "predicted"] = 1

    print(f"  Predicted as binding site: {n_predicted}/{len(residues)} "
          f"(top {top_percent*100:.0f}%)")
          
    print(f"\n  Top 10 residues:")
    print(results[["chain", "resnum", "resname", "binding_prob"]].head(10)
          .to_string(index=False))

    # ── 4. Save results ───────────────────────────────────────────────────
    print("\n[4/4] Saving results ...")
    pdb_stem = pdb_path.stem
    csv_out  = RESULTS_DIR / f"predictions_{pdb_stem}.csv"
    pml_out  = RESULTS_DIR / f"predictions_{pdb_stem}.pml"

    results.to_csv(csv_out, index=False)
    print(f"  Predictions CSV : {csv_out}")

    export_pymol_script(results, pdb_path, pml_out)

    return results


################################################################################
########## Use predict.py as command line ######################################
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict binding sites in a new PDB structure."
    )
    parser.add_argument("--pdb", required=True,
                        help="Path to the .pdb file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Probability threshold for reporting only "
                             "(does not control final predictions)")
    parser.add_argument("--top_percent", type=float, default=TOP_PERCENT,
                        help=f"Fraction of residues to predict as binding site "
                             f"(default: {TOP_PERCENT})")
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    results = predict_binding_sites(
        pdb_path    = args.pdb,
        threshold   = args.threshold,
        top_percent = args.top_percent,
    )
