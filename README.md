# Protein-Binding-Predictor
SBI -PYT Project
## Overview
 
This project implements a Random Forest classifier to predict ligand binding sites in protein structures directly from their 3D coordinates. 
Given a PDB file, the model assigns a binding probability to each residue and generates a PyMOL visualization script to inspect the predictions.
 
The approach is inspired by [P2Rank](https://github.com/rdk/p2rank) (Krivak & Hoksza, 2018) and uses a similar set of physicochemical, geometric,
 and structural features computed per residue.
 
The training data comes from [BioLiP](https://zhanggroup.org/BioLiP/), a curated database of biologically relevant protein-ligand interactions. We use 2000 proteins selected by random.

 
## Requirements
 
### Python packages
conda install -c conda-forge numpy pandas scipy scikit-learn biopython tqdm requests matplotlib joblib dssp freesasa

### Visualization
- [PyMOL](https://pymol.org/) — to open the generated `.pml` scripts
 
 
## Project Structure

project/
│
├── BioLip_nr.txt.gz              # BioLiP dataset (download separately)
│
├── download_biolip.py            # Step 1 — reads BioLiP, selects diverse PDBs, downloads them
├── parse_structures.py           # Step 2 — parses PDB files into JSON
├── build_dataset.py              # Step 3 — computes residue-ligand labels (0/1)
├── feature_engineering.py        # Step 4 — computes 39 features per residue
├── merge_dataset.py              # Step 5 — merges labels.csv + features.csv into dataset.csv
├── train.py                      # Step 6 — trains Random Forest, evaluates, saves model
├── predict.py                    # Step 7 — predicts binding sites on a new PDB
│
├── data/
│   ├── raw_pdbs/                 # Downloaded .pdb files
│   ├── parsed/                   # Parsed JSON files (one per PDB)
│   ├── labels.csv                # Residue labels: pdb_id, chain, resnum, ca_x, ca_y, ca_z, label
│   ├── features.csv              # 39 features per residue
│   ├── dataset.csv               # labels + features merged (input to train.py)
│   ├── pdb_ids.txt               # List of selected PDB IDs
│   ├── ligand_index.json         # BioLiP ligand index {pdb_id: [{ligand, chain}]}
│   └── failed_ids.txt            # PDB IDs that failed to download
│
├── models/
│   ├── binding_site_model.pkl    # Trained pipeline (StandardScaler + RandomForest)
│   └── model_meta.json           # Feature names, threshold, and test metrics
│
└── results/
    ├── metrics.txt               # Full evaluation report
    ├── roc_curve.png             # ROC curve
    ├── pr_curve.png              # Precision-Recall curve
    ├── feature_importance.png    # Feature importances (Gini)
    ├── confusion_matrix.png      # Confusion matrix
    ├── predictions_PDBID.csv     # Per-residue binding probabilities
    └── predictions_PDBID.pml     # PyMOL visualization script

---
 
## Pipeline
 
The pipeline consists of 7 sequential scripts. Each script reads the output of the previous one.
 
### Step 1 — `download_biolip.py`
 
Reads `BioLip_nr.txt.gz` to extract PDB IDs and their associated ligands. Then choose 2000 pdb structures by chance

Output:
- `data/pdb_ids.txt` — selected PDB IDs
- `data/ligand_index.json` — maps each PDB to its BioLiP ligands
- `data/raw_pdbs/*.pdb` — downloaded structure files
---
 
### Step 2 — `parse_structures.py`
 
Parses each `.pdb` file using BioPython's `PDBParser`. Extracts for each residue: chain, residue number, residue name, Cα coordinates, and all atom coordinates. Also extracts ligand information: name, number of atoms, atom coordinates, and center of mass.
 
Stores results as JSON files — one per PDB — for fast access in subsequent steps.
 
Output:
- `data/parsed/*.json` — one JSON per PDB with residues and ligands
- `data/parse_errors.txt` — IDs that failed to parse
 
---
 
### Step 3 — `build_dataset.py`
 
Computes binary labels for each residue. A residue is labeled 1 (binding site) if any of its atoms is within 4 Å of any atom of a BioLiP-relevant ligand (with ≥5 atoms). Otherwise it is labeled 0.
 
Distances are computed efficiently using `scipy.spatial.cKDTree`. PDBs where no residue is labeled as a binding site are discarded.
 
Output:
- `data/labels.csv` — columns: `pdb_id, chain, resnum, icode, resname, ca_x, ca_y, ca_z, label`
 
> Note: `ca_x`, `ca_y`, `ca_z` are Cα coordinates stored as auxiliary data for `feature_engineering.py`. They are not used as model features, they are dropped before training in `merge_dataset.py`.
 
---
 
### Step 4 — `feature_engineering.py`
 
Computes 39 features per residue organized in 5 groups:
 
| Group | Features | Tool |
|-------|----------|------|
| Amino acid properties | hydrophobic, hydrophilic, hydrophatyIndex, aliphatic, aromatic, sulfur, hydroxyl, basic, acidic, amide, posCharge, negCharge, hBondDonor, hBondAcceptor, hBondDonorAcceptor, polar, ionizable | Lookup table |
| Geometric (neighbourhood) | atoms, atomDensity, atomC, atomO, atomN, hDonorAtoms, hAcceptorAtoms, protrusion | KDTree (scipy) |
| Secondary structure + flexibility | ss_helix, ss_sheet, ss_coil, bfactor, rsa | DSSP (BioPython) |
| Solvent accessibility | sasa_total, sasa_relative | FreeSASA |
| VolSite + atomic hydrophobicity | vsAromatic, vsCation, vsAnion, vsHydrophobic, vsAcceptor, vsDonor, atomicHydrophobicity | Lookup table |
 
Output:
- `data/features.csv` — 39 features per residue, indexed by `pdb_id, chain, resnum, icode`
 
---
 
### Step 5 — `merge_dataset.py`
 
Merges `labels.csv` (residue identifiers, Cα coordinates, and binding site labels) with `features.csv` (39 computed features) using an inner join on `(pdb_id, chain, resnum, icode)`.
 
Drops the Cα coordinates (`ca_x`, `ca_y`, `ca_z`) from the final dataset, these were needed by `feature_engineering.py` to compute geometric features (neighbour counts, protrusion) 
but must not be used as model inputs, as they encode absolute protein position in space rather than residue properties.
 
Output:
- `data/dataset.csv` — 39 features + label per residue, ready for training
 
---
 
### Step 6 — `train.py`
 
Trains a `RandomForestClassifier` (300 trees, `class_weight="balanced"`) wrapped in a `Pipeline` with `StandardScaler`.
 
Key design decisions:
 
- Split by PDB using `GroupShuffleSplit` (70% train / 15% val / 15% test) — prevents data leakage from the same protein appearing in both train and test sets. 
Splitting by residue instead would give artificially inflated metrics.
- 5-fold cross-validation with `GroupKFold` on the training set — respects PDB boundaries, giving an honest performance estimate without leakage.
- Optimal threshold searched on the validation set by maximizing F1 — not on the test set, which would inflate reported metrics.
 
Output:
- `models/binding_site_model.pkl` — trained pipeline
- `models/model_meta.json` — feature names, threshold, and test metrics
- `results/` — ROC curve, PR curve, feature importances, confusion matrix, metrics report
 
Usage:
     bash
python train.py
python train.py --input data/dataset.csv   # custom CSV

---
 
### Step 7 — `predict.py`
 
Given a new PDB file (not seen during training), computes the same 39 features using the same functions as `feature_engineering.py`, loads the trained model, and predicts a binding probability for each residue.
 
Generates a ranked CSV and a PyMOL script coloring residues by predicted confidence.
 
**Color scheme in PyMOL:**
 
| Color | Binding probability |
|-------|-------------------|
| Red | ≥ 0.7 — high confidence |
| Orange | ≥ 0.5 — medium confidence |
| Yellow | ≥ 0.3 — low confidence |
| Grey | < 0.3 — not predicted as binding |
| Green | Original ligand in PDB (reference) |
 
Usage:
```bash
python predict.py --pdb data/raw_pdbs/1abc.pdb
python predict.py --pdb my_protein.pdb --threshold 0.6
```
 
Output:
- `results/predictions_PDBID.csv` — residues ranked by binding probability
- `results/predictions_PDBID.pml` — PyMOL visualization script
 
> `train.py --predict` vs `predict.py --pdb`: both do exactly the same thing. `train.py --predict` is a convenience shortcut that internally calls `predict_binding_sites()` from `predict.py`. 
Using `predict.py --pdb` directly is cleaner and more explicit, prefer it when you only want to run predictions without thinking about training.
 
---
 
## Usage
 
### Full pipeline from scratch
 
```bash
# 1. Select diverse proteins and download PDB files
python download_biolip.py
 
# 2. Parse PDB files into JSON
python parse_structures.py
 
# 3. Compute residue-ligand labels
python build_dataset.py
 
# 4. Compute 39 features per residue
python feature_engineering.py
 
# 5. Merge labels and features
python merge_dataset.py
 
# 6. Train and evaluate the model
python train.py
 
# 7. Predict on a new protein
python predict.py --pdb my_protein.pdb
```
 
# Predict
python predict.py --pdb 1hsg.pdb --threshold 0.5
 
# Open in PyMOL
pymol results/predictions_1hsg.pml

 
## Output & Visualization
 
After running `predict.py`, open PyMOL and run the generated script:
 
```
# In PyMOL console:
run results/predictions_1hsg.pml
 
# Or from terminal:
pymol results/predictions_1hsg.pml
```
 
The protein appears as a grey cartoon. Predicted binding residues are colored red/orange/yellow by confidence level and shown as sticks with a transparent surface. The original ligand (if present in the PDB) appears in green for visual comparison. PyMOL automatically zooms into the predicted binding region.
 
---
 
## Model Performance  -> change parameters (those are from cluster approach model)
 
Trained on ~2000 PDBs selected by chance from BioLiP:
 
| Metric | Value |
|--------|-------|
| ROC-AUC (test) | 0.782 |
| AUC-PR (test) | 0.181 |
| Binding site recall | 37% |
| Binding site precision | 18% |
| Dataset balance | 5.9% positive / 94.1% negative |
| Neg/pos ratio | 16.0x |
 
Most informative features (by correlation with binding label):
 
| Feature | Correlation | Biological meaning |
|---------|-------------|-------------------|
| bfactor | 0.072 | Binding sites are more rigid (lower B-factor) |
| aromatic | 0.058 | Aromatic residues are enriched in binding pockets |
| vsAromatic | 0.058 | Atomic-level aromaticity confirms the above |
| negCharge | 0.045 | Negative charges frequent in binding interfaces |
| protrusion | 0.041 | Pocket geometry — less exposed means more buried |
 
Performance improves substantially with more training data. Recommended minimum: 2000 PDBs.
 
---
 
## Known Limitations
 
- Small dataset: 2000 PDBs is well below the recommended minimum for this task. P2Rank uses 4000+ structures. Increasing `MAX_PDBS` in `download_biolip.py` is the single most impactful improvement available.
- High false positive rate: Precision at binding sites is ~18% — for every 100 residues predicted as binding site, ~82 are false positives. Use a higher threshold (`--threshold 0.6` or `0.7`) in `predict.py` to reduce false positives at the cost of lower recall.
- DSSP failures: DSSP fails silently on ~5-10% of PDBs with discontinuous chains and returns default values (`ss_coil=1`, `bfactor=0`), adding noise to secondary structure features.
- No evolutionary features: PSSM and conservation scores (among the most informative features in the literature) are not implemented due to computational cost. Adding them would require PSI-BLAST or ConSurf.
- Residue-level prediction: The model predicts at residue level, whereas P2Rank uses solvent-accessible surface points, which gives finer spatial resolution and better pocket localization.
- CV vs test gap: The difference between CV ROC-AUC (~0.92) and test ROC-AUC (0.782) reflects genuine overfitting to the small training set, not a methodological error — the CV correctly uses `GroupKFold`.
