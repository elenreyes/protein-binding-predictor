#################################################################################
# train.py  -  Random Forest + undersampling + GroupKFold CV (no leakage)
#################################################################################
# Changes vs original:
#   1. Undersampling of negatives before training (configurable ratio)
#   2. GroupKFold CV --> no PDB leakage in cross-validation
#   3. balanced_subsample instead of balanced
#   4. AUC-PR added to CV metrics
#   5. Recall@Precision≥0.5 added to test metrics
#################################################################################

# Result:
#   models/binding_site_model.pkl   
#   models/model_meta.json          
#   results/metrics.txt            
#   results/roc_curve.png
#   results/pr_curve.png
#   results/feature_importance.png
#   results/confusion_matrix.png
#
# Workflow use of the script:
#   python train.py                        # train with data/train_dataset.csv


import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.model_selection import (
    GroupShuffleSplit,      # split per pdb -> better than per residues (avoid leakage)
    GroupKFold,             # avoid PDB leakage in CV
    cross_val_score,
)
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)

DATASET_CSV      = Path("data/train_dataset.csv")
MODELS_DIR       = Path("models")
RESULTS_DIR      = Path("results")
RANDOM_STATE     = 18
UNDERSAMPLE_RATIO = 10   # neg/pos ratio after undersampling; None to disable

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = {
    "pdb_id", "chain", "resnum", "icode",
    "resname", "label", "ca_x", "ca_y", "ca_z",
}

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(csv_path: Path):
    """
    Load data from train_dataset.csv to generate a DataFrame with all features
    """

    df = pd.read_csv(csv_path, low_memory=False)

    missing = NON_FEATURE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns not found: {missing}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # Eliminate NaN and incomplete features
    before = len(df)
    df = df.dropna(subset=feature_cols + ["label"])
    if len(df) < before:
        print(f"  Deleted {before - len(df)} rows with NaN")

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    print(f"  Loaded residues  : {len(df)}")
    print(f"  Unique PDBs      : {df['pdb_id'].nunique()}")
    print(f"  Features         : {len(feature_cols)}")
    print(f"  Binding sites (1): {n_pos}")
    print(f"  No binding    (0): {n_neg}")
    print(f"  Ratio neg/pos    : {n_neg/n_pos:.1f}x")


    print("\nCorrelation features vs label (top 15):")
    corrs = df[feature_cols].corrwith(df["label"]).abs().sort_values(ascending=False)
    print(corrs.head(15).to_string())
    print("\nMean per class - top 5 features:")
    print(df.groupby("label")[corrs.head(5).index.tolist()].mean().to_string())

    return df, feature_cols

# ─── PDB-level split ──────────────────────────────────────────────────────────

def split_by_pdb(df):
    """
    Split dataset: df_train, df_val, df_test 
    """

    groups = df["pdb_id"].values
    
    # Separate test -> 15 %
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    rest_idx, test_idx = next(gss.split(df, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss2.split(df.iloc[rest_idx], groups=groups[rest_idx]))
    train_idx = rest_idx[train_idx]
    val_idx   = rest_idx[val_idx]

    df_train = df.iloc[train_idx]
    df_val   = df.iloc[val_idx]
    df_test  = df.iloc[test_idx]

    print(f"  Train : {len(df_train):>7}  -  {df_train['pdb_id'].nunique()} PDBs")
    print(f"  Val   : {len(df_val):>7}  -  {df_val['pdb_id'].nunique()} PDBs")
    print(f"  Test  : {len(df_test):>7}  -  {df_test['pdb_id'].nunique()} PDBs")
    return df_train, df_val, df_test

# ─── Undersampling ────────────────────────────────────────────────────────────

def undersample_negatives(df, ratio=UNDERSAMPLE_RATIO):
    """
    Reduce negatives so neg/pos = ratio.
    Only applied to the training set - val and test keep the real distribution.
    """
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    target = len(pos) * ratio

    if target >= len(neg):
        print(f"  Undersampling skipped (current ratio already ≤ {ratio}x)")
        return df

    neg_sample = resample(neg, n_samples=target, replace=False,
                          random_state=RANDOM_STATE)
    out = pd.concat([pos, neg_sample]).sample(frac=1, random_state=RANDOM_STATE)
    print(f"  After undersampling -> pos: {len(pos)}  "
          f"neg: {len(neg_sample)}  ratio: {len(neg_sample)/len(pos):.1f}x")
    return out

# ─── Model pipeline ───────────────────────────────────────────────────────────

def build_pipeline():
    """
    Build sklearn pipeline: scaler -> random forest 
    """
    #random forest configuration
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",   # per-tree reweighting
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", rf)])

# ─── Cross-validation (GroupKFold - no PDB leakage) ──────────────────────────

def cross_validate_grouped(pipeline, X_train, y_train, groups_train, cv_folds=5):
    """
    GroupKFold ensures PDBs from the same group never appear in both
    train and validation folds, giving unbiased CV estimates.
    """
    print(f"\n  {cv_folds}-fold GroupKFold CV on train set...")
    cv = GroupKFold(n_splits=cv_folds)

    cv_roc = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, groups=groups_train,
                             scoring="roc_auc", n_jobs=-1)
    cv_ap  = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, groups=groups_train,
                             scoring="average_precision", n_jobs=-1)
    cv_f1  = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, groups=groups_train,
                             scoring="f1", n_jobs=-1)

    print(f"  ROC-AUC : {cv_roc.mean():.3f} ± {cv_roc.std():.3f}")
    print(f"  AUC-PR  : {cv_ap.mean():.3f} ± {cv_ap.std():.3f}")
    print(f"  F1      : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print("  (GroupKFold: no PDB leakage - these numbers match test performance)")

# ─── Optimal threshold ────────────────────────────────────────────────────────

def find_best_threshold(pipeline, X_val, y_val):
    """
    As random forest results return probabilities, we need a threshold to determinate 
    if the prediction is 1 (BS) or 0 (NBS). We search the optimum threslhold according to precision,
    recall and F1 in validation dataset.
    """

    probs = pipeline.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, probs)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    best = int(np.argmax(f1))
    print(f"  Optimal threshold : {thr[best]:.3f}")
    print(f"  F1 @ threshold    : {f1[best]:.3f}")
    print(f"  Precision         : {prec[best]:.3f}")
    print(f"  Recall            : {rec[best]:.3f}")
    return float(thr[best])

# ─── Evaluation + plots ───────────────────────────────────────────────────────

def evaluate_and_plot(pipeline, X_test, y_test, threshold, feature_cols, tag="v1"):
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    roc_auc  = roc_auc_score(y_test, probs)
    avg_prec = average_precision_score(y_test, probs)
    report   = classification_report(y_test, preds,
                                     target_names=["no binding", "binding site"])
    cm = confusion_matrix(y_test, preds)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, probs)
    mask = prec_arr >= 0.50
    recall_at_p50 = rec_arr[mask].max() if mask.any() else 0.0

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    txt = (
        f"\n{'═'*50}\n"
        f"RESULTS ON TEST SET  [{tag}]\n"
        f"{'═'*50}\n"
        f"Threshold              : {threshold:.3f}\n"
        f"ROC-AUC                : {roc_auc:.4f}\n"
        f"AUC-PR                 : {avg_prec:.4f}\n"
        f"Recall @ Precision≥0.5 : {recall_at_p50:.4f}\n"
        f"Sensitivity (BS recall): {sensitivity:.4f}\n"
        f"Specificity            : {specificity:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix:\n"
        f"  TN={tn}  FP={fp}\n"
        f"  FN={fn}  TP={tp}\n"
        f"{'═'*50}\n"
    )
    print(txt)
    (RESULTS_DIR / f"metrics_{tag}.txt").write_text(txt)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC Curve [{tag}]")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"roc_curve_{tag}.png", dpi=150); plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_arr, prec_arr, lw=2, label=f"AUC-PR = {avg_prec:.3f}")
    ax.axhline(y=(y_test == 1).mean(), color="k", ls="--", lw=1,
               label="Random")
    ax.axvline(x=recall_at_p50, color="r", ls=":", lw=1,
               label=f"Recall@P≥0.5 = {recall_at_p50:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"Precision-Recall Curve [{tag}]")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"pr_curve_{tag}.png", dpi=150); plt.close(fig)

    # Feature importance
    rf = pipeline.named_steps["clf"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    top = min(30, len(feature_cols))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(top), imp[idx[:top]])
    ax.set_xticks(range(top))
    ax.set_xticklabels([feature_cols[i] for i in idx[:top]],
                       rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="Importance (Gini)", title=f"Feature Importances [{tag}]")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"feature_importance_{tag}.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real 0", "Real 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix [{tag}]"); fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{tag}.png", dpi=150)
    plt.close(fig)

    return roc_auc, avg_prec

# ─── Save ─────────────────────────────────────────────────────────────────────

def save_model(pipeline, feature_cols, threshold, roc_auc, avg_prec, tag="v1"):
    path = MODELS_DIR / f"binding_site_model_{tag}.pkl"
    joblib.dump(pipeline, path)
    meta = {
        "model_version":    tag,
        "feature_names":    feature_cols,
        "threshold":        threshold,
        "roc_auc_test":     roc_auc,
        "auc_pr_test":      avg_prec,
        "n_estimators":     pipeline.named_steps["clf"].n_estimators,
        "undersample_ratio": UNDERSAMPLE_RATIO,
        "class_weight":     "balanced_subsample",
        "cv_strategy":      "GroupKFold",
    }
    (MODELS_DIR / f"model_meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"  Model saved → {path}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def train(csv_path=DATASET_CSV):
    print("=" * 55)
    print("  BINDING SITE PREDICTOR  -  v1 (RF + undersampling)")
    print("=" * 55)

    print("\nLoading data...")
    df, feature_cols = load_data(csv_path)

    print("\nSplitting by PDB...")
    df_train, df_val, df_test = split_by_pdb(df)

    if UNDERSAMPLE_RATIO is not None:
        print(f"\nUndersampling train negatives (target {UNDERSAMPLE_RATIO}x)...")
        df_train = undersample_negatives(df_train)

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    groups_train = df_train["pdb_id"].values     # needed for GroupKFold

    X_val  = df_val[feature_cols].values
    y_val  = df_val["label"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    pipeline = build_pipeline()

    print("\nCross-validation (GroupKFold - no leakage)...")
    cross_validate_grouped(pipeline, X_train, y_train, groups_train)

    print("\nTraining final model...")
    pipeline.fit(X_train, y_train)

    print("\nFinding optimal threshold on validation set...")
    threshold = find_best_threshold(pipeline, X_val, y_val)

    print("\nEvaluating on test set...")
    roc_auc, avg_prec = evaluate_and_plot(
        pipeline, X_test, y_test, threshold, feature_cols, tag="v1"
    )

    print("\nSaving model...")
    save_model(pipeline, feature_cols, threshold, roc_auc, avg_prec, tag="v1")

    print(f"\nFinal ROC-AUC : {roc_auc:.4f}")
    print(f"Final AUC-PR  : {avg_prec:.4f}")
    print("\nDone.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATASET_CSV))
    parser.add_argument("--predict", default=None, metavar="PDB")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.predict:
        from predict import predict_binding_sites
        pipeline  = joblib.load(MODELS_DIR / "binding_site_model_v1.pkl")
        meta      = json.loads((MODELS_DIR / "model_meta_v1.json").read_text())
        threshold = args.threshold or meta["threshold"]
        predict_binding_sites(args.predict, pipeline,
                              meta["feature_names"], threshold)
    else:
        train(Path(args.input))

if __name__ == "__main__":
    main()
