#################################################################################
########################### Summary data generated ##############################
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
#   python train.py --input otro.csv       # CSV of new dataset ffor validation
#   python train.py --predict prot.pdb     # predict BS for new .pdb structure

import argparse                        # command line args
import json                            # save in json files
import joblib               
import numpy as np                     # arrays
import pandas as pd                    # dataframes
import matplotlib                      # build plot
matplotlib.use("Agg")                  # backend 
import matplotlib.pyplot as plt

from pathlib import Path                                # use routes
from sklearn.ensemble import RandomForestClassifier     # random forest 
from sklearn.preprocessing import StandardScaler        # scale features
from sklearn.pipeline import Pipeline                   # scaler -> model
from sklearn.model_selection import (           
    GroupShuffleSplit,                 # split per pdb -> better than per residues (avoid leakage)
    GroupKFold,                   # cross validation
    cross_val_score                    # calculate metrics
)
from sklearn.metrics import (                          # calculate metrics:
    roc_auc_score, roc_curve,                          #    ROC-AUC
    precision_recall_curve, average_precision_score,   #    Precision-Recall
    confusion_matrix, classification_report,           #    Confusion maatrix
)                                                      #    F1 score

DATASET_CSV = Path("data/train_dataset.csv")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Columns of .cvs which are not features
NON_FEATURE_COLS = {"pdb_id", "chain", "resnum", "icode", "resname", "label", "ca_x", "ca_y", "ca_z"}


##################################################################################
############################### Load data ########################################
##################################################################################

def load_data(csv_path: Path):
    """
    Load data from train_dataset.csv to generate a DataFrame with all features
    """
    df = pd.read_csv(csv_path)

    missing = NON_FEATURE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns not found: {missing}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS] #detect features

    # Eliminate NaN and incomplete features
    before = len(df)
    df = df.dropna(subset=feature_cols + ["label"])
    if len(df) < before:
        print(f" Delete {before - len(df)} rows with NaN")

    print(f" Loaded residues: {len(df)}")
    print(f" Unique PDBs: {df['pdb_id'].nunique()}")
    print(f" Detected features: {len(feature_cols)}")
    print(f" Binding sites (1): {(df['label']==1).sum()}")
    print(f" No binding (0): {(df['label']==0).sum()}")
    print(f" Ratio neg/pos: {(df['label']==0).sum()/(df['label']==1).sum():.1f}x")

    # Añade esto en train.py justo después del merge, antes del split
    print("\nCorrelación features con label (top 15):")
    corrs = df[feature_cols].corrwith(df["label"]).abs().sort_values(ascending=False)
    print(corrs.head(15).to_string())

    print("\nMedia por clase de las top 5 features:")
    top5 = corrs.head(5).index.tolist()
    print(df.groupby("label")[top5].mean().to_string()) 

    return df, feature_cols  #return clean dataframe with features and list of features


###################################################################################
################### Split per PDB (avoid leakage) #################################
###################################################################################


def split_by_pdb(df: pd.DataFrame):
    """
    Split dataset: df_train, df_val, df_test 
    """
    groups = df["pdb_id"].values

    # Separate test -> 15 %
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15,
                            random_state=RANDOM_STATE)
    rest_idx, test_idx = next(gss.split(df, groups=groups))

    # Separate validation -> 15 %
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.176,
                             random_state=RANDOM_STATE)
    train_idx, val_idx = next(
        gss2.split(df.iloc[rest_idx], groups=groups[rest_idx])
    )
    train_idx = rest_idx[train_idx]
    val_idx   = rest_idx[val_idx]
    # Dataset train -> 70 %
    df_train = df.iloc[train_idx]
    df_val   = df.iloc[val_idx]
    df_test  = df.iloc[test_idx]

    print(f" Train: {len(df_train):>6} residues — {df_train['pdb_id'].nunique()} PDBs")
    print(f" Val: {len(df_val):>6} residues — {df_val['pdb_id'].nunique()} PDBs")
    print(f" Test: {len(df_test):>6} residues — {df_test['pdb_id'].nunique()} PDBs")

    return df_train, df_val, df_test #split dataframes of train, validation and test dataset


############################################################################
################## Pipeline: scaler + random forest ########################
############################################################################

def build_pipeline() -> Pipeline:
    """
    Build sklearn pipeline: scaler -> random forest 
    """
    #random forest configuration
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    rf),
    ])


###########################################################################
#################### Cross-validation in train ############################
###########################################################################

def cross_validate(pipeline: Pipeline,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   groups_train,
                   cv_folds: int = 5):
    """
    Stimation of: ROC-AUC and F1 with 5-fold CV of train dataset -> Yeild stimation (not final metrics)
    """
    print(f"\n  {cv_folds}-fold CV in train set...")
    #cross validation 5-fold
    cv = GroupKFold(n_splits=cv_folds) 
    #ROC-AUC
    cv_roc = cross_val_score(pipeline, X_train, y_train, groups=groups_train, scoring="roc_auc", n_jobs=-1)
    #F1
    cv_f1  = cross_val_score(pipeline, X_train, y_train, groups=groups_train, scoring="f1", n_jobs=-1)

    print(f"  ROC-AUC: {cv_roc.mean():.3f} ± {cv_roc.std():.3f}")
    print(f"  F1:      {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

#############################################################################
################# Optimum threshold for validation dataset###################
#############################################################################

def find_best_threshold(pipeline: Pipeline,
                        X_val: np.ndarray,
                        y_val: np.ndarray) -> float:
    """
    As random forest results return probabilities, we need a threshold to determinate 
    if the prediction is 1 (BS) or 0 (NBS). We search the optimum threslhold according to precision,
    recall and F1 in validation dataset.
    """
    val_probs = pipeline.predict_proba(X_val)[:, 1]    #probabilities od validation dataset
    prec, rec, thresholds = precision_recall_curve(y_val, val_probs)   #precision-recall curve
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)   
    best_threshold = float(thresholds[np.argmax(f1)])      #find best threshold
    print(f" Optimun threshold (val): {best_threshold:.3f}  "
          f"(F1 val = {f1.max():.3f})")
    return best_threshold


############################################################################
################################# Evaluation ###############################
############################################################################

def evaluate_and_plot(pipeline: Pipeline,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      threshold: float,
                      feature_cols: list[str]):
    """
    
    """
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    roc_auc  = roc_auc_score(y_test, test_probs)
    avg_prec = average_precision_score(y_test, test_probs)
    report   = classification_report(y_test, test_preds,
                                     target_names=["no binding", "binding site"])
    cm = confusion_matrix(y_test, test_preds)

    metrics_text = (
        f"\n{'═'*45}\n"
        f"RESULTS IN TEST SET\n"
        f"{'═'*45}\n"
        f"Threshold:        {threshold:.3f}\n"
        f"ROC-AUC:          {roc_auc:.4f}\n"
        f"AUC-PR:           {avg_prec:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix:\n"
        f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"  FN={cm[1,0]}  TP={cm[1,1]}\n"
        f"{'═'*45}\n"
    )
    print(metrics_text)
    (RESULTS_DIR / "metrics.txt").write_text(metrics_text)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # precision curve
    prec, rec, _ = precision_recall_curve(y_test, test_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"AUC-PR = {avg_prec:.3f}")
    ax.axhline(y=(y_test == 1).mean(), color="k", linestyle="--",
               lw=1, label="Random classifier")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    # Feature importances
    rf          = pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    top_n       = min(30, len(feature_cols))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(top_n), importances[sorted_idx[:top_n]])
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_cols[i] for i in sorted_idx[:top_n]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Importance (Gini)")
    ax.set_title("Feature importances — Random Forest")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
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
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print("Graphics in results/")
    return roc_auc, avg_prec


#######################################################################
###################### Save model #####################################
#######################################################################

def save_model(pipeline: Pipeline, feature_cols: list[str],
               threshold: float, roc_auc: float, avg_prec: float):
    joblib.dump(pipeline, MODELS_DIR / "binding_site_model.pkl")

    meta = {
        "feature_names": feature_cols,
        "threshold":     threshold,
        "roc_auc_test":  roc_auc,
        "auc_pr_test":   avg_prec,
        "n_estimators":  pipeline.named_steps["clf"].n_estimators,
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Modelo guardado en: {MODELS_DIR / 'binding_site_model.pkl'}")


#########################################################################
########################### train ####################################### 
#########################################################################

def train(csv_path: Path = DATASET_CSV):
    print("Loading data...")
    df, feature_cols = load_data(csv_path)

    print("\nSplit per PDB...")
    df_train, df_val, df_test = split_by_pdb(df)

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    X_val   = df_val[feature_cols].values
    y_val   = df_val["label"].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test["label"].values

    pipeline = build_pipeline()

    groups_train = df_train["pdb_id"].values

    print("\nCross-validation in train...")
    cross_validate(pipeline, X_train, y_train, groups_train)

    print("\nTrain final model...")
    pipeline.fit(X_train, y_train)

    print("\nSearch optimun threshold in validation set...")
    threshold = find_best_threshold(pipeline, X_val, y_val)

    print("\nEvaluating test...")
    roc_auc, avg_prec = evaluate_and_plot(
        pipeline, X_test, y_test, threshold, feature_cols
    )

    print("\nSaving modelo...")
    save_model(pipeline, feature_cols, threshold, roc_auc, avg_prec)

    print(f"\nROC-AUC final: {roc_auc:.4f}")
    print("Done. For predict a new PDB:")
    print("  python train.py --predict new_protein.pdb")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or use binding site predictor."
    )
    parser.add_argument("--input", default=str(DATASET_CSV),
                        help="CSV generated by feature_engineering.py")
    parser.add_argument("--predict", default=None, metavar="PDB",
                        help="PDB sobre el que predecir (omite entrenamiento)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Manual threshold (it use the optimum for the model by default)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.predict:
        # import the script predict.py 
        from predict import predict_binding_sites
        pipeline = joblib.load(MODELS_DIR / "binding_site_model.pkl")
        meta     = json.loads((MODELS_DIR / "model_meta.json").read_text())
        threshold = args.threshold or meta["threshold"]
        predict_binding_sites(args.predict, pipeline,
                              meta["feature_names"], threshold)
    else:
        train(Path(args.input))


if __name__ == "__main__":
    main()