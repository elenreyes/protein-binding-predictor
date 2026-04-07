################################################################################
################## Merge info dataset.csv + features.csv #######################

#Generate .csv file for train the model

import pandas as pd
from pathlib import Path

FEATURES = Path("data/features.csv")
LABELS   = Path("data/labels.csv")
OUTPUT   = Path("data/train_dataset.csv")

features = pd.read_csv(FEATURES)
labels   = pd.read_csv(LABELS)

df = features.merge(
    labels,
    on=["pdb_id", "chain", "resnum", "icode"],
    how="inner"
)

df.to_csv(OUTPUT, index=False)

print(f"Dataset final guardado en: {OUTPUT}")
print(f"Filas: {len(df)}")
print(f"Columnas: {len(df.columns)}")