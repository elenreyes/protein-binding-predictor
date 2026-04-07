#!/bin/bash
source venv/bin/activate

python download_biolip.py
python parse_structures.py
python build_dataset.py
python feature_engineering.py
python merge_dataset.py
python train.py

# Only run predict if arguments are provided
if [ "$#" -gt 0 ]; then
    python predict.py "$@"
fi
