"""
predict.py
===========
Loads the saved preprocessor + trained Random Forest model and runs
inference on new, unlabeled samples -- the "use the model" step that would
sit behind a real clinical screening tool (or, more realistically for this
project, a command-line demo you can run in a viva).

Usage:
    python predict.py                     # runs the built-in example patient
    python predict.py --csv new_samples.csv   # predicts for every row in a CSV

Input CSV/record must have the same raw columns preprocess.py expects:
    age, sex, patient_cohort, sample_origin,
    plasma_CA19_9, creatinine, LYVE1, REG1B, TFF1, REG1A
(diagnosis/stage/benign_sample_diagnosis are NOT needed -- those are only
ever known after a diagnosis, which is exactly what we're predicting.)
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data_loader import DIAGNOSIS_LABELS
from preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# 0-indexed (matches the diagnosis-1 shift done in preprocess.py)
CLASS_NAMES = [DIAGNOSIS_LABELS[k] for k in sorted(DIAGNOSIS_LABELS)]

# A single made-up example patient, useful for a quick sanity check /
# live demo without needing an external file.
EXAMPLE_PATIENT = pd.DataFrame(
    [
        {
            "age": 68,
            "sex": "F",
            "patient_cohort": "Cohort1",
            "sample_origin": "BPTB",
            "plasma_CA19_9": 145.0,
            "creatinine": 1.2,
            "LYVE1": 6.8,
            "REG1B": 38.5,
            "TFF1": 120.0,
            "REG1A": 22.0,
        }
    ]
)


def load_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    return preprocessor, model


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Predict diagnosis class + class probabilities for each row of df."""
    preprocessor, model = load_artifacts()

    missing_cols = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {sorted(missing_cols)}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_transformed = preprocessor.transform(X)

    predicted_idx = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)

    results = df.copy()
    results["predicted_diagnosis"] = [CLASS_NAMES[i] for i in predicted_idx]
    for i, class_name in enumerate(CLASS_NAMES):
        results[f"probability_{class_name}"] = probabilities[:, i]
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict pancreatic cancer status from urinary biomarkers.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a CSV of new samples to predict.")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        print("[predict] No --csv given, running the built-in example patient.\n")
        df = EXAMPLE_PATIENT

    results = predict(df)
    print(results[["predicted_diagnosis"] + [c for c in results.columns if c.startswith("probability_")]])


if __name__ == "__main__":
    main()
