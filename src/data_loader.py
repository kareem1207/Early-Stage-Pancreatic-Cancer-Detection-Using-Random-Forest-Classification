"""
data_loader.py
================
Responsible for one thing: getting a raw, tabular urinary-biomarker dataset
onto disk as a CSV that the rest of the pipeline (preprocess.py) can read.

WHY a synthetic generator exists here
--------------------------------------
This project targets the real "Urinary biomarkers for pancreatic cancer"
dataset (Debernardi et al., 2020, PLOS Medicine). The environment this repo
was scaffolded in has no network access to Kaggle/UCI, so
`generate_synthetic_dataset()` fabricates data with the SAME columns, SAME
value ranges, and a SIMILAR class-imbalance pattern as the real dataset, so
the ML pipeline (preprocessing -> training -> evaluation) is fully runnable
end to end. See data/README.md for exactly how to swap in the real CSV.

Columns (matching the real dataset):
    sample_id                : unique sample identifier
    patient_cohort            : "Cohort1" or "Cohort2" (batch the sample was collected in)
    sample_origin              : lab/site code where the sample was processed
    age                        : patient age in years
    sex                        : "M" or "F"
    diagnosis                  : TARGET. 1=control, 2=benign, 3=PDAC (pancreatic cancer)
    stage                      : cancer stage (only meaningful when diagnosis == 3)
    benign_sample_diagnosis    : free-text benign condition (only when diagnosis == 2)
    plasma_CA19_9               : blood biomarker, units U/mL (many missing values in real data)
    creatinine                  : urine creatinine, mg/mL
    LYVE1                       : urinary biomarker protein, ng/mL
    REG1B                        : urinary biomarker protein, ng/mL
    TFF1                         : urinary biomarker protein, ng/mL
    REG1A                         : urinary biomarker protein, ng/mL (mostly missing in real data)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --- Constants describing the dataset schema -------------------------------
# If you swap in the real CSV and its column names differ, only these two
# constants need to change; the rest of the pipeline reads TARGET_COLUMN.
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "urinary_biomarkers.csv"
TARGET_COLUMN = "diagnosis"

# Class labels, kept human-readable for plots/reports.
DIAGNOSIS_LABELS = {1: "Control", 2: "Benign", 3: "PDAC"}


def generate_synthetic_dataset(n_samples: int = 590, random_state: int = 42) -> pd.DataFrame:
    """
    Create a synthetic urinary-biomarker dataset that mimics the real
    Debernardi et al. dataset's schema and class imbalance.

    n_samples=590 matches the real dataset's approximate size (183 controls,
    208 benign, 199 PDAC in the original paper).

    The biomarker distributions are deliberately overlapping but shifted so
    that PDAC samples tend to have higher CA19-9, LYVE1, REG1B and TFF1 than
    control samples -- this is the real, published clinical signal these
    biomarkers carry, and it's what lets Random Forest learn something
    genuine rather than fit pure noise.
    """
    rng = np.random.default_rng(random_state)

    # Roughly reproduce the real class proportions: control ~31%, benign ~35%, PDAC ~34%
    diagnosis = rng.choice([1, 2, 3], size=n_samples, p=[0.31, 0.35, 0.34])
    n = n_samples

    age = rng.normal(loc=60, scale=14, size=n).clip(20, 90).round(0)
    sex = rng.choice(["M", "F"], size=n)
    patient_cohort = rng.choice(["Cohort1", "Cohort2"], size=n, p=[0.6, 0.4])
    sample_origin = rng.choice(["BPTB", "ESP", "LIV", "UCL"], size=n)

    # Biomarker base distributions (log-normal, since concentrations are
    # strictly positive and right-skewed in real assay data).
    def lognormal(mean_log, sigma, shift=0.0):
        return np.exp(rng.normal(mean_log, sigma, size=n)) + shift

    creatinine = lognormal(mean_log=0.2, sigma=0.6)  # mg/mL, not strongly diagnosis-linked

    # Diagnosis-dependent shift: PDAC (3) > Benign (2) > Control (1)
    diag_shift = np.select(
        [diagnosis == 1, diagnosis == 2, diagnosis == 3],
        [0.0, 0.5, 1.3],
    )

    plasma_ca19_9 = lognormal(mean_log=2.0, sigma=1.1) + diag_shift * rng.uniform(15, 40, n)
    lyve1 = lognormal(mean_log=0.5, sigma=0.9) + diag_shift * rng.uniform(1.0, 3.0, n)
    reg1b = lognormal(mean_log=1.0, sigma=1.0) + diag_shift * rng.uniform(5, 15, n)
    tff1 = lognormal(mean_log=2.5, sigma=1.2) + diag_shift * rng.uniform(20, 60, n)
    reg1a = lognormal(mean_log=1.5, sigma=1.1) + diag_shift * rng.uniform(5, 20, n)

    # Real dataset has substantial missingness in CA19-9 and REG1A (not
    # collected for every cohort) -- reproduce that so preprocess.py has a
    # genuine missing-value-handling job to do.
    ca19_9_missing_mask = rng.random(n) < 0.13
    reg1a_missing_mask = rng.random(n) < 0.45
    plasma_ca19_9 = plasma_ca19_9.astype(object)
    reg1a = reg1a.astype(object)
    plasma_ca19_9[ca19_9_missing_mask] = np.nan
    reg1a[reg1a_missing_mask] = np.nan

    stage_values = rng.choice(["I", "II", "III", "IV"], size=n, p=[0.15, 0.45, 0.25, 0.15]).astype(object)
    stage = np.where(diagnosis == 3, stage_values, np.nan)

    benign_condition_pool = ["Chronic pancreatitis", "Gallstones", "Cyst", "Other"]
    benign_values = rng.choice(benign_condition_pool, size=n).astype(object)
    benign_sample_diagnosis = np.where(diagnosis == 2, benign_values, np.nan)

    df = pd.DataFrame(
        {
            "sample_id": [f"S{idx:04d}" for idx in range(1, n + 1)],
            "patient_cohort": patient_cohort,
            "sample_origin": sample_origin,
            "age": age,
            "sex": sex,
            "diagnosis": diagnosis,
            "stage": stage,
            "benign_sample_diagnosis": benign_sample_diagnosis,
            "plasma_CA19_9": plasma_ca19_9,
            "creatinine": creatinine,
            "LYVE1": lyve1,
            "REG1B": reg1b,
            "TFF1": tff1,
            "REG1A": reg1a,
        }
    )
    return df


def load_raw_data(path: Path = RAW_DATA_PATH, regenerate: bool = False) -> pd.DataFrame:
    """
    Load the raw dataset from disk, generating the synthetic placeholder
    first if it doesn't exist yet (or if regenerate=True).
    """
    if regenerate or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_dataset()
        df.to_csv(path, index=False)
        print(f"[data_loader] Generated synthetic dataset -> {path} ({len(df)} rows)")
    else:
        df = pd.read_csv(path)
        print(f"[data_loader] Loaded existing dataset <- {path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    data = load_raw_data(regenerate=True)
    print(data.head())
    print("\nClass balance (diagnosis):")
    print(data["diagnosis"].map(DIAGNOSIS_LABELS).value_counts())
