"""
preprocess.py
==============
Cleans the raw dataset and turns it into model-ready train/test splits.

Key decisions (explained here so they're easy to defend in a viva):

1. DROP `sample_id`
   Pure identifier, carries no signal, would just let the model memorize
   rows.

2. DROP `stage` and `benign_sample_diagnosis`
   These are TARGET LEAKAGE. `stage` is only ever filled in when
   diagnosis == PDAC, and `benign_sample_diagnosis` only when
   diagnosis == Benign. A model could "predict" the class just by checking
   which of these two columns is non-null -- that's not biomarker-based
   prediction, it's cheating. In a real clinical setting these fields
   wouldn't be known before diagnosis anyway, so keeping them would make
   the whole exercise meaningless.

3. IMPUTE missing biomarker values with the MEDIAN (per training fold only)
   `plasma_CA19_9` and `REG1A` have real missingness (not every cohort
   measured them). Median imputation is robust to the right-skewed,
   outlier-heavy distributions typical of biomarker concentrations, and is
   computed on the training split only to avoid leaking test-set
   statistics into training.

4. ONE-HOT ENCODE categoricals (`sex`, `patient_cohort`, `sample_origin`)
   Random Forest can technically split on raw label-encoded categories, but
   one-hot keeps the "which category" decision explicit and avoids
   implying a false ordering between categories.

5. SCALE numeric features with StandardScaler
   Random Forest itself doesn't need scaling (it splits on raw thresholds),
   but we scale anyway so the same preprocessed features could be reused
   with a different, scale-sensitive model later, and so a human/plot
   comparing feature magnitudes isn't misled by raw unit differences.

6. STRATIFIED train/test split (80/20)
   Stratifying on `diagnosis` keeps the same class proportions in both
   splits -- important here because the classes are already imbalanced.

Outputs:
    data/processed/train.csv, data/processed/test.csv
        Human-readable, cleaned (imputed + encoded, pre-scaling) splits --
        useful for a quick sanity look in a spreadsheet.
    models/preprocessor.pkl
        The fitted ColumnTransformer (imputers + encoder + scaler), reused
        by train.py and predict.py so new data is transformed identically.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import RAW_DATA_PATH, TARGET_COLUMN, load_raw_data

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

LEAKY_COLUMNS = ["stage", "benign_sample_diagnosis"]
ID_COLUMNS = ["sample_id"]
CATEGORICAL_FEATURES = ["sex", "patient_cohort", "sample_origin"]
NUMERIC_FEATURES = ["age", "plasma_CA19_9", "creatinine", "LYVE1", "REG1B", "TFF1", "REG1A"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) sklearn ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def load_and_split(df: pd.DataFrame | None = None):
    """
    Full pipeline: load raw data (if not passed in), drop leaky/id columns,
    split into stratified train/test, fit the preprocessor on train only,
    and return transformed arrays plus the fitted preprocessor.
    """
    if df is None:
        df = load_raw_data(RAW_DATA_PATH)

    df = df.drop(columns=LEAKY_COLUMNS + ID_COLUMNS)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    # Diagnosis is 1/2/3 in the raw data; shift to 0/1/2 so it's a clean
    # class index for scikit-learn / plotting.
    y = df[TARGET_COLUMN] - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return {
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
    }


def save_processed_splits(split: dict) -> None:
    """Save human-readable (pre-scaling) train/test CSVs and the fitted preprocessor."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_readable = split["X_train_raw"].copy()
    train_readable[TARGET_COLUMN] = split["y_train"].values
    test_readable = split["X_test_raw"].copy()
    test_readable[TARGET_COLUMN] = split["y_test"].values

    train_readable.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_readable.to_csv(PROCESSED_DIR / "test.csv", index=False)
    joblib.dump(split["preprocessor"], MODELS_DIR / "preprocessor.pkl")

    print(f"[preprocess] Saved {len(train_readable)} train / {len(test_readable)} test rows -> {PROCESSED_DIR}")
    print(f"[preprocess] Saved fitted preprocessor -> {MODELS_DIR / 'preprocessor.pkl'}")


if __name__ == "__main__":
    split = load_and_split()
    save_processed_splits(split)
    print(f"\n[preprocess] Transformed feature count: {len(split['feature_names'])}")
    print(f"[preprocess] Features: {list(split['feature_names'])}")
    print(f"\n[preprocess] Train class balance:\n{split['y_train'].value_counts().sort_index()}")
    print(f"\n[preprocess] Test class balance:\n{split['y_test'].value_counts().sort_index()}")
