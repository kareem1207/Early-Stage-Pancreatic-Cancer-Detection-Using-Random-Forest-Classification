"""
train.py
=========
Trains a Random Forest classifier to predict pancreatic cancer status
(Control / Benign / PDAC) from urinary biomarker features, with:

  - class-imbalance handling via SMOTE (Synthetic Minority Oversampling),
    applied INSIDE the cross-validation loop so no synthetic samples ever
    leak into a validation fold or the held-out test set
  - hyperparameter tuning via GridSearchCV with stratified 5-fold CV,
    optimizing macro-F1 (treats all 3 classes as equally important,
    appropriate for a diagnostic task where missing the minority PDAC
    class is the costly error)

Why Random Forest (and only Random Forest)?
  - Handles the mix of biomarker scales/distributions without needing
    perfectly-normal features.
  - Gives us feature importances for free -- directly useful for the
    project's key insight (which biomarkers matter most).
  - Robust to the moderate class imbalance and modest sample size here,
    especially combined with SMOTE + class_weight.
  - Easy to explain in a viva: it's just many decision trees voting.

Why SMOTE *and* class_weight='balanced' together?
  SMOTE rebalances the training data itself (synthesizes new minority-class
  samples), while class_weight='balanced' additionally penalizes
  minority-class mistakes more during tree splitting. Using both is
  belt-and-suspenders here since our classes are only mildly imbalanced
  (~30/35/35%) -- either alone would likely suffice, but combining them
  costs nothing and makes the imbalance-handling story concrete for
  the report/viva.
"""

from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from preprocess import load_and_split, save_processed_splits

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42

# Hyperparameter grid: kept deliberately small so GridSearchCV runs in
# seconds, but still covers the knobs that matter most for Random Forest:
#   n_estimators     -> how many trees (more = more stable, diminishing returns)
#   max_depth         -> how deep each tree can grow (controls overfitting)
#   min_samples_leaf  -> smallest allowed leaf size (higher = smoother, less overfit)
#   max_features       -> how many features each split considers (controls tree diversity)
PARAM_GRID = {
    "rf__n_estimators": [100, 200, 400],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2"],
}


def build_pipeline() -> ImbPipeline:
    """
    SMOTE + RandomForestClassifier chained in an imblearn Pipeline.
    Using imblearn's Pipeline (not sklearn's) is essential: it re-fits SMOTE
    on only the training portion of each CV fold, so validation folds
    always contain real (never synthetic) samples.
    """
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "rf",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(X_train, y_train):
    """Run GridSearchCV over PARAM_GRID with stratified 5-fold CV, optimizing macro-F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


if __name__ == "__main__":
    print("[train] Loading and preprocessing data...")
    split = load_and_split()
    save_processed_splits(split)

    print("[train] Running GridSearchCV (SMOTE + RandomForest, 5-fold CV, scoring=f1_macro)...")
    search = train_model(split["X_train"], split["y_train"])

    print(f"\n[train] Best CV macro-F1: {search.best_score_:.4f}")
    print(f"[train] Best hyperparameters: {search.best_params_}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "random_forest_model.pkl"
    joblib.dump(search.best_estimator_, model_path)
    print(f"[train] Saved best model -> {model_path}")

    # Also persist the exact feature names/order and test split so
    # evaluate.py doesn't need to re-run preprocessing (and therefore
    # can't accidentally use a different train/test split).
    joblib.dump(
        {
            "X_test": split["X_test"],
            "y_test": split["y_test"],
            "feature_names": split["feature_names"],
            "best_params": search.best_params_,
            "cv_best_score": search.best_score_,
        },
        MODELS_DIR / "test_split_and_meta.pkl",
    )
    print(f"[train] Saved test split + metadata -> {MODELS_DIR / 'test_split_and_meta.pkl'}")
