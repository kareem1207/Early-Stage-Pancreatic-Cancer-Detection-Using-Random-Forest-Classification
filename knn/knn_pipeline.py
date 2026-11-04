"""
knn_pipeline.py
================
K-Nearest Neighbors (KNN) variant of the pancreatic-cancer pipeline. Reuses the
exact same data, cleaning, leakage removal, scaling and stratified 80/20 split
as the Random Forest version (src/preprocess.py), so the two are comparable.

Why KNN needs care:
  * KNN is distance-based, so feature scaling is essential (the shared
    preprocessor already standardizes numeric features).
  * KNN has no built-in feature importance, so permutation importance on the
    held-out test set is used instead.
  * SMOTE is applied inside each CV fold (imblearn Pipeline), as before.

Run from the repo root:   python knn/knn_pipeline.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from data_loader import DIAGNOSIS_LABELS  # noqa: E402
from preprocess import load_and_split  # noqa: E402

RANDOM_STATE = 42
OUT = ROOT / "reports"
CLASS_NAMES = [DIAGNOSIS_LABELS[k] for k in sorted(DIAGNOSIS_LABELS)]

# Small grid: neighbourhood size, vote weighting, and distance metric (p=1 Manhattan, p=2 Euclidean).
PARAM_GRID = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 41],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}


def build_pipeline() -> ImbPipeline:
    return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("knn", KNeighborsClassifier())])


def train_model(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(build_pipeline(), PARAM_GRID, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    return search.fit(X_train, y_train)


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
    return {"accuracy": accuracy_score(y_test, y_pred), "roc_auc_macro_ovr": auc,
            "classification_report": report}, y_pred, y_proba, y_bin


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print("[knn] Loading and preprocessing data (shared with the Random Forest pipeline)...")
    split = load_and_split()
    print("[knn] Running GridSearchCV (SMOTE + KNN, 5-fold CV, macro-F1)...")
    search = train_model(split["X_train"], split["y_train"])
    print(f"\n[knn] Best CV macro-F1: {search.best_score_:.4f}")
    print(f"[knn] Best hyperparameters: {search.best_params_}")
    model = search.best_estimator_
    joblib.dump(model, ROOT / "models" / "knn_model.pkl")

    X_test, y_test = split["X_test"], split["y_test"]
    metrics, y_pred, y_proba, y_bin = compute_metrics(model, X_test, y_test)
    print(f"\n[knn] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[knn] Macro ROC-AUC (OvR): {metrics['roc_auc_macro_ovr']:.4f}")
    print("\n[knn] Classification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix — KNN (test set)"); fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, name in enumerate(CLASS_NAMES):
        RocCurveDisplay.from_predictions(y_bin[:, i], y_proba[:, i], name=f"{name} vs rest", ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance"); ax.set_title("ROC Curves (one-vs-rest) — KNN")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=150); plt.close(fig)

    # KNN has no feature_importances_, so use permutation importance (drop in macro-F1 when a feature is shuffled).
    perm = permutation_importance(model, X_test, y_test, scoring="f1_macro", n_repeats=30, random_state=RANDOM_STATE)
    names = [n.split("__", 1)[-1] for n in split["feature_names"]]
    order = np.argsort(perm.importances_mean)[::-1][:10]
    top = [(names[i], float(perm.importances_mean[i])) for i in order]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=[t[1] for t in top], y=[t[0] for t in top], ax=ax, color="#2F5496")
    ax.set_xlabel("Permutation importance (mean drop in macro-F1)"); ax.set_title("KNN Permutation Feature Importance")
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=150); plt.close(fig)

    metrics.update(top_features=top, best_hyperparameters=search.best_params_, cv_best_macro_f1=search.best_score_,
                   confusion_matrix=confusion_matrix(y_test, y_pred).tolist())
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print("\n[knn] Top 5 most important biomarkers (permutation):")
    for n, v in top[:5]:
        print(f"  {n}: {v:.4f}")
