"""
evaluate.py
============
Loads the trained model + held-out test split (both saved by train.py) and
produces:

  - console metrics: accuracy, precision/recall/F1 (per-class + macro avg),
    multi-class ROC-AUC (one-vs-rest, macro-averaged)
  - reports/confusion_matrix.png
  - reports/roc_curve.png            (one curve per class, one-vs-rest)
  - reports/feature_importance.png   -- the key insight: which biomarkers
    drive predictions
  - reports/metrics.json             -- machine-readable metrics for the report

We only ever touch the TEST split here (never re-fit anything on it), so
these numbers reflect genuine generalization performance, not training fit.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from data_loader import DIAGNOSIS_LABELS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# y was shifted from {1,2,3} to {0,1,2} in preprocess.py; keep a 0-indexed
# label list matching that shift for readable plots.
CLASS_NAMES = [DIAGNOSIS_LABELS[k] for k in sorted(DIAGNOSIS_LABELS)]


def load_model_and_test_data():
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    meta = joblib.load(MODELS_DIR / "test_split_and_meta.pkl")
    return model, meta


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    # Multi-class ROC-AUC needs one-hot ("binarized") true labels.
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

    metrics = {
        "accuracy": accuracy,
        "roc_auc_macro_ovr": roc_auc_macro,
        "classification_report": report,
    }
    return metrics, y_pred, y_proba, y_test_bin


def plot_confusion_matrix(y_test, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix — Random Forest (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved confusion matrix -> {out_path}")


def plot_roc_curves(y_test_bin, y_proba, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, class_name in enumerate(CLASS_NAMES):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            y_proba[:, i],
            name=f"{class_name} vs rest",
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves (one-vs-rest) — Random Forest (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved ROC curves -> {out_path}")


def plot_feature_importance(model, feature_names, out_path: Path, top_n: int = 15) -> None:
    """
    Extract feature_importances_ from the fitted RandomForestClassifier
    (the "rf" step of the imblearn Pipeline) and plot the top-N features.
    This is the project's key clinical insight: which biomarkers the model
    actually relies on.
    """
    rf = model.named_steps["rf"]
    importances = rf.feature_importances_

    # Clean up sklearn's "num__"/"cat__" ColumnTransformer prefixes for readability.
    clean_names = [name.split("__", 1)[-1] for name in feature_names]

    order = np.argsort(importances)[::-1][:top_n]
    sorted_names = [clean_names[i] for i in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sorted_importances, y=sorted_names, ax=ax, color="steelblue")
    ax.set_xlabel("Feature importance (Gini-based, mean decrease in impurity)")
    ax.set_title("Random Forest Feature Importance — Which Biomarkers Matter Most")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved feature importance plot -> {out_path}")

    return list(zip(sorted_names, sorted_importances.tolist()))


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model, meta = load_model_and_test_data()
    X_test, y_test = meta["X_test"], meta["y_test"]

    metrics, y_pred, y_proba, y_test_bin = compute_metrics(model, X_test, y_test)

    print(f"\n[evaluate] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[evaluate] Macro ROC-AUC (OvR): {metrics['roc_auc_macro_ovr']:.4f}")
    print("\n[evaluate] Classification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(y_test, y_pred, REPORTS_DIR / "confusion_matrix.png")
    plot_roc_curves(y_test_bin, y_proba, REPORTS_DIR / "roc_curve.png")
    top_features = plot_feature_importance(model, meta["feature_names"], REPORTS_DIR / "feature_importance.png")

    metrics["top_features"] = top_features
    metrics["best_hyperparameters"] = meta["best_params"]
    metrics["cv_best_macro_f1"] = meta["cv_best_score"]

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[evaluate] Saved metrics.json -> {REPORTS_DIR / 'metrics.json'}")

    print("\n[evaluate] Top 5 most important biomarkers:")
    for name, importance in top_features[:5]:
        print(f"  {name}: {importance:.4f}")


if __name__ == "__main__":
    main()
