# Early-Stage Pancreatic Cancer Detection Using Random Forest Classification on Clinical Biomarker Data

An end-to-end machine learning pipeline that classifies pancreatic cancer
status (**Control / Benign / PDAC**) from urinary biomarker data (CA19-9,
creatinine, LYVE1, REG1B, TFF1, age, sex, sample origin) using a **Random
Forest classifier** — the only data mining technique used in this project,
as required by the assignment (no ensembling with other model families, no
deep learning).

> **⚠️ Dataset assumption:** this repo targets the real
> ["Urinary biomarkers for pancreatic cancer"](https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer)
> dataset (Debernardi et al., 2020). Because this environment has no
> network access to download it, the pipeline currently runs on a
> **synthetic dataset generated to match its schema and class balance**.
> See [`data/README.md`](data/README.md) for exactly what to download and
> where to drop it in — the pipeline itself does not need to change.

## Project structure

```
.
├── data/
│   ├── raw/urinary_biomarkers.csv     # synthetic placeholder (see data/README.md)
│   ├── processed/train.csv, test.csv  # cleaned, split output of preprocess.py
│   └── README.md                      # dataset source, license, assumptions
├── notebooks/
│   └── eda.ipynb                      # exploratory data analysis
├── src/
│   ├── data_loader.py                 # loads / generates the raw dataset
│   ├── preprocess.py                  # cleaning, encoding, scaling, train/test split
│   ├── train.py                       # SMOTE + GridSearchCV Random Forest training
│   ├── evaluate.py                    # metrics, confusion matrix, ROC, feature importance
│   ├── predict.py                     # inference on new samples (CLI)
│   └── app.py                         # Streamlit UI for interactive predictions
├── models/
│   ├── preprocessor.pkl               # fitted preprocessing pipeline
│   └── random_forest_model.pkl        # trained, tuned Random Forest
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png         # key insight: which biomarkers matter most
│   ├── metrics.json
│   └── report.md                      # full write-up: methodology, results, limitations
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

Run each stage in order (each script can also be run standalone — later
stages will regenerate what they need from earlier ones):

```bash
cd src

# 1. Generate/load the raw dataset
python data_loader.py

# 2. Clean, encode, scale, split into train/test
python preprocess.py

# 3. Train the Random Forest (SMOTE + GridSearchCV hyperparameter tuning)
python train.py

# 4. Evaluate on the held-out test set (metrics + plots -> ../reports/)
python evaluate.py

# 5. Predict on a new sample (built-in example, or --csv path/to/file.csv)
python predict.py
```

Or explore interactively: `jupyter notebook ../notebooks/eda.ipynb`.

## Interactive UI

Once `train.py` has been run at least once (so `models/*.pkl` exist), launch
the Streamlit app to enter a patient's biomarkers by hand and get a live
prediction, instead of editing a CSV:

```bash
cd src
streamlit run app.py
```

This opens a form (age, sex, cohort, origin, CA19-9, creatinine, LYVE1,
REG1B, TFF1, REG1A) and shows the predicted class, per-class probabilities,
and the feature-importance plot. It's a thin wrapper around `predict.py` —
no separate model logic, so the UI can't drift from the CLI's behavior.

## Hindsight memory agent (surveillance layer)

The Random Forest scores each sample in isolation. `src/agent.py` wraps it in an
agent backed by [Hindsight](https://github.com/vectorize-io/hindsight) memory
so it can track **each patient across visits** and learn from **clinician
feedback**:

- **retain**: every visit (biomarkers, RF call) and every clinician verdict, tagged `patient:<id>`
- **recall**: prior visits and confounder notes before each new prediction
- **trend flags**: marker rises vs. the last visit surface risk before the classifier crosses PDAC
- **confounder memory**: a clinician-overruled false alarm is remembered, so the repeat is routed to review instead of re-escalated

```bash
pip install -r requirements.txt
cd src
python demo.py                       # offline JSON fallback memory
HINDSIGHT_BASE_URL=https://... HINDSIGHT_API_KEY=... python demo.py   # real Hindsight
```

`demo.py` prints stateless RF vs. memory-backed agent on two synthetic patients.
Code: `src/memory.py` (Hindsight + fallback), `src/agent.py`, `src/demo.py`.
Content drafts and a submission checklist are in [`submission/`](submission/).

> Research prototype on synthetic data, not a diagnostic device. The Hindsight
> backend is written against the `hindsight-client` SDK but has not yet been
> exercised against a live server; the offline fallback is what `demo.py` ran on.

## Results summary

On the current (synthetic) dataset, the tuned Random Forest achieves:

| Metric | Value |
|---|---|
| Accuracy | 0.915 |
| Macro ROC-AUC (one-vs-rest) | 0.983 |
| Macro F1 | 0.917 |

The most predictive biomarkers, by feature importance, are **REG1B**,
**TFF1**, and **plasma CA19-9** — see
[`reports/feature_importance.png`](reports/feature_importance.png).

Full methodology, per-class metrics, limitations, and future work are in
[`reports/report.md`](reports/report.md) — written to be adaptable
directly for a project submission or viva presentation.

## Key design decisions (for viva prep)

- **Target leakage removed:** `stage` and `benign_sample_diagnosis` are
  dropped before modeling — both are only populated *after* a diagnosis is
  known, so keeping them would let the model "cheat."
- **Class imbalance:** handled with SMOTE (applied inside cross-validation
  folds via an `imblearn` pipeline, so no synthetic data leaks into
  validation/test) plus `class_weight='balanced'` on the forest itself.
- **Hyperparameter tuning:** `GridSearchCV` with stratified 5-fold CV,
  optimizing macro-F1 (all three classes weighted equally, since missing a
  minority PDAC case is the costly error).
- **Evaluation integrity:** the test set is never touched during fitting or
  tuning — all reported metrics reflect genuine held-out performance.

## License / data source

See [`data/README.md`](data/README.md) for the target dataset's citation
and license (CC BY 4.0, per Debernardi et al., 2020).
