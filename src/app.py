"""
app.py
=======
A small Streamlit UI on top of predict.py, so a user can enter a patient's
biomarker values by hand and see the Random Forest's prediction instead of
editing a CSV or calling predict.py from the command line.

Run with:
    streamlit run app.py        (from inside src/)

This is a thin presentation layer only -- all the actual ML logic (loading
the preprocessor/model, transforming inputs, predicting) lives in
predict.py and is reused unchanged here, so the UI can never drift from the
CLI's behavior.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from predict import CLASS_NAMES, load_artifacts, predict
from preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

st.set_page_config(page_title="Pancreatic Cancer Risk Classifier", page_icon="🩺", layout="centered")

st.title("🩺 Pancreatic Cancer Screening — Random Forest")
st.caption(
    "Enter a patient's urinary biomarker values to get a Control / Benign / PDAC "
    "prediction from the trained Random Forest model. This is a course-project "
    "demo, not a diagnostic tool."
)

# Fail loudly (and early) with a helpful message rather than a stack trace if
# the model hasn't been trained yet.
missing_artifacts = [
    f for f in ["preprocessor.pkl", "random_forest_model.pkl"] if not (MODELS_DIR / f).exists()
]
if missing_artifacts:
    st.error(
        f"Missing model artifact(s): {', '.join(missing_artifacts)}. "
        "Run `python train.py` from the src/ directory first to train and save the model."
    )
    st.stop()


@st.cache_resource
def get_artifacts():
    return load_artifacts()


# Warm the cache once up front so the form below doesn't re-load the model
# on every rerun.
get_artifacts()

st.subheader("Patient details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=60, step=1)
    sex = st.selectbox("Sex", options=["F", "M"])
    patient_cohort = st.selectbox("Patient cohort", options=["Cohort1", "Cohort2"])
    sample_origin = st.selectbox("Sample origin", options=["BPTB", "ESP", "LIV", "UCL"])
with col2:
    plasma_ca19_9 = st.number_input("Plasma CA19-9 (U/mL)", min_value=0.0, value=50.0, step=1.0)
    creatinine = st.number_input("Creatinine (mg/mL)", min_value=0.0, value=1.0, step=0.1)
    lyve1 = st.number_input("LYVE1 (ng/mL)", min_value=0.0, value=2.0, step=0.1)

col3, col4 = st.columns(2)
with col3:
    reg1b = st.number_input("REG1B (ng/mL)", min_value=0.0, value=10.0, step=0.5)
    tff1 = st.number_input("TFF1 (ng/mL)", min_value=0.0, value=30.0, step=1.0)
with col4:
    reg1a = st.number_input("REG1A (ng/mL)", min_value=0.0, value=8.0, step=0.5)

if st.button("Predict", type="primary"):
    patient = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "patient_cohort": patient_cohort,
                "sample_origin": sample_origin,
                "plasma_CA19_9": plasma_ca19_9,
                "creatinine": creatinine,
                "LYVE1": lyve1,
                "REG1B": reg1b,
                "TFF1": tff1,
                "REG1A": reg1a,
            }
        ]
    )

    result = predict(patient).iloc[0]
    prob_cols = [f"probability_{c}" for c in CLASS_NAMES]
    probs = {c: float(result[f"probability_{c}"]) for c in CLASS_NAMES}

    st.subheader("Prediction")
    predicted = result["predicted_diagnosis"]
    if predicted == "PDAC":
        st.error(f"Predicted class: **{predicted}**  (confidence: {probs[predicted]:.1%})")
    elif predicted == "Benign":
        st.warning(f"Predicted class: **{predicted}**  (confidence: {probs[predicted]:.1%})")
    else:
        st.success(f"Predicted class: **{predicted}**  (confidence: {probs[predicted]:.1%})")

    st.bar_chart(pd.Series(probs, name="Probability"))

    with st.expander("Raw probabilities"):
        st.table(pd.DataFrame([probs]).T.rename(columns={0: "Probability"}))

st.divider()
st.subheader("Model reference")
st.caption(
    "Trained with Random Forest + SMOTE + GridSearchCV. "
    "See reports/report.md for full methodology, results, and limitations."
)

feature_importance_path = REPORTS_DIR / "feature_importance.png"
if feature_importance_path.exists():
    st.image(str(feature_importance_path), caption="Which biomarkers the model relies on most")

st.caption(
    "⚠️ This app runs on a synthetic dataset generated to match the schema of the real "
    "Debernardi et al. urinary biomarkers dataset (no network access to Kaggle/UCI in this "
    "environment) — see data/README.md. Predictions are illustrative of the pipeline, not "
    "clinically meaningful until the real dataset is substituted."
)
