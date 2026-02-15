# app.py
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    load_feature_names, ensure_column_order, plot_confusion_matrix,
    plot_roc, make_test_template_csv, classification_report_df
)

st.set_page_config(page_title="BITS ML Assignment 2 – Classification Demo", layout="centered")

st.title("ML Assignment 2 – Classification Models (UCI Breast Cancer Diagnostic)")
st.caption("Upload **test CSV**, pick a model, and view metrics, confusion matrix, and ROC curve.")

MODEL_DIR = Path(__file__).parent / "model"

# --- Load saved artifacts ---
feature_names = load_feature_names(MODEL_DIR)
# Preload models for dropdown
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "XGBoost (Ensemble)": "xgboost_ensemble.joblib",
}
models = {}
for k, v in MODEL_FILES.items():
    models[k] = joblib.load(MODEL_DIR / v)

# Class labels: per sklearn dataset, 0=malignant, 1=benign
target_names = ["malignant", "benign"]
labels = np.array([0, 1])  # keep numeric form for metrics

# --- Sidebar ---
st.sidebar.header("Model & Data")
model_name = st.sidebar.selectbox("Choose model", list(MODEL_FILES.keys()), index=5)
st.sidebar.write("**Positive class** is treated as: `malignant (label=0)` for metrics/ROC.")

st.sidebar.download_button(
    label="Download test template CSV",
    data=make_test_template_csv(feature_names),
    file_name="test_template_breast_cancer.csv",
    mime="text/csv",
    help="A one-row CSV with the required feature columns."
)

uploaded = st.file_uploader("Upload test CSV (features only, columns must match training schema)", type=["csv"])

# Optional ground-truth column handling
st.markdown(
    """
    **Ground truth (optional, for metrics):**
    - If your CSV contains a `target` column with values 0/1 (0=malignant, 1=benign), it will be used.
    - Alternatively, a `diagnosis` column with values `M`/`B` will be mapped to 0/1 respectively.
    """
)

def extract_X_y(df: pd.DataFrame):
    y = None
    if "target" in df.columns:
        y = df["target"].astype(int).values
        df = df.drop(columns=["target"])
    elif "diagnosis" in df.columns:
        mapped = df["diagnosis"].map({"M": 0, "B": 1})
        if mapped.isna().any():
            st.warning("Found unexpected values in 'diagnosis' column. Expected M/B.")
        y = mapped.astype(int).values
        df = df.drop(columns=["diagnosis"])
    X = ensure_column_order(df, feature_names)
    return X, y

if uploaded is None:
    st.info("Upload a test CSV to begin. You can also test with the template after filling values.")
else:
    # Read and validate
    raw_df = pd.read_csv(uploaded)
    try:
        X, y_true = extract_X_y(raw_df.copy())
    except Exception as e:
        st.error(f"CSV schema error: {e}")
        st.stop()

    model = models[model_name]
    # Predict
    y_pred = model.predict(X)

    # Probabilities (for AUC/ROC)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            y_proba = np.vstack([1 - s, s]).T
        else:
            s = (scores - scores.min(axis=1, keepdims=True)) / (
                scores.max(axis=1, keepdims=True) - scores.min(axis=1, keepdims=True) + 1e-9
            )
            y_proba = s

    st.subheader("Predictions")
    st.write(f"Rows predicted: **{len(y_pred)}**")
    pred_df = pd.DataFrame({
        "predicted_label": y_pred,
        "predicted_class": [target_names[i] for i in y_pred]
    })
    st.dataframe(pred_df.head(20), use_container_width=True)

    if y_true is not None:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
        )

        # Metrics with positive class = 0 (malignant)
        pos_label = 0
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc_score = None
        if y_proba is not None:
            auc_score = roc_auc_score((y_true == pos_label), y_proba[:, np.where(labels == pos_label)[0][0]])

        st.subheader("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision (pos=malignant)", f"{precision:.3f}")
        col3.metric("Recall (pos=malignant)", f"{recall:.3f}")
        col1.metric("F1 (pos=malignant)", f"{f1:.3f}")
        col2.metric("MCC", f"{mcc:.3f}")
        col3.metric("AUC", f"{auc_score:.3f}" if auc_score is not None else "—")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_true, y_pred, labels=[0, 1], title="Confusion Matrix (0=malignant, 1=benign)")
        st.pyplot(fig_cm)

        # Classification report
        st.subheader("Classification Report")
        st.dataframe(classification_report_df(y_true, y_pred, target_names), use_container_width=True)

        # ROC
        if y_proba is not None:
            st.subheader("ROC Curve")
            st.caption("Positive class = malignant (label 0)")
            fig_roc = plot_roc(y_true, y_proba, labels=[0, 1], pos_label=0, title=f"ROC – {model_name}")
            st.pyplot(fig_roc)
    else:
        st.info("Ground truth not provided — metrics/CM/ROC will appear once your CSV includes `target` or `diagnosis`.")
