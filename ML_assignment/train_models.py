# train_models.py
# Trains 6 classifiers on UCI Breast Cancer Wisconsin (Diagnostic), computes metrics,
# and saves models + preprocessing artifacts for Streamlit inference.

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_uci_breast_cancer():
    """
    Loads the scikit-learn copy of the UCI Breast Cancer Wisconsin (Diagnostic) dataset.
    Returns X (DataFrame), y (Series), target_names (list).
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Standardize column naming (safer for CSV I/O)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    X = df.drop(columns=["target"])
    y = df["target"]  # 0 = malignant, 1 = benign (per sklearn docs)
    target_names = list(data.target_names)
    return X, y, target_names

def build_models():
    """
    Returns a dict of name -> estimator (already wrapped in pipelines if needed).
    For binary classification (our case), XGBoost uses objective='binary:logistic'.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced_subsample"
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }
    return models

def compute_metrics(y_true, y_pred, y_proba, labels, pos_label=None):
    """
    Computes Accuracy, AUC, Precision, Recall, F1, MCC.
    Handles binary or multiclass:
    - For binary: pos_label controls positive class (default: 0 for this dataset = 'malignant').
    - For multiclass: macro-average one-vs-rest AUC.
    """
    metrics = {}
    average = "binary" if len(labels) == 2 else "macro"

    # Accuracy, Precision, Recall, F1
    if average == "binary":
        # Our dataset: target 0=malignant, 1=benign. Treat 'malignant' (0) as positive.
        if pos_label is None:
            pos_label = 0
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics["F1"] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

        # AUC
        if y_proba is not None:
            # Find column index for positive class
            classes_ = np.array(labels)
            if y_proba.ndim == 1:
                # If we ever provide only positive class probability
                y_pos = y_proba
            else:
                pos_idx = int(np.where(classes_ == pos_label)[0][0])
                y_pos = y_proba[:, pos_idx]
            metrics["AUC"] = roc_auc_score(y_true == pos_label, y_pos)
        else:
            metrics["AUC"] = np.nan
    else:
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
        if y_proba is not None:
            Y = label_binarize(y_true, classes=labels)
            metrics["AUC"] = roc_auc_score(Y, y_proba, average="macro", multi_class="ovr")
        else:
            metrics["AUC"] = np.nan

    return metrics

def main():
    X, y, target_names = load_uci_breast_cancer()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    models = build_models()

    comparison_rows = []
    # Save a StandardScaler for any inference that needs it (KNN/LogReg)
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # Persist feature names for schema validation in Streamlit
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    for name, est in models.items():
        # Fit
        est.fit(X_train, y_train)

        # Save model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(est, MODEL_DIR / f"{safe_name}.joblib")

        # Predict
        y_pred = est.predict(X_test)

        # Predicted probabilities for AUC/ROC (if available)
        y_proba = None
        if hasattr(est, "predict_proba"):
            y_proba = est.predict_proba(X_test)
        elif hasattr(est, "decision_function"):
            # decision_function -> convert to probabilities via logistic for 2-class
            # but we’ll skip here to keep metrics comparable; AUC needs proba/score.
            scores = est.decision_function(X_test)
            # shape -> (n_samples,) or (n_samples, n_classes)
            if scores.ndim == 1:
                # Convert to 2D [p_pos, p_neg] proxy using rank-based normalization
                s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                y_proba = np.vstack([1 - s, s]).T
            else:
                # Normalize per-row
                s = (scores - scores.min(axis=1, keepdims=True)) / (
                    scores.max(axis=1, keepdims=True) - scores.min(axis=1, keepdims=True) + 1e-9
                )
                y_proba = s

        labels = np.unique(y_train)
        m = compute_metrics(y_test, y_pred, y_proba, labels=labels, pos_label=0)

        comparison_rows.append({
            "ML Model Name": name,
            "Accuracy": m["Accuracy"],
            "AUC": m["AUC"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
            "MCC": m["MCC"]
        })

        # Also save per-model classification report
        report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        pd.DataFrame(report_dict).to_csv(MODEL_DIR / f"{safe_name}_classification_report.csv", index=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(MODEL_DIR / f"{safe_name}_confusion_matrix.csv")

    # Save comparison table
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.sort_values(by="AUC", ascending=False, inplace=True)
    df_comp.to_csv(MODEL_DIR / "metrics_comparison.csv", index=False)

    print("Training complete. Artifacts written to ./model")
    print(df_comp)

if __name__ == "__main__":
    main()