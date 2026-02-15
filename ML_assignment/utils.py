# utils.py
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize

def load_feature_names(model_dir: Path):
    with open(model_dir / "feature_names.json", "r") as f:
        return json.load(f)

def ensure_column_order(df: pd.DataFrame, expected_cols):
    # Accept extra columns (will drop). Require all expected columns to be present.
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in uploaded CSV: {missing}")
    return df[expected_cols].copy()

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_roc(y_true, y_proba, labels, pos_label=0, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(labels) == 2:
        # Binary ROC
        # y_proba: [:, idx_of_pos]
        classes_ = np.array(labels)
        pos_idx = int(np.where(classes_ == pos_label)[0][0])
        y_score = y_proba[:, pos_idx]
        fpr, tpr, _ = roc_curve(y_true == pos_label, y_score)
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        # Multiclass OvR macro-average
        Y = label_binarize(y_true, classes=labels)
        fprs, tprs = [], []
        for i, cls in enumerate(labels):
            fpr, tpr, _ = roc_curve(Y[:, i], y_proba[:, i])
            fprs.append(fpr); tprs.append(tpr)
        # Simple overlay
        for i, cls in enumerate(labels):
            ax.plot(fprs[i], tprs[i], lw=1, alpha=0.7, label=f"Class {cls}")
        try:
            auc_macro = roc_auc_score(Y, y_proba, average="macro", multi_class="ovr")
            ax.plot([0,1], [0,1], "k--", lw=0.8)
            ax.legend(title=f"Macro AUC={auc_macro:.3f}")
        except Exception:
            pass
    ax.plot([0,1], [0,1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def make_test_template_csv(expected_cols):
    # Create a 1-row CSV template for users
    df = pd.DataFrame([np.zeros(len(expected_cols))], columns=expected_cols)
    return df.to_csv(index=False).encode("utf-8")

def classification_report_df(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    return pd.DataFrame(report).round(3)