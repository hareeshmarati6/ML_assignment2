# ML Assignment 2 – Classification on UCI Breast Cancer Wisconsin (Diagnostic)

## Problem statement
Build and compare multiple classification models on a real dataset. Provide an interactive Streamlit app with model selection, metrics, confusion matrix / classification report, and deploy it on Streamlit Community Cloud.

## Dataset description
We use the **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine Learning Repository. It contains **569 instances** and **30 real-valued features**, with no missing values; task is to classify tumors as **malignant** or **benign**. Features are computed from digitized FNA images of breast masses. [1](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)  
This dataset is also mirrored in scikit‑learn (`load_breast_cancer`), which confirms **569×30** shape and that 0/1 encode **malignant/benign**. [2](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

## Repository layout

```
app.py                  # Streamlit app for interactive model inference
train_models.py         # Script to train and save all models and metrics
utils.py                # Utility functions for data processing and plotting
requirements.txt        # Python dependencies
model/                  # Saved models, metrics, and artifacts
    decision_tree_classification_report.csv
    decision_tree_confusion_matrix.csv
    decision_tree.joblib
    feature_names.json
    knn_classification_report.csv
    knn_confusion_matrix.csv
    knn.joblib
    logistic_regression_classification_report.csv
    logistic_regression_confusion_matrix.csv
    logistic_regression.joblib
    metrics_comparison.csv
    naive_bayes_gaussian_classification_report.csv
    naive_bayes_gaussian_confusion_matrix.csv
    naive_bayes_gaussian.joblib
    random_forest_ensemble_classification_report.csv
    random_forest_ensemble_confusion_matrix.csv
    random_forest_ensemble.joblib
    scaler.joblib
    xgboost_ensemble_classification_report.csv
    xgboost_ensemble_confusion_matrix.csv
    xgboost_ensemble.joblib
```

## Setup instructions

1. **Clone the repository and install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Train models and generate artifacts:**
   ```sh
   python train_models.py
   ```
   This will create the `model/` directory with all required files.
3. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

## Usage
- Open the app in your browser (Streamlit will provide a local URL).
- Upload a test CSV file with the same feature columns as the training data (see template download in the app sidebar).
- Optionally, include a `target` (0/1) or `diagnosis` (M/B) column for metrics.
- Select a model from the sidebar to view predictions, metrics, confusion matrix, classification report, and ROC curve.

## Models trained
- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

All models are trained on the UCI Breast Cancer Wisconsin (Diagnostic) dataset with standardized features. Metrics and confusion matrices are saved for comparison.

## References
1. [UCI ML Repository: Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)
2. [scikit-learn breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)