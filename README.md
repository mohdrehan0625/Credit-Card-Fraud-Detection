# 💳 Credit Card Fraud Detection System

An end-to-end machine learning system designed to detect fraudulent credit card transactions from highly skewed tabular data using unsupervised anomaly detection, supervised gradient boosting, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Credit card fraud represents significant annual losses across global banking networks. Detecting fraudulent events is uniquely challenging because legitimate transactions vastly outnumber fraudulent ones. 

This project explores the Kaggle Credit Card Fraud dataset (284,807 transactions with only 492 fraud cases, representing a severe 0.17% class imbalance). The repository covers the full pipeline:
1. **Data Preprocessing & Balancing**: Outlier-resilient scaling (`RobustScaler`) and synthetic minority sampling (`SMOTE`).
2. **Unsupervised Anomaly Detection**: Evaluation of baseline models (`Isolation Forest` and `Local Outlier Factor`).
3. **Supervised Classification**: Training an optimized `XGBoost` classifier evaluated on Area Under the ROC Curve (AUC-ROC) and confusion matrices.
4. **Web Dashboard**: An interactive application built with `Streamlit` for real-time transaction scoring.

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.10+
* **Data Manipulation & Analysis**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn
* **Data Visualization**: Matplotlib, Seaborn
* **Model Deployment & UI**: Streamlit

---

## 📊 Dataset Description

The dataset consists of credit card transactions made by European cardholders in September 2013:
* **Total Transactions**: 284,807
* **Fraudulent Transactions**: 492 (0.172%)
* **Legitimate Transactions**: 284,315 (99.828%)
* **Features**:
  * `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
  * `Amount`: Transaction amount.
  * `V1 - V28`: 28 numerical principal components obtained via Principal Component Analysis (PCA) for confidentiality.
  * `Class`: Target variable (1 for fraud, 0 for legitimate).

---

## 📂 Repository Structure

```text
├── 01_Data_Preprocessing.ipynb   # Scaling, distribution analysis, and SMOTE balancing
├── 02_Anomaly_Detection.ipynb    # Isolation Forest & Local Outlier Factor (LOF)
├── 03_XGBoost_Classifier.ipynb   # Supervised XGBoost training, ROC-AUC & Confusion Matrix
├── app.py                        # Streamlit web dashboard for live prediction
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation
