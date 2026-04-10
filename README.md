# Fraud Detection in Financial Transactions

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)


Project Overview

An end-to-end machine learning solution for detecting fraudulent transactions in mobile money systems using a synthetic dataset of over 6.3 million records. The solution focuses on building a robust, scalable, and interpretable fraud detection model using modern data science techniques.

Problem Statement

Fraud in mobile financial systems poses a significant risk to both service providers and users. The objective of this project is to develop a model that can accurately identify fraudulent transactions while minimizing false negatives, as missing fraud can lead to severe financial losses.

Dataset Overview

The dataset is a synthetic financial dataset generated using the PaySim simulator, containing over 6.3 million transactions. It includes transaction details such as type, amount, account balances, and fraud labels. The dataset is highly imbalanced, with fraudulent transactions accounting for approximately 0.13% of all records.

Approach

The project follows a structured data science workflow including exploratory data analysis, feature engineering, model training, evaluation, and interpretation.

Exploratory Data Analysis (EDA)

· Fraud vs. non-fraud distribution analysis
· Transaction type patterns by fraud occurrence
· Amount distribution comparison between fraud and legitimate transactions
· Correlation heatmap of numerical features

Feature Engineering

Key engineered features include:

· Balance error metrics: orig_balance_error and dest_balance_error to detect account inconsistencies
· One-hot encoding of transaction types (PAYMENT, TRANSFER, CASH_OUT, etc.)
· Removal of ID columns (nameOrig, nameDest) to prevent data leakage

Models Used

The project evaluates multiple machine learning models:

· Logistic Regression (baseline with class balancing)
· Random Forest (with balanced class weights)
· XGBoost (primary model with hyperparameter tuning)

Handling Class Imbalance

Instead of undersampling, the project uses:

· class_weight='balanced' for Logistic Regression and Random Forest
· scale_pos_weight parameter in XGBoost

This approach preserves valuable data while improving fraud detection performance.

Hyperparameter Tuning

XGBoost is optimized using RandomizedSearchCV with cross-validation (3 folds) over 8 parameter combinations, evaluating on ROC-AUC score. Parameters tuned include:

· n_estimators (100, 200)
· max_depth (4, 6)
· learning_rate (0.05, 0.1)
· subsample (0.8, 1.0)
· colsample_bytree (0.8, 1.0)

Evaluation Metrics

Models are evaluated using metrics suitable for imbalanced classification:

· Precision, Recall, and F1-score (primary metrics)
· Confusion Matrix
· ROC-AUC (used for model selection during tuning)

Model Explainability

SHAP (SHapley Additive exPlanations) is used to interpret model predictions, providing:

· Feature importance visualization
· Insights into which features contribute most to fraud detection
· Enhanced transparency and trust in the model

The SHAP summary plot reveals that balance errors and transaction patterns are the strongest indicators of fraud.

Key Results

· Logistic Regression: Poor precision for fraud class (1%), despite high recall (95%)
· Random Forest: Excellent performance (96% recall, 100% precision for fraud)
· XGBoost (tuned): Best balance with 96% recall and 57% precision for fraud class

## Visualizations

### Class Imbalance
![Fraud vs Non-Fraud Distribution](images/fraud_distribution.png)
*Only 0.13% of transactions are fraudulent*

### Model Performance
![XGBoost Classification Report](images/classification_report.png)

### Feature Importance (SHAP)
![SHAP Summary Plot](images/shap_summary.png)
*Balance errors are the strongest fraud indicators*

Feature importance analysis confirms that balance inconsistencies (orig_balance_error, dest_balance_error) and transaction amount are the strongest predictors of fraudulent activity.

Business Impact

The model can be deployed in real-time systems to flag suspicious transactions, reducing financial losses and improving security in mobile payment platforms. High recall ensures that most fraudulent activities are detected, while SHAP explainability provides auditability and regulatory compliance support.

Requirements

· Python 3.10+
· pandas, numpy, matplotlib, seaborn
· scikit-learn
· xgboost
· shap

Future Work

· Deploy model as a real-time API endpoint
· Implement cost-sensitive learning to optimize for business metrics
· Explore anomaly detection techniques (Isolation Forest, Autoencoders)
· Add additional behavioral features from transaction sequences
