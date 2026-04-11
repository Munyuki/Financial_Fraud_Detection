# Fraud Detection in Financial Transactions

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)


## Project Overview

An end-to-end machine learning solution for detecting fraudulent transactions in mobile money systems using a synthetic dataset of over 6.3 million records. The solution focuses on building a robust, scalable, and interpretable fraud detection model using modern data science techniques.

## Problem Statement

Fraud in mobile financial systems poses a significant risk to both service providers and users. The objective of this project is to develop a model that can accurately identify fraudulent transactions while minimizing false negatives, as missing fraud can lead to severe financial losses.

## Dataset Overview

The dataset is a synthetic financial dataset generated using the PaySim simulator, containing over 6.3 million transactions. It includes transaction details such as type, amount, account balances, and fraud labels. The dataset is highly imbalanced, with fraudulent transactions accounting for approximately 0.13% of all records.

## Approach

The project follows a structured data science workflow including exploratory data analysis, feature engineering, model training, evaluation, and interpretation.

## Exploratory Data Analysis (EDA)

· Fraud vs. non-fraud distribution analysis
· Transaction type patterns by fraud occurrence
· Amount distribution comparison between fraud and legitimate transactions
· Correlation heatmap of numerical features

## Feature Engineering

Key engineered features include:

· Balance error metrics: orig_balance_error and dest_balance_error to detect account inconsistencies
· One-hot encoding of transaction types (PAYMENT, TRANSFER, CASH_OUT, etc.)
· Removal of ID columns (nameOrig, nameDest) to prevent data leakage

## Models Used

The project evaluates multiple machine learning models:

· Logistic Regression (baseline with class balancing)
· Random Forest (with balanced class weights)
· XGBoost (primary model with hyperparameter tuning)

## Handling Class Imbalance

Instead of undersampling, the project uses:

· class_weight='balanced' for Logistic Regression and Random Forest

· scale_pos_weight parameter in XGBoost

This approach preserves valuable data while improving fraud detection performance.

## Hyperparameter Tuning

XGBoost is optimized using RandomizedSearchCV with cross-validation (3 folds) over 8 parameter combinations, evaluating on ROC-AUC score. Parameters tuned include:

· n_estimators (100, 200)
· max_depth (4, 6)
· learning_rate (0.05, 0.1)
· subsample (0.8, 1.0)
· colsample_bytree (0.8, 1.0)

## Evaluation Metrics

Models are evaluated using metrics suitable for imbalanced classification:

· Precision, Recall, and F1-score (primary metrics)
· Confusion Matrix
· ROC-AUC (used for model selection during tuning)
· Precision-Recall Curve (PR Curve)
· PR-AUC (Area Under Precision-Recall Curve)

Given the extreme class imbalance (~0.13% fraud), PR-AUC is a more informative metric than ROC-AUC, as it focuses specifically on the model’s ability to correctly identify the minority (fraud) class without being influenced by the majority class.

The Precision-Recall Curve provides a clear view of the trade-off between precision and recall, which is critical in fraud detection where both false positives and false negatives carry significant costs.

## Model Explainability

SHAP (SHapley Additive exPlanations) is used to interpret model predictions, providing:

· Feature importance visualization

· Insights into which features contribute most to fraud detection

· Enhanced transparency and trust in the model

The SHAP summary plot reveals that balance errors and transaction patterns are the strongest indicators of fraud.

## Key Results

While Random Forest achieved near-perfect precision (100%) and high recall (96%) on the fraud class, this performance may indicate potential overfitting or sensitivity to the synthetic dataset characteristics. Such unusually high precision is rare in real-world fraud detection scenarios and may not generalize well to unseen data.

In contrast, the tuned XGBoost model provides a more realistic and robust performance, achieving 90% recall and 81% precision, with a PR-AUC of 0.9367. This reflects a more practical trade-off between detecting fraudulent transactions and minimizing false alarms.

Therefore, XGBoost is selected as the preferred model due to its better generalization capability and higher reliability in a production environment.

## Visualizations

### Class Imbalance
<img width="504" height="350" alt="Fraud vs Non-Fraud Distribution" src="https://github.com/user-attachments/assets/648309fd-e36b-4a43-b52a-45b6aa3f05e3" />

*Only 0.13% of transactions are fraudulent*

### Model Performance
<img width="677" height="139" alt="XGBoost Classification Report" src="https://github.com/user-attachments/assets/5c20b62a-f46c-407c-93d8-6b5076532fb2" />

### Feature Importance (SHAP)
<img width="547" height="449" alt="SHAP Summary Plot" src="https://github.com/user-attachments/assets/283fa3d9-cd9e-4239-92ad-56a774c629a4" />

*Balance errors are the strongest fraud indicators*

Feature importance analysis confirms that balance inconsistencies (orig_balance_error, dest_balance_error) and transaction amount are the strongest predictors of fraudulent activity.

### Precision-Recall Curve
<img width="414" height="325" alt="Precision Recall Curve (2)" src="https://github.com/user-attachments/assets/9bba6edc-07dd-472c-8542-5bfe2f548ee7" />

*The PR curve highlights the trade-off between precision and recall for the fraud class, providing a more realistic evaluation of model performance on imbalanced data.*

## Business Impact

The model can be deployed in real-time transaction processing systems to flag suspicious activities before completion, significantly reducing financial losses.

With a recall of 90% (achieved by the final tuned XGBoost model), the model is capable of detecting approximately 90 out of every 100 fraudulent transactions, ensuring the majority of fraud attempts are identified.

While there is a trade-off with false positives (precision of 81%), this level of accuracy is exceptionally high for fraud detection. It means that for every 10 flagged transactions, roughly 8 are actual fraud, which minimizes the operational burden of manual investigation compared to traditional models like Logistic Regression (which only had 1% precision).

Additionally, the use of SHAP explainability enhances transparency, enabling financial institutions to justify flagged transactions for auditing and regulatory compliance purposes.

## Requirements

· Python 3.10+
· pandas, numpy, matplotlib, seaborn
· scikit-learn
· xgboost
· shap

## Future Work

· Deploy the trained XGBoost model as a REST API using frameworks such as Flask or FastAPI for real-time fraud detection  
· Integrate the model into a streaming pipeline (e.g., Kafka or cloud-based event systems) for continuous transaction monitoring  
· Implement cost-sensitive learning to explicitly optimize business impact by assigning higher penalties to false negatives  
· Explore advanced anomaly detection techniques such as Isolation Forest and Autoencoders for unsupervised fraud detection  
· Incorporate temporal and behavioral features derived from transaction sequences to improve model accuracy  
· Perform model monitoring and drift detection to ensure sustained performance in production environments  
