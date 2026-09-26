📌 Project Overview

Financial fraud detection is an imbalanced classification problem where fraudulent transactions are relatively rare but can have significant financial impact.

This project implements a complete fraud-detection pipeline that:

- Loads and validates transaction data
- Performs ETL and preprocessing
- Engineers behavioral fraud-related features
- Handles class imbalance using SMOTE on the training set only
- Trains and compares four supervised ML models
- Selects a model using a configurable F1-based criterion
- Converts fraud probability into a configurable 0–100 risk score
- Stores transactions, predictions, alerts, and model versions in SQLite
- Provides a FastAPI prediction service
- Provides a 7-page Streamlit analytics dashboard
- Simulates real-time transactions without requiring Kafka
- Generates console/database alerts and optional email alerts
- Provides feature-importance based model explainability

The core system runs locally without Kafka or PySpark. Those technologies are available as optional advanced modules.

---

## ✨ Key Features

- **Data Validation & ETL** — checks missing values, duplicates, invalid amounts, timestamps, categorical values, class imbalance, and potential leakage.
- **Behavioral Feature Engineering** — creates 9 additional fraud-relevant features.
- **Class Imbalance Handling** — SMOTE is applied only after the train/test split and only to the training fold.
- **Multi-Model Training** — Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Risk Scoring** — converts fraud probability into a 0–100 score with Low, Medium, High, and Critical levels.
- **SQLite Persistence** — stores transactions, predictions, alerts, and model versions.
- **FastAPI** — exposes prediction, health, transaction, alert, metrics, and retraining endpoints.
- **Streamlit Dashboard** — provides seven analytical and investigation pages.
- **Real-Time Simulator** — replays transactions through the same prediction → risk → database → alert pipeline.
- **Alerting** — console and database alerts with optional SMTP email.
- **Explainability** — feature importance is saved for analysis and dashboard visualization.
- **Testing** — pytest suite covering preprocessing, prediction, risk scoring, and API behavior.

---

## 📊 Dataset

### Primary Dataset

`financial_fraud_detection_dataset.csv`

| Property | Value |
|---|---:|
| Transactions | 5,000 |
| Columns | 14 |
| Genuine | 4,518 |
| Fraudulent | 482 |
| Fraud Rate | 9.64% |

The selected dataset contains useful behavioral attributes such as previous transactions, average spending, account age, international transactions, and suspicious keywords.

### Dataset Schema

| Column | Description |
|---|---|
| `Transaction_ID` | Unique transaction identifier |
| `Customer_ID` | Customer identifier |
| `Transaction_Date` | Transaction timestamp |
| `Transaction_Amount` | Transaction amount |
| `Merchant_Category` | Merchant category |
| `Payment_Method` | Payment method |
| `Device_Type` | Device used for transaction |
| `Location` | Transaction location |
| `Is_International` | International transaction flag |
| `Previous_Transactions` | Customer's previous transaction count |
| `Average_Spend` | Customer's historical average spending |
| `Account_Age_Days` | Account age in days |
| `Suspicious_Keyword` | Suspicious keyword flag |
| `Fraudulent` | Binary fraud target |

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │   Transaction Data   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Validation / ETL│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Engineering │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴───────────┐
                 ▼                      ▼
        ┌─────────────────┐    ┌─────────────────┐
        │ Supervised ML   │    │ Isolation Forest│
        │ LogReg / DT /   │    │   Optional      │
        │ RF / XGBoost    │    └─────────────────┘
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ Model Comparison    │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Fraud Probability   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Risk Score 0–100    │
        └───────┬───────┬─────┘
                │       │
                ▼       ▼
        ┌──────────┐ ┌────────────┐
        │ SQLite   │ │ Alert      │
        │ Database │ │ Engine     │
        └────┬─────┘ └─────┬──────┘
             │             │
             └──────┬──────┘
                    ▼
             ┌─────────────┐
             │  FastAPI    │
             └──────┬──────┘
                    ▼
             ┌─────────────┐
             │  Streamlit  │
             │  Dashboard  │
             └─────────────┘
