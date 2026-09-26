# Financial Fraud Detection Model with Dashboard

An end-to-end, locally-runnable financial fraud detection system: ETL, behavioral
feature engineering, multi-model training with class-imbalance handling, a
configurable 0–100 risk-scoring engine, a SQLite datastore, a FastAPI prediction
service, a 7-page Streamlit analytics dashboard, real-time transaction simulation,
and an alerting system. Kafka, PySpark, and Isolation-Forest anomaly detection are
included as **optional** advanced modules — the core system runs without them.

---

## Overview

Financial institutions need to detect fraudulent transactions quickly and explain
*why* a transaction was flagged. This project builds that pipeline end-to-end on a
real (synthetic) transaction dataset, using models that are genuinely trained and
evaluated — no invented metrics anywhere in this repo.

## Problem Statement

Fraudulent transactions are rare (~9.6% of this dataset) but costly. A naive model
that always predicts "not fraud" would already be ~90% accurate while catching zero
fraud — which is why this project evaluates models on Precision, Recall, F1,
ROC-AUC, and PR-AUC instead of accuracy alone.

## Objectives

- Load, validate, and clean real transaction data
- Engineer behavioral, fraud-relevant features without leaking the target
- Handle class imbalance correctly (SMOTE on the training fold only)
- Train and compare Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Automatically select the best model on a documented metric (F1)
- Convert fraud probability into an interpretable 0–100 risk score and category
- Persist transactions/predictions/alerts in SQLite
- Serve predictions through a REST API
- Visualize everything in an interactive dashboard
- Simulate a live transaction feed with automatic alerting
- Explain model decisions via feature importance

## Features

- Reusable, documented ETL + validation pipeline (`src/preprocessing.py`)
- 9 engineered behavioral features (`src/feature_engineering.py`)
- 4 supervised models trained and compared on identical folds
- SMOTE class-imbalance handling applied correctly (train-only, post-split)
- Configurable risk-scoring engine (`src/risk_scoring.py`)
- SQLite database with a real indexed schema (no placeholder SQL)
- FastAPI service with 8 documented endpoints
- Streamlit dashboard with 7 pages and shared filters
- No-Kafka real-time simulator + optional Kafka/PySpark modules
- Console + database + optional email alerting, credentials via `.env` only
- Isolation Forest as an optional unsupervised anomaly signal
- Pytest suite (17 tests, all passing) covering preprocessing, prediction,
  risk scoring, and the API

## Dataset

**Primary dataset:** `financial_fraud_detection_dataset.csv`

Several datasets were provided. They were inspected individually — not blindly
merged, since their schemas differ substantially:

| Dataset | Rows | Columns | Fraud Rate | Notes |
|---|---|---|---|---|
| **financial_fraud_detection_dataset.csv** | 5,000 | 14 | 9.64% | **Selected as primary.** Zero missing values, zero duplicates, richest behavioral columns (Previous_Transactions, Average_Spend, Account_Age_Days, Is_International, Suspicious_Keyword). Realistic imbalance ratio. |
| credit_card_fraud_dataset.csv | 100,000 | 7 | 1.0% | Very few features (Amount, MerchantID, TransactionType, Location only) — not enough behavioral signal, and the fraud rate is extremely low, making it harder to build a meaningful demo model without heavier sampling. Kept for optional comparison. |
| synthetic_fraud_dataset1.csv | 50,000 | 14 | 32.1% | Decent feature set, but a 32% fraud rate is far higher than realistic real-world fraud rates and suggests fully synthetic/balanced generation. Kept for optional comparison/experimentation, not as ground truth. |

This matches the task's stated preference for `financial_fraud_detection_dataset.csv`
as the primary implementation dataset.

### Dataset Schema (primary dataset)

| Column | Type | Description |
|---|---|---|
| Transaction_ID | string | Unique transaction identifier (5,000 unique values, no duplicates) |
| Customer_ID | string | Customer identifier (3,847 unique customers) |
| Transaction_Date | datetime | Transaction timestamp |
| Transaction_Amount | float | Transaction amount |
| Merchant_Category | categorical | 8 categories (Travel, Utilities, Entertainment, Food, Grocery, Health, Electronics, Fashion) |
| Payment_Method | categorical | 5 methods (Credit Card, Debit Card, PayPal, NetBanking, UPI) |
| Device_Type | categorical | 3 types (POS, Mobile, Desktop) |
| Location | categorical | 7 cities |
| Is_International | binary | 1 = international transaction |
| Previous_Transactions | int | Customer's prior transaction count |
| Average_Spend | float | Customer's historical average spend |
| Account_Age_Days | int | Account age in days |
| Suspicious_Keyword | categorical | Yes/No flag |
| **Fraudulent** | binary | **Target column.** 4,518 genuine (90.36%) / 482 fraud (9.64%) |

No missing values, no duplicate rows, no duplicate transaction IDs, no negative
amounts, no invalid timestamps were found (see `notebooks/fraud_detection_analysis.ipynb`,
section 14, for the full generated data-quality report).

## Architecture

```
Transaction Dataset
        |
        v
  Data Validation
        |
        v
 ETL / Preprocessing
        |
        v
Feature Engineering
        |
   +----+-----------------+
   |                       |
   v                       v
Supervised ML       Unsupervised ML
(LogReg, DT,         (Isolation Forest,
 RF, XGBoost)          optional)
   |                       |
   +----------+------------+
              |
              v
      Model Comparison
              |
              v
        Best ML Model
              |
              v
      Fraud Probability
              |
              v
       Risk Score 0-100
              |
    +---------+----------+
    |                     |
    v                     v
SQLite Database      Alert Engine
    |
    v
FastAPI Prediction API
    |
    v
Streamlit Dashboard
    |
    v
Real-Time Monitor
```

## Technology Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) ·
SQLite · FastAPI · Streamlit · Plotly · Matplotlib/Seaborn · Joblib ·
python-dotenv · pytest — plus **optional** Kafka (kafka-python) and PySpark.

## Project Structure

```
financial-fraud-detection/
├── data/
│   ├── raw/financial_fraud_detection_dataset.csv
│   └── processed/processed_transactions.csv
├── notebooks/fraud_detection_analysis.ipynb
├── models/
│   ├── best_fraud_model.pkl
│   ├── preprocessor.pkl
│   ├── model_metadata.json
│   ├── feature_importance.csv
│   └── isolation_forest.pkl        (optional anomaly model)
├── database/
│   ├── fraud_detection.db
│   └── schema.sql
├── src/
│   ├── config.py, utils.py
│   ├── preprocessing.py, feature_engineering.py
│   ├── train_model.py, model_comparison.py
│   ├── predict.py, risk_scoring.py, anomaly_detection.py
│   ├── database.py, alerts.py, realtime.py
├── api/main.py
├── dashboard/
│   ├── app.py, components.py
│   └── pages/ (7 page modules)
├── streaming/
│   ├── simulator.py               (no Kafka required)
│   ├── kafka_producer.py, kafka_consumer.py   (optional)
│   └── spark_streaming_example.py             (optional)
├── tests/
│   ├── test_preprocessing.py, test_prediction.py
│   ├── test_risk_scoring.py, test_api.py
├── reports/  (model_comparison.csv + generated PNG charts)
├── .env.example, .gitignore, requirements.txt
├── README.md, PROJECT_REPORT.md
└── run.py
```

## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment variables template
cp .env.example .env
```

## Environment Variables

See `.env.example`. All alerting/email/API/simulator settings are read from `.env`
via `python-dotenv` — nothing is hard-coded in source. Gmail users need an
**App Password** (not their normal password) if `ALERT_EMAIL_ENABLED=true`.

## Data Preprocessing

`src/preprocessing.py` implements Extract → Validate → Transform → Load:
- Extract: load the raw CSV
- Validate: missing values, duplicate rows/IDs, negative amounts, invalid
  timestamps, unexpected categorical values, class imbalance ratio, and a
  correlation-based data-leakage check — all captured in a `DataQualityReport`
- Transform: drop duplicates, impute missing values (median/mode), drop invalid
  amounts, parse timestamps
- Load: save to `data/processed/processed_transactions.csv` and (via
  `src/database.py`) into SQLite

The original spec's placeholder `cursor.execute("")` was replaced with a real,
indexed schema in `database/schema.sql`.

## Feature Engineering

9 behavioral features are added in `src/feature_engineering.py`, each documented
with its rationale in the module docstring (transaction hour/day/weekend,
amount-to-average ratio, amount deviation, log-amount, account age in years,
new-account flag, suspicious-keyword flag). None of these use the `Fraudulent`
column, so there is no target leakage — confirmed by the correlation-based
leakage check in `validate_data()`, which found no columns correlated > 0.9
with the target.

## Machine Learning

Four models are trained on an identical, stratified 80/20 split
(`random_state=42`): Logistic Regression, Decision Tree, Random Forest, XGBoost.
**SMOTE is applied to the training fold only, after the split** — never to the
test set — to avoid leakage while addressing the ~9.6% fraud rate.

## Model Comparison (actual results from this run)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Decision Tree** ⭐ | 0.843 | 0.3216 | 0.5729 | **0.4120** | 0.8218 | 0.3076 |
| Logistic Regression | 0.789 | 0.2797 | 0.7604 | 0.4090 | 0.8750 | 0.4258 |
| Random Forest | 0.889 | 0.4176 | 0.3958 | 0.4064 | 0.8934 | 0.3798 |
| XGBoost | 0.889 | 0.3810 | 0.2500 | 0.3019 | 0.8918 | 0.4081 |

(Full numbers regenerated each run in `reports/model_comparison.csv`.)

## Model Evaluation & Selection

The system selects the best model by **highest F1** (configurable in
`src/config.py` via `MODEL_SELECTION_METRIC`), since F1 balances precision
(minimizing false alarms) against recall (minimizing missed fraud) on this
imbalanced dataset — accuracy alone would favor a model that just predicts
"not fraud" most of the time. On this run, **Decision Tree** was selected.
Note Logistic Regression and Random Forest have competitive/better ROC-AUC and
PR-AUC — this is documented as a real trade-off, not hidden. You can change
`MODEL_SELECTION_METRIC` to `"Recall"` or `"PR_AUC"` if your business priority
differs (e.g. prioritizing catching more fraud over fewer false positives).

## Risk Scoring

`src/risk_scoring.py` converts a fraud probability into a 0–100 score and a
level using configurable thresholds (`src/config.py: RISK_THRESHOLDS`):

| Score | Level |
|---|---|
| 0–30 | Low |
| 31–60 | Medium |
| 61–80 | High |
| 81–100 | Critical |

A probability is a model estimate, not proof of fraud — the dashboard and API
responses are written to reflect that.

## Database

SQLite (`database/fraud_detection.db`), schema in `database/schema.sql`:
`transactions`, `predictions`, `alerts`, `model_versions`, with indexes on
customer_id, fraud flag, transaction_id (FK), and risk_level/status.

## FastAPI

`api/main.py` — run with `uvicorn api.main:app --reload`, docs at `/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Model/DB readiness check |
| `/predict` | POST | Score a single transaction (Pydantic-validated) |
| `/transactions` | GET | Recent transactions |
| `/fraud-transactions` | GET | Recent labeled-fraud transactions |
| `/alerts` | GET | Alerts, optionally filtered by status |
| `/metrics` | GET | Current model metadata/metrics |
| `/retrain` | POST | Re-run training; reports new metrics for human review before deployment |

## Dashboard

`streamlit run dashboard/app.py` — **Financial Fraud Intelligence Platform**,
7 pages via the sidebar, with shared filters (date range, location, device,
payment method, merchant category, fraud status):

1. **Executive Overview** — KPI cards + fraud/transaction/amount trend charts
2. **Fraud Analysis** — fraud rate by payment method, merchant category, device, location, international flag
3. **Risk Analysis** — risk-level counts, score distribution, average-risk trend
4. **Customer Behavior** — frequency/spend/deviation/account-age distributions + per-customer lookup
5. **Real-Time Monitor** — latest scored transactions, color-highlighted by risk level
6. **Model Performance** — live metrics, model comparison table, confusion matrix/ROC/PR/feature-importance charts
7. **Transaction Investigation** — look up any transaction by ID, view details and (stored or live) risk assessment

## Real-Time Monitoring

`streaming/simulator.py` replays transactions one at a time through the full
predict → risk-score → persist → alert pipeline with a configurable delay —
**no Kafka needed**:
```bash
python -m streaming.simulator --limit 20 --delay 0.5
```

## Alert System

`src/alerts.py` fires when `risk_score >= ALERT_RISK_SCORE_THRESHOLD` (default
61, configurable in `.env`): prints to console, writes to the `alerts` table,
and optionally emails via SMTP if `ALERT_EMAIL_ENABLED=true` and credentials are
set in `.env`. No credentials are ever hard-coded.

## Optional Kafka

`streaming/kafka_producer.py` / `kafka_consumer.py` implement a real
`KafkaProducer`/`KafkaConsumer` pair with JSON serialization (the original
spec's `KfkaConsumer`/`KafkaComsumer`/`json/loads` typos are fixed). To enable:
```bash
pip install kafka-python
# start a local Kafka broker (e.g. via Docker: bitnami/kafka or confluentinc images)
python -m streaming.kafka_producer
python -m streaming.kafka_consumer
```
The core project works fully without this.

## Optional PySpark

`streaming/spark_streaming_example.py` is a corrected Spark Structured
Streaming example (fixes: `.getOrCreate()` call, `"timestamp"` type typo,
missing `StringIndexer` stage, missing imports). Install with `pip install
pyspark`. Not required for the core project.

## Model Explainability

Feature importance is extracted from the trained tree/linear model
(`feature_importances_` or `|coef_|`) and saved to
`models/feature_importance.csv` / `reports/feature_importance.png`. Top drivers
from this run: `transaction_hour`, `suspicious_keyword_flag`, and
`Is_International` were the strongest signals — see the dashboard's Model
Performance page or the CSV for the full ranked list.

## Adaptive Learning (Retraining)

`POST /retrain` re-runs the full training pipeline and returns the new model's
metrics for review. It does **not** automatically replace the production model —
a human should compare the new metrics against the current `models/model_metadata.json`
before promoting it, per the spec's "no unsafe automatic replacement" requirement.

## Testing

```bash
pytest tests/ -v
```
17 tests covering preprocessing (ETL correctness), prediction (output shape,
probability bounds, unseen-category handling), risk scoring (threshold
boundaries), and the API (all major endpoints, including validation-error
handling). All 17 currently pass.

## How to Run

```bash
# One-shot setup: ETL + train + init DB + load transactions
python run.py

# Or step by step:
python -m src.train_model          # train models, save artifacts, generate reports
python -m src.database              # initialize the database schema

# Start the API
uvicorn api.main:app --reload

# Start the dashboard (separate terminal)
streamlit run dashboard/app.py

# Run tests
pytest tests/ -v

# Run the real-time simulator
python -m streaming.simulator --limit 20 --delay 0.5
```

## Screenshots

_Add screenshots of the running dashboard here after your first local run
(Executive Overview, Real-Time Monitor, and Model Performance pages are good
choices)._

## Future Enhancements

- AI-driven creditworthiness/risk scoring beyond fraud
- Graph-based fraud-ring detection across customers/merchants
- Blockchain-based fraud prevention exploration
- Mobile push alerts
- SHAP-based per-transaction explainability in the dashboard

## Limitations

- Trained on a synthetic 5,000-row dataset; production deployment would need a
  much larger, continuously-refreshed dataset and periodic retraining.
- Precision on the selected model (~0.32) means roughly 2 in 3 "flagged" alerts
  would be false positives in production-scale terms — acceptable for a
  portfolio/demo system but should be tuned (e.g. threshold, sampling, or model
  choice) before production use.
- Email alerting requires manual SMTP/App Password setup; not deployed as a
  managed service.
- Kafka/PySpark modules are illustrative and untested against a live cluster in
  this environment (no broker available here); syntax and logic were corrected
  and reviewed, but end-to-end execution requires a local Kafka/Spark setup.

## Author

Built as a portfolio / academic project — Financial Fraud Detection Model with
Interactive Dashboard.
