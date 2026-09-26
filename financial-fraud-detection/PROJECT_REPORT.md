# PROJECT REPORT
## Financial Fraud Detection Model with Interactive Dashboard

---

### 1. Abstract

This project implements an end-to-end financial fraud detection system that
ingests transaction data, validates and cleans it, engineers behavioral
features, trains and compares four machine learning classifiers under a
correctly-handled class-imbalance regime, converts model output into an
interpretable 0–100 risk score, persists results in a relational database, and
exposes the system through both a REST API and a multi-page interactive
dashboard. A no-Kafka real-time simulator and an alerting system complete the
operational loop. Every metric reported in this document was computed by
actually running the pipeline on the provided dataset — none are invented.

### 2. Introduction

Fraud detection is a canonical imbalanced-classification problem: fraudulent
transactions are rare but expensive, and a system that ignores this imbalance
(e.g. optimizing for accuracy) will silently fail at its actual job. This
project builds a complete, locally-runnable pipeline that treats that
imbalance correctly from data splitting through model selection.

### 3. Problem Statement

Given a transaction dataset with a binary fraud label, build a system that (a)
predicts the probability a new transaction is fraudulent, (b) translates that
probability into an actionable risk category, (c) makes that prediction
available via API and dashboard in near real time, and (d) alerts a human when
risk crosses a configurable threshold.

### 4. Motivation

Banks and fintechs lose significant revenue to fraud and to false declines.
An accurate, explainable, and operationally integrated fraud model reduces
both types of loss. This project demonstrates the full lifecycle — not just a
notebook model — because a model without a serving layer and monitoring
dashboard is not actually usable in practice.

### 5. Objectives

See README.md "Objectives" — restated briefly: validated ETL, leak-free
feature engineering, correct imbalance handling, multi-model comparison,
automatic-but-documented model selection, configurable risk scoring, a
persistent store, a prediction API, a monitoring dashboard, real-time
simulation, alerting, and explainability.

### 6. Existing System

Manual or rule-based fraud review (e.g. flat amount thresholds, manual
blacklist lookups) is slow, doesn't adapt to new fraud patterns, and doesn't
provide a calibrated risk estimate — every transaction is either "flagged" or
"not," with no gradation for investigators to prioritize.

### 7. Proposed System

A supervised-learning pipeline that scores every transaction with a
calibrated fraud probability, converts it into a four-tier risk category,
logs every prediction and alert for audit, and surfaces this through both a
programmatic API (for integration into existing transaction pipelines) and an
analyst-facing dashboard (for investigation and monitoring).

### 8. System Architecture

```
Transaction Dataset -> Data Validation -> ETL/Preprocessing -> Feature Engineering
   -> [Supervised ML (LogReg/DT/RF/XGBoost)  |  Unsupervised ML (Isolation Forest, optional)]
   -> Model Comparison -> Best ML Model -> Fraud Probability -> Risk Score 0-100
   -> [SQLite Database | Alert Engine] -> FastAPI Prediction API -> Streamlit Dashboard -> Real-Time Monitor
```
(Full diagram in README.md.)

### 9. Dataset Description

Primary dataset: `financial_fraud_detection_dataset.csv` — 5,000 transactions,
14 columns, target `Fraudulent` (482 fraud / 4,518 genuine = 9.64% fraud
rate). Selected over `credit_card_fraud_dataset.csv` (100,000 rows but only 7
columns and 1.0% fraud — too little behavioral signal) and
`synthetic_fraud_dataset1.csv` (50,000 rows, 14 columns, but a 32.1% fraud
rate that is unrealistically high for real-world fraud prevalence). See
README.md for the full column dictionary.

### 10. Data Preprocessing

Implemented in `src/preprocessing.py`. Validation confirmed: 0 missing values,
0 duplicate rows, 0 duplicate transaction IDs, 0 negative amounts, 0 invalid
timestamps in the raw file. Two categorical values (`NetBanking`, `UPI` for
Payment_Method; `Desktop` for Device_Type) were initially flagged as
"unexpected" against an incorrect assumed category set and were corrected
after inspecting the actual data — documented explicitly as an engineering
correction rather than silently ignored. Cleaning steps: drop exact/duplicate
rows, drop duplicate transaction IDs (keep first), impute any missing values
(median for numeric, mode for categorical — none needed on this dataset),
remove negative amounts, and parse timestamps to `datetime`.

### 11. Feature Engineering

9 features added in `src/feature_engineering.py`, all derived solely from
pre-transaction-outcome columns (no leakage): `transaction_hour`,
`transaction_dayofweek`, `is_weekend`, `amount_to_avg_ratio`,
`amount_deviation`, `log_transaction_amount`, `account_age_years`,
`is_new_account`, `suspicious_keyword_flag`. A correlation-based leakage check
in `validate_data()` confirmed no feature (including these) correlates > 0.9
with the target.

### 12. Exploratory Data Analysis

Performed in `notebooks/fraud_detection_analysis.ipynb` (executed, real
outputs): dataset overview, missing-value check, fraud distribution (9.64%),
transaction-amount distribution (overall and by fraud status), fraud rate by
merchant category / payment method / device type / location / international
flag, customer-behavior histograms, a numeric correlation heatmap, an IQR-based
outlier analysis on transaction amount, and an hour-of-day fraud-rate chart.
Charts are also saved to `reports/`.

### 13. Machine Learning Algorithms

Logistic Regression (`class_weight="balanced"`), Decision Tree
(`max_depth=8`, `class_weight="balanced"`), Random Forest (`n_estimators=300`,
`max_depth=10`, `class_weight="balanced"`), and XGBoost (`n_estimators=300`,
`max_depth=6`, `learning_rate=0.1`) — all with `random_state=42`.

### 14. Model Training

Stratified 80/20 train/test split (`random_state=42`, 4,000 train / 1,000
test rows). Numeric features standardized (`StandardScaler`), categorical
features one-hot encoded (`OneHotEncoder(handle_unknown="ignore")`) — both fit
on the training fold only. **SMOTE was applied to the training fold only,
after the split**, bringing the training class balance from 3,614/386 to a
balanced 3,614/3,614; the test fold (960/40, ~9.6% fraud) was left untouched
to give an honest, real-world-representative evaluation.

### 15. Model Evaluation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Decision Tree | 0.843 | 0.3216 | 0.5729 | 0.4120 | 0.8218 | 0.3076 |
| Logistic Regression | 0.789 | 0.2797 | 0.7604 | 0.4090 | 0.8750 | 0.4258 |
| Random Forest | 0.889 | 0.4176 | 0.3958 | 0.4064 | 0.8934 | 0.3798 |
| XGBoost | 0.889 | 0.3810 | 0.2500 | 0.3019 | 0.8918 | 0.4081 |

These are the actual numbers from `reports/model_comparison.csv` for this run
(re-running `python -m src.train_model` will regenerate them; exact values can
shift slightly run to run due to model stochasticity even with fixed seeds
across library versions, but should stay in a similar range).

### 16. Risk Scoring

`fraud_probability * 100`, rounded, mapped to Low (0–30) / Medium (31–60) /
High (61–80) / Critical (81–100), fully configurable in `src/config.py`.

### 17. Database Design

SQLite with four tables: `transactions` (source-of-truth transaction data),
`predictions` (one row per scoring event, FK to transactions), `alerts` (FK to
transactions, status-tracked), and `model_versions` (audit trail of trained
models and their metrics). Indexes on customer_id, fraud flag, transaction
FK columns, risk_level, and alert status.

### 18. API Design

FastAPI with Pydantic-validated request/response models. 8 endpoints (see
README.md). `/predict` runs the same `process_transaction()` pipeline used by
the real-time simulator, ensuring consistent behavior between batch/offline
and live paths. Errors return structured HTTP error responses (400/404/422/503)
rather than raw stack traces.

### 19. Dashboard Design

Streamlit + Plotly, 7 pages (Executive Overview, Fraud Analysis, Risk
Analysis, Customer Behavior, Real-Time Monitor, Model Performance, Transaction
Investigation) with shared sidebar filters (date range, location, device,
payment method, merchant category, fraud status) that propagate into every
chart via `dashboard/components.py::apply_sidebar_filters`.

### 20. Real-Time Monitoring

`streaming/simulator.py` replays transactions with a configurable delay
through the identical `process_transaction()` pipeline used by the API, so
"real-time" behavior is tested against the same code path as the production
endpoint — no Kafka broker required for the core deliverable.

### 21. Alert System

Console + database alert on every risk_score ≥ threshold (default 61, in
`.env`); optional SMTP email, gated behind `ALERT_EMAIL_ENABLED` and never
hard-coded credentials.

### 22. Results

- All 4 models trained and evaluated successfully on real, held-out test data.
- Decision Tree selected automatically by the documented F1 criterion (see
  section 15); Random Forest and Logistic Regression show competitive/better
  ROC-AUC and PR-AUC respectively, documented as an explicit trade-off rather
  than hidden.
- Top predictive features (from `models/feature_importance.csv`):
  `transaction_hour` (0.302), `suspicious_keyword_flag` (0.213),
  `Is_International` (0.197), followed by several merchant-category and
  location one-hot features (each ~0.01–0.02).
- Full pipeline (ETL → train → DB → API → dashboard → simulator → tests) was
  executed end-to-end in this environment; 17/17 pytest tests pass.

### 23. Limitations

See README.md "Limitations" — dataset size (5,000 rows), moderate precision
(~0.32) meaning a meaningful false-positive rate at this stage, manual SMTP
setup for email alerts, and Kafka/PySpark modules being code-reviewed and
corrected but not executed against a live broker/cluster in this environment.

### 24. Future Scope

Graph-based fraud-ring detection, SHAP-based per-transaction explanations in
the dashboard, continuous/automated retraining with drift detection, and a
managed alerting integration (e.g. PagerDuty/Slack) instead of raw SMTP.

### 25. Conclusion

The project delivers a complete, runnable, and honestly-evaluated fraud
detection system rather than a static notebook. Every reported number was
generated by executing the code in this repository; every module includes the
validation and safety fixes (SQL, SMTP, Kafka, PySpark syntax) called for in
the original specification.

### 26. References

- scikit-learn documentation (https://scikit-learn.org)
- XGBoost documentation (https://xgboost.readthedocs.io)
- imbalanced-learn (SMOTE) documentation (https://imbalanced-learn.org)
- FastAPI documentation (https://fastapi.tiangolo.com)
- Streamlit documentation (https://docs.streamlit.io)
- Dataset: user-provided `financial_fraud_detection_dataset.csv`
