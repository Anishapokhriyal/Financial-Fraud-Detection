# Viva / Interview Questions — Financial Fraud Detection Model with Dashboard

These 25+ questions are based specifically on decisions actually made in this
implementation (see `src/`, `reports/model_comparison.csv`, and
`models/model_metadata.json` for the real numbers referenced below).

---

**1. Why was `financial_fraud_detection_dataset.csv` chosen over the other two CSVs you were given?**
It had zero missing values, zero duplicate rows, a realistic ~9.6% fraud rate, and
the richest set of behavioral columns (`Previous_Transactions`, `Average_Spend`,
`Account_Age_Days`, `Is_International`, `Suspicious_Keyword`). `credit_card_fraud_dataset.csv`
only had 7 columns and a 1% fraud rate (too sparse for behavioral features);
`synthetic_fraud_dataset1.csv` had a 32% fraud rate, which is unrealistically high
for a production-like model.

**2. Why not just combine all three datasets into one big training set?**
They have different schemas, different column semantics, and very different fraud
rates (1%, 9.6%, 32%). Merging them would blend inconsistent label distributions and
force-fit unrelated columns, quietly corrupting the model rather than improving it.

**3. Why is accuracy not used to select the best model?**
The dataset is imbalanced (~90.4% genuine vs ~9.6% fraud). A model that predicts
"not fraud" for every transaction would score ~90% accuracy while catching zero
fraud. Precision, Recall, F1, ROC-AUC, and PR-AUC are used instead because they
reflect performance on the minority (fraud) class.

**4. Which metric did you use to automatically select the best model, and why?**
F1 score (harmonic mean of precision and recall), configurable in
`src/config.py` via `MODEL_SELECTION_METRIC`. It was chosen because it balances
false alarms (precision) against missed fraud (recall) — both matter in fraud
detection, unlike optimizing for just one.

**5. What was the actual result of the model comparison?**
Decision Tree had the best F1 (0.4120), narrowly ahead of Logistic Regression
(0.4090) and Random Forest (0.4064); XGBoost trailed at 0.3019. Logistic
Regression had the highest ROC-AUC (0.8750) and Random Forest the highest
accuracy (0.889). See `reports/model_comparison.csv` for exact figures.

**6. If Random Forest had better accuracy and ROC-AUC, why wasn't it selected?**
Because the selection criterion is F1, not accuracy or ROC-AUC — and that
criterion was chosen deliberately (see Q4) before looking at results, to avoid
cherry-picking. The threshold is configurable if the business priority changes.

**7. How is class imbalance handled, and why is the order important?**
SMOTE (Synthetic Minority Over-sampling) is applied only to the training set,
and only after the train/test split. Applying SMOTE before splitting would leak
synthetic copies of test-set patterns into training, artificially inflating
evaluation metrics.

**8. What is data leakage, and how did you check for it?**
Data leakage happens when a feature indirectly encodes the target, letting the
model "cheat" during training but fail in production. `validate_data()` in
`src/preprocessing.py` computes each numeric column's correlation with
`Fraudulent` and flags any above 0.9 as suspicious; none were found in this
dataset.

**9. Walk through the feature engineering. Why these features specifically?**
Nine features were engineered from existing columns only: `transaction_hour`,
`transaction_dayofweek`, `is_weekend` (timing patterns), `amount_to_avg_ratio`
and `amount_deviation` (spending anomaly vs. the customer's own history),
`log_transaction_amount` (reduces skew for linear models), `account_age_years`
and `is_new_account` (new accounts carry higher risk), and
`suspicious_keyword_flag`. None of them use the target column.

**10. How do you prevent the model from crashing on an unseen category at inference time?**
The `OneHotEncoder` in the preprocessing pipeline is built with
`handle_unknown="ignore"`, so a category never seen in training (e.g. a new
`Merchant_Category`) is encoded as all-zeros instead of raising an error.

**11. Explain the risk scoring formula.**
`risk_scoring.py` converts the model's fraud probability (0–1) into a 0–100
integer score by multiplying by 100 and rounding, then maps that score to Low
(0–30), Medium (31–60), High (61–80), or Critical (81–100) using thresholds
defined in `config.RISK_THRESHOLDS`, which are easy to change without touching
the scoring logic.

**12. Is a "Critical" risk score proof of fraud?**
No. It is a probability-based estimate meaning the transaction is statistically
similar to past fraud patterns. It should trigger investigation, not automatic
blocking — this distinction is documented in the code and README.

**13. What does the Isolation Forest module add that the supervised model doesn't?**
It's an unsupervised anomaly detector that doesn't need fraud labels. It flags
transactions that are statistically unusual, which can catch novel fraud
patterns the supervised model was never trained on. It's optional and the core
system works without it; `anomaly_detection.py` also provides a configurable
function to blend both signals.

**14. Why SQLite instead of PostgreSQL/MySQL for the database?**
SQLite requires no separate server process, ships with Python, and is
sufficient for a single-instance student/portfolio project. The schema
(`database/schema.sql`) uses standard SQL with foreign keys and indexes, so
migrating to PostgreSQL later would mainly require changing the connection
string and minor syntax adjustments.

**15. What real database design mistake from the original spec did you fix?**
The original spec used `cursor.execute("")` — an empty string — to "create" the
transactions table, which does nothing. It was replaced with an actual schema
(`database/schema.sql`) defining `transactions`, `predictions`, `alerts`, and
`model_versions` tables with primary keys, foreign keys, and indexes.

**16. Why do `predictions` and `alerts` reference `transaction_id` as a foreign key, and what problem did that cause?**
It enforces that every prediction/alert is tied to a real transaction. During
testing, calling `/predict` with a brand-new `Transaction_ID` failed with a
foreign key constraint error because that transaction wasn't in the
`transactions` table yet. This was fixed by adding `upsert_transaction()`,
which inserts the incoming transaction before inserting its prediction.

**17. How does the FastAPI `/retrain` endpoint avoid unsafely replacing a working model?**
It runs the full training pipeline synchronously and returns the new model's
metrics in the response, but the actual production model file is only
overwritten if that run's evaluation is accepted — the design intentionally
avoids automatic, unreviewed deployment, matching the adaptive-learning
principle: new models must be evaluated before promotion.

**18. What happens if you call `/predict` before training has ever been run?**
`predict_transaction()` raises `FileNotFoundError` if `models/best_fraud_model.pkl`
doesn't exist; the API catches this and returns HTTP 503 with a clear message
instead of crashing.

**19. Why is SMTP email alerting disabled by default, and how are credentials handled?**
`ALERT_EMAIL_ENABLED` defaults to `false` in `.env.example` so the project runs
out-of-the-box without requiring email setup. Credentials (`ALERT_EMAIL`,
`ALERT_EMAIL_PASSWORD`, `ALERT_RECEIVER`) are read only from environment
variables via `python-dotenv`, never hard-coded — and `.env` is in
`.gitignore` so real credentials are never committed.

**20. What bug existed in the original spec's alert email code, and how was it fixed?**
The original `send_alert()` function declared variables like `sender_email`,
`subject`, and `body` with no values and no logic — it wouldn't run.
`alerts.py` implements a complete, working `send_email_alert()` using
`smtplib` and `email.mime.text.MIMEText`, with credentials from `.env` and a
graceful no-op if email alerting is disabled or misconfigured.

**21. Why is Kafka optional rather than required?**
The assignment requires the core system to run locally without external
infrastructure. `streaming/simulator.py` replays transactions through the same
prediction/risk/alert pipeline with a configurable delay, giving the "real-time"
experience without a Kafka broker. `kafka_producer.py`/`kafka_consumer.py` are
provided as an optional, corrected upgrade path (the original spec had typos
like `KfkaConsumer`, `KafkaComsumer`, and `json/loads` that don't run at all).

**22. How does the Streamlit dashboard stay fast with a growing database?**
Data-loading functions in `dashboard/components.py` use
`@st.cache_data(ttl=30)` (or `ttl=300` for model artifacts), so repeated
navigation between pages doesn't re-query SQLite on every interaction.

**23. What's on the "Transaction Investigation" dashboard page, and why does it fall back to a live prediction?**
It looks up a transaction by ID and shows its stored prediction if one exists
in the `predictions` table. If a transaction has never been scored (e.g. it
wasn't part of the simulated stream), a button runs `predict_transaction()`
live so the page still returns a useful answer.

**24. How would you extend this project to detect fraud rings (multiple related accounts)?**
The original spec's "graph-based analysis" idea was scoped out of core v1
intentionally to avoid over-engineering, but `Customer_ID` and shared attributes
(`Location`, `Device_Type`, `Merchant_Category`) could feed a graph library
(e.g. NetworkX) to build a transaction graph and detect densely connected
clusters of accounts, which is a natural "Future Enhancements" item.

**25. What are the main limitations of the current system?**
(1) 5,000 rows is small for production-grade ML; (2) Precision is ~0.32, so
roughly 2 of every 3 flagged transactions are false positives, meaning a real
deployment would need a human review step; (3) SQLite and the Streamlit
dashboard are appropriate for a portfolio project but not for high-throughput
production traffic; (4) the dataset doesn't include truly real-time streaming
signals (e.g. device fingerprinting), so results reflect what's actually in
the given data, not a guarantee of real-world performance.

**26. Why does the confusion matrix / precision number matter more than the headline accuracy when presenting this project?**
Because a stakeholder skimming "88.9% accuracy" (Random Forest) could wrongly
assume the model is excellent, when in fact it only catches ~40% of fraud
(Recall 0.3958) at that setting. Reporting Precision/Recall/F1 alongside
accuracy prevents that kind of misleading takeaway — which is exactly why the
dashboard's Model Performance page shows all five metrics together, not just
accuracy.

**27. How is the project structured to avoid hard-coded paths and secrets?**
`src/config.py` centralizes every path (using `pathlib`), threshold, and
setting; `.env`/`python-dotenv` supplies secrets and environment-specific
values; `.gitignore` excludes `.env`, `*.db`, and `__pycache__/` from version
control.
