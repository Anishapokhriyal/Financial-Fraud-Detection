-- schema.sql
-- Financial Fraud Detection database schema.
-- Replaces the broken `cursor.execute("")` placeholder from the original spec
-- with a real, indexed relational schema.

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id          TEXT PRIMARY KEY,
    customer_id              TEXT,
    transaction_date         TEXT,
    transaction_amount       REAL,
    merchant_category        TEXT,
    payment_method            TEXT,
    device_type               TEXT,
    location                  TEXT,
    is_international          INTEGER,
    previous_transactions     INTEGER,
    average_spend             REAL,
    account_age_days          INTEGER,
    suspicious_keyword        TEXT,
    fraudulent                INTEGER,
    created_at                TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id        TEXT NOT NULL,
    fraud_probability     REAL NOT NULL,
    risk_score            INTEGER NOT NULL,
    risk_level            TEXT NOT NULL,
    predicted_class       INTEGER NOT NULL,
    model_version         TEXT NOT NULL,
    prediction_timestamp  TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT NOT NULL,
    risk_level      TEXT NOT NULL,
    message         TEXT,
    status          TEXT DEFAULT 'OPEN',
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name    TEXT NOT NULL,
    version       TEXT NOT NULL,
    metrics       TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud ON transactions(fraudulent);
CREATE INDEX IF NOT EXISTS idx_predictions_txn ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_alerts_txn ON alerts(transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
