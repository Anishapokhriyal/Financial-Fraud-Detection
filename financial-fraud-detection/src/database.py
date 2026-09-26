"""
database.py
------------
SQLite database access layer. Uses a real schema (database/schema.sql)
instead of the broken `cursor.execute("")` placeholder from the original
project specification.
"""
from __future__ import annotations

import sqlite3
import json
import pandas as pd
from contextlib import contextmanager
from typing import Optional

from src import config
from src.predict import predict_transaction
from src.utils import get_logger

logger = get_logger(__name__)


@contextmanager
def get_connection():
    """Context-managed SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(str(config.DATABASE_PATH))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """Create all tables/indexes from schema.sql if they don't already exist."""
    schema_sql = config.SCHEMA_PATH.read_text()
    with get_connection() as conn:
        conn.executescript(schema_sql)
    logger.info(f"Database initialized at {config.DATABASE_PATH}")


def load_transactions_from_dataframe(df: pd.DataFrame) -> int:
    """Bulk-load processed transactions into the transactions table."""
    records = df.rename(
        columns={
            "Transaction_ID": "transaction_id",
            "Customer_ID": "customer_id",
            "Transaction_Date": "transaction_date",
            "Transaction_Amount": "transaction_amount",
            "Merchant_Category": "merchant_category",
            "Payment_Method": "payment_method",
            "Device_Type": "device_type",
            "Location": "location",
            "Is_International": "is_international",
            "Previous_Transactions": "previous_transactions",
            "Average_Spend": "average_spend",
            "Account_Age_Days": "account_age_days",
            "Suspicious_Keyword": "suspicious_keyword",
            "Fraudulent": "fraudulent",
        }
    )
    columns = [
        "transaction_id", "customer_id", "transaction_date", "transaction_amount",
        "merchant_category", "payment_method", "device_type", "location",
        "is_international", "previous_transactions", "average_spend",
        "account_age_days", "suspicious_keyword", "fraudulent",
    ]
    records = records[[c for c in columns if c in records.columns]]
    records["transaction_date"] = records["transaction_date"].astype(str)

    # Insert rows through the schema created by schema.sql (preserves the
    # PRIMARY KEY / FOREIGN KEY / created_at DEFAULT definitions). Using
    # `to_sql(if_exists="replace")` here would silently drop that schema, which
    # previously broke the predictions/alerts foreign keys -- fixed below.
    placeholders = ", ".join(["?"] * len(columns))
    col_list = ", ".join(columns)
    rows = [tuple(r) for r in records[columns].itertuples(index=False, name=None)]

    with get_connection() as conn:
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM transactions")
        conn.executemany(
            f"INSERT OR REPLACE INTO transactions ({col_list}) VALUES ({placeholders})",
            rows,
        )

    logger.info(f"Loaded {len(records)} transactions into the database")
    return len(records)


def upsert_transaction(transaction: dict) -> None:
    """Insert (or replace) a single transaction, e.g. one submitted live via the API,
    so that predictions/alerts can safely reference it via foreign key."""
    mapping = {
        "Transaction_ID": "transaction_id", "Customer_ID": "customer_id",
        "Transaction_Date": "transaction_date", "Transaction_Amount": "transaction_amount",
        "Merchant_Category": "merchant_category", "Payment_Method": "payment_method",
        "Device_Type": "device_type", "Location": "location",
        "Is_International": "is_international", "Previous_Transactions": "previous_transactions",
        "Average_Spend": "average_spend", "Account_Age_Days": "account_age_days",
        "Suspicious_Keyword": "suspicious_keyword", "Fraudulent": "fraudulent",
    }
    row = {mapping[k]: v for k, v in transaction.items() if k in mapping}
    row.setdefault("fraudulent", None)
    columns = list(row.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_list = ", ".join(columns)
    with get_connection() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO transactions ({col_list}) VALUES ({placeholders})",
            tuple(row.values()),
        )


def insert_prediction(transaction_id: str, fraud_probability: float, risk_score: int,
                       risk_level: str, predicted_class: int, model_version: str) -> int:
    """Insert a single prediction row and return its prediction_id."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (transaction_id, fraud_probability, risk_score, risk_level, predicted_class, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (transaction_id, fraud_probability, risk_score, risk_level, predicted_class, model_version),
        )
        return cursor.lastrowid


def insert_alert(transaction_id: str, risk_level: str, message: str, status: str = "OPEN") -> int:
    """Insert an alert row and return its alert_id."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO alerts (transaction_id, risk_level, message, status) VALUES (?, ?, ?, ?)",
            (transaction_id, risk_level, message, status),
        )
        return cursor.lastrowid


def insert_model_version(model_name: str, version: str, metrics: dict) -> int:
    """Record a trained model version and its metrics."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO model_versions (model_name, version, metrics) VALUES (?, ?, ?)",
            (model_name, version, json.dumps(metrics)),
        )
        return cursor.lastrowid


def get_transactions(limit: int = 100, fraud_only: bool = False) -> pd.DataFrame:
    """Fetch transactions, optionally filtered to fraudulent-only."""
    query = "SELECT * FROM transactions"
    if fraud_only:
        query += " WHERE fraudulent = 1"
    query += " ORDER BY created_at DESC LIMIT ?"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=(limit,))


def backfill_predictions_from_transactions(limit: int | None = None) -> int:
    """Generate predictions for all transactions if none exist yet."""
    with get_connection() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        if existing > 0:
            return existing

        query = "SELECT * FROM transactions"
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        query += " ORDER BY created_at DESC"
        rows = pd.read_sql_query(query, conn, params=params)

    if rows.empty:
        return 0

    count = 0
    for _, row in rows.iterrows():
        payload = {
            "Transaction_ID": row.get("transaction_id"),
            "Customer_ID": row.get("customer_id"),
            "Transaction_Date": row.get("transaction_date"),
            "Transaction_Amount": row.get("transaction_amount"),
            "Merchant_Category": row.get("merchant_category"),
            "Payment_Method": row.get("payment_method"),
            "Device_Type": row.get("device_type"),
            "Location": row.get("location"),
            "Is_International": row.get("is_international"),
            "Previous_Transactions": row.get("previous_transactions"),
            "Average_Spend": row.get("average_spend"),
            "Account_Age_Days": row.get("account_age_days"),
            "Suspicious_Keyword": row.get("suspicious_keyword"),
        }
        result = predict_transaction(payload)
        insert_prediction(
            transaction_id=result["transaction_id"],
            fraud_probability=result["fraud_probability"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            predicted_class=result["is_fraud"],
            model_version=result["model_version"],
        )
        count += 1

    logger.info(f"Backfilled {count} predictions from existing transactions")
    return count


def get_alerts(status: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
    """Fetch alerts, optionally filtered by status."""
    query = "SELECT * FROM alerts"
    params: tuple = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY created_at DESC LIMIT ?"
    params = params + (limit,)
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_predictions(limit: int = 100) -> pd.DataFrame:
    """Fetch recent predictions."""
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY prediction_timestamp DESC LIMIT ?",
            conn, params=(limit,),
        )


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully.")
