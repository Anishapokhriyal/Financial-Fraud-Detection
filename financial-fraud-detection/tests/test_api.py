"""Tests for api/main.py"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.database import get_connection, init_database, load_transactions_from_dataframe

client = TestClient(app)


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()


def test_predict_endpoint_valid_transaction():
    payload = {
        "Transaction_ID": "TX_PYTEST_001",
        "Customer_ID": "CUST0001",
        "Transaction_Date": "2024-05-01 12:00",
        "Transaction_Amount": 150.0,
        "Merchant_Category": "Food",
        "Payment_Method": "UPI",
        "Device_Type": "Mobile",
        "Location": "Delhi",
        "Is_International": 0,
        "Previous_Transactions": 50,
        "Average_Spend": 140.0,
        "Account_Age_Days": 800,
        "Suspicious_Keyword": "No",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "TX_PYTEST_001"
    assert 0 <= body["risk_score"] <= 100


def test_predict_endpoint_rejects_invalid_payload():
    r = client.post("/predict", json={"Transaction_ID": "TX_BAD"})
    assert r.status_code == 422  # Pydantic validation error


def test_transactions_endpoint():
    r = client.get("/transactions?limit=5")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "model_name" in r.json()


def test_load_transactions_clears_dependent_records_before_reload():
    init_database()
    row = {
        "Transaction_ID": "TX_DB_RELOAD_001",
        "Customer_ID": "CUST0001",
        "Transaction_Date": "2024-05-01 12:00",
        "Transaction_Amount": 150.0,
        "Merchant_Category": "Food",
        "Payment_Method": "UPI",
        "Device_Type": "Mobile",
        "Location": "Delhi",
        "Is_International": 0,
        "Previous_Transactions": 50,
        "Average_Spend": 140.0,
        "Account_Age_Days": 800,
        "Suspicious_Keyword": "No",
        "Fraudulent": 0,
    }
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO transactions (transaction_id, customer_id, transaction_date, transaction_amount, merchant_category, payment_method, device_type, location, is_international, previous_transactions, average_spend, account_age_days, suspicious_keyword, fraudulent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["Transaction_ID"], row["Customer_ID"], row["Transaction_Date"],
                row["Transaction_Amount"], row["Merchant_Category"], row["Payment_Method"],
                row["Device_Type"], row["Location"], row["Is_International"],
                row["Previous_Transactions"], row["Average_Spend"], row["Account_Age_Days"],
                row["Suspicious_Keyword"], row["Fraudulent"],
            ),
        )
        conn.execute(
            "INSERT INTO predictions (transaction_id, fraud_probability, risk_score, risk_level, predicted_class, model_version) VALUES (?, ?, ?, ?, ?, ?)",
            (row["Transaction_ID"], 0.6, 60, "High", 1, "1.0"),
        )
        conn.execute(
            "INSERT INTO alerts (transaction_id, risk_level, message, status) VALUES (?, ?, ?, ?)",
            (row["Transaction_ID"], "High", "test alert", "OPEN"),
        )

    n = load_transactions_from_dataframe(pd.DataFrame([row]))
    assert n == 1

    with get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0
