"""Tests for src/predict.py"""
import pytest
from src.predict import predict_transaction

SAMPLE_TRANSACTION = {
    "Transaction_ID": "TX_TEST_001",
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


def test_predict_transaction_output_keys():
    result = predict_transaction(SAMPLE_TRANSACTION)
    expected_keys = {
        "transaction_id", "is_fraud", "fraud_probability",
        "risk_score", "risk_level", "model_version",
    }
    assert expected_keys.issubset(result.keys())


def test_predict_transaction_probability_range():
    result = predict_transaction(SAMPLE_TRANSACTION)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert 0 <= result["risk_score"] <= 100


def test_predict_transaction_handles_unknown_category():
    """Unseen categorical values must not crash the prediction (handle_unknown='ignore')."""
    txn = dict(SAMPLE_TRANSACTION)
    txn["Merchant_Category"] = "SomeBrandNewCategoryNeverSeenBefore"
    txn["Location"] = "AtlantisCity"
    result = predict_transaction(txn)
    assert result["transaction_id"] == "TX_TEST_001"
