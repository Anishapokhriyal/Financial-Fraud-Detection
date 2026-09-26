"""Tests for src/preprocessing.py"""
import pandas as pd
from src.preprocessing import clean_data, validate_data, load_raw_data
from src import config


def test_load_raw_data():
    df = load_raw_data()
    assert len(df) > 0
    assert config.TARGET_COLUMN in df.columns


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "Transaction_ID": ["T1", "T1", "T2"],
        "Customer_ID": ["C1", "C1", "C2"],
        "Transaction_Date": ["01-01-2024 10:00", "01-01-2024 10:00", "02-01-2024 11:00"],
        "Transaction_Amount": [100.0, 100.0, 50.0],
        "Merchant_Category": ["Food", "Food", "Travel"],
        "Payment_Method": ["UPI", "UPI", "UPI"],
        "Device_Type": ["Mobile", "Mobile", "POS"],
        "Location": ["Delhi", "Delhi", "Mumbai"],
        "Is_International": [0, 0, 0],
        "Previous_Transactions": [5, 5, 10],
        "Average_Spend": [90.0, 90.0, 45.0],
        "Account_Age_Days": [100, 100, 200],
        "Suspicious_Keyword": ["No", "No", "No"],
        "Fraudulent": [0, 0, 0],
    })
    cleaned = clean_data(df)
    assert cleaned["Transaction_ID"].duplicated().sum() == 0
    assert len(cleaned) == 2


def test_clean_data_removes_negative_amounts():
    df = pd.DataFrame({
        "Transaction_ID": ["T1", "T2"],
        "Customer_ID": ["C1", "C2"],
        "Transaction_Date": ["01-01-2024 10:00", "02-01-2024 11:00"],
        "Transaction_Amount": [-50.0, 50.0],
        "Merchant_Category": ["Food", "Travel"],
        "Payment_Method": ["UPI", "UPI"],
        "Device_Type": ["Mobile", "POS"],
        "Location": ["Delhi", "Mumbai"],
        "Is_International": [0, 0],
        "Previous_Transactions": [5, 10],
        "Average_Spend": [90.0, 45.0],
        "Account_Age_Days": [100, 200],
        "Suspicious_Keyword": ["No", "No"],
        "Fraudulent": [0, 0],
    })
    cleaned = clean_data(df)
    assert (cleaned["Transaction_Amount"] >= 0).all()
    assert len(cleaned) == 1


def test_validate_data_report_structure():
    df = load_raw_data()
    report = validate_data(df)
    assert report.total_rows == len(df)
    assert "class_distribution" in report.to_dict()
