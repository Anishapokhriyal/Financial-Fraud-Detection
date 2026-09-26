"""
config.py
---------
Centralized configuration for the Financial Fraud Detection project.
All paths, thresholds, and settings live here so nothing is hard-coded
elsewhere in the codebase.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Base paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, DATABASE_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset configuration
# --------------------------------------------------------------------------
# Primary dataset selected after inspecting all uploaded files (see README /
# PROJECT_REPORT for the full justification). financial_fraud_detection_dataset.csv
# was chosen because it has zero missing values, zero duplicates, a realistic
# ~9.6% fraud rate, and the richest set of behavioral columns
# (Previous_Transactions, Average_Spend, Account_Age_Days, Is_International,
# Suspicious_Keyword) needed for behavioral fraud-detection features.
RAW_DATASET_PATH = RAW_DATA_DIR / "financial_fraud_detection_dataset.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "processed_transactions.csv"

TARGET_COLUMN = "Fraudulent"
TRANSACTION_ID_COLUMN = "Transaction_ID"
CUSTOMER_ID_COLUMN = "Customer_ID"
DATE_COLUMN = "Transaction_Date"
AMOUNT_COLUMN = "Transaction_Amount"

CATEGORICAL_COLUMNS = [
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Suspicious_Keyword",
]

NUMERIC_COLUMNS = [
    "Transaction_Amount",
    "Is_International",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
]

# --------------------------------------------------------------------------
# Model / training configuration
# --------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2

MODEL_PATH = MODELS_DIR / "best_fraud_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

MODEL_VERSION = "1.0"

# Metric used to automatically pick the "best" model. Chosen because fraud
# datasets are imbalanced and F1 balances precision (avoid false alarms)
# against recall (avoid missed fraud).
MODEL_SELECTION_METRIC = "F1"

# --------------------------------------------------------------------------
# Risk scoring configuration (fully configurable thresholds)
# --------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "LOW": (0, 30),
    "MEDIUM": (31, 60),
    "HIGH": (61, 80),
    "CRITICAL": (81, 100),
}

# --------------------------------------------------------------------------
# Alerting configuration
# --------------------------------------------------------------------------
ALERT_RISK_SCORE_THRESHOLD = int(os.getenv("ALERT_RISK_SCORE_THRESHOLD", "61"))
ALERT_EMAIL_ENABLED = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# --------------------------------------------------------------------------
# Database configuration
# --------------------------------------------------------------------------
DATABASE_PATH = DATABASE_DIR / "fraud_detection.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"

# --------------------------------------------------------------------------
# API configuration
# --------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Financial Fraud Detection API"

# --------------------------------------------------------------------------
# Dashboard configuration
# --------------------------------------------------------------------------
DASHBOARD_TITLE = "Financial Fraud Intelligence Platform"

# --------------------------------------------------------------------------
# Real-time simulation configuration
# --------------------------------------------------------------------------
SIMULATOR_DELAY_SECONDS = float(os.getenv("SIMULATOR_DELAY_SECONDS", "1.0"))

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "fraud_detection.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
