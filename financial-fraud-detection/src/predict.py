"""
predict.py
----------
Prediction engine: loads the trained model + preprocessor and scores a
single transaction (dict or Series), returning fraud prediction, probability,
risk score, and risk level.

Unknown categorical values at inference time are handled safely because the
saved preprocessor's OneHotEncoder was fit with handle_unknown='ignore'.
"""
from __future__ import annotations

import joblib
import pandas as pd
import json

from src import config
from src.feature_engineering import engineer_features, get_model_feature_columns
from src.risk_scoring import calculate_risk
from src.utils import get_logger

logger = get_logger(__name__)

_model = None
_preprocessor = None
_metadata = None


def _load_artifacts():
    """Lazy-load model artifacts once per process."""
    global _model, _preprocessor, _metadata
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {config.MODEL_PATH}. Run: python -m src.train_model"
            )
        _model = joblib.load(config.MODEL_PATH)
        _preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        _metadata = json.loads(config.MODEL_METADATA_PATH.read_text())
        logger.info(f"Loaded model '{_metadata['model_name']}' v{_metadata['model_version']}")
    return _model, _preprocessor, _metadata


def predict_transaction(transaction: dict) -> dict:
    """
    Score a single transaction dict (raw column names, e.g. Transaction_ID,
    Customer_ID, Transaction_Date, Transaction_Amount, Merchant_Category,
    Payment_Method, Device_Type, Location, Is_International,
    Previous_Transactions, Average_Spend, Account_Age_Days,
    Suspicious_Keyword) and return a structured prediction result.
    """
    model, preprocessor, metadata = _load_artifacts()

    df = pd.DataFrame([transaction])
    df = engineer_features(df)

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    # Ensure every expected column exists (fill safely if missing from input)
    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0
    for col in categorical_features:
        if col not in df.columns:
            df[col] = "Unknown"

    X = df[numeric_features + categorical_features]

    try:
        X_proc = preprocessor.transform(X)
    except Exception as exc:
        logger.error(f"Preprocessing failed for transaction, using safe fallback: {exc}")
        raise ValueError(f"Invalid transaction payload: {exc}")

    fraud_probability = float(model.predict_proba(X_proc)[:, 1][0])
    predicted_class = int(fraud_probability >= 0.5)

    risk = calculate_risk(fraud_probability)

    result = {
        "transaction_id": transaction.get("Transaction_ID", "UNKNOWN"),
        "is_fraud": predicted_class,
        "fraud_probability": risk["fraud_probability"],
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "model_version": metadata["model_version"],
    }
    return result


if __name__ == "__main__":
    sample_transaction = {
        "Transaction_ID": "TX_DEMO_001",
        "Customer_ID": "CUST9999",
        "Transaction_Date": "2024-05-01 23:45",
        "Transaction_Amount": 4800.0,
        "Merchant_Category": "Electronics",
        "Payment_Method": "Credit Card",
        "Device_Type": "Mobile",
        "Location": "Mumbai",
        "Is_International": 1,
        "Previous_Transactions": 2,
        "Average_Spend": 120.0,
        "Account_Age_Days": 5,
        "Suspicious_Keyword": "Yes",
    }
    print(predict_transaction(sample_transaction))
