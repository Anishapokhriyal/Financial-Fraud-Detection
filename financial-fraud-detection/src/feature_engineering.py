"""
feature_engineering.py
-----------------------
Builds fraud-relevant behavioral features from the ACTUAL columns present in
financial_fraud_detection_dataset.csv. Only features supported by real
columns are created (no invented columns).

Engineered features and rationale:
- transaction_hour            : hour of day extracted from Transaction_Date.
                                 Fraud often clusters at unusual hours.
- transaction_dayofweek       : day of week (0=Mon..6=Sun).
- is_weekend                  : weekend transactions can carry different risk.
- amount_to_avg_ratio         : Transaction_Amount / Average_Spend. A large
                                 deviation from a customer's historical
                                 average spend is a classic fraud signal.
- amount_deviation            : absolute difference between the transaction
                                 amount and the customer's average spend.
- log_transaction_amount      : log1p transform to reduce skew for models
                                 sensitive to scale (Logistic Regression).
- customer_txn_frequency      : how many prior transactions this customer has
                                 made (directly available as Previous_Transactions,
                                 renamed/kept for clarity).
- account_age_years           : Account_Age_Days / 365, easier to interpret.
- is_new_account              : flag for accounts younger than 30 days
                                 (new accounts are higher fraud risk).
- suspicious_keyword_flag     : binary encoding of Suspicious_Keyword (Yes/No).
- is_international            : already binary in the raw data, kept as-is.

No target leakage: none of these features use the Fraudulent column, and none
are derived from post-transaction/outcome information.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

ENGINEERED_FEATURE_DOCS = {
    "transaction_hour": "Hour of day (0-23) extracted from Transaction_Date.",
    "transaction_dayofweek": "Day of week (0=Mon .. 6=Sun) extracted from Transaction_Date.",
    "is_weekend": "1 if the transaction occurred on Saturday/Sunday, else 0.",
    "amount_to_avg_ratio": "Transaction_Amount divided by the customer's Average_Spend.",
    "amount_deviation": "Absolute difference between Transaction_Amount and Average_Spend.",
    "log_transaction_amount": "log1p(Transaction_Amount), reduces right-skew for linear models.",
    "account_age_years": "Account_Age_Days / 365.",
    "is_new_account": "1 if Account_Age_Days < 30, else 0.",
    "suspicious_keyword_flag": "Binary encoding of Suspicious_Keyword (Yes=1, No=0).",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioral fraud-detection features to a cleaned transaction dataframe."""
    df = df.copy()

    if config.DATE_COLUMN in df.columns:
        dt = pd.to_datetime(df[config.DATE_COLUMN], errors="coerce")
        df["transaction_hour"] = dt.dt.hour
        df["transaction_dayofweek"] = dt.dt.dayofweek
        df["is_weekend"] = (df["transaction_dayofweek"] >= 5).astype(int)

    if "Average_Spend" in df.columns and config.AMOUNT_COLUMN in df.columns:
        safe_avg = df["Average_Spend"].replace(0, np.nan)
        df["amount_to_avg_ratio"] = (df[config.AMOUNT_COLUMN] / safe_avg).fillna(0)
        df["amount_deviation"] = (df[config.AMOUNT_COLUMN] - df["Average_Spend"]).abs()

    if config.AMOUNT_COLUMN in df.columns:
        df["log_transaction_amount"] = np.log1p(df[config.AMOUNT_COLUMN])

    if "Account_Age_Days" in df.columns:
        df["account_age_years"] = df["Account_Age_Days"] / 365.0
        df["is_new_account"] = (df["Account_Age_Days"] < 30).astype(int)

    if "Suspicious_Keyword" in df.columns:
        df["suspicious_keyword_flag"] = (
            df["Suspicious_Keyword"].astype(str).str.strip().str.lower().eq("yes").astype(int)
        )

    logger.info(f"Engineered {len(ENGINEERED_FEATURE_DOCS)} new behavioral features")
    return df


def get_model_feature_columns() -> tuple[list, list]:
    """Return (numeric_features, categorical_features) used for model training."""
    numeric_features = [
        config.AMOUNT_COLUMN,
        "Is_International",
        "Previous_Transactions",
        "Average_Spend",
        "Account_Age_Days",
        "transaction_hour",
        "transaction_dayofweek",
        "is_weekend",
        "amount_to_avg_ratio",
        "amount_deviation",
        "log_transaction_amount",
        "account_age_years",
        "is_new_account",
        "suspicious_keyword_flag",
    ]
    categorical_features = [
        "Merchant_Category",
        "Payment_Method",
        "Device_Type",
        "Location",
    ]
    return numeric_features, categorical_features


if __name__ == "__main__":
    from src.preprocessing import run_etl

    data = run_etl(save=False)
    data = engineer_features(data)
    print(data.head())
    print(f"\nColumns after feature engineering: {list(data.columns)}")
