"""
preprocessing.py
-----------------
Data validation and ETL (Extract, Transform, Load) pipeline for the
financial_fraud_detection_dataset.csv dataset.

Fixes applied vs. the original project specification:
- No `cursor.execute("")` placeholder SQL; a real schema is used (see database.py).
- Robust, explicit validation with a documented data-quality report instead of
  silent assumptions.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------
def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw transaction CSV file."""
    path = path or config.RAW_DATASET_PATH
    logger.info(f"Loading raw dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@dataclass
class DataQualityReport:
    total_rows: int = 0
    total_columns: int = 0
    missing_values: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_transaction_ids: int = 0
    negative_amounts: int = 0
    zero_or_negative_account_age: int = 0
    invalid_timestamps: int = 0
    unexpected_categorical_values: dict = field(default_factory=dict)
    class_distribution: dict = field(default_factory=dict)
    class_imbalance_ratio: float = 0.0
    potential_leakage_columns: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


EXPECTED_CATEGORIES = {
    "Payment_Method": {"Credit Card", "Debit Card", "PayPal", "NetBanking", "UPI"},
    "Device_Type": {"POS", "Mobile", "Desktop"},
    "Suspicious_Keyword": {"Yes", "No"},
}


def validate_data(df: pd.DataFrame) -> DataQualityReport:
    """Run a full validation pass and return a structured data-quality report."""
    report = DataQualityReport()
    report.total_rows = len(df)
    report.total_columns = len(df.columns)
    report.missing_values = df.isnull().sum().to_dict()
    report.duplicate_rows = int(df.duplicated().sum())

    if config.TRANSACTION_ID_COLUMN in df.columns:
        report.duplicate_transaction_ids = int(
            df[config.TRANSACTION_ID_COLUMN].duplicated().sum()
        )

    if config.AMOUNT_COLUMN in df.columns:
        report.negative_amounts = int((df[config.AMOUNT_COLUMN] < 0).sum())

    if "Account_Age_Days" in df.columns:
        report.zero_or_negative_account_age = int((df["Account_Age_Days"] <= 0).sum())

    if config.DATE_COLUMN in df.columns:
        parsed = pd.to_datetime(df[config.DATE_COLUMN], errors="coerce", dayfirst=True)
        report.invalid_timestamps = int(parsed.isnull().sum())

    for col, expected in EXPECTED_CATEGORIES.items():
        if col in df.columns:
            actual = set(df[col].dropna().unique())
            unexpected = actual - expected
            if unexpected:
                report.unexpected_categorical_values[col] = list(unexpected)

    if config.TARGET_COLUMN in df.columns:
        dist = df[config.TARGET_COLUMN].value_counts().to_dict()
        report.class_distribution = dist
        if dist.get(1, 0) > 0:
            report.class_imbalance_ratio = round(dist.get(0, 0) / dist.get(1, 1), 2)

    # Data-leakage check: flag columns that are (near) perfectly correlated
    # with the target, which would indicate the model is "cheating".
    numeric_df = df.select_dtypes(include=[np.number])
    if config.TARGET_COLUMN in numeric_df.columns:
        corr = numeric_df.corr()[config.TARGET_COLUMN].abs().sort_values(ascending=False)
        suspicious = corr[(corr > 0.9) & (corr.index != config.TARGET_COLUMN)]
        report.potential_leakage_columns = list(suspicious.index)

    logger.info(f"Data quality report generated: {report.to_dict()}")
    return report


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns, handle duplicates/missing values, and fix data types."""
    df = df.copy()

    # Standardize column names (strip whitespace only; names are already clean)
    df.columns = [c.strip() for c in df.columns]

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        logger.info(f"Dropped {before - len(df)} duplicate rows")

    # Drop rows with duplicate transaction IDs (keep first occurrence)
    if config.TRANSACTION_ID_COLUMN in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[config.TRANSACTION_ID_COLUMN], keep="first")
        if before != len(df):
            logger.info(f"Dropped {before - len(df)} duplicate transaction IDs")

    # Handle missing values: numeric -> median, categorical -> mode
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Remove invalid (negative) transaction amounts
    if config.AMOUNT_COLUMN in df.columns:
        before = len(df)
        df = df[df[config.AMOUNT_COLUMN] >= 0]
        if before != len(df):
            logger.info(f"Removed {before - len(df)} rows with negative amounts")

    # Convert timestamp column
    if config.DATE_COLUMN in df.columns:
        df[config.DATE_COLUMN] = pd.to_datetime(
            df[config.DATE_COLUMN], errors="coerce", dayfirst=True
        )
        df = df.dropna(subset=[config.DATE_COLUMN])

    df = df.reset_index(drop=True)
    return df


def run_etl(save: bool = True) -> pd.DataFrame:
    """Full Extract -> Validate -> Transform -> Load pipeline."""
    df = load_raw_data()
    validate_data(df)
    df = clean_data(df)

    if save:
        config.PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.PROCESSED_DATASET_PATH, index=False)
        logger.info(f"Saved processed data to {config.PROCESSED_DATASET_PATH}")

    return df


if __name__ == "__main__":
    data = run_etl()
    print(data.head())
    print(f"\nFinal shape after ETL: {data.shape}")
