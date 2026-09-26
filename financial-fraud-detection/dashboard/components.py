"""
components.py
--------------
Reusable data-loading and UI helper functions shared across all dashboard pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import json

from src import config
from src.database import get_connection


@st.cache_data(ttl=30)
def load_transactions() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_predictions() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
    if df.empty:
        from src.database import backfill_predictions_from_transactions
        backfill_predictions_from_transactions()
        with get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM predictions", conn)
    if "prediction_timestamp" in df.columns:
        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_alerts() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM alerts", conn)
    return df


@st.cache_data(ttl=300)
def load_model_metadata() -> dict:
    if config.MODEL_METADATA_PATH.exists():
        return json.loads(config.MODEL_METADATA_PATH.read_text())
    return {}


@st.cache_data(ttl=300)
def load_model_comparison() -> pd.DataFrame:
    if config.MODEL_COMPARISON_PATH.exists():
        return pd.read_csv(config.MODEL_COMPARISON_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_feature_importance() -> pd.DataFrame:
    if config.FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(config.FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame()


def kpi_card(label: str, value: str, delta: str = None):
    st.metric(label, value, delta)


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render the shared sidebar filters and return the filtered transactions dataframe."""
    st.sidebar.markdown("### Filters")

    filtered = df.copy()

    if "transaction_date" in filtered.columns and filtered["transaction_date"].notna().any():
        min_date = filtered["transaction_date"].min().date()
        max_date = filtered["transaction_date"].max().date()
        date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                            min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            filtered = filtered[
                (filtered["transaction_date"].dt.date >= start)
                & (filtered["transaction_date"].dt.date <= end)
            ]

    def multiselect_filter(col, label):
        nonlocal filtered
        if col in filtered.columns:
            options = sorted(filtered[col].dropna().unique().tolist())
            selected = st.sidebar.multiselect(label, options, default=[])
            if selected:
                filtered = filtered[filtered[col].isin(selected)]

    multiselect_filter("location", "Location")
    multiselect_filter("device_type", "Device")
    multiselect_filter("payment_method", "Payment Method")
    multiselect_filter("merchant_category", "Merchant Category")

    if "fraudulent" in filtered.columns:
        fraud_choice = st.sidebar.selectbox("Fraud status", ["All", "Fraud only", "Genuine only"])
        if fraud_choice == "Fraud only":
            filtered = filtered[filtered["fraudulent"] == 1]
        elif fraud_choice == "Genuine only":
            filtered = filtered[filtered["fraudulent"] == 0]

    return filtered
