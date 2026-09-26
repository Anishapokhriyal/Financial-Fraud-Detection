"""Fraud Analysis page: fraud breakdown by categorical dimensions."""
import streamlit as st
import plotly.express as px
import pandas as pd


def _fraud_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    grouped = df.groupby(col)["fraudulent"].agg(["sum", "count"]).reset_index()
    grouped.columns = [col, "fraud_count", "total_count"]
    grouped["fraud_rate_%"] = (grouped["fraud_count"] / grouped["total_count"] * 100).round(2)
    return grouped.sort_values("fraud_rate_%", ascending=False)


def render(df: pd.DataFrame):
    st.subheader("Fraud Analysis")

    if "fraudulent" not in df.columns:
        st.warning("No fraud label available in the loaded data.")
        return

    dims = [
        ("payment_method", "Payment Method"),
        ("merchant_category", "Merchant Category"),
        ("device_type", "Device Type"),
        ("location", "Location"),
    ]

    for i in range(0, len(dims), 2):
        cols = st.columns(2)
        for j, (col_name, label) in enumerate(dims[i:i + 2]):
            if col_name in df.columns:
                data = _fraud_rate_by(df, col_name)
                fig = px.bar(data, x=col_name, y="fraud_rate_%",
                             title=f"Fraud Rate by {label}", color="fraud_rate_%",
                             color_continuous_scale="Reds")
                cols[j].plotly_chart(fig, use_container_width=True)

    if "is_international" in df.columns:
        intl = df.copy()
        intl["International"] = intl["is_international"].map({1: "International", 0: "Domestic"})
        data = _fraud_rate_by(intl, "International")
        fig = px.bar(data, x="International", y="fraud_rate_%", title="Fraud Rate: International vs Domestic",
                     color="fraud_rate_%", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
