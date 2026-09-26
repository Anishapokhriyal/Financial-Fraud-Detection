"""Real-Time Monitor page: latest simulated/live transactions with risk highlighting."""
import streamlit as st
import pandas as pd


def _highlight_risk(row):
    color_map = {
        "Critical": "background-color: #fed7d7",
        "High": "background-color: #feebc8",
        "Medium": "background-color: #fefcbf",
        "Low": "background-color: #c6f6d5",
    }
    style = color_map.get(row.get("risk_level"), "")
    return [style] * len(row)


def render(predictions: pd.DataFrame, transactions: pd.DataFrame):
    st.subheader("Real-Time Transaction Monitor")
    st.caption(
        "Shows the latest transactions scored by the model. Run "
        "`python -m streaming.simulator` to feed new live transactions into this view."
    )

    if predictions.empty:
        st.info("No predictions yet. Run the real-time simulator to populate this view.")
        return

    merged = predictions.merge(
        transactions[[c for c in ["transaction_id", "location", "device_type"] if c in transactions.columns]],
        on="transaction_id", how="left",
    )

    latest = merged.sort_values("prediction_timestamp", ascending=False).head(25)
    display_cols = [c for c in [
        "transaction_id", "fraud_probability", "risk_score", "risk_level",
        "location", "device_type", "prediction_timestamp",
    ] if c in latest.columns]

    styled = latest[display_cols].style.apply(_highlight_risk, axis=1)
    st.dataframe(styled, use_container_width=True)

    critical = latest[latest["risk_level"] == "Critical"]
    if not critical.empty:
        st.error(f"{len(critical)} critical-risk transaction(s) in the latest batch — investigate immediately.")
