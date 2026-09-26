"""Executive Overview page: KPI cards + high-level trend charts."""
import streamlit as st
import plotly.express as px
import pandas as pd


def render(df: pd.DataFrame, predictions: pd.DataFrame):
    st.subheader("Executive Overview")

    total_txn = len(df)
    fraud_txn = int(df["fraudulent"].sum()) if "fraudulent" in df.columns else 0
    genuine_txn = total_txn - fraud_txn
    fraud_rate = (fraud_txn / total_txn * 100) if total_txn else 0
    total_amount = df["transaction_amount"].sum() if "transaction_amount" in df.columns else 0
    fraud_amount = (
        df.loc[df["fraudulent"] == 1, "transaction_amount"].sum()
        if "fraudulent" in df.columns else 0
    )
    avg_risk_score = predictions["risk_score"].mean() if not predictions.empty else 0
    critical_alerts = (
        int((predictions["risk_level"] == "Critical").sum()) if not predictions.empty else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total_txn:,}")
    c2.metric("Fraud Transactions", f"{fraud_txn:,}")
    c3.metric("Genuine Transactions", f"{genuine_txn:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Transaction Amount", f"${total_amount:,.0f}")
    c6.metric("Fraudulent Amount", f"${fraud_amount:,.0f}")
    c7.metric("Avg. Risk Score", f"{avg_risk_score:.1f}")
    c8.metric("Critical Alerts", f"{critical_alerts:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if "fraudulent" in df.columns:
            counts = df["fraudulent"].map({0: "Genuine", 1: "Fraud"}).value_counts().reset_index()
            counts.columns = ["Status", "Count"]
            fig = px.pie(counts, names="Status", values="Count", title="Fraud vs Genuine",
                         color="Status", color_discrete_map={"Genuine": "#2b6cb0", "Fraud": "#e53e3e"})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "transaction_date" in df.columns:
            trend = df.copy()
            trend["date"] = trend["transaction_date"].dt.date
            daily = trend.groupby("date").size().reset_index(name="Transactions")
            fig = px.line(daily, x="date", y="Transactions", title="Transaction Trend")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "transaction_date" in df.columns and "fraudulent" in df.columns:
            trend = df.copy()
            trend["date"] = trend["transaction_date"].dt.date
            fraud_daily = trend[trend["fraudulent"] == 1].groupby("date").size().reset_index(name="Fraud Count")
            fig = px.bar(fraud_daily, x="date", y="Fraud Count", title="Fraud Trend")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "transaction_date" in df.columns and "fraudulent" in df.columns:
            trend = df.copy()
            trend["date"] = trend["transaction_date"].dt.date
            fraud_amt_daily = trend[trend["fraudulent"] == 1].groupby("date")["transaction_amount"].sum().reset_index()
            fig = px.line(fraud_amt_daily, x="date", y="transaction_amount", title="Fraud Amount Trend")
            st.plotly_chart(fig, use_container_width=True)
