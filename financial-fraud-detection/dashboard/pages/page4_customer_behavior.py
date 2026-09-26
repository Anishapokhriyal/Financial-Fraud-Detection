"""Customer Behavior page: per-customer spending and risk patterns."""
import streamlit as st
import plotly.express as px
import pandas as pd


def render(df: pd.DataFrame):
    st.subheader("Customer Behavior")

    if "customer_id" not in df.columns:
        st.info("No customer_id column available for customer-level analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="previous_transactions", nbins=30,
                            title="Transaction Frequency Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="average_spend", nbins=30, title="Average Spend Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        df_dev = df.copy()
        df_dev["amount_deviation"] = (df_dev["transaction_amount"] - df_dev["average_spend"]).abs()
        fig = px.histogram(df_dev, x="amount_deviation", nbins=30,
                            title="Amount Deviation from Customer Average")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(df, x="account_age_days", nbins=30, title="Account Age Distribution (days)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Customer Lookup")
    customer_ids = sorted(df["customer_id"].dropna().unique().tolist())
    selected = st.selectbox("Select a customer", ["-- choose --"] + customer_ids)
    if selected != "-- choose --":
        cust_df = df[df["customer_id"] == selected]
        st.write(f"**{len(cust_df)} transactions** for customer `{selected}`")
        st.dataframe(cust_df[[c for c in [
            "transaction_id", "transaction_date", "transaction_amount",
            "merchant_category", "location", "fraudulent"
        ] if c in cust_df.columns]], use_container_width=True)
