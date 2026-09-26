"""Transaction Investigation page: look up a single transaction ID and score a new transaction."""
import streamlit as st
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.predict import predict_transaction


def _render_prediction_result(result: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
    c2.metric("Risk Score", result["risk_score"])
    c3.metric("Risk Level", result["risk_level"])

    st.progress(min_value=0, max_value=100, value=result["risk_score"])
    if result["risk_score"] >= 81:
        st.error("Critical alert: this transaction is highly suspicious and should be reviewed immediately.")
    elif result["risk_score"] >= 61:
        st.warning("High-risk transaction: validate the merchant, device, and customer behavior.")
    elif result["risk_score"] >= 31:
        st.info("Medium risk: monitor closely for behavioral anomalies or repeat activity.")
    else:
        st.success("Low risk: transaction appears consistent with normal customer behavior.")


def render(transactions: pd.DataFrame, predictions: pd.DataFrame):
    st.subheader("Transaction Investigation")
    st.caption("Investigate an existing transaction or score a new one with the live fraud model.")

    with st.form("new_transaction_prediction"):
        st.markdown("### Predict a new transaction")
        col1, col2 = st.columns(2)

        with col1:
            transaction_id = st.text_input("Transaction ID", value=f"TX_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}")
            customer_id = st.text_input("Customer ID", value="CUST_001")
            transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=10.0, value=250.0)
            average_spend = st.number_input("Average Spend", min_value=0.0, step=10.0, value=120.0)
            previous_transactions = st.number_input("Previous Transactions", min_value=0, step=1, value=12)
            account_age_days = st.number_input("Account Age (days)", min_value=0, step=1, value=270)

        with col2:
            transaction_date = st.date_input("Transaction Date", value=pd.Timestamp.today().date())
            transaction_time = st.time_input("Transaction Time", value=pd.Timestamp.now().time())
            merchant_category = st.selectbox("Merchant Category", ["Food", "Travel", "Utilities", "Entertainment", "Grocery", "Health", "Electronics", "Fashion"], index=0)
            payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "NetBanking", "UPI"], index=0)
            device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "POS"], index=0)
            location = st.selectbox("Location", ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune"], index=0)
            is_international = st.selectbox("International transaction?", [0, 1], index=0)
            suspicious_keyword = st.selectbox("Suspicious Keyword", ["No", "Yes"], index=0)

        submitted = st.form_submit_button("Run fraud prediction", use_container_width=True)

    if submitted:
        raw = {
            "Transaction_ID": transaction_id,
            "Customer_ID": customer_id,
            "Transaction_Date": f"{transaction_date} {transaction_time.strftime('%H:%M')}",
            "Transaction_Amount": float(transaction_amount),
            "Merchant_Category": merchant_category,
            "Payment_Method": payment_method,
            "Device_Type": device_type,
            "Location": location,
            "Is_International": int(is_international),
            "Previous_Transactions": int(previous_transactions),
            "Average_Spend": float(average_spend),
            "Account_Age_Days": int(account_age_days),
            "Suspicious_Keyword": suspicious_keyword,
        }
        result = predict_transaction(raw)
        st.success("Prediction complete")
        _render_prediction_result(result)

    st.markdown("---")
    txn_id = st.text_input("Enter an existing Transaction ID to investigate", key="existing_txn_lookup")

    if not txn_id:
        st.caption("Type a transaction ID above to review its stored details and historical assessment.")
        return

    txn_rows = transactions[transactions["transaction_id"] == txn_id]
    if txn_rows.empty:
        st.warning(f"Transaction '{txn_id}' not found in the database.")
        return

    txn = txn_rows.iloc[0]
    st.markdown("#### Transaction Details")
    st.json({k: (str(v) if not pd.isna(v) else None) for k, v in txn.to_dict().items()})

    pred_rows = predictions[predictions["transaction_id"] == txn_id]
    if not pred_rows.empty:
        pred = pred_rows.sort_values("prediction_timestamp", ascending=False).iloc[0]
        st.markdown("#### Stored Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Probability", f"{pred['fraud_probability']:.2%}")
        c2.metric("Risk Score", int(pred["risk_score"]))
        c3.metric("Risk Level", pred["risk_level"])
        st.caption(f"Predicted at {pred['prediction_timestamp']} using model v{pred['model_version']}")
    else:
        st.markdown("#### Live Assessment")
        if st.button("Run fraud prediction now", key="live_txn_prediction"):
            raw = {
                "Transaction_ID": txn["transaction_id"],
                "Customer_ID": txn.get("customer_id"),
                "Transaction_Date": str(txn.get("transaction_date")),
                "Transaction_Amount": txn.get("transaction_amount"),
                "Merchant_Category": txn.get("merchant_category"),
                "Payment_Method": txn.get("payment_method"),
                "Device_Type": txn.get("device_type"),
                "Location": txn.get("location"),
                "Is_International": txn.get("is_international"),
                "Previous_Transactions": txn.get("previous_transactions"),
                "Average_Spend": txn.get("average_spend"),
                "Account_Age_Days": txn.get("account_age_days"),
                "Suspicious_Keyword": txn.get("suspicious_keyword"),
            }
            result = predict_transaction(raw)
            _render_prediction_result(result)
