"""
dashboard/app.py
-----------------
Financial Fraud Intelligence Platform - main Streamlit entry point.

Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config
from dashboard.components import (
    load_transactions, load_predictions, load_alerts,
    load_model_metadata, load_model_comparison, load_feature_importance,
    apply_sidebar_filters,
)
from dashboard.pages import (
    page1_overview, page2_fraud_analysis, page3_risk_analysis,
    page4_customer_behavior, page5_realtime_monitor,
    page6_model_performance, page7_investigation,
)

st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b1020 0%, #111827 40%, #0f172a 100%);
            color: #e5eefb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4, h5 {
            color: #f8fbff;
        }
        .stSidebar > div {
            background: rgba(15, 23, 42, 0.9);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        [data-testid="stMetricValue"] {
            color: #86efac;
            font-weight: 700;
        }
        .stButton > button {
            background: linear-gradient(135deg, #22c55e, #10b981);
            color: white;
            border: none;
            border-radius: 0.6rem;
            font-weight: 600;
        }
        .stTextInput input, .stNumberInput input, .stSelectbox select, .stDateInput input {
            background: rgba(15, 23, 42, 0.8);
            color: #e2e8f0;
        }
        div[data-testid="stForm"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 1rem;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"🛡️ {config.DASHBOARD_TITLE}")
st.caption("Real-time transaction risk monitoring, model analytics, and fraud investigation.")

PAGES = {
    "1. Executive Overview": page1_overview,
    "2. Fraud Analysis": page2_fraud_analysis,
    "3. Risk Analysis": page3_risk_analysis,
    "4. Customer Behavior": page4_customer_behavior,
    "5. Real-Time Monitor": page5_realtime_monitor,
    "6. Model Performance": page6_model_performance,
    "7. Transaction Investigation": page7_investigation,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")

transactions = load_transactions()
predictions = load_predictions()
alerts = load_alerts()
metadata = load_model_metadata()
comparison = load_model_comparison()
feature_importance = load_feature_importance()

filtered_transactions = apply_sidebar_filters(transactions)

if selection == "1. Executive Overview":
    page1_overview.render(filtered_transactions, predictions)
elif selection == "2. Fraud Analysis":
    page2_fraud_analysis.render(filtered_transactions)
elif selection == "3. Risk Analysis":
    page3_risk_analysis.render(predictions)
elif selection == "4. Customer Behavior":
    page4_customer_behavior.render(filtered_transactions)
elif selection == "5. Real-Time Monitor":
    page5_realtime_monitor.render(predictions, transactions)
elif selection == "6. Model Performance":
    page6_model_performance.render(metadata, comparison, feature_importance)
elif selection == "7. Transaction Investigation":
    page7_investigation.render(transactions, predictions)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model version: {metadata.get('model_version', 'N/A')}")
st.sidebar.caption(f"Total transactions: {len(transactions):,}")
