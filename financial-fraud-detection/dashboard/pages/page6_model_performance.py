"""Model Performance page: metrics, confusion matrix, ROC/PR curves, feature importance."""
import streamlit as st
import pandas as pd
from pathlib import Path

from src import config


def render(metadata: dict, comparison: pd.DataFrame, feature_importance: pd.DataFrame):
    st.subheader("Model Performance")

    if not metadata:
        st.info("No trained model found. Run: python -m src.train_model")
        return

    st.markdown(f"**Best model:** `{metadata['model_name']}` (v{metadata['model_version']}) "
                f"— selected by highest **{metadata['selection_metric']}**")

    m = metadata["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", m["Precision"])
    c2.metric("Recall", m["Recall"])
    c3.metric("F1 Score", m["F1"])
    c4.metric("ROC-AUC", m["ROC_AUC"])
    c5.metric("PR-AUC", m["PR_AUC"])

    st.markdown("#### Model Comparison")
    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True)

    st.markdown("#### Diagnostic Charts")
    img_cols = st.columns(2)
    chart_files = [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("roc_curve.png", "ROC Curve"),
        ("precision_recall_curve.png", "Precision-Recall Curve"),
        ("feature_importance.png", "Feature Importance"),
    ]
    for i, (fname, title) in enumerate(chart_files):
        path = config.REPORTS_DIR / fname
        if path.exists():
            img_cols[i % 2].image(str(path), caption=title, use_container_width=True)

    if not feature_importance.empty:
        st.markdown("#### Top 10 Fraud-Related Features")
        st.dataframe(feature_importance.head(10), use_container_width=True)
