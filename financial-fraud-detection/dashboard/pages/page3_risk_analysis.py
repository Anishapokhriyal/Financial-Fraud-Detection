"""Risk Analysis page: risk level breakdown and score distribution."""
import streamlit as st
import plotly.express as px
import pandas as pd


def render(predictions: pd.DataFrame):
    st.subheader("Risk Analysis")

    if predictions.empty:
        st.info("No predictions have been generated yet. Run the training pipeline or the "
                 "real-time simulator to populate predictions.")
        return

    level_counts = predictions["risk_level"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"]
    ).fillna(0).reset_index()
    level_counts.columns = ["Risk Level", "Count"]

    c1, c2, c3, c4 = st.columns(4)
    for col, level in zip([c1, c2, c3, c4], ["Low", "Medium", "High", "Critical"]):
        count = int(level_counts.loc[level_counts["Risk Level"] == level, "Count"].values[0])
        col.metric(f"{level} Risk", f"{count:,}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(level_counts, names="Risk Level", values="Count", title="Risk Distribution",
                     color="Risk Level",
                     color_discrete_map={"Low": "#38a169", "Medium": "#d69e2e",
                                          "High": "#dd6b20", "Critical": "#e53e3e"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(predictions, x="risk_score", nbins=20, title="Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if "prediction_timestamp" in predictions.columns:
        trend = predictions.copy()
        trend["date"] = trend["prediction_timestamp"].dt.date
        daily_avg = trend.groupby("date")["risk_score"].mean().reset_index()
        fig = px.line(daily_avg, x="date", y="risk_score", title="Average Risk Score Trend")
        st.plotly_chart(fig, use_container_width=True)
