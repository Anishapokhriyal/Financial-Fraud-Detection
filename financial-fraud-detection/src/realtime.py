"""
realtime.py
-----------
Shared logic for processing a single transaction in "real time": predict ->
score risk -> persist -> alert if needed. Used by streaming/simulator.py and
can also be called from the FastAPI /predict endpoint.
"""
from __future__ import annotations

from src.predict import predict_transaction
from src.database import insert_prediction, upsert_transaction
from src.alerts import should_alert, raise_alert
from src.utils import get_logger

logger = get_logger(__name__)


def process_transaction(transaction: dict) -> dict:
    """Run one transaction through the full real-time pipeline and return the result."""
    result = predict_transaction(transaction)

    # Ensure the transaction row exists so predictions/alerts can reference it
    # via foreign key (needed for transactions submitted live through the API).
    upsert_transaction(transaction)

    insert_prediction(
        transaction_id=result["transaction_id"],
        fraud_probability=result["fraud_probability"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        predicted_class=result["is_fraud"],
        model_version=result["model_version"],
    )

    if should_alert(result["risk_score"]):
        raise_alert(result["transaction_id"], result["risk_level"], result["risk_score"])

    return result
