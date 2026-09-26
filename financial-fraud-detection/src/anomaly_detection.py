"""
anomaly_detection.py
---------------------
OPTIONAL unsupervised anomaly detector using Isolation Forest.

This module is independent of the supervised model. If disabled, the rest of
the system (training, prediction API, dashboard) still works unchanged.

IMPORTANT: an anomaly flag means "statistically unusual", NOT "confirmed
fraud". Anomalies should be treated as an additional signal, not a verdict.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

ANOMALY_MODEL_PATH = config.MODELS_DIR / "isolation_forest.pkl"


def train_isolation_forest(X: pd.DataFrame, contamination: float = 0.1) -> IsolationForest:
    """Train an Isolation Forest anomaly detector on numeric features only."""
    numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=config.RANDOM_SEED,
    )
    model.fit(numeric_X)
    joblib.dump(model, ANOMALY_MODEL_PATH)
    logger.info(f"Isolation Forest trained on {numeric_X.shape[1]} numeric features and saved")
    return model


def score_anomalies(X: pd.DataFrame) -> pd.DataFrame:
    """Return anomaly_score (higher = more anomalous) and anomaly_flag (1/-1 -> 1/0)."""
    if not ANOMALY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Isolation Forest model not found. Run train_isolation_forest() first."
        )
    model = joblib.load(ANOMALY_MODEL_PATH)
    numeric_X = X.select_dtypes(include=[np.number]).fillna(0)

    raw_scores = model.decision_function(numeric_X)  # lower = more anomalous
    predictions = model.predict(numeric_X)  # -1 = anomaly, 1 = normal

    result = pd.DataFrame(index=X.index)
    # Flip sign so higher score = more anomalous, then min-max scale to 0-1
    inverted = -raw_scores
    result["anomaly_score"] = (inverted - inverted.min()) / (inverted.max() - inverted.min() + 1e-9)
    result["anomaly_flag"] = (predictions == -1).astype(int)
    return result


def combine_supervised_and_anomaly(fraud_probability: float, anomaly_score: float,
                                    weight_supervised: float = 0.7) -> float:
    """
    Optional configurable blend of the supervised fraud probability and the
    unsupervised anomaly score into a single combined risk signal (0-1).
    """
    weight_supervised = max(0.0, min(1.0, weight_supervised))
    weight_anomaly = 1.0 - weight_supervised
    return weight_supervised * fraud_probability + weight_anomaly * anomaly_score
