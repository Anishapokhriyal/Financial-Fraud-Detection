"""
risk_scoring.py
----------------
Converts a fraud probability (0.0 - 1.0) into a 0-100 risk score and a
human-readable risk level. Thresholds are fully configurable via
src/config.py (RISK_THRESHOLDS).

Note: a probability is a model estimate, not proof of fraud. Higher-risk
transactions should be investigated, not automatically blocked.
"""
from __future__ import annotations

from src import config


def probability_to_score(fraud_probability: float) -> int:
    """Convert a 0-1 probability into a 0-100 integer risk score."""
    fraud_probability = max(0.0, min(1.0, float(fraud_probability)))
    return round(fraud_probability * 100)


def score_to_level(risk_score: int) -> str:
    """Map a 0-100 risk score to a risk level using configurable thresholds."""
    for level, (low, high) in config.RISK_THRESHOLDS.items():
        if low <= risk_score <= high:
            return level.title()
    return "Unknown"


def calculate_risk(fraud_probability: float) -> dict:
    """Return the full risk-scoring payload for a given fraud probability."""
    risk_score = probability_to_score(fraud_probability)
    risk_level = score_to_level(risk_score)
    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    for p in [0.05, 0.35, 0.65, 0.9]:
        print(p, "->", calculate_risk(p))
