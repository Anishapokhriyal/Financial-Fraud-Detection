"""
model_comparison.py
--------------------
Helper functions to evaluate a trained classifier and produce a metrics
dictionary used to compare multiple models on a common, business-relevant
basis (precision, recall, F1, ROC-AUC, PR-AUC). Accuracy is intentionally
NOT used as the selection criterion because the dataset is imbalanced
(~9.6% fraud).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    """Compute the full metric suite for a single model's predictions."""
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_proba), 4),
        "PR_AUC": round(average_precision_score(y_true, y_proba), 4),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["Confusion_Matrix"] = cm.tolist()
    return metrics


def select_best_model(results: dict, metric: str = "F1") -> str:
    """
    Given {model_name: metrics_dict}, return the name of the best model
    according to the chosen metric (default F1, documented in config.py).
    """
    return max(results.keys(), key=lambda name: results[name][metric])
