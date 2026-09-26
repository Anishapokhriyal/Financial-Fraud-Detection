"""
train_model.py
---------------
End-to-end training pipeline:
ETL -> Feature Engineering -> Train/Test Split -> SMOTE (train only) ->
Train 4 models -> Evaluate -> Select best model -> Save artifacts ->
Generate report charts.

Run: python -m src.train_model
"""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src import config
from src.utils import get_logger
from src.preprocessing import run_etl
from src.feature_engineering import engineer_features, get_model_feature_columns
from src.model_comparison import evaluate_model, select_best_model
from src.anomaly_detection import train_isolation_forest

logger = get_logger(__name__)


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """ColumnTransformer: scale numeric features, one-hot encode categoricals.
    handle_unknown='ignore' ensures unseen categories at inference time do not crash."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=config.RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=config.RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=config.RANDOM_SEED,
        ),
    }


def main():
    logger.info("=== Starting training pipeline ===")

    # 1. ETL
    df = run_etl(save=True)

    # 2. Feature engineering
    df = engineer_features(df)
    numeric_features, categorical_features = get_model_feature_columns()

    X = df[numeric_features + categorical_features]
    y = df[config.TARGET_COLUMN]

    # 3. Train/test split (stratified, reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Train fraud rate: {y_train.mean():.4f}, Test fraud rate: {y_test.mean():.4f}")

    # 4. Preprocess (fit on train only to avoid leakage)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 5. Handle class imbalance: SMOTE applied to TRAINING data only, after the split
    logger.info("Applying SMOTE to training data only (never to test data)")
    smote = SMOTE(random_state=config.RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    logger.info(f"Post-SMOTE train class distribution: {np.bincount(y_train_res)}")

    # 6. Train & evaluate all models
    models = get_models()
    results = {}
    fitted_models = {}

    for name, model in models.items():
        logger.info(f"Training {name} ...")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)
        results[name] = metrics
        fitted_models[name] = model
        logger.info(f"{name} metrics: { {k: v for k, v in metrics.items() if k != 'Confusion_Matrix'} }")

    # 7. Save model comparison report
    comparison_rows = []
    for name, m in results.items():
        comparison_rows.append({
            "Model": name, "Accuracy": m["Accuracy"], "Precision": m["Precision"],
            "Recall": m["Recall"], "F1": m["F1"], "ROC_AUC": m["ROC_AUC"], "PR_AUC": m["PR_AUC"],
        })
    comparison_df = pd.DataFrame(comparison_rows).sort_values("F1", ascending=False)
    comparison_df.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Saved model comparison to {config.MODEL_COMPARISON_PATH}")

    # 8. Select the best model (documented criterion: highest F1 -- see config.py)
    best_name = select_best_model(results, metric=config.MODEL_SELECTION_METRIC)
    best_model = fitted_models[best_name]
    logger.info(f"Best model selected: {best_name} (criterion: {config.MODEL_SELECTION_METRIC})")

    # 9. Save best model + preprocessor
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    metadata = {
        "model_name": best_name,
        "model_version": config.MODEL_VERSION,
        "selection_metric": config.MODEL_SELECTION_METRIC,
        "metrics": {k: v for k, v in results[best_name].items() if k != "Confusion_Matrix"},
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_fraud_rate": float(round(y_train.mean(), 4)),
        "test_fraud_rate": float(round(y_test.mean(), 4)),
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Saved model metadata to {config.MODEL_METADATA_PATH}")

    # 10. Feature importance (tree-based models expose feature_importances_)
    feature_names = list(numeric_features) + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    )
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)
    logger.info(f"Saved feature importance to {config.FEATURE_IMPORTANCE_PATH}")

    # 11. Charts
    generate_charts(best_model, best_name, X_test_proc, y_test, fi_df)

    # 12. Optional: train Isolation Forest anomaly detector
    train_isolation_forest(X_train)

    logger.info("=== Training pipeline complete ===")
    print("\nModel comparison:\n", comparison_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Best model metrics: {metadata['metrics']}")


def generate_charts(model, model_name, X_test_proc, y_test, fi_df):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"ROC Curve - {model_name}")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Precision-Recall curve
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # Feature importance (top 15)
    top_fi = fi_df.head(15)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_fi["Feature"][::-1], top_fi["Importance"][::-1], color="#2b6cb0")
    ax.set_title(f"Top 15 Feature Importances - {model_name}")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    logger.info("Saved all report charts to reports/")


if __name__ == "__main__":
    main()
