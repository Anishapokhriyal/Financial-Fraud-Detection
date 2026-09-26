"""
api/main.py
-----------
FastAPI prediction API for the Financial Fraud Detection system.

Run: uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.predict import predict_transaction
from src.database import (
    init_database, get_transactions, get_alerts, get_predictions,
)
from src.realtime import process_transaction
from src.utils import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    logger.info("API startup complete")
    yield


app = FastAPI(title=config.API_TITLE, version=config.MODEL_VERSION, lifespan=lifespan)


# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------
class TransactionInput(BaseModel):
    Transaction_ID: str = Field(...)
    Customer_ID: str = Field(...)
    Transaction_Date: str = Field(...)
    Transaction_Amount: float = Field(..., ge=0)
    Merchant_Category: str = Field(...)
    Payment_Method: str = Field(...)
    Device_Type: str = Field(...)
    Location: str = Field(...)
    Is_International: int = Field(..., ge=0, le=1)
    Previous_Transactions: int = Field(..., ge=0)
    Average_Spend: float = Field(..., ge=0)
    Account_Age_Days: int = Field(..., ge=0)
    Suspicious_Keyword: str = Field(...)


class PredictionOutput(BaseModel):
    transaction_id: str
    is_fraud: int
    fraud_probability: float
    risk_score: int
    risk_level: str
    model_version: str


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": config.API_TITLE,
        "status": "running",
        "model_version": config.MODEL_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    model_ready = config.MODEL_PATH.exists()
    db_ready = config.DATABASE_PATH.exists()
    return {
        "status": "ok" if (model_ready and db_ready) else "degraded",
        "model_loaded": model_ready,
        "database_ready": db_ready,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    try:
        result = process_transaction(transaction.model_dump())
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.get("/transactions")
def transactions(limit: int = 50):
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    df = get_transactions(limit=limit)
    return json.loads(df.to_json(orient="records"))


@app.get("/fraud-transactions")
def fraud_transactions(limit: int = 50):
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    df = get_transactions(limit=limit, fraud_only=True)
    return json.loads(df.to_json(orient="records"))


@app.get("/alerts")
def alerts(status: Optional[str] = None, limit: int = 50):
    df = get_alerts(status=status, limit=limit)
    return json.loads(df.to_json(orient="records"))


@app.get("/metrics")
def metrics():
    if not config.MODEL_METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="No trained model metadata found. Run training first.")
    metadata = json.loads(config.MODEL_METADATA_PATH.read_text())
    return metadata


@app.post("/retrain")
def retrain():
    """
    Trigger a model retrain. This runs synchronously and evaluates the new
    model before it could ever replace the production model (see
    src/train_model.py + README "Adaptive Learning" section). It does NOT
    automatically deploy an unreviewed model -- the response reports the new
    metrics so a human can approve the swap.
    """
    try:
        from src.train_model import main as train_main
        train_main()
        metadata = json.loads(config.MODEL_METADATA_PATH.read_text())
        return {"status": "retrained", "new_model_metadata": metadata}
    except Exception as exc:
        logger.error(f"Retrain failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {exc}")
