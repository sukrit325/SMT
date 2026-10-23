from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.policy import Borrower, check_policies
from src.verifier import verify_policy
from typing import Any
import logging
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "combined_models.pkl"))

app = FastAPI(title="LendVerify API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model bundle once at startup ----
_bundle: dict | None = None

@app.on_event("startup")
def _startup():
    global _bundle
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
        logger.info("Loaded model bundle from %s", MODEL_PATH)
    else:
        logger.warning("Model bundle not found at %s; /ml/predict will fail.", MODEL_PATH)

def _models() -> dict:
    if not _bundle:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return _bundle.get("models", {})

def _feature_columns() -> list[str]:
    if not _bundle:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return list(_bundle.get("feature_columns", []))


class AssessmentRequest(BaseModel):
    profitable_fy1: bool
    profitable_fy2: bool
    bank_credits: float = Field(ge=0)
    turnover: float = Field(gt=0)
    repayment_bounces: int = Field(ge=0, le=12)
    bounces_cleared_within_3_days: bool
    deviation_approved: bool
    monthly_income: float = Field(ge=0)
    existing_emi: float = Field(ge=0)
    monthly_savings: float = Field(ge=0)
    family_expenses: float = Field(ge=0)
    proposed_emi: float = Field(ge=0)


class VerificationRequest(BaseModel):
    defect: str | None = None


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "service": "lendverify-api",
        "models_loaded": bool(_bundle),
        "models": list(_models().keys()) if _bundle else [],
    }


@app.post("/api/v1/assess")
def assess_borrower(request: AssessmentRequest):
    borrower = Borrower(**request.model_dump())
    result = check_policies(borrower)
    return {
        "rules": result["rules"],
        "available_surplus": result["available_surplus"],
        "compliant": result["compliant"],
    }


@app.post("/api/v1/verify")
def verify_lending_policy(request: VerificationRequest):
    try:
        return verify_policy(request.defect)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Verification failed")
        raise HTTPException(status_code=500, detail="Verification engine failed") from exc


@app.post("/api/v1/ml/predict")
def predict_borrower_endpoint(request: PredictionRequest):
    models = _models()
    cols = _feature_columns()
    try:
        X = pd.DataFrame([request.features])[cols]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing feature: {exc}") from exc

    predictions, probabilities = {}, {}
    for name, model in models.items():
        try:
            predictions[name] = int(model.predict(X)[0])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", []))
                probabilities[name] = {
                    str(cls): float(p) for cls, p in zip(classes, proba)
                }
        except Exception as exc:
            logger.exception("Model %s failed", name)
            predictions[name] = None
            probabilities[name] = {"error": str(exc)}

    # Backward-compatible singular fields (first model)
    first = next(iter(models)) if models else None
    return {
        "prediction": predictions.get(first),
        "class_probabilities": probabilities.get(first),
        "all_predictions": predictions,
        "all_probabilities": probabilities,
        "surrogate_prediction": predictions.get(first),
        "surrogate_path": [],   # fill in if you have a surrogate
    }