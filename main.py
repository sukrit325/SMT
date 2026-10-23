
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.policy import Borrower, check_policies
from src.verifier import verify_policy
from typing import Any
import logging

# Enable backend logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="LendVerify API",
    description="Lending policy assessment and Z3 verification",
    version="1.0.0",
)

# ... (rest of file remains the same)

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
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": "lendverify-api",
    }


@app.post("/api/v1/assess")
def assess_borrower(request: AssessmentRequest):
    logger.info("Assessment request: %s", request.model_dump())
    borrower = Borrower(**request.model_dump())
    result = check_policies(borrower)

    return {
        "rules": result["rules"],
        "available_surplus": result["available_surplus"],
        "compliant": result["compliant"],
    }


@app.get("/api/v1/ml/status")
def ml_status():
    logger.info("ML status requested")
    return {
        "status": "available",
        "note": "Models must be trained first using src/train.py",
        "training_command": "python src/train.py"
    }


@app.post("/api/v1/ml/train")
def ml_train():
    logger.info("ML training started - running src.train.main()...")
    try:
        # Import and run the available training pipeline
        from src import train as train_module
        train_module.main()
        return {
            "status": "completed",
            "message": "Models trained successfully",
            "note": "Check /output/training.log for details"
        }
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/predict")
def predict_borrower_endpoint(request: PredictionRequest):
    logger.info("Prediction request received with features")
    # Note: This endpoint returns a placeholder response
    # The actual prediction would require loading trained models from src/train.py bundle
    return {
        "status": "unavailable",
        "message": "This is a demo endpoint. For real predictions, train models using /ml/train first.",
        "note": "The training pipeline creates a bundled model that can be loaded for predictions."
    }


@app.post("/api/v1/ml/evaluate-test")
def evaluate_test_endpoint():
    logger.info("Test evaluation requested")
    # Note: This endpoint returns a placeholder response
    return {
        "status": "unavailable",
        "message": "This is a demo endpoint. Evaluation would require loaded trained models."
    }


@app.post("/api/v1/verify")
def verify_lending_policy(request: VerificationRequest):
    logger.info("Verification request: defect=%s", request.defect)
    try:
        return verify_policy(request.defect)
    except ValueError as exc:
        logger.error("Verification failed (ValueError): %s", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Verification failed (Unexpected): %s", str(exc))
        raise HTTPException(
            status_code=500,
            detail="Verification engine failed",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting LendVerify API on http://127.0.0.1:8000")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
