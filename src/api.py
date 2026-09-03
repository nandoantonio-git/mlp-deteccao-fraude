"""FastAPI prediction service for the promoted fraud detection model."""
from __future__ import annotations

import json
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from .preprocessing import BEST_RUN_PATH, FEATURE_COLUMNS, MODEL_PATH, PREPROCESSOR_PATH, transaction_to_frame

TransactionInput = create_model(
    "TransactionInput",
    __config__=ConfigDict(extra="forbid"),
    **{feature: (float, Field(..., description=f"Numeric transaction field {feature}")) for feature in FEATURE_COLUMNS},
)


class PredictionResult(BaseModel):
    prediction: Annotated[int, Field(ge=0, le=1)]
    label: str
    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]


app = FastAPI(
    title="CP4 Fraud Detection Prediction API",
    version="1.0.0",
    description="Local API for predicting credit card fraud with the promoted model.",
)

_model = None
_preprocessor = None
_best_run: dict | None = None


def load_artifacts() -> None:
    global _model, _preprocessor, _best_run
    missing = [str(path) for path in [MODEL_PATH, PREPROCESSOR_PATH, BEST_RUN_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model artifacts not found. Run `python -m src.train` before starting the API. Missing: "
            + ", ".join(missing)
        )
    _model = joblib.load(MODEL_PATH)
    _preprocessor = joblib.load(PREPROCESSOR_PATH)
    _best_run = json.loads(BEST_RUN_PATH.read_text(encoding="utf-8"))


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_artifacts()
    except FileNotFoundError:
        # Keep app importable for docs/tests; /predict returns a clear 503 until training runs.
        pass


@app.get("/health")
def health() -> dict[str, object]:
    ready = MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and BEST_RUN_PATH.exists()
    return {"status": "ready" if ready else "missing_artifacts", "ready": ready}


@app.post("/predict", response_model=PredictionResult)
def predict(transaction: TransactionInput) -> PredictionResult:  # type: ignore[valid-type]
    global _model, _preprocessor, _best_run
    if _model is None or _preprocessor is None or _best_run is None:
        try:
            load_artifacts()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    frame = transaction_to_frame(transaction.model_dump())
    prepared = _preprocessor.transform(frame)
    probability = float(_model.predict_proba(prepared)[:, 1][0])
    threshold = float(_best_run.get("decision_threshold", 0.5))
    prediction = int(probability >= threshold)
    label = "fraude" if prediction == 1 else "legitima"
    return PredictionResult(prediction=prediction, label=label, fraud_probability=probability)
