"""FastAPI prediction service for the promoted fraud detection model."""
from __future__ import annotations

import json
import logging
import math
from contextlib import asynccontextmanager
from typing import Annotated

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from .preprocessing import BEST_RUN_PATH, FEATURE_COLUMNS, MLFLOW_TRACKING_URI, transaction_to_frame

TransactionInput = create_model(
    "TransactionInput",
    __config__=ConfigDict(extra="forbid"),
    **{feature: (float, Field(..., strict=True, allow_inf_nan=False, description=f"Numeric transaction field {feature}")) for feature in FEATURE_COLUMNS},
)


class PredictionResult(BaseModel):
    prediction: Annotated[int, Field(ge=0, le=1)]
    label: str
    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
    except RuntimeError:
        logging.getLogger(__name__).exception("Unable to load the selected MLflow run")
    yield


app = FastAPI(
    title="CP4 Fraud Detection Prediction API",
    version="1.0.0",
    description="Local API for predicting credit card fraud with the promoted model.",
    lifespan=lifespan,
)

_model = None
_preprocessor = None
_best_run: dict | None = None


def load_artifacts() -> None:
    global _model, _preprocessor, _best_run
    _model = _preprocessor = _best_run = None
    try:
        selected = json.loads(BEST_RUN_PATH.read_text(encoding="utf-8"))
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run_id = selected["best_run_id"]
        run = client.get_run(run_id)
        if run.info.status != "FINISHED" or run.info.lifecycle_stage != "active":
            raise ValueError("Selected run is not active and finished")
        threshold = float(run.data.params["decision_threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid decision threshold in selected run")
        model_uri = f"runs:/{run_id}/sklearn_model"
        model = mlflow.sklearn.load_model(model_uri)
        preprocessor = joblib.load(client.download_artifacts(run_id, "preprocessor/preprocessor.joblib"))
        # Publish state only after every artifact from the same run has loaded.
        _model, _preprocessor = model, preprocessor
        _best_run = {"best_run_id": run_id, "model_uri": model_uri, "decision_threshold": threshold}
    except Exception as exc:
        raise RuntimeError(
            "Selected MLflow model unavailable. Check models/best_run.json and mlruns; "
            "run `python -m src.train` if no complete training is available."
        ) from exc


@app.get("/health")
def health() -> dict[str, object]:
    ready = _model is not None and _preprocessor is not None and _best_run is not None
    return {"status": "ready" if ready else "missing_artifacts", "ready": ready,
            "run_id": _best_run["best_run_id"] if ready else None}


@app.post("/predict", response_model=PredictionResult)
def predict(transaction: TransactionInput) -> PredictionResult:  # type: ignore[valid-type]
    global _model, _preprocessor, _best_run
    if _model is None or _preprocessor is None or _best_run is None:
        try:
            load_artifacts()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    frame = transaction_to_frame(transaction.model_dump())
    prepared = _preprocessor.transform(frame)
    probability = float(_model.predict_proba(prepared)[:, 1][0])
    threshold = float(_best_run.get("decision_threshold", 0.5))
    prediction = int(probability >= threshold)
    label = "fraude" if prediction == 1 else "legitima"
    return PredictionResult(prediction=prediction, label=label, fraud_probability=probability)
