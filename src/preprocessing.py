"""Shared preprocessing and validation helpers for credit card fraud detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
SCALED_FEATURES = ["Time", "Amount"]
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", *PCA_FEATURES, "Amount"]
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
BEST_RUN_PATH = MODELS_DIR / "best_run.json"


def build_preprocessor() -> ColumnTransformer:
    """Create the reusable feature transformer used for training and prediction."""
    return ColumnTransformer(
        transformers=[("scale_time_amount", StandardScaler(), SCALED_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def validate_required_columns(df: pd.DataFrame, *, require_target: bool = True) -> None:
    """Validate that a dataframe contains the expected feature columns and target."""
    required = REQUIRED_COLUMNS if require_target else FEATURE_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset/input is missing required columns: " + ", ".join(missing)
        )


def validate_transaction_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    """Validate one prediction transaction and return ordered numeric features."""
    missing = [col for col in FEATURE_COLUMNS if col not in payload]
    if missing:
        raise ValueError("Missing required transaction fields: " + ", ".join(missing))

    ordered: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        value = payload[col]
        if isinstance(value, bool):
            raise ValueError(f"Field {col} must be numeric, got boolean")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field {col} must be numeric") from exc
        if not np.isfinite(numeric_value):
            raise ValueError(f"Field {col} must be finite")
        ordered[col] = numeric_value
    return ordered


def transaction_to_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert one validated transaction payload into a one-row dataframe."""
    ordered = validate_transaction_payload(payload)
    return pd.DataFrame([ordered], columns=FEATURE_COLUMNS)


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load and validate the Kaggle credit card fraud dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Download Kaggle dataset 'mlg-ulb/creditcardfraud' "
            f"and save creditcard.csv at {dataset_path}"
        )
    df = pd.read_csv(dataset_path)
    validate_required_columns(df, require_target=True)
    return df[REQUIRED_COLUMNS].copy()
