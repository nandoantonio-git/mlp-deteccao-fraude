"""Train, track, compare, and promote the best fraud detection MLP."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .experiments import EXPERIMENTS, ExperimentConfig, as_mlflow_params
from .preprocessing import (
    BEST_RUN_PATH,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    build_preprocessor,
    load_dataset,
)

RECALL_FLOOR = 0.80
MLFLOW_EXPERIMENT_NAME = "cp4-credit-card-fraud-mlp"


def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def rebalance_training_data(X_train, y_train, strategy: str, random_state: int):
    if strategy == "smote":
        return SMOTE(random_state=random_state, k_neighbors=5).fit_resample(X_train, y_train)
    if strategy == "undersampling":
        return RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)
    raise ValueError(f"Unsupported imbalance strategy: {strategy}")


def build_model(config: ExperimentConfig) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=config.hidden_layers,
        learning_rate_init=config.learning_rate,
        activation=config.activation,
        solver=config.solver,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        random_state=config.random_state,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.10,
    )


def predict_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def calculate_metrics(y_true, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = predict_with_threshold(probabilities, threshold)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }
    return {key: float(value) for key, value in metrics.items()}


def find_best_threshold(y_true, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.01, 0.99, 99)
    scored: list[tuple[float, dict[str, float]]] = [
        (float(threshold), calculate_metrics(y_true, probabilities, float(threshold)))
        for threshold in candidates
    ]
    eligible = [item for item in scored if item[1]["recall"] >= RECALL_FLOOR]
    pool = eligible or scored
    return max(pool, key=lambda item: (item[1]["f1_score"], item[1]["roc_auc"], item[1]["recall"]))


def train_one_experiment(config: ExperimentConfig, X_train, y_train, X_val, y_val) -> dict[str, object]:
    preprocessor = build_preprocessor()
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_val_prepared = preprocessor.transform(X_val)
    X_balanced, y_balanced = rebalance_training_data(
        X_train_prepared, y_train, config.imbalance_strategy, config.random_state
    )
    model = build_model(config)
    model.fit(X_balanced, y_balanced)
    val_probabilities = model.predict_proba(X_val_prepared)[:, 1]
    threshold, val_metrics = find_best_threshold(y_val, val_probabilities)

    with mlflow.start_run(run_name=config.name) as run:
        mlflow.log_params(as_mlflow_params(config))
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_metrics({f"val_{key}": value for key, value in val_metrics.items()})
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_file = tmp / "model.joblib"
            preprocessor_file = tmp / "preprocessor.joblib"
            joblib.dump(model, model_file)
            joblib.dump(preprocessor, preprocessor_file)
            mlflow.log_artifact(str(model_file), artifact_path="model")
            mlflow.log_artifact(str(preprocessor_file), artifact_path="preprocessor")
            mlflow.sklearn.log_model(model, artifact_path="sklearn_model")
        artifact_uri = mlflow.get_artifact_uri()
        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "artifact_uri": artifact_uri,
        "experiment_name": config.name,
        "config": config,
        "model": model,
        "preprocessor": preprocessor,
        "decision_threshold": threshold,
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }


def choose_best_run(results: list[dict[str, object]]) -> dict[str, object]:
    eligible = [run for run in results if float(run["val_recall"]) >= RECALL_FLOOR]
    pool = eligible or results
    return max(
        pool,
        key=lambda run: (
            float(run["val_f1_score"]),
            float(run["val_roc_auc"]),
            float(run["val_recall"]),
        ),
    )


def write_reports(results: list[dict[str, object]], best: dict[str, object], test_metrics: dict[str, float]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in results:
        config: ExperimentConfig = run["config"]  # type: ignore[assignment]
        rows.append(
            {
                "run_id": run["run_id"],
                "experiment_name": run["experiment_name"],
                "hidden_layers": str(config.hidden_layers),
                "learning_rate": config.learning_rate,
                "activation": config.activation,
                "solver": config.solver,
                "batch_size": config.batch_size,
                "max_iter": config.max_iter,
                "imbalance_strategy": config.imbalance_strategy,
                "decision_threshold": run["decision_threshold"],
                "val_accuracy": run["val_accuracy"],
                "val_precision": run["val_precision"],
                "val_recall": run["val_recall"],
                "val_f1_score": run["val_f1_score"],
                "val_roc_auc": run["val_roc_auc"],
                "artifact_uri": run["artifact_uri"],
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "experiment_results.csv", index=False)

    best_name = str(best["experiment_name"])
    selection = (
        f"# Model Selection\n\n"
        f"Selected model: **{best_name}**\n\n"
        f"The selected model had validation recall {float(best['val_recall']):.4f}, "
        f"precision {float(best['val_precision']):.4f}, F1-score {float(best['val_f1_score']):.4f}, "
        f"and ROC-AUC {float(best['val_roc_auc']):.4f}. It was selected by maximizing "
        f"fraud-class F1 among candidates with acceptable recall, using ROC-AUC as tie-breaker.\n\n"
        f"## Final held-out test metrics\n\n"
        f"- Accuracy: {test_metrics['accuracy']:.4f}\n"
        f"- Fraud precision: {test_metrics['precision']:.4f}\n"
        f"- Fraud recall: {test_metrics['recall']:.4f}\n"
        f"- Fraud F1-score: {test_metrics['f1_score']:.4f}\n"
        f"- ROC-AUC: {test_metrics['roc_auc']:.4f}\n\n"
        f"If any target threshold is not met, this section should be discussed honestly in the video and README as the best observed tradeoff from the experiment matrix.\n"
    )
    (REPORTS_DIR / "model_selection.md").write_text(selection, encoding="utf-8")


def promote_artifacts(best: dict[str, object], test_metrics: dict[str, float]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(best["preprocessor"], PREPROCESSOR_PATH)
    best_run = {
        "best_run_id": best["run_id"],
        "experiment_name": best["experiment_name"],
        "decision_threshold": best["decision_threshold"],
        "validation_metrics": {
            key.removeprefix("val_"): best[key]
            for key in ["val_accuracy", "val_precision", "val_recall", "val_f1_score", "val_roc_auc"]
        },
        "final_test_metrics": test_metrics,
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "selection_reason": "Selected by fraud-class F1 with recall floor preference and ROC-AUC tie-breaker.",
    }
    BEST_RUN_PATH.write_text(json.dumps(best_run, indent=2), encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("file:" + str(Path("mlruns").resolve()))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_dataset(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    results = [train_one_experiment(config, X_train, y_train, X_val, y_val) for config in EXPERIMENTS]
    best = choose_best_run(results)
    X_test_prepared = best["preprocessor"].transform(X_test)  # type: ignore[union-attr]
    test_probabilities = best["model"].predict_proba(X_test_prepared)[:, 1]  # type: ignore[union-attr]
    test_metrics = calculate_metrics(y_test, test_probabilities, float(best["decision_threshold"]))

    promote_artifacts(best, test_metrics)
    write_reports(results, best, test_metrics)
    print(f"Selected model: {best['experiment_name']}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
