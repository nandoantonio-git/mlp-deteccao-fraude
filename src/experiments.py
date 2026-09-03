"""Experiment matrix for the CP4 fraud detection MLP project."""
from __future__ import annotations

from dataclasses import dataclass

from .preprocessing import RANDOM_STATE


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    activation: str
    solver: str
    batch_size: int
    max_iter: int
    imbalance_strategy: str
    random_state: int = RANDOM_STATE


EXPERIMENTS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig("MLP-01", (32,), 0.001, "relu", "adam", 256, 200, "smote"),
    ExperimentConfig("MLP-02", (64, 32), 0.001, "relu", "adam", 256, 200, "smote"),
    ExperimentConfig("MLP-03", (64, 32), 0.0001, "relu", "adam", 256, 300, "smote"),
    ExperimentConfig("MLP-04", (128, 64), 0.001, "relu", "adam", 256, 200, "undersampling"),
    ExperimentConfig("MLP-05", (64, 32), 0.001, "tanh", "adam", 256, 200, "smote"),
)


def as_mlflow_params(config: ExperimentConfig) -> dict[str, object]:
    return {
        "experiment_name": config.name,
        "hidden_layers": str(config.hidden_layers),
        "learning_rate": config.learning_rate,
        "activation": config.activation,
        "solver": config.solver,
        "batch_size": config.batch_size,
        "max_iter": config.max_iter,
        "imbalance_strategy": config.imbalance_strategy,
        "random_state": config.random_state,
    }
