# Implementation Plan: CP4 Fraud Detection Prediction System

**Branch**: `001-fraud-detection-mlp` | **Date**: 2026-09-03 | **Spec**: `spec.md`
**Input**: Feature specification from `/specs/001-fraud-detection-mlp/spec.md`

## Summary

Build an academic, reproducible fraud detection project using the Kaggle credit card fraud dataset. The delivery will include mandatory EDA evidence in a notebook, a reusable training/prediction preprocessing path, at least five tracked model experiments, selection of the best fraud detector using validation-driven thresholding and final held-out metrics, a local prediction service, static experiment reports, and a five-minute demo guide.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: pandas, numpy, scikit-learn, imbalanced-learn, MLflow, FastAPI, uvicorn, joblib, pydantic, matplotlib, seaborn, jupyter
**Modeling Constraint**: Use scikit-learn `MLPClassifier`; do not use TensorFlow, Keras, or PyTorch.
**Dataset**: Kaggle `mlg-ulb/creditcardfraud`, local file `data/raw/creditcard.csv`, excluded from Git.
**Data Shape**: 30 feature columns (`Time`, `Amount`, `V1` through `V28`) plus target `Class`.
**Split Strategy**: Stratified train/validation/test split, approximately 70%/15%/15%.
**Imbalance Strategy**: Apply SMOTE in most experiments and one undersampling experiment; never resample validation, test, or prediction inputs.
**Tracking**: Local MLflow tracking under `mlruns/`, excluded from Git; static CSV/markdown reports committed under `reports/`.
**Prediction Runtime**: Local FastAPI app loads promoted artifacts from `models/`.
**Repository Root**: `/Users/nando_mac/mlp-deteccao-fraude`.
**Performance Goal**: Complete full-dataset training using controlled five-experiment matrix with `early_stopping=True`; no quick mode.

## Constitution Check

- **Scope Discipline**: PASS. Plan stays academic and excludes cloud, auth, frontend, monitoring, and retraining.
- **Evaluation Evidence First**: PASS. Notebook, MLflow runs, reports, model-selection file, API test, and video checklist are planned.
- **Honest Imbalanced Classification**: PASS. Selection uses fraud-class F1 with recall floor and reports precision, recall, F1, ROC-AUC.
- **Reproducibility and Consistency**: PASS. Fixed random state, serialized preprocessor, stable promoted artifacts, requirements, and README workflow are planned.
- **Academic Acceptance Requirements**: PASS. Notebook is mandatory for EDA; scripts support reproducible training and API validation.

## Project Structure

```text
mlp-deteccao-fraude/
├── data/
│   ├── raw/                  # creditcard.csv ignored by Git
│   └── processed/            # optional generated outputs ignored if large
├── models/                   # promoted best_model.joblib, preprocessor.joblib, best_run.json
├── notebooks/
│   └── eda.ipynb             # mandatory EDA evidence
├── reports/
│   ├── experiment_results.csv
│   ├── model_selection.md
│   └── video_script.md
├── scripts/
│   └── test_api.py
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── experiments.py
│   ├── preprocessing.py
│   └── train.py
├── specs/001-fraud-detection-mlp/
│   ├── spec.md
│   ├── plan.md
│   ├── research.md
│   ├── data-model.md
│   ├── quickstart.md
│   ├── contracts/
│   │   └── openapi.yaml
│   └── checklists/
│       └── requirements.md
├── .gitignore
├── README.md
└── requirements.txt
```

## Phase 0: Research

Research decisions are documented in `research.md`. No unresolved technical clarifications remain.

## Phase 1: Design and Contracts

Design artifacts are documented in:

- `data-model.md`
- `contracts/openapi.yaml`
- `quickstart.md`

## Implementation Strategy

1. Create repository skeleton and dependency files.
2. Implement `src/experiments.py` with five named experiment configurations.
3. Implement `src/preprocessing.py` with feature schema validation and a reusable transformation pipeline that scales only `Time` and `Amount` while passing PCA features through.
4. Implement `src/train.py` to load data, split train/validation/test, apply balancing on train only, train each candidate, log MLflow runs, select threshold on validation, evaluate final selected model on test, write reports, and promote artifacts.
5. Implement mandatory `notebooks/eda.ipynb` as lightweight EDA evidence, not as the training engine.
6. Implement `src/api.py` to load `models/best_model.joblib`, `models/preprocessor.joblib`, and `models/best_run.json` and serve `POST /predict`.
7. Implement `scripts/test_api.py` with a complete sample payload.
8. Write README as evaluator runbook and `reports/video_script.md` as the five-minute recording guide.
9. Verify syntax, importability, contract parsing, artifact generation paths, and API behavior after model artifacts exist.

## Post-Design Constitution Check

- **Scope Discipline**: PASS. Design contains only academic local execution artifacts.
- **Evaluation Evidence First**: PASS. Every acceptance area maps to a concrete artifact or command.
- **Honest Imbalanced Classification**: PASS. Metrics and threshold selection are explicit and validation/test roles are separated.
- **Reproducibility and Consistency**: PASS. Shared schema, preprocessor persistence, fixed randomness, and local promoted artifacts are specified.
- **Academic Acceptance Requirements**: PASS. Notebook remains mandatory and scripts remain reproducible.

## Complexity Tracking

No constitution violations or unjustified complexity exceptions.
