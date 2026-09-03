# Quickstart: CP4 Fraud Detection Prediction System

This guide defines the validation path the evaluator should be able to run after implementation.

## Prerequisites

- Python 3.10 or newer.
- Kaggle credit card fraud dataset downloaded from `mlg-ulb/creditcardfraud`.
- Dataset saved as `data/raw/creditcard.csv` from the project root.

## 1. Create environment and install dependencies

```bash
cd /Users/nando_mac/mlp-deteccao-fraude
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected outcome: dependencies install successfully in an isolated environment.

## 2. Confirm dataset location

```bash
test -f data/raw/creditcard.csv && echo "dataset ok"
```

Expected outcome: `dataset ok`.

## 3. Run the mandatory EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

Expected outcome: notebook displays dataset size, missing values, class counts, fraud percentage, and basic summaries for `Time` and `Amount`.

## 4. Train and track experiments

```bash
python -m src.train
```

Expected outcome:

- At least five experiment runs are recorded.
- `reports/experiment_results.csv` is generated.
- `reports/model_selection.md` explains the selected model.
- `models/best_model.joblib`, `models/preprocessor.joblib`, and `models/best_run.json` are generated.

## 5. Inspect experiment comparison

```bash
mlflow ui
```

Expected outcome: MLflow UI opens locally and shows runs with parameters, metrics, and artifacts.

## 6. Start prediction service

```bash
uvicorn src.api:app --reload
```

Expected outcome: local prediction service starts and exposes `POST /predict`.

## 7. Test prediction

In a second terminal with the same environment active:

```bash
python scripts/test_api.py
```

Expected outcome: response includes `prediction`, `label`, and `fraud_probability`.

## 8. Validate clean repository

```bash
git status --short
```

Expected outcome: raw dataset, local MLflow directory, caches, virtual environment, and notebook checkpoints are not staged for commit.

## Contract Reference

The prediction contract is documented in `specs/001-fraud-detection-mlp/contracts/openapi.yaml`.
