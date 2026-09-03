# Tasks: CP4 Fraud Detection Prediction System

**Input**: Design documents from `specs/001-fraud-detection-mlp/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/openapi.yaml`, `quickstart.md`
**Generation mode**: Manual fallback because `.specify/scripts/bash/setup-tasks.sh` is not present.

## User Stories

- **US1 — Reproducible project setup and EDA evidence**: Evaluator can install the project, place the dataset, and inspect the mandatory notebook evidence.
- **US2 — Train, track, compare, and select fraud models**: Evaluator can run at least five full-dataset experiments, inspect tracked metrics/artifacts, and understand why the best model was selected.
- **US3 — Local prediction service**: Evaluator can start a local prediction service and send one complete transaction to receive prediction, label, and fraud probability.
- **US4 — Evaluation packaging and demo readiness**: Evaluator can follow the README/video script and verify the repository is clean and ready for submission.

## Phase 1: Setup

- [X] T001 Create project directories `data/raw`, `data/processed`, `models`, `notebooks`, `reports`, `scripts`, and `src` in `/Users/nando_mac/mlp-deteccao-fraude`
- [X] T002 Create Python package marker in `/Users/nando_mac/mlp-deteccao-fraude/src/__init__.py`
- [X] T003 Create `.gitignore` in `/Users/nando_mac/mlp-deteccao-fraude/.gitignore` excluding `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `data/raw/*.csv`, `data/processed/`, `mlruns/`, `.mlflow/`, and transient model/report caches
- [X] T004 Create pinned dependency file in `/Users/nando_mac/mlp-deteccao-fraude/requirements.txt` with Python ML, notebook, MLflow, FastAPI, and serialization dependencies
- [X] T005 Create root README skeleton in `/Users/nando_mac/mlp-deteccao-fraude/README.md` with project title, dataset path requirement, setup commands, and evaluation workflow headings

## Phase 2: Foundational

- [X] T006 Implement shared constants for feature order, target column, random state, and artifact paths in `/Users/nando_mac/mlp-deteccao-fraude/src/preprocessing.py`
- [X] T007 Implement transaction schema validation helper in `/Users/nando_mac/mlp-deteccao-fraude/src/preprocessing.py` requiring `Time`, `Amount`, and `V1` through `V28` as numeric fields
- [X] T008 Implement reusable feature preprocessor builder in `/Users/nando_mac/mlp-deteccao-fraude/src/preprocessing.py` that standardizes `Time` and `Amount` and passes `V1` through `V28` unchanged
- [X] T009 Implement dataset loading helper in `/Users/nando_mac/mlp-deteccao-fraude/src/preprocessing.py` that reads `data/raw/creditcard.csv`, validates required columns, and raises a clear error if missing
- [X] T010 Define the five required experiment configurations in `/Users/nando_mac/mlp-deteccao-fraude/src/experiments.py` using names `MLP-01` through `MLP-05`
- [X] T011 Include SMOTE for four configurations and undersampling for one configuration in `/Users/nando_mac/mlp-deteccao-fraude/src/experiments.py`
- [X] T012 Ensure every experiment configuration in `/Users/nando_mac/mlp-deteccao-fraude/src/experiments.py` specifies hidden layers, learning rate, activation, solver, batch size, max iterations, imbalance strategy, and fixed random state

## Phase 3: User Story 1 — Reproducible project setup and EDA evidence

**Goal**: Evaluator can install the project, place the dataset, and inspect the mandatory notebook evidence.

**Independent Test Criteria**: With `data/raw/creditcard.csv` present, opening `notebooks/eda.ipynb` shows dataset shape, column list, missing-value checks, class counts, fraud percentage, and basic `Time`/`Amount` summaries without running training.

- [X] T013 [P] [US1] Create mandatory EDA notebook in `/Users/nando_mac/mlp-deteccao-fraude/notebooks/eda.ipynb` with dataset loading and clear markdown introduction
- [X] T014 [P] [US1] Add dataset shape, column list, and missing-value checks to `/Users/nando_mac/mlp-deteccao-fraude/notebooks/eda.ipynb`
- [X] T015 [P] [US1] Add target distribution counts, fraud percentage, and imbalance explanation to `/Users/nando_mac/mlp-deteccao-fraude/notebooks/eda.ipynb`
- [X] T016 [P] [US1] Add descriptive statistics and simple plots for `Time`, `Amount`, and `Class` distribution to `/Users/nando_mac/mlp-deteccao-fraude/notebooks/eda.ipynb`
- [X] T017 [US1] Update dataset instructions in `/Users/nando_mac/mlp-deteccao-fraude/README.md` explaining Kaggle source `mlg-ulb/creditcardfraud` and required local path `data/raw/creditcard.csv`
- [X] T018 [US1] Verify notebook JSON structure by running `python -m json.tool notebooks/eda.ipynb` from `/Users/nando_mac/mlp-deteccao-fraude`

## Phase 4: User Story 2 — Train, track, compare, and select fraud models

**Goal**: Evaluator can run at least five full-dataset experiments, inspect tracked metrics/artifacts, and understand why the best model was selected.

**Independent Test Criteria**: Running `python -m src.train` with the dataset present creates MLflow runs, `reports/experiment_results.csv`, `reports/model_selection.md`, `models/best_model.joblib`, `models/preprocessor.joblib`, and `models/best_run.json`.

- [X] T019 [US2] Implement stratified train/validation/test splitting in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` using approximately 70/15/15 proportions and fixed random state
- [X] T020 [US2] Implement training-only imbalance handling in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` using SMOTE or undersampling based on each experiment configuration
- [X] T021 [US2] Implement MLPClassifier construction in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` using each experiment configuration with `early_stopping=True`
- [X] T022 [US2] Implement validation-threshold search in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` maximizing fraud-class F1 while respecting recall preference
- [X] T023 [US2] Implement metric calculation in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` for accuracy, fraud precision, fraud recall, fraud F1, and ROC-AUC
- [X] T024 [US2] Implement MLflow experiment logging in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` for parameters, metrics, trained model artifact, and preprocessing artifact
- [X] T025 [US2] Implement run comparison table generation in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` writing `/Users/nando_mac/mlp-deteccao-fraude/reports/experiment_results.csv`
- [X] T026 [US2] Implement best-model selection in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` using fraud-class F1 among candidates with acceptable recall and ROC-AUC tie-breaker
- [X] T027 [US2] Implement final held-out test evaluation for the selected model in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` without using test data for threshold selection
- [X] T028 [US2] Promote selected artifacts in `/Users/nando_mac/mlp-deteccao-fraude/src/train.py` to `models/best_model.joblib`, `models/preprocessor.joblib`, and `models/best_run.json`
- [X] T029 [US2] Generate model-selection explanation in `/Users/nando_mac/mlp-deteccao-fraude/reports/model_selection.md` from `/Users/nando_mac/mlp-deteccao-fraude/src/train.py`
- [X] T030 [US2] Update training and MLflow instructions in `/Users/nando_mac/mlp-deteccao-fraude/README.md` with commands for `python -m src.train` and `mlflow ui`

## Phase 5: User Story 3 — Local prediction service

**Goal**: Evaluator can start a local prediction service and send one complete transaction to receive prediction, label, and fraud probability.

**Independent Test Criteria**: After model artifacts exist, `uvicorn src.api:app --reload` starts successfully and `python scripts/test_api.py` returns JSON containing `prediction`, `label`, and `fraud_probability`.

- [X] T031 [P] [US3] Implement Pydantic request and response models matching `/Users/nando_mac/mlp-deteccao-fraude/specs/001-fraud-detection-mlp/contracts/openapi.yaml` in `/Users/nando_mac/mlp-deteccao-fraude/src/api.py`
- [X] T032 [US3] Implement startup artifact loading in `/Users/nando_mac/mlp-deteccao-fraude/src/api.py` for `models/best_model.joblib`, `models/preprocessor.joblib`, and `models/best_run.json`
- [X] T033 [US3] Implement `POST /predict` in `/Users/nando_mac/mlp-deteccao-fraude/src/api.py` applying the saved preprocessor and selected decision threshold
- [X] T034 [US3] Implement clear missing-artifact error handling in `/Users/nando_mac/mlp-deteccao-fraude/src/api.py` explaining that training must run first
- [X] T035 [P] [US3] Implement complete sample prediction client in `/Users/nando_mac/mlp-deteccao-fraude/scripts/test_api.py` with all 30 required input fields
- [X] T036 [US3] Update prediction-service instructions in `/Users/nando_mac/mlp-deteccao-fraude/README.md` with `uvicorn src.api:app --reload`, request contract, and sample test command

## Phase 6: User Story 4 — Evaluation packaging and demo readiness

**Goal**: Evaluator can follow the README/video script and verify the repository is clean and ready for submission.

**Independent Test Criteria**: README contains a linear evaluator runbook, `reports/video_script.md` fits a five-minute demo, generated heavy files are ignored, and commands from `specs/001-fraud-detection-mlp/quickstart.md` are reflected in the root README.

- [X] T037 [P] [US4] Write five-minute video roteiro in `/Users/nando_mac/mlp-deteccao-fraude/reports/video_script.md` covering problem, notebook EDA, MLP experiments, MLflow comparison, API prediction, and final model justification
- [X] T038 [US4] Complete evaluator runbook in `/Users/nando_mac/mlp-deteccao-fraude/README.md` with setup, dataset, notebook, training, MLflow, API, test, and troubleshooting sections
- [X] T039 [US4] Add model-selection answer template to `/Users/nando_mac/mlp-deteccao-fraude/README.md` explaining how metrics justify the best model after training
- [X] T040 [US4] Ensure `.gitignore` in `/Users/nando_mac/mlp-deteccao-fraude/.gitignore` excludes raw dataset, `mlruns/`, virtual environments, notebook checkpoints, caches, and generated local-only files
- [X] T041 [US4] Align root README commands with `/Users/nando_mac/mlp-deteccao-fraude/specs/001-fraud-detection-mlp/quickstart.md`

## Final Phase: Polish and Cross-Cutting Concerns

- [X] T042 Run Python syntax validation with `python -m py_compile src/*.py scripts/*.py` from `/Users/nando_mac/mlp-deteccao-fraude`
- [X] T043 Validate OpenAPI contract by parsing `/Users/nando_mac/mlp-deteccao-fraude/specs/001-fraud-detection-mlp/contracts/openapi.yaml`
- [X] T044 Run import smoke checks for `/Users/nando_mac/mlp-deteccao-fraude/src/preprocessing.py`, `/Users/nando_mac/mlp-deteccao-fraude/src/experiments.py`, and `/Users/nando_mac/mlp-deteccao-fraude/src/api.py`
- [X] T045 If dataset is available, run `python -m src.train` from `/Users/nando_mac/mlp-deteccao-fraude` and verify all expected reports and model artifacts exist
- [X] T046 If model artifacts exist, run local API startup and prediction smoke test using `/Users/nando_mac/mlp-deteccao-fraude/scripts/test_api.py`
- [X] T047 Run repository cleanliness check with `git status --short` from `/Users/nando_mac/mlp-deteccao-fraude` and confirm ignored heavy/runtime files are not staged
- [X] T048 Run final unresolved-marker scan across `/Users/nando_mac/mlp-deteccao-fraude/README.md`, `/Users/nando_mac/mlp-deteccao-fraude/reports`, and `/Users/nando_mac/mlp-deteccao-fraude/specs/001-fraud-detection-mlp`

## Dependencies

### Phase Dependencies

- Phase 1 Setup must complete before all other phases.
- Phase 2 Foundational must complete before US2 and US3.
- US1 can proceed after Phase 1 and does not require model training.
- US2 requires Phase 2 and produces artifacts needed by US3.
- US3 requires promoted model artifacts from US2 for full end-to-end validation.
- US4 can begin after Phase 1 but should be finalized after US1, US2, and US3 content exists.
- Final Polish depends on all user-story phases.

### User Story Completion Order

1. **US1** first for mandatory notebook acceptance and dataset understanding.
2. **US2** second because training, tracking, reports, and promoted artifacts are the core project value.
3. **US3** third because prediction depends on promoted selected-model artifacts.
4. **US4** last because packaging must reflect the final implemented workflow.

## Parallel Execution Examples

### US1 Parallel Work

- T013, T014, T015, and T016 can be drafted in parallel if each contributor edits separate notebook cells carefully or one contributor owns the notebook while another updates README with T017.

### US2 Parallel Work

- T019 through T023 should be sequential inside `src/train.py` because later logic depends on earlier data and metric functions.
- T030 can proceed in parallel with T019 through T029 because README command structure can be drafted before final metric values exist.

### US3 Parallel Work

- T031 and T035 can run in parallel because the API schema and sample client are separate files.
- T032 through T034 should be sequential because prediction behavior depends on artifact loading.

### US4 Parallel Work

- T037 can run in parallel with T038 because video script and README are separate files.
- T040 can run in parallel with documentation tasks because `.gitignore` is independent.

## Implementation Strategy

### MVP First

Complete Phases 1 and 2, then US1 and US2. At that point the academic core is demonstrable: dataset exploration, five experiments, tracking, comparison, and model selection.

### Incremental Delivery

1. Deliver setup and mandatory notebook evidence.
2. Deliver full training and MLflow/report artifacts.
3. Deliver local prediction service and sample client.
4. Deliver README, video roteiro, and final repository hygiene.

### Validation Strategy

Validate each user story independently using its listed criteria before moving to the next dependent story. Do not claim final completion until syntax checks, contract parsing, training artifact checks, API smoke test, and repository hygiene checks have been performed where prerequisites are available.
