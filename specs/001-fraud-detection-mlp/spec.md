# Feature Specification: CP4 Fraud Detection Prediction System

**Feature Branch**: `001-fraud-detection-mlp`  
**Created**: 2026-09-03  
**Status**: Draft  
**Input**: User description: "CP4 — Detecção de Fraude em Cartão de Crédito; MLP + experiment tracking + local prediction API; mandatory EDA notebook; academic delivery focused on reproducible evaluation."

## User Scenarios & Testing

### Primary User Story

As a course evaluator, I want to review a complete fraud detection machine learning project so that I can verify the team explored the dataset, trained multiple candidate models, selected the best candidate using appropriate imbalanced-classification metrics, and demonstrated local prediction on a new transaction.

### Acceptance Scenarios

1. **Given** the evaluator has the repository and the required dataset file in the documented location, **When** they follow the setup and training instructions, **Then** the project records at least five comparable experiment runs with parameters, metrics, and model artifacts.
2. **Given** experiment runs exist, **When** the evaluator reviews the comparison evidence, **Then** they can identify which model was selected and why that model is best for fraud detection.
3. **Given** the selected model has been promoted for prediction, **When** the evaluator submits a valid transaction containing all required transaction fields, **Then** the system returns whether the transaction is legitimate or fraudulent and includes a fraud probability.
4. **Given** the evaluator opens the required exploratory notebook, **When** they inspect it, **Then** they see dataset shape, target distribution, missing-value checks, class imbalance analysis, and basic feature summaries relevant to fraud detection.

### Edge Cases

- Dataset file is missing from the expected local path.
- A prediction request omits one or more required transaction fields.
- A prediction request includes non-numeric values for transaction fields.
- No experiment satisfies the target metric thresholds simultaneously.
- Multiple experiments have similar fraud-class F1 scores.
- The selected model artifacts are missing or inconsistent with the preprocessing artifacts.

## Requirements

### Functional Requirements

- **FR-001**: The project MUST document how to obtain the credit card fraud dataset and where to place it locally without storing the raw dataset in version control.
- **FR-002**: The project MUST include mandatory exploratory analysis evidence in a notebook showing dataset size, columns, missing values, class distribution, fraud proportion, and basic summaries for transaction time and amount.
- **FR-003**: The project MUST split data into training, validation, and final test portions while preserving the fraud/legitimate class proportion in each split.
- **FR-004**: The project MUST apply a reusable preprocessing process to transaction fields so that training and prediction use the same feature transformations.
- **FR-005**: The project MUST address class imbalance during model training without applying resampling to validation, final test, or single prediction inputs.
- **FR-006**: The project MUST run at least five distinct model experiments that vary meaningful training configuration choices.
- **FR-007**: Each experiment MUST record its configuration parameters, fraud-class and overall evaluation metrics, and trained model artifact in an experiment tracking record.
- **FR-008**: The project MUST calculate at minimum accuracy, fraud-class precision, fraud-class recall, fraud-class F1-score, and ROC-AUC for experiment comparison.
- **FR-009**: The project MUST select the best model by maximizing fraud-class F1-score among candidates with acceptable fraud recall; ties MUST be resolved using ROC-AUC.
- **FR-010**: The project MUST choose any prediction threshold using validation evidence only and MUST report final selected-model metrics on a held-out final test set.
- **FR-011**: The project MUST promote the selected model and matching preprocessing artifact to stable local files for prediction demonstration.
- **FR-012**: The project MUST expose a local prediction capability that accepts one transaction with all required numeric fields and returns a class prediction, human-readable label, and fraud probability.
- **FR-013**: The project MUST include a reproducible prediction test example that an evaluator can run without inventing input data.
- **FR-014**: The project MUST include a written explanation answering why the selected model was considered best and which metrics support the decision.
- **FR-015**: The project MUST provide a clean repository with generated caches, raw data, local tracking stores, and large runtime artifacts excluded from version control unless explicitly required for evaluation.
- **FR-016**: The project MUST include a video demonstration script or checklist covering the problem, model experimentation, experiment comparison, prediction demonstration, and final justification within five minutes.

### Key Entities

- **Transaction**: A single card transaction represented by anonymized numeric components, transaction time, transaction amount, and optionally a known class label for evaluation data.
- **Experiment Run**: One trained candidate with a named configuration, recorded parameters, evaluation metrics, and model artifact.
- **Preprocessing Artifact**: The saved transformation process used consistently before model training and prediction.
- **Selected Model**: The promoted candidate chosen from experiment evidence for local prediction.
- **Prediction Result**: The output for one transaction, including numeric class prediction, readable label, and fraud probability.
- **EDA Notebook**: The required exploratory evidence artifact used to document dataset understanding before training.

## Success Criteria

### Measurable Outcomes

- **SC-001**: An evaluator can complete environment setup from the README using the documented command sequence without needing undocumented steps.
- **SC-002**: The exploratory notebook documents 100% of the required dataset checks: shape, column list, missing values, class counts, fraud percentage, and basic time/amount summaries.
- **SC-003**: At least five experiment runs are available for comparison, and each run contains the required parameters, metrics, and a recoverable trained model artifact.
- **SC-004**: The selected model reports final held-out metrics for fraud-class precision, fraud-class recall, fraud-class F1-score, and ROC-AUC.
- **SC-005**: The selected model targets recall of at least 0.80, precision of at least 0.60, F1-score of at least 0.70, and ROC-AUC of at least 0.90; if any target is not met, the README explains the observed tradeoff and why the selected model remains the best candidate.
- **SC-006**: A valid sample transaction returns a prediction response containing exactly the prediction class, readable label, and fraud probability.
- **SC-007**: Invalid prediction input is rejected with a clear error instead of returning a misleading prediction.
- **SC-008**: The final repository excludes raw dataset files, local caches, notebook checkpoints, and local experiment tracking directories from version control.
- **SC-009**: The demonstration can be completed in five minutes or less while showing the notebook, experiment comparison, selected-model justification, and a working prediction.

## Assumptions

- The project is evaluated as an academic CP4 delivery, not as a production antifraud service.
- The raw dataset is downloaded by the evaluator or student and placed locally according to the README.
- The mandatory notebook is used for exploratory analysis evidence; training and prediction are demonstrated through reproducible project commands.
- Class imbalance is handled during training only, and validation/test evaluation uses the natural class distribution.
- If final metric targets are not all reached, honest reporting and selection rationale are acceptable as long as experimentation and tradeoffs are clear.

## Out of Scope

- Cloud deployment.
- Authentication or authorization for the local prediction service.
- User interface or dashboard beyond experiment comparison and local prediction demonstration.
- Automated retraining, monitoring, alerting, or production operations.
- Storing the raw dataset in the repository.
