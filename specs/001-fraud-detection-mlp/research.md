# Phase 0 Research: CP4 Fraud Detection Prediction System

## Decision: Use a three-way stratified split

**Rationale**: Fraud is rare, so every split must preserve class proportions. A training/validation/test split supports honest model selection: training fits the candidate, validation chooses the prediction threshold and compares candidates, and final test reports selected-model performance only once.

**Alternatives considered**:
- Train/test only: simpler, but threshold choice on test would overstate performance.
- Cross-validation only: useful but more complex and slower for an academic MLP deliverable with MLflow runs.

## Decision: Scale `Time` and `Amount`; pass through `V1`-`V28`

**Rationale**: `V1` through `V28` are already PCA-derived numeric components. `Time` and `Amount` are raw numeric fields on different scales and should be standardized. A reusable feature preprocessor ensures training and prediction consistency.

**Alternatives considered**:
- Scale all columns: acceptable but unnecessary for PCA columns and less aligned with dataset documentation.
- No scaling: risky for MLP convergence because amount/time magnitudes can dominate optimization.

## Decision: Use SMOTE for most experiments and undersampling once

**Rationale**: The target class is extremely rare. SMOTE increases minority examples in training and generally supports recall. A single undersampling experiment demonstrates comparison against a simpler balancing strategy. Resampling is restricted to the training set to avoid leakage.

**Alternatives considered**:
- Class weights: `MLPClassifier` does not natively support `class_weight`, making it less direct.
- Oversampling by duplication: simpler but can overfit duplicate minority points.
- All undersampling: faster but discards many legitimate transactions.

## Decision: Select by fraud-class F1 subject to recall floor

**Rationale**: In fraud detection, accuracy is misleading. Recall matters because missed fraud is expensive, while precision matters because false alarms burden review. F1 balances the two. ROC-AUC is retained as a tie-breaker and general separability signal.

**Alternatives considered**:
- Accuracy: rejected because a trivial all-legitimate classifier scores very high.
- Highest recall only: can create too many false positives.
- ROC-AUC only: threshold-independent and useful, but less directly tied to operating decisions.

## Decision: Tune decision threshold on validation predictions

**Rationale**: The default 0.5 threshold may not be appropriate for highly imbalanced fraud detection. Selecting a threshold from validation evidence can improve the recall/precision tradeoff while preserving a final test set for honest reporting.

**Alternatives considered**:
- Always use 0.5: simpler but may underperform for the minority class.
- Tune threshold on test: rejected due to evaluation leakage.

## Decision: Promote stable local prediction artifacts while retaining MLflow tracking

**Rationale**: MLflow provides traceability, but local demos are more reliable when the prediction service loads stable promoted files such as `models/best_model.joblib`, `models/preprocessor.joblib`, and `models/best_run.json`. Static reports protect evaluation even when the MLflow UI is not opened by the evaluator.

**Alternatives considered**:
- Load directly from internal `mlruns/` paths: traceable but brittle for demo and path portability.
- Commit all MLflow artifacts: can bloat the repository and conflicts with a clean academic delivery.

## Decision: Keep notebook mandatory but limited to EDA

**Rationale**: The course requires notebook evidence. Keeping EDA in the notebook satisfies acceptance while scripts provide reproducible training, tracking, and API behavior.

**Alternatives considered**:
- Notebook does everything: easier to inspect but weaker for API and repeatable training.
- No notebook: rejected because notebook is mandatory for acceptance.
