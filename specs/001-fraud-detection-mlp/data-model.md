# Data Model: CP4 Fraud Detection Prediction System

## Entity: Transaction

Represents one credit card transaction.

### Fields

- `Time`: numeric elapsed time value from the dataset.
- `Amount`: numeric transaction amount.
- `V1` through `V28`: numeric anonymized PCA components.
- `Class`: optional known label for historical/evaluation data; `0` means legitimate and `1` means fraud.

### Validation Rules

- Prediction input must include `Time`, `Amount`, and every field from `V1` through `V28`.
- All prediction fields must be numeric.
- `Class` is required for training/evaluation rows and not accepted as a required prediction input.
- Extra input fields should not influence prediction.

### Relationships

- A transaction may produce one prediction result.
- Historical transactions are split into training, validation, and final test partitions.

## Entity: Experiment Run

Represents one candidate model evaluation.

### Fields

- `experiment_name`: stable name such as `MLP-01`.
- `hidden_layers`: candidate architecture descriptor.
- `learning_rate`: numeric learning rate.
- `activation`: activation choice.
- `solver`: optimization choice.
- `batch_size`: training batch size.
- `max_iter`: maximum training iterations.
- `imbalance_strategy`: training-only class imbalance treatment.
- `decision_threshold`: selected threshold based on validation evidence.
- `accuracy`: evaluation accuracy.
- `precision`: fraud-class precision.
- `recall`: fraud-class recall.
- `f1_score`: fraud-class F1 score.
- `roc_auc`: separability score.
- `artifact_uri`: recoverable model artifact location.

### Validation Rules

- Every run must contain all required parameters and metrics.
- At least five distinct runs must exist.
- Metrics used for comparison must be calculated on validation data during selection.

### Relationships

- Many experiment runs compete to become one selected model.
- Each run references one trained model artifact.

## Entity: Preprocessing Artifact

Represents the saved feature transformation used consistently by training and prediction.

### Fields

- `feature_order`: ordered list of expected prediction fields.
- `scaled_features`: fields standardized during preprocessing.
- `passthrough_features`: fields passed through unchanged.
- `artifact_path`: local saved path.

### Validation Rules

- The feature order must match training and prediction inputs.
- The artifact used by the prediction service must match the selected model.

### Relationships

- One selected model has one matching preprocessing artifact.
- Prediction requests must be transformed by this artifact before classification.

## Entity: Selected Model

Represents the promoted best candidate for local prediction.

### Fields

- `best_run_id`: identifier of the selected experiment run.
- `experiment_name`: selected candidate name.
- `selection_reason`: written explanation of why it was selected.
- `decision_threshold`: threshold used for final prediction.
- `final_test_metrics`: held-out performance metrics.
- `model_path`: stable local model artifact path.
- `preprocessor_path`: stable local preprocessing artifact path.

### Validation Rules

- Selection must be based on validation comparison and final test reporting.
- The selected model must be recoverable without retraining.
- The selection reason must mention fraud-class precision, recall, F1-score, and ROC-AUC.

### Relationships

- Exactly one selected model is promoted for the local prediction service.
- One selected model produces many prediction results.

## Entity: Prediction Result

Represents the response for one prediction request.

### Fields

- `prediction`: integer class, `0` or `1`.
- `label`: readable label, `legitima` or `fraude`.
- `fraud_probability`: numeric probability from 0.0 to 1.0.

### Validation Rules

- `prediction` must match the configured decision threshold applied to fraud probability.
- `label` must correspond to `prediction`.
- `fraud_probability` must be bounded between 0.0 and 1.0.
