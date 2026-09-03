# Model Selection

Selected model: **MLP-04**

The selected model had validation recall 0.8108, precision 0.6316, F1-score 0.7101, and ROC-AUC 0.9813. It was selected by maximizing fraud-class F1 among candidates that reached the recall floor of 0.80, using ROC-AUC as tie-breaker. MLP-02 and MLP-05 had higher validation F1, but both were rejected by the selection rule because their validation recall stayed below 0.80.

## Final held-out test metrics

- Accuracy: 0.9991
- Fraud precision: 0.6977
- Fraud recall: 0.8108
- Fraud F1-score: 0.7500
- ROC-AUC: 0.9657

The selected model met all reference targets on the held-out test set: recall >= 0.80, precision >= 0.60, F1-score >= 0.70, and ROC-AUC >= 0.90.
