# Model Selection

Selected model: **MLP-03**

Run ID: `80642936d52a4121858484fb3acb1877`. Dataset policy: `exact_duplicates_removed_v1`. Decision threshold: 0.83.

The selected model had validation recall 0.8028, precision 0.7600, F1-score 0.7808, and ROC-AUC 0.9617. It was selected by maximizing fraud-class F1 among candidates with acceptable recall, using ROC-AUC as tie-breaker.

## Final held-out test metrics

- Accuracy: 0.9990
- Fraud precision: 0.6667
- Fraud recall: 0.7887
- Fraud F1-score: 0.7226
- ROC-AUC: 0.9535

Reference targets: recall >= 0.80, precision >= 0.60, F1 >= 0.70, ROC-AUC >= 0.90. Targets are reference goals, not guarantees. Selection uses validation only. Exact duplicate records were removed before splitting; see data_preparation.json.

Test reference targets not met: recall. The test set was not used to retune the model or threshold.
