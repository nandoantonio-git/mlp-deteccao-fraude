"""Integration checks against the locally selected MLflow run (no retraining)."""
import json
import unittest
from unittest.mock import patch

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from src import api
from src.preprocessing import BEST_RUN_PATH, MLFLOW_TRACKING_URI, transaction_to_frame
from scripts.test_api import SAMPLE_TRANSACTION


@unittest.skipUnless(BEST_RUN_PATH.exists(), "Train a model before running local integration checks")
class PredictionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.context = TestClient(api.app)
        self.client = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def test_prediction_matches_mlflow_artifacts_and_threshold(self):
        run_id = json.loads(BEST_RUN_PATH.read_text())["best_run_id"]
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run = client.get_run(run_id)
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/sklearn_model")
        preprocessor = joblib.load(client.download_artifacts(run_id, "preprocessor/preprocessor.joblib"))
        probability = float(model.predict_proba(preprocessor.transform(transaction_to_frame(SAMPLE_TRANSACTION)))[0, 1])
        result = self.client.post("/predict", json=SAMPLE_TRANSACTION)
        self.assertEqual(result.status_code, 200, result.text)
        body = result.json()
        self.assertAlmostEqual(body["fraud_probability"], probability)
        self.assertEqual(body["prediction"], int(probability >= float(run.data.params["decision_threshold"])))
        self.assertEqual(body["label"], "fraude" if body["prediction"] else "legitima")
        self.assertEqual(self.client.get("/health").json()["run_id"], run_id)

    def test_invalid_inputs_are_rejected(self):
        for value in [True, "abc", "NaN", "Infinity", None]:
            with self.subTest(value=value):
                response = self.client.post("/predict", json={**SAMPLE_TRANSACTION, "Amount": value})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post("/predict", json={}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={**SAMPLE_TRANSACTION, "extra": 1}).status_code, 422)

    def test_prediction_uses_run_threshold_instead_of_default_half(self):
        threshold = api._best_run["decision_threshold"]
        if threshold == 0.5:
            self.skipTest("Selected run uses the default threshold")
        probability = (threshold + 0.5) / 2
        with patch.object(api._model, "predict_proba", return_value=np.array([[1 - probability, probability]])):
            response = self.client.post("/predict", json=SAMPLE_TRANSACTION)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], int(probability >= threshold))
        self.assertNotEqual(response.json()["prediction"], int(probability >= 0.5))

    def test_unavailable_run_clears_readiness_and_returns_503(self):
        with patch.object(api.MlflowClient, "get_run", side_effect=RuntimeError("Unavailable run")):
            with self.assertRaises(RuntimeError):
                api.load_artifacts()
            self.assertFalse(self.client.get("/health").json()["ready"])
            self.assertEqual(self.client.post("/predict", json=SAMPLE_TRANSACTION).status_code, 503)

    def test_invalid_threshold_cannot_mark_model_ready(self):
        run = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).get_run(api._best_run["best_run_id"])
        run.data.params["decision_threshold"] = "nan"
        with patch.object(api.MlflowClient, "get_run", return_value=run):
            with self.assertRaises(RuntimeError):
                api.load_artifacts()
            self.assertFalse(self.client.get("/health").json()["ready"])


if __name__ == "__main__":
    unittest.main()
