import os
from pathlib import Path

import joblib


class TestModel:
    def predict(self, frame):
        return [0] * len(frame)


def pytest_sessionstart(session):
    artifact_path = Path(session.config._tmp_path_factory.getbasetemp()) / "model.joblib"
    joblib.dump(
        {
            "model": TestModel(),
            "encoders": {},
            "categorical_indices": [],
            "feature_names": [
                "transaction_amount",
                "transaction_type",
                "payment_mode",
                "device_type",
                "device_location",
                "account_age_days",
                "transaction_hour",
                "previous_failed_attempts",
                "avg_transaction_amount",
                "is_international",
                "ip_risk_score",
                "login_attempts_last_24h",
            ],
            "metadata": {"model_version": "test"},
        },
        artifact_path,
    )
    os.environ["MODEL_PATH"] = str(artifact_path)
