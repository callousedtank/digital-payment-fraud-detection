import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.main import (
    PredictionRequest,
    get_metrics,
    health,
    predict,
    ready,
)


VALID_FEATURES = {
    "transaction_amount": 5000,
    "transaction_type": "Transfer",
    "payment_mode": "UPI",
    "device_type": "Android",
    "device_location": "Chennai",
    "account_age_days": 365,
    "transaction_hour": 14,
    "previous_failed_attempts": 0,
    "avg_transaction_amount": 4500,
    "is_international": 0,
    "ip_risk_score": 0.2,
    "login_attempts_last_24h": 2,
}


def test_predict_valid_transaction():
    request = PredictionRequest(
        features=VALID_FEATURES
    )

    data = predict(request)

    assert "fraud_prediction" in data
    assert data["fraud_prediction"] in [0, 1]
    assert data["fraud_probability"] == 0.1
    assert data["decision_threshold"] == 0.5
    assert data["schema_version"] == "2.0.0"
    assert data["model_version"] == "test"


def test_predict_missing_features():
    features = VALID_FEATURES.copy()
    features.pop("transaction_amount")

    request = PredictionRequest(
        features=features
    )

    with pytest.raises(HTTPException) as exc_info:
        predict(request)

    assert exc_info.value.status_code == 400
    assert "Missing required features" in exc_info.value.detail


def test_predict_extra_features():
    features = {
        **VALID_FEATURES,
        "unexpected_feature": 123,
    }

    request = PredictionRequest(
        features=features
    )

    with pytest.raises(HTTPException) as exc_info:
        predict(request)

    assert exc_info.value.status_code == 400
    assert "Unexpected features" in exc_info.value.detail


def test_health_and_readiness():
    assert health() == {
        "status": "ok"
    }

    readiness = ready()

    assert readiness["status"] == "ready"
    assert readiness["model_version"] == "test"
    assert readiness["schema_version"] == "2.0.0"
    assert readiness["feature_count"] == 12
    assert readiness["classifier"] == "TestModel"


def test_metrics_include_request_counters():
    response = get_metrics()

    assert response.status_code == 200
    assert b"fraud_api_requests_total" in response.body
    assert b"fraud_api_request_latency_ms_total" in response.body
    assert b"fraud_api_predictions_total" in response.body
    assert b"fraud_api_prediction_latency_ms_total" in response.body


def test_predict_accepts_unknown_categories():
    features = {
        **VALID_FEATURES,
        "transaction_type": "UNKNOWN_TRANSACTION_TYPE",
        "payment_mode": "UNKNOWN_PAYMENT_MODE",
        "device_type": "UNKNOWN_DEVICE",
        "device_location": "UNKNOWN_LOCATION",
    }

    request = PredictionRequest(
        features=features
    )

    data = predict(request)

    assert data["fraud_prediction"] in [0, 1]


def test_prediction_request_rejects_nan():
    features = {
        **VALID_FEATURES,
        "transaction_amount": float("nan"),
    }

    request = PredictionRequest(
        features=features
    )

    with pytest.raises(HTTPException) as exc_info:
        predict(request)

    assert exc_info.value.status_code == 400
    assert "non-finite" in exc_info.value.detail