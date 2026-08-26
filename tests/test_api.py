import pytest
from pydantic import ValidationError

from app.main import Transaction, get_metrics, health, predict, ready


VALID_TRANSACTION = {
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
    data = predict(Transaction(**VALID_TRANSACTION))

    assert "fraud_prediction" in data
    assert data["fraud_prediction"] in [0, 1]
    assert data["fraud_probability"] == 0.1
    assert data["decision_threshold"] == 0.5
    assert data["schema_version"] == "1.0.0"
    assert data["model_version"] == "test"


def test_predict_missing_fields():
    with pytest.raises(ValidationError):
        Transaction()


def test_health_and_readiness():
    assert health() == {"status": "ok"}
    assert ready() == {
        "status": "ready",
        "model_version": "test",
        "schema_version": "1.0.0",
    }


def test_metrics_include_request_counters():
    response = get_metrics()

    assert response.status_code == 200
    assert b"fraud_api_requests_total" in response.body
    assert b"fraud_api_request_latency_ms_total" in response.body
    assert b"fraud_api_predictions_total" in response.body
    assert b"fraud_api_prediction_latency_ms_total" in response.body


def test_predict_unknown_categories():
    transaction = {
        **VALID_TRANSACTION,
        "transaction_type": "UNKNOWN_TRANSACTION_TYPE",
        "payment_mode": "UNKNOWN_PAYMENT_MODE",
        "device_type": "UNKNOWN_DEVICE",
        "device_location": "UNKNOWN_LOCATION",
        "account_age_days": 365,
        "transaction_hour": 14,
        "previous_failed_attempts": 0,
        "avg_transaction_amount": 4500,
        "is_international": 0,
        "ip_risk_score": 0.2,
    }
    data = predict(Transaction(**transaction))

    assert "fraud_prediction" in data
    assert data["fraud_prediction"] in [0, 1]


@pytest.mark.parametrize(
    ("field", "value"),
    [("transaction_hour", 24), ("ip_risk_score", 1.1), ("is_international", 2)],
)
def test_predict_rejects_invalid_domain_values(field, value):
    transaction = {**VALID_TRANSACTION, field: value}
    with pytest.raises(ValidationError):
        Transaction(**transaction)
