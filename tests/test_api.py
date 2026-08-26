from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_valid_transaction():
    transaction = {
        "transaction_amount": 5000,
        "transaction_type": "TRANSFER",
        "payment_mode": "UPI",
        "device_type": "Mobile",
        "device_location": "Chennai",
        "account_age_days": 365,
        "transaction_hour": 14,
        "previous_failed_attempts": 0,
        "avg_transaction_amount": 4500,
        "is_international": 0,
        "ip_risk_score": 0.2,
        "login_attempts_last_24h": 2
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_prediction" in data
    assert data["fraud_prediction"] in [0, 1]
    assert data["schema_version"] == "1.0.0"
    assert data["model_version"] == "test"


def test_predict_missing_fields():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


def test_health_and_readiness():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {
        "status": "ready",
        "model_version": "test",
        "schema_version": "1.0.0",
    }


def test_metrics_include_request_counters():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "fraud_api_requests_total" in response.text
    assert "fraud_api_request_latency_ms_total" in response.text
    assert "fraud_api_predictions_total" in response.text
    assert "fraud_api_prediction_latency_ms_total" in response.text


def test_predict_unknown_categories():
    transaction = {
        "transaction_amount": 5000,
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
        "login_attempts_last_24h": 2
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_prediction" in data
    assert data["fraud_prediction"] in [0, 1]
