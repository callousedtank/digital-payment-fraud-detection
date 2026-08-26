import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from src.model_registry import resolve_model_path
from src.predict import load_model, predict_transaction

LEGACY_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION")
MODEL_REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "models/model_registry.json")
API_SCHEMA_VERSION = "1.0.0"


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for field in ("method", "path", "status_code", "latency_ms"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload)


logger = logging.getLogger("fraud_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

metrics = Counter()
metrics_lock = Lock()

app = FastAPI(
    title="Digital Payment Fraud Detection API",
    version="1.0.0"
)

model_path, resolved_model_version = resolve_model_path(
    MODEL_VERSION,
    MODEL_REGISTRY_PATH,
    LEGACY_MODEL_PATH,
)

if not model_path.exists():
    raise RuntimeError(
        f"Model artifact not found: {model_path}. "
        "Run `python src/train.py` before starting the API."
    )

artifact = load_model(model_path)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - started_at) * 1000
        with metrics_lock:
            metrics["api_errors_total"] += 1
        logger.exception(
            "request_failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "latency_ms": round(latency_ms, 2),
            },
        )
        raise

    latency_ms = (time.perf_counter() - started_at) * 1000
    with metrics_lock:
        metrics["api_requests_total"] += 1
        metrics[f"api_responses_{response.status_code}_total"] += 1
        metrics["api_request_latency_ms_total"] += latency_ms
        if response.status_code >= 500:
            metrics["api_errors_total"] += 1
    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )
    return response


class Transaction(BaseModel):
    transaction_amount: float
    transaction_type: str
    payment_mode: str
    device_type: str
    device_location: str
    account_age_days: int
    transaction_hour: int
    previous_failed_attempts: int
    avg_transaction_amount: float
    is_international: int
    ip_risk_score: float
    login_attempts_last_24h: int


@app.get("/")
def root():
    return {"message": "Digital Payment Fraud Detection API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    metadata = artifact.get("metadata", {})
    return {
        "status": "ready",
        "model_version": metadata.get("model_version", resolved_model_version),
        "schema_version": API_SCHEMA_VERSION,
    }


@app.get("/metrics", response_class=Response)
def get_metrics():
    with metrics_lock:
        snapshot = dict(metrics)

    lines = [
        "# TYPE fraud_api_requests_total counter",
        f"fraud_api_requests_total {snapshot.get('api_requests_total', 0)}",
        "# TYPE fraud_api_errors_total counter",
        f"fraud_api_errors_total {snapshot.get('api_errors_total', 0)}",
        "# TYPE fraud_api_request_latency_ms_total counter",
        "fraud_api_request_latency_ms_total "
        f"{snapshot.get('api_request_latency_ms_total', 0)}",
        "# TYPE fraud_api_predictions_total counter",
        f"fraud_api_predictions_total {snapshot.get('predictions_total', 0)}",
        "# TYPE fraud_api_prediction_latency_ms_total counter",
        "fraud_api_prediction_latency_ms_total "
        f"{snapshot.get('prediction_latency_ms_total', 0)}",
    ]
    for key, value in sorted(snapshot.items()):
        if key.startswith("api_responses_"):
            status_code = key.removeprefix("api_responses_").removesuffix("_total")
            lines.append(
                "fraud_api_responses_total"
                f'{{status_code="{status_code}"}} {value}'
            )
    return Response("\n".join(lines) + "\n", media_type="text/plain")


@app.post("/predict")
def predict(transaction: Transaction):
    started_at = time.perf_counter()
    prediction = predict_transaction(transaction.model_dump(), artifact)
    latency_ms = (time.perf_counter() - started_at) * 1000

    with metrics_lock:
        metrics["predictions_total"] += 1
        metrics["prediction_latency_ms_total"] += latency_ms

    logger.info(
        "prediction_completed",
        extra={
            "method": "POST",
            "path": "/predict",
            "status_code": 200,
            "latency_ms": round(latency_ms, 2),
        },
    )

    return {
        "fraud_prediction": prediction,
        "schema_version": API_SCHEMA_VERSION,
        "model_version": artifact.get("metadata", {}).get(
            "model_version", resolved_model_version
        ),
    }
