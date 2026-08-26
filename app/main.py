import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.model_registry import resolve_model_path
from src.predict import load_model, predict_transaction_details

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model_path, resolved_model_version = resolve_model_path(
    MODEL_VERSION,
    MODEL_REGISTRY_PATH,
    LEGACY_MODEL_PATH,
)

if not model_path.exists():
    raise RuntimeError(
        f"Model artifact not found: {model_path}. "
        "Run `python -m src.train` before starting the API."
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
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, allow_inf_nan=False
    )

    transaction_amount: float = Field(ge=0)
    transaction_type: str = Field(min_length=1, max_length=100)
    payment_mode: str = Field(min_length=1, max_length=100)
    device_type: str = Field(min_length=1, max_length=100)
    device_location: str = Field(min_length=1, max_length=100)
    account_age_days: int = Field(ge=0)
    transaction_hour: int = Field(ge=0, le=23)
    previous_failed_attempts: int = Field(ge=0)
    avg_transaction_amount: float = Field(ge=0)
    is_international: int = Field(ge=0, le=1)
    ip_risk_score: float = Field(ge=0, le=1)
    login_attempts_last_24h: int = Field(ge=0)


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
    prediction = predict_transaction_details(transaction.model_dump(), artifact)
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
        **prediction,
        "schema_version": API_SCHEMA_VERSION,
        "model_version": artifact.get("metadata", {}).get(
            "model_version", resolved_model_version
        ),
    }
