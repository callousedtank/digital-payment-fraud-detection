import json
import logging
import math
import os
import time
from collections import Counter
from functools import lru_cache
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from src.model_registry import (
    load_registry,
    resolve_model_path,
)
from src.predict import load_model, predict_transaction_details


LEGACY_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/model.joblib",
)

MODEL_VERSION = os.getenv("MODEL_VERSION")

MODEL_REGISTRY_PATH = os.getenv(
    "MODEL_REGISTRY_PATH",
    "models/model_registry.json",
)

API_SCHEMA_VERSION = "2.1.0"


logger = logging.getLogger("fraud_api")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        for field in (
            "method",
            "path",
            "status_code",
            "latency_ms",
        ):
            value = getattr(record, field, None)

            if value is not None:
                payload[field] = value

        return json.dumps(payload)


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
    version=API_SCHEMA_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOW_ORIGINS",
            "*",
        ).split(",")
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def resolve_artifact(version=None):
    selected_version = version or MODEL_VERSION

    model_path, resolved_version = resolve_model_path(
        selected_version,
        MODEL_REGISTRY_PATH,
        LEGACY_MODEL_PATH,
    )

    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found: {model_path}. "
            "Run `python -m src.train` before starting the API."
        )

    return model_path, resolved_version


@lru_cache(maxsize=16)
def load_registered_model(version):
    model_path, resolved_version = resolve_artifact(
        version
    )

    artifact = load_model(model_path)

    return artifact, resolved_version


def get_model(version=None):
    requested_version = version or MODEL_VERSION

    return load_registered_model(
        requested_version
    )


# Load the default model once for metadata/readiness endpoints.
default_artifact, default_model_version = get_model()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    features: dict
    model_version: str | None = None


def validate_request_features(
    features,
    feature_names,
):
    expected_features = set(
        feature_names
    )

    received_features = set(
        features
    )

    missing_features = sorted(
        expected_features - received_features
    )

    extra_features = sorted(
        received_features - expected_features
    )

    if missing_features:
        raise ValueError(
            f"Missing required features: "
            f"{missing_features}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features: "
            f"{extra_features}"
        )

    for name, value in features.items():
        if isinstance(value, bool):
            continue

        if isinstance(
            value,
            (int, float),
        ) and not math.isfinite(value):
            raise ValueError(
                f"Feature '{name}' contains "
                "a non-finite value"
            )


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    started_at = time.perf_counter()

    try:
        response = await call_next(
            request
        )

    except Exception:
        latency_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        with metrics_lock:
            metrics[
                "api_errors_total"
            ] += 1

        logger.exception(
            "request_failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "latency_ms": round(
                    latency_ms,
                    2,
                ),
            },
        )

        raise

    latency_ms = (
        time.perf_counter()
        - started_at
    ) * 1000

    with metrics_lock:
        metrics[
            "api_requests_total"
        ] += 1

        metrics[
            f"api_responses_{response.status_code}_total"
        ] += 1

        metrics[
            "api_request_latency_ms_total"
        ] += latency_ms

        if response.status_code >= 500:
            metrics[
                "api_errors_total"
            ] += 1

    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(
                latency_ms,
                2,
            ),
        },
    )

    return response


@app.get("/")
def root():
    return {
        "message": (
            "Digital Payment Fraud Detection API"
        )
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/ready")
def ready():
    metadata = default_artifact.get(
        "metadata",
        {},
    )

    training_configuration = metadata.get(
        "training_configuration",
        {},
    )

    return {
        "status": "ready",
        "model_version": metadata.get(
            "model_version",
            default_model_version,
        ),
        "schema_version": API_SCHEMA_VERSION,
        "feature_count": len(
            default_artifact[
                "feature_names"
            ]
        ),
        "classifier": metadata.get(
            "classifier"
        ),
        "dataset": training_configuration.get(
            "dataset"
        ),
    }


@app.get("/model")
def model_info():
    metadata = default_artifact.get(
        "metadata",
        {},
    )

    training_configuration = metadata.get(
        "training_configuration",
        {},
    )

    return {
        "model_version": metadata.get(
            "model_version",
            default_model_version,
        ),
        "classifier": metadata.get(
            "classifier"
        ),
        "dataset": training_configuration.get(
            "dataset"
        ),
        "resampling": training_configuration.get(
            "resampling"
        ),
        "feature_count": len(
            default_artifact[
                "feature_names"
            ]
        ),
        "feature_names": default_artifact[
            "feature_names"
        ],
        "schema_version": API_SCHEMA_VERSION,
    }


@app.get("/models")
def available_models():
    try:
        registry = load_registry(
            MODEL_REGISTRY_PATH
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to load model registry: "
                f"{exc}"
            ),
        ) from exc

    models = []

    for version, entry in registry.get(
        "models",
        {},
    ).items():
        metadata = entry.get(
            "metadata",
            {},
        )

        training_configuration = metadata.get(
            "training_configuration",
            {},
        )

        models.append(
            {
                "model_version": version,
                "dataset": training_configuration.get(
                    "dataset"
                ),
                "classifier": metadata.get(
                    "classifier"
                ),
                "resampling": training_configuration.get(
                    "resampling"
                ),
                "validation_status": entry.get(
                    "validation_status"
                ),
            }
        )

    return {
        "active_version": registry.get(
            "active_version"
        ),
        "models": models,
    }


@app.get(
    "/metrics",
    response_class=Response,
)
def get_metrics():
    with metrics_lock:
        snapshot = dict(metrics)

    lines = [
        "# TYPE fraud_api_requests_total counter",
        (
            "fraud_api_requests_total "
            f"{snapshot.get('api_requests_total', 0)}"
        ),
        "# TYPE fraud_api_errors_total counter",
        (
            "fraud_api_errors_total "
            f"{snapshot.get('api_errors_total', 0)}"
        ),
        "# TYPE fraud_api_request_latency_ms_total counter",
        (
            "fraud_api_request_latency_ms_total "
            f"{snapshot.get('api_request_latency_ms_total', 0)}"
        ),
        "# TYPE fraud_api_predictions_total counter",
        (
            "fraud_api_predictions_total "
            f"{snapshot.get('predictions_total', 0)}"
        ),
        "# TYPE fraud_api_prediction_latency_ms_total counter",
        (
            "fraud_api_prediction_latency_ms_total "
            f"{snapshot.get('prediction_latency_ms_total', 0)}"
        ),
    ]

    for key, value in sorted(
        snapshot.items()
    ):
        if key.startswith(
            "api_responses_"
        ):
            status_code = (
                key
                .removeprefix(
                    "api_responses_"
                )
                .removesuffix(
                    "_total"
                )
            )

            lines.append(
                "fraud_api_responses_total"
                f'{{status_code="{status_code}"}} '
                f"{value}"
            )

    return Response(
        "\n".join(lines) + "\n",
        media_type="text/plain",
    )


@app.post("/predict")
def predict(
    request: PredictionRequest,
):
    try:
        artifact, resolved_version = (
            get_model(
                request.model_version
            )
        )

        validate_request_features(
            request.features,
            artifact[
                "feature_names"
            ],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load requested "
                f"model: {exc}"
            ),
        ) from exc

    started_at = time.perf_counter()

    prediction = (
        predict_transaction_details(
            request.features,
            artifact,
        )
    )

    latency_ms = (
        time.perf_counter()
        - started_at
    ) * 1000

    with metrics_lock:
        metrics[
            "predictions_total"
        ] += 1

        metrics[
            "prediction_latency_ms_total"
        ] += latency_ms

    metadata = artifact.get(
        "metadata",
        {},
    )

    training_configuration = metadata.get(
        "training_configuration",
        {},
    )

    logger.info(
        "prediction_completed",
        extra={
            "method": "POST",
            "path": "/predict",
            "status_code": 200,
            "latency_ms": round(
                latency_ms,
                2,
            ),
        },
    )

    return {
        **prediction,
        "model_version": metadata.get(
            "model_version",
            resolved_version,
        ),
        "dataset": training_configuration.get(
            "dataset"
        ),
        "resampling": training_configuration.get(
            "resampling"
        ),
        "schema_version": API_SCHEMA_VERSION,
    }