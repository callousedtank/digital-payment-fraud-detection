import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


EXPERIMENT_SCHEMA_VERSION = 1
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_experiment_name(name):
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Experiment names may contain only letters, numbers, dots, underscores, "
            "and hyphens."
        )
    return name


def dataset_fingerprint(path):
    dataset_path = Path(path)
    digest = hashlib.sha256()
    with dataset_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(dataset_path),
        "sha256": digest.hexdigest(),
        "size_bytes": dataset_path.stat().st_size,
    }


def record_experiment(experiments_dir, experiment_name, record):
    experiment_name = validate_experiment_name(experiment_name)
    experiments_path = Path(experiments_dir)
    experiments_path.mkdir(parents=True, exist_ok=True)
    record_path = experiments_path / f"{experiment_name}.jsonl"
    with record_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")
    return record_path


def build_experiment_record(
    model_version,
    model_type,
    training_configuration,
    dataset,
    validation_metrics,
    feature_names,
):
    return {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "model_type": model_type,
        "training_configuration": training_configuration,
        "dataset": dataset,
        "validation_metrics": validation_metrics,
        "feature_names": feature_names,
    }


def load_experiments(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
