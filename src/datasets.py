from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: str
    target_column: str
    identifier_columns: tuple[str, ...]


DATASETS = {
    "original": DatasetConfig(
        name="original",
        path="data/Digital_Payment_Fraud_Detection_Dataset.csv",
        target_column="fraud_label",
        identifier_columns=("transaction_id", "user_id"),
    ),
    "ulb": DatasetConfig(
        name="ulb",
        path="data/creditcard.csv",
        target_column="Class",
        identifier_columns=(),
    ),
}


def get_dataset_config(name):
    try:
        return DATASETS[name]
    except KeyError as exc:
        available = ", ".join(DATASETS)
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        ) from exc


def validate_dataset_path(config):
    path = Path(config.path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {path}"
        )

    return path