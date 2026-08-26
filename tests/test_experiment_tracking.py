from src.experiment_tracking import (
    build_experiment_record,
    load_experiments,
    record_experiment,
)


def test_experiment_record_is_reproducible_jsonl(tmp_path):
    record = build_experiment_record(
        "1.0.0",
        "random_forest",
        {"random_state": 42},
        {"sha256": "dataset-hash"},
        {"accuracy": 0.95},
        ["transaction_amount"],
    )

    path = record_experiment(tmp_path, "fraud-detection", record)

    assert load_experiments(path) == [record]
