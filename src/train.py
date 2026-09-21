import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.datasets import get_dataset_config
from src.experiment_tracking import (
    build_experiment_record,
    dataset_fingerprint,
    record_experiment,
)
from src.model_registry import (
    activate_model,
    register_model,
    validate_version,
)
from src.preprocessing import preprocess_data


MODELS_DIR = Path("models")
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
EXPERIMENTS_DIR = Path("experiments")
ARTIFACT_VERSION = 1


def apply_resampling(
    X_train,
    y_train,
    categorical_indices,
    resampling,
):
    if resampling == "none":
        return X_train, y_train

    if resampling == "smotenc":
        if not categorical_indices:
            raise ValueError(
                "SMOTENC requires categorical features. "
                "Use --resampling smote or none for this dataset."
            )

        sampler = SMOTENC(
            categorical_features=categorical_indices,
            random_state=42,
        )

    elif resampling == "smote":
        sampler = SMOTE(random_state=42)

    else:
        raise ValueError(
            f"Unsupported resampling strategy: {resampling}"
        )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train,
    )

    return X_resampled, y_resampled


def train_model(
    X_train,
    y_train,
    categorical_indices,
    model_type,
    resampling,
):
    X_train_resampled, y_train_resampled = apply_resampling(
        X_train,
        y_train,
        categorical_indices,
        resampling,
    )

    print("Training class distribution:")
    print("Before resampling:", Counter(y_train))
    print("After resampling:", Counter(y_train_resampled))

    if model_type == "random_forest":
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        )

    elif model_type == "logistic_regression":
        model = LogisticRegression(
            random_state=42,
            max_iter=1_000,
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    model.fit(
        X_train_resampled,
        y_train_resampled,
    )

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Predictions:", Counter(predictions))

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }

    print("Validation metrics:", metrics)

    return metrics


def save_model(
    model,
    encoders,
    categorical_indices,
    feature_names,
    model_version,
    validation_metrics,
    training_configuration,
):
    model_version = validate_version(model_version)

    path = MODELS_DIR / f"model-{model_version}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "classifier": type(model).__name__,
        "feature_names": feature_names,
        "training_configuration": training_configuration,
        "validation_metrics": validation_metrics,
    }

    artifact = {
        "model": model,
        "encoders": encoders,
        "categorical_indices": categorical_indices,
        "feature_names": feature_names,
        "metadata": metadata,
    }

    joblib.dump(
        artifact,
        path,
    )

    register_model(
        model_version,
        path,
        metadata,
        REGISTRY_PATH,
    )

    print(f"Model saved to: {path}")


def default_model_version():
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train or activate a fraud detection model."
    )

    parser.add_argument(
        "--dataset",
        choices=("original", "ulb"),
        default="original",
        help="Dataset to use for training.",
    )

    parser.add_argument(
        "--model-version",
        default=default_model_version(),
        help=(
            "Version for the new model artifact "
            "(default: UTC timestamp)."
        ),
    )

    parser.add_argument(
        "--activate-version",
        help=(
            "Activate a previously evaluated model "
            "version without training."
        ),
    )

    parser.add_argument(
        "--model-type",
        choices=(
            "random_forest",
            "logistic_regression",
        ),
        default="random_forest",
        help="Model family to train.",
    )

    parser.add_argument(
        "--resampling",
        choices=(
            "none",
            "smote",
            "smotenc",
        ),
        default="smotenc",
        help="Training-set resampling strategy.",
    )

    parser.add_argument(
        "--experiment-name",
        default="fraud-detection",
        help="Name of the JSONL experiment record.",
    )

    return parser.parse_args()


def validate_resampling(dataset_name, resampling):
    if dataset_name == "original":
        if resampling == "smote":
            raise ValueError(
                "Use SMOTENC for the original dataset because "
                "it contains categorical features."
            )

    if dataset_name == "ulb":
        if resampling == "smotenc":
            raise ValueError(
                "The ULB dataset contains numerical features only. "
                "Use SMOTE or none."
            )


def main():
    arguments = parse_arguments()

    if arguments.activate_version:
        activate_model(
            arguments.activate_version,
            REGISTRY_PATH,
        )

        print(
            f"Activated model version: "
            f"{arguments.activate_version}"
        )

        return

    validate_resampling(
        arguments.dataset,
        arguments.resampling,
    )

    dataset_config = get_dataset_config(
        arguments.dataset
    )

    processed = preprocess_data(
        arguments.dataset
    )

    model = train_model(
        processed["X_train"],
        processed["y_train"],
        processed["categorical_indices"],
        arguments.model_type,
        arguments.resampling,
    )

    validation_metrics = evaluate_model(
        model,
        processed["X_test"],
        processed["y_test"],
    )

    training_configuration = {
        "random_state": 42,
        "dataset": arguments.dataset,
        "resampling": arguments.resampling,
        "classifier": type(model).__name__,
        "model_type": arguments.model_type,
        "test_size": 0.2,
    }

    save_model(
        model,
        processed["encoders"],
        processed["categorical_indices"],
        list(processed["X_train"].columns),
        arguments.model_version,
        validation_metrics,
        training_configuration,
    )

    record_path = record_experiment(
        EXPERIMENTS_DIR,
        arguments.experiment_name,
        build_experiment_record(
            arguments.model_version,
            arguments.model_type,
            training_configuration,
            dataset_fingerprint(
                dataset_config.path
            ),
            validation_metrics,
            list(processed["X_train"].columns),
        ),
    )

    print(
        f"Experiment recorded in: {record_path}"
    )


if __name__ == "__main__":
    main()