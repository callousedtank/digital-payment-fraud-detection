import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from src.experiment_tracking import (
    build_experiment_record,
    dataset_fingerprint,
    record_experiment,
)
from src.model_registry import activate_model, register_model, validate_version
from src.preprocessing import preprocess_data

DATA_PATH = "data/Digital_Payment_Fraud_Detection_Dataset.csv"
MODELS_DIR = Path("models")
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
EXPERIMENTS_DIR = Path("experiments")
ARTIFACT_VERSION = 1


def train_model(X_train, y_train, categorical_indices, model_type):
    smote = SMOTENC(
        categorical_features=categorical_indices,
        random_state=42
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("Before SMOTE:", Counter(y_train))
    print("After SMOTE:", Counter(y_train_resampled))

    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1_000)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train_resampled, y_train_resampled)

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Predictions:", Counter(predictions))
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))
    metrics = {
        "accuracy": accuracy,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)

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

    joblib.dump(artifact, path)
    register_model(model_version, path, metadata, REGISTRY_PATH)

    print(f"Model saved to: {path}")


def default_model_version():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train or activate a fraud model.")
    parser.add_argument(
        "--model-version",
        default=default_model_version(),
        help="Version for the new model artifact (default: UTC timestamp).",
    )
    parser.add_argument(
        "--activate-version",
        help="Activate a previously evaluated model version without training.",
    )
    parser.add_argument(
        "--model-type",
        choices=("random_forest", "logistic_regression"),
        default="random_forest",
        help="Model family to train.",
    )
    parser.add_argument(
        "--experiment-name",
        default="fraud-detection",
        help="Name of the JSONL experiment record.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.activate_version:
        activate_model(arguments.activate_version, REGISTRY_PATH)
        print(f"Activated model version: {arguments.activate_version}")
        return

    (
        X_train,
        X_test,
        y_train,
        y_test,
        encoders,
        categorical_indices
    ) = preprocess_data(DATA_PATH)

    model = train_model(
        X_train,
        y_train,
        categorical_indices,
        arguments.model_type,
    )

    validation_metrics = evaluate_model(model, X_test, y_test)
    training_configuration = {
        "random_state": 42,
        "resampling": "SMOTENC",
        "classifier": type(model).__name__,
        "model_type": arguments.model_type,
    }

    save_model(
        model,
        encoders,
        categorical_indices,
        list(X_train.columns),
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
            dataset_fingerprint(DATA_PATH),
            validation_metrics,
            list(X_train.columns),
        ),
    )
    print(f"Experiment recorded in: {record_path}")


if __name__ == "__main__":
    main()
