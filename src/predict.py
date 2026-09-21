import argparse
import os
import joblib
import pandas as pd

from src.model_registry import resolve_model_path


REGISTRY_PATH = "models/model_registry.json"
LEGACY_MODEL_PATH = "models/model.joblib"


def load_model(path):
    return joblib.load(path)


def prepare_input(transaction, artifact):
    feature_names = artifact["feature_names"]
    encoders = artifact["encoders"]

    missing_features = [
        feature
        for feature in feature_names
        if feature not in transaction
    ]

    if missing_features:
        raise ValueError(
            f"Input is missing required features: {missing_features}"
        )

    df = pd.DataFrame([transaction])

    df = df[feature_names].copy()

    for column, encoder in encoders.items():
        df[column] = encoder.transform(
            df[[column]]
        ).ravel()

    return df


def predict_transaction(transaction, artifact):
    df = prepare_input(transaction, artifact)

    model = artifact["model"]

    probability = float(
        model.predict_proba(df)[0][1]
    )

    threshold = float(
        artifact.get("metadata", {})
        .get("decision_threshold", 0.5)
    )

    return int(probability >= threshold)


def predict_transaction_details(transaction, artifact):
    df = prepare_input(transaction, artifact)

    model = artifact["model"]

    probability = float(
        model.predict_proba(df)[0][1]
    )

    threshold = float(
        artifact.get("metadata", {})
        .get("decision_threshold", 0.5)
    )

    return {
        "fraud_prediction": int(
            probability >= threshold
        ),
        "fraud_probability": probability,
        "decision_threshold": threshold,
        "model_version": artifact.get(
            "metadata", {}
        ).get("model_version"),
        "classifier": artifact.get(
            "metadata", {}
        ).get("classifier"),
    }


def load_active_model(version=None):
    version = version or os.getenv("MODEL_VERSION")

    model_path, selected_version = resolve_model_path(
        version,
        REGISTRY_PATH,
        LEGACY_MODEL_PATH,
    )

    artifact = load_model(model_path)

    return artifact, selected_version


def run_demo(dataset_path, target_column, artifact):
    df = pd.read_csv(dataset_path)

    feature_names = artifact["feature_names"]

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Demo dataset does not match model schema. "
            f"Missing features: {missing_features}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Demo dataset is missing target column: "
            f"{target_column}"
        )

    samples = pd.concat(
        [
            df[df[target_column] == 0].head(5),
            df[df[target_column] == 1].head(5),
        ]
    )

    for _, row in samples.iterrows():
        actual = row[target_column]

        transaction = row.drop(
            target_column
        ).to_dict()

        details = predict_transaction_details(
            transaction,
            artifact,
        )

        print(
            f"Actual: {actual} | "
            f"Prediction: {details['fraud_prediction']} | "
            f"Probability: {details['fraud_probability']:.4f}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run fraud model inference."
    )

    parser.add_argument(
        "--model-version",
        help=(
            "Specific registered model version. "
            "Defaults to the active model."
        ),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a local inference demonstration.",
    )

    parser.add_argument(
        "--dataset-path",
        help="CSV dataset path used by --demo.",
    )

    parser.add_argument(
        "--target-column",
        help="Target column used by --demo.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    artifact, selected_version = load_active_model(
        arguments.model_version
    )

    print(
        f"Loaded model version: {selected_version}"
    )

    metadata = artifact.get("metadata", {})

    print(
        f"Classifier: "
        f"{metadata.get('classifier', 'unknown')}"
    )

    print(
        f"Features: "
        f"{len(artifact['feature_names'])}"
    )

    if arguments.demo:
        if not arguments.dataset_path:
            raise ValueError(
                "--dataset-path is required with --demo"
            )

        if not arguments.target_column:
            raise ValueError(
                "--target-column is required with --demo"
            )

        run_demo(
            arguments.dataset_path,
            arguments.target_column,
            artifact,
        )


if __name__ == "__main__":
    main()