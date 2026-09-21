import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from src.datasets import get_dataset_config, validate_dataset_path


def load_data(path):
    return pd.read_csv(path)


def clean_data(df, identifier_columns):
    missing_columns = [
        column
        for column in identifier_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing identifier columns: {missing_columns}"
        )

    if identifier_columns:
        return df.drop(columns=list(identifier_columns))

    return df.copy()


def split_features_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(
            f"Dataset is missing target column: {target_column}"
        )

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.nunique() != 2:
        raise ValueError(
            "Fraud target must contain exactly two classes"
        )

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def encode_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    encoders = {}

    categorical_columns = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_train[column] = encoder.fit_transform(
            X_train[[column]]
        ).ravel()

        X_test[column] = encoder.transform(
            X_test[[column]]
        ).ravel()

        encoders[column] = encoder

    categorical_indices = [
        X_train.columns.get_loc(column)
        for column in categorical_columns
    ]

    return (
        X_train,
        X_test,
        encoders,
        categorical_indices,
    )


def preprocess_data(dataset_name):
    config = get_dataset_config(dataset_name)
    dataset_path = validate_dataset_path(config)

    df = load_data(dataset_path)

    df = clean_data(
        df,
        config.identifier_columns,
    )

    X, y = split_features_target(
        df,
        config.target_column,
    )

    X_train, X_test, y_train, y_test = split_data(X, y)

    (
        X_train,
        X_test,
        encoders,
        categorical_indices,
    ) = encode_features(
        X_train,
        X_test,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "categorical_indices": categorical_indices,
        "dataset_name": config.name,
        "dataset_path": str(dataset_path),
        "target_column": config.target_column,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect dataset preprocessing."
    )
    parser.add_argument(
        "--dataset",
        choices=("original", "ulb"),
        required=True,
    )

    args = parser.parse_args()

    result = preprocess_data(args.dataset)

    print(f"Dataset: {result['dataset_name']}")
    print(f"Path: {result['dataset_path']}")
    print(f"Target: {result['target_column']}")
    print(f"X_train: {result['X_train'].shape}")
    print(f"X_test: {result['X_test'].shape}")
    print(f"y_train: {result['y_train'].shape}")
    print(f"y_test: {result['y_test'].shape}")
    print(f"Encoders: {list(result['encoders'].keys())}")
    print(f"Categorical indices: {result['categorical_indices']}")