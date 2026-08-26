import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

IDENTIFIER_COLUMNS = ["transaction_id", "user_id"]
TARGET_COLUMN = "fraud_label"


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    missing_columns = [column for column in IDENTIFIER_COLUMNS if column not in df]
    if missing_columns:
        raise ValueError(f"Dataset is missing identifier columns: {missing_columns}")
    return df.drop(columns=IDENTIFIER_COLUMNS)


def split_features_target(df):
    if TARGET_COLUMN not in df:
        raise ValueError(f"Dataset is missing target column: {TARGET_COLUMN}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    if y.nunique() != 2:
        raise ValueError("Fraud target must contain exactly two classes")

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def encode_features(X_train, X_test):
    encoders = {}

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in categorical_columns:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X_train[col] = encoder.fit_transform(
            X_train[[col]]
        ).ravel()

        X_test[col] = encoder.transform(
            X_test[[col]]
        ).ravel()

        encoders[col] = encoder

    categorical_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_columns
    ]

    return X_train, X_test, encoders, categorical_indices



def preprocess_data(path):
    df = load_data(path)
    df = clean_data(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, encoders, categorical_indices = encode_features(
        X_train,
        X_test
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        encoders,
        categorical_indices
    )


if __name__ == "__main__":
    DATA_PATH = "data/Digital_Payment_Fraud_Detection_Dataset.csv"

    X_train, X_test, y_train, y_test, encoders, categorical_indices = preprocess_data(
        DATA_PATH
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"Encoders: {list(encoders.keys())}")
    print(f"Categorical indices: {categorical_indices}")
