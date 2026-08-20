import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    columns_to_drop = ["transaction_id", "user_id"]

    return df.drop(columns=columns_to_drop)


def split_features_target(df):
    X = df.drop(columns=["fraud_label"])
    y = df["fraud_label"]

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