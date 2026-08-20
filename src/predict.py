import joblib
import pandas as pd

MODEL_PATH = "models/model.joblib"


def load_model(path):
    return joblib.load(path)


def prepare_input(transaction, artifact):
    feature_names = artifact["feature_names"]
    encoders = artifact["encoders"]

    df = pd.DataFrame([transaction])

    df = df[feature_names]

    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[[col]]).ravel()

    return df


def predict_transaction(transaction, artifact):
    df = prepare_input(transaction, artifact)

    prediction = artifact["model"].predict(df)[0]

    return int(prediction)

if __name__ == "__main__":
    artifact = load_model(MODEL_PATH)

    df = pd.read_csv(
        "data/Digital_Payment_Fraud_Detection_Dataset.csv"
    )

    df = df.drop(columns=["transaction_id", "user_id"])

    samples = pd.concat([
        df[df["fraud_label"] == 0].head(5),
        df[df["fraud_label"] == 1].head(5)
    ])

    for _, row in samples.iterrows():
        actual = row["fraud_label"]

        transaction = row.drop("fraud_label").to_dict()

        prediction = predict_transaction(
            transaction,
            artifact
        )

        print(
            f"Actual: {actual} | Prediction: {prediction}"
        )