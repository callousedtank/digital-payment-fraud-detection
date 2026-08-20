from collections import Counter
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_data

DATA_PATH = "data/Digital_Payment_Fraud_Detection_Dataset.csv"
MODEL_PATH = "models/model.joblib"


def train_model(X_train, y_train, categorical_indices):
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

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("Predictions:", Counter(predictions))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))


def save_model(model, encoders, categorical_indices, feature_names, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "encoders": encoders,
        "categorical_indices": categorical_indices,
        "feature_names": feature_names
    }

    joblib.dump(artifact, path)

    print(f"Model saved to: {path}")


def main():
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
        categorical_indices
    )

    save_model(
        model,
        encoders,
        categorical_indices,
        list(X_train.columns),
        MODEL_PATH
    )

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()