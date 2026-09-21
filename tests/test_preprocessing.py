import pandas as pd
import pytest

from src.preprocessing import clean_data, split_features_target


def test_preprocessing_rejects_missing_identifier_columns():
    df = pd.DataFrame(
        {
            "fraud_label": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="identifier columns"):
        clean_data(
            df,
            identifier_columns=("transaction_id", "user_id"),
        )


def test_preprocessing_rejects_missing_values():
    df = pd.DataFrame(
        {
            "feature": [1, None],
            "fraud_label": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        split_features_target(
            df,
            target_column="fraud_label",
        )


def test_preprocessing_rejects_single_class_target():
    df = pd.DataFrame(
        {
            "feature": [1, 2],
            "fraud_label": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="exactly two classes"):
        split_features_target(
            df,
            target_column="fraud_label",
        )


def test_clean_data_drops_identifier_columns():
    df = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "user_id": [10, 20],
            "feature": [100, 200],
        }
    )

    result = clean_data(
        df,
        identifier_columns=("transaction_id", "user_id"),
    )

    assert list(result.columns) == ["feature"]


def test_split_features_target():
    df = pd.DataFrame(
        {
            "feature": [1, 2],
            "fraud_label": [0, 1],
        }
    )

    X, y = split_features_target(
        df,
        target_column="fraud_label",
    )

    assert list(X.columns) == ["feature"]
    assert y.tolist() == [0, 1]