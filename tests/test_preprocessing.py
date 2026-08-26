import pandas as pd
import pytest

from src.preprocessing import clean_data, split_features_target


def test_preprocessing_rejects_missing_identifier_columns():
    with pytest.raises(ValueError, match="identifier columns"):
        clean_data(pd.DataFrame({"fraud_label": [0, 1]}))


def test_preprocessing_rejects_missing_values_and_single_class_target():
    with pytest.raises(ValueError, match="missing values"):
        split_features_target(pd.DataFrame({"feature": [1, None], "fraud_label": [0, 1]}))

    with pytest.raises(ValueError, match="exactly two classes"):
        split_features_target(pd.DataFrame({"feature": [1, 2], "fraud_label": [0, 0]}))
