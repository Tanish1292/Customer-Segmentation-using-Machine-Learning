"""Preprocessing helpers for label encoding and feature scaling."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS


def encode_gender(data: pd.DataFrame) -> pd.DataFrame:
    """Convert Gender values to numeric form: Female=0 and Male=1."""
    encoded_data = data.copy()
    encoded_data["Gender_Encoded"] = encoded_data["Gender"].map(
        {"Female": 0, "Male": 1}
    )
    return encoded_data


def scale_features(
    data: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
):
    """Scale model features using StandardScaler."""
    feature_data = data[FEATURE_COLUMNS]

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        scaled_features = scaler.fit_transform(feature_data)
    else:
        scaled_features = scaler.transform(feature_data)

    return scaled_features, scaler


def prepare_features(data: pd.DataFrame):
    """Encode gender and scale features for clustering."""
    encoded_data = encode_gender(data)
    scaled_features, scaler = scale_features(encoded_data)
    return encoded_data, scaled_features, scaler
