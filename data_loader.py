"""Functions for loading, validating, and cleaning the customer dataset."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from src.config import DATASET_URLS, REQUIRED_COLUMNS


def download_dataset(destination_path: Path) -> bool:
    """Try to download the Mall Customers dataset from public GitHub URLs."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    for url in DATASET_URLS:
        try:
            urlretrieve(url, destination_path)
            downloaded_data = pd.read_csv(destination_path)
            if validate_columns(downloaded_data):
                return True
        except Exception:
            continue

    return False


def generate_sample_dataset(destination_path: Path, records: int = 200) -> pd.DataFrame:
    """Generate a Mall-Customers-like dataset when no file/download is available."""
    np.random.seed(42)

    ages = np.random.randint(18, 71, size=records)
    genders = np.random.choice(["Male", "Female"], size=records, p=[0.44, 0.56])
    annual_income = np.random.randint(15, 138, size=records)

    spending_score = []
    for age, income in zip(ages, annual_income):
        base_score = 100 - age * 0.45 + income * 0.15
        noisy_score = base_score + np.random.normal(0, 18)
        spending_score.append(int(np.clip(noisy_score, 1, 100)))

    sample_data = pd.DataFrame(
        {
            "CustomerID": range(1, records + 1),
            "Gender": genders,
            "Age": ages,
            "Annual Income (k$)": annual_income,
            "Spending Score (1-100)": spending_score,
        }
    )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sample_data.to_csv(destination_path, index=False)
    return sample_data


def validate_columns(data: pd.DataFrame) -> bool:
    """Check whether all required columns are present."""
    return all(column in data.columns for column in REQUIRED_COLUMNS)


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the dataset, downloading or generating it automatically if needed."""
    if not data_path.exists():
        downloaded = download_dataset(data_path)
        if not downloaded:
            return generate_sample_dataset(data_path)

    data = pd.read_csv(data_path)

    if not validate_columns(data):
        raise ValueError(
            "Dataset columns are not valid. Expected columns: "
            f"{', '.join(REQUIRED_COLUMNS)}"
        )

    return data


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and basic data type corrections."""
    cleaned_data = data.copy()

    cleaned_data.drop_duplicates(inplace=True)
    cleaned_data["Gender"] = cleaned_data["Gender"].astype(str).str.strip().str.title()

    for column in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
        cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors="coerce")
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())

    cleaned_data["CustomerID"] = pd.to_numeric(
        cleaned_data["CustomerID"], errors="coerce"
    )
    missing_customer_ids = cleaned_data["CustomerID"].isna()
    if missing_customer_ids.any():
        cleaned_data.loc[missing_customer_ids, "CustomerID"] = range(
            1,
            missing_customer_ids.sum() + 1,
        )
    cleaned_data["CustomerID"] = cleaned_data["CustomerID"].astype(int)

    cleaned_data["Gender"] = cleaned_data["Gender"].replace({"Nan": np.nan, "": np.nan})
    cleaned_data["Gender"] = cleaned_data["Gender"].fillna(
        cleaned_data["Gender"].mode()[0]
    )
    cleaned_data = cleaned_data[cleaned_data["Gender"].isin(["Male", "Female"])]

    return cleaned_data.reset_index(drop=True)
