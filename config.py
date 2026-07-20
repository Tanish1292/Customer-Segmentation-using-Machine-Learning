"""Project-wide paths and constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
SAMPLE_OUTPUT_DIR = PROJECT_ROOT / "sample_outputs"

DATA_FILE = DATA_DIR / "Mall_Customers.csv"
MODEL_FILE = MODEL_DIR / "kmeans_customer_segmentation.joblib"
TRAINED_DATA_FILE = SAMPLE_OUTPUT_DIR / "clustered_customers.csv"
METRICS_FILE = SAMPLE_OUTPUT_DIR / "model_metrics.csv"

RANDOM_STATE = 42
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10

FEATURE_COLUMNS = [
    "Gender_Encoded",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

NUMERIC_COLUMNS = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

DATASET_URLS = [
    "https://raw.githubusercontent.com/sharmaroshan/Mall-Customers-Clustering-Analysis/master/Mall_Customers.csv",
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Mall_Customers.csv",
]
