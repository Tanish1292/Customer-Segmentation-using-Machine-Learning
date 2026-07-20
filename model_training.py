"""Train, evaluate, and save the K-Means customer segmentation model."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import (
    DATA_FILE,
    FEATURE_COLUMNS,
    FIGURE_DIR,
    MAX_CLUSTERS,
    METRICS_FILE,
    MIN_CLUSTERS,
    MODEL_FILE,
    RANDOM_STATE,
    SAMPLE_OUTPUT_DIR,
    TRAINED_DATA_FILE,
)
from src.data_loader import clean_dataset, load_dataset
from src.eda import generate_eda_plots
from src.preprocessing import prepare_features
from src.visualization import (
    save_cluster_plot,
    save_elbow_plot,
    save_silhouette_plot,
)


def evaluate_cluster_range(scaled_features) -> pd.DataFrame:
    """Train K-Means for multiple K values and collect evaluation metrics."""
    metrics = []

    for clusters in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        model = KMeans(
            n_clusters=clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(scaled_features)

        metrics.append(
            {
                "Clusters": clusters,
                "Inertia": model.inertia_,
                "Silhouette Score": silhouette_score(scaled_features, labels),
                "Davies-Bouldin Index": davies_bouldin_score(
                    scaled_features, labels
                ),
            }
        )

    return pd.DataFrame(metrics)


def choose_best_cluster_count(metrics_data: pd.DataFrame) -> int:
    """Choose K using an elbow-style knee point on the inertia curve."""
    x_values = metrics_data["Clusters"].to_numpy(dtype=float)
    y_values = metrics_data["Inertia"].to_numpy(dtype=float)

    points = np.column_stack((x_values, y_values))
    start = points[0]
    end = points[-1]
    line_vector = end - start
    line_length = np.linalg.norm(line_vector)

    if line_length == 0:
        return int(metrics_data.iloc[0]["Clusters"])

    distances = np.abs(
        line_vector[0] * (start[1] - points[:, 1])
        - (start[0] - points[:, 0]) * line_vector[1]
    ) / line_length
    knee_index = int(np.argmax(distances))
    return int(metrics_data.iloc[knee_index]["Clusters"])


def create_cluster_descriptions(cluster_profile: pd.DataFrame) -> dict[int, str]:
    """Create meaningful student-friendly names for the learned clusters."""
    income_median = cluster_profile["Annual Income (k$)"].median()
    spending_median = cluster_profile["Spending Score (1-100)"].median()
    age_median = cluster_profile["Age"].median()
    descriptions = {}

    for cluster_id, row in cluster_profile.iterrows():
        high_income = row["Annual Income (k$)"] >= income_median
        high_spending = row["Spending Score (1-100)"] >= spending_median
        young = row["Age"] <= age_median

        if high_income and high_spending:
            segment = "Premium Customers"
            meaning = "High income and high spending customers."
        elif high_income and not high_spending:
            segment = "Careful Customers"
            meaning = "High income customers who spend cautiously."
        elif not high_income and high_spending and young:
            segment = "Young High Spenders"
            meaning = "Younger customers with strong spending interest."
        elif not high_income and high_spending:
            segment = "Budget Enthusiasts"
            meaning = "Lower income customers who still spend actively."
        elif not high_income and not high_spending:
            segment = "Low Spending Customers"
            meaning = "Lower income and lower spending customers."
        else:
            segment = "Average Customers"
            meaning = "Customers with balanced income and spending behavior."

        descriptions[int(cluster_id)] = f"{segment}: {meaning}"

    return descriptions


def train_final_model(scaled_features, cluster_count: int) -> KMeans:
    """Train the final K-Means model."""
    model = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    model.fit(scaled_features)
    return model


def run_training_pipeline() -> dict:
    """Run the full training workflow and save all important outputs."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset(DATA_FILE)
    cleaned_data = clean_dataset(raw_data)
    encoded_data, scaled_features, scaler = prepare_features(cleaned_data)

    generate_eda_plots(cleaned_data, FIGURE_DIR)

    metrics_data = evaluate_cluster_range(scaled_features)
    best_cluster_count = choose_best_cluster_count(metrics_data)
    final_model = train_final_model(scaled_features, best_cluster_count)

    labels = final_model.predict(scaled_features)
    clustered_data = encoded_data.copy()
    clustered_data["Cluster"] = labels

    cluster_profile = clustered_data.groupby("Cluster")[FEATURE_COLUMNS].mean()
    cluster_descriptions = create_cluster_descriptions(cluster_profile)
    clustered_data["Segment_Name"] = clustered_data["Cluster"].map(
        lambda value: cluster_descriptions[int(value)].split(":")[0]
    )

    final_metrics = {
        "Best Cluster Count": best_cluster_count,
        "Best Silhouette K": int(
            metrics_data.loc[metrics_data["Silhouette Score"].idxmax(), "Clusters"]
        ),
        "Silhouette Score": silhouette_score(scaled_features, labels),
        "Inertia": final_model.inertia_,
        "Davies-Bouldin Index": davies_bouldin_score(scaled_features, labels),
    }

    metrics_data.to_csv(METRICS_FILE, index=False)
    clustered_data.to_csv(TRAINED_DATA_FILE, index=False)
    save_elbow_plot(metrics_data, FIGURE_DIR)
    save_silhouette_plot(metrics_data, FIGURE_DIR)
    save_cluster_plot(clustered_data, final_model, scaler, FIGURE_DIR)

    model_package = {
        "model": final_model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "cluster_descriptions": cluster_descriptions,
        "cluster_profile": cluster_profile,
        "metrics": final_metrics,
    }
    joblib.dump(model_package, MODEL_FILE)

    return final_metrics


if __name__ == "__main__":
    results = run_training_pipeline()
    print("Training completed successfully.")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value}")
