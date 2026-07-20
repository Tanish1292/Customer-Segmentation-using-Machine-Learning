"""Model visualisation functions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def save_elbow_plot(metrics_data: pd.DataFrame, output_dir: Path) -> Path:
    """Save the Elbow Method plot."""
    output_path = output_dir / "elbow_method.png"
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_data["Clusters"], metrics_data["Inertia"], marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_silhouette_plot(metrics_data: pd.DataFrame, output_dir: Path) -> Path:
    """Save the Silhouette Score comparison plot."""
    output_path = output_dir / "silhouette_scores.png"
    plt.figure(figsize=(8, 5))
    plt.plot(
        metrics_data["Clusters"],
        metrics_data["Silhouette Score"],
        marker="o",
        color="#D1495B",
    )
    plt.title("Silhouette Scores for Different K Values")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_cluster_plot(data: pd.DataFrame, model, scaler, output_dir: Path) -> Path:
    """Save final customer clusters using income and spending score axes."""
    output_path = output_dir / "final_customer_clusters.png"
    centroids = scaler.inverse_transform(model.cluster_centers_)
    centroid_data = pd.DataFrame(
        centroids,
        columns=[
            "Gender_Encoded",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
        ],
    )

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Segment_Name",
        palette="tab10",
        s=75,
        edgecolor="black",
        alpha=0.85,
    )
    plt.scatter(
        centroid_data["Annual Income (k$)"],
        centroid_data["Spending Score (1-100)"],
        marker="X",
        s=280,
        c="black",
        label="Cluster Centroids",
    )
    plt.title("Customer Segments using K-Means Clustering")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(title="Customer Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
