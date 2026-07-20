"""Exploratory Data Analysis plotting functions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import NUMERIC_COLUMNS


sns.set_theme(style="whitegrid")


def save_histograms(data, output_dir: Path) -> Path:
    """Save histograms for age, annual income, and spending score."""
    output_path = output_dir / "histograms.png"
    data[NUMERIC_COLUMNS].hist(figsize=(12, 8), bins=20, color="#2E86AB")
    plt.suptitle("Customer Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_boxplots(data, output_dir: Path) -> Path:
    """Save box plots to identify spread and possible outliers."""
    output_path = output_dir / "boxplots.png"
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data[NUMERIC_COLUMNS], palette="Set2")
    plt.title("Box Plot of Numeric Customer Features")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_pairplot(data, output_dir: Path) -> Path:
    """Save pair plot showing feature relationships by gender."""
    output_path = output_dir / "pairplot.png"
    pair_grid = sns.pairplot(
        data[["Gender"] + NUMERIC_COLUMNS],
        hue="Gender",
        diag_kind="hist",
        palette="Set1",
    )
    pair_grid.fig.suptitle("Pair Plot of Customer Features", y=1.02)
    pair_grid.savefig(output_path, dpi=150)
    plt.close("all")
    return output_path


def save_correlation_heatmap(data, output_dir: Path) -> Path:
    """Save a correlation heatmap for numeric columns."""
    output_path = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(8, 6))
    correlation = data[NUMERIC_COLUMNS].corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_gender_distribution(data, output_dir: Path) -> Path:
    """Save a chart showing male/female customer counts."""
    output_path = output_dir / "gender_distribution.png"
    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="Gender", hue="Gender", palette="Set1", legend=False)
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_individual_distributions(data, output_dir: Path) -> list[Path]:
    """Save separate distributions for each numeric customer feature."""
    generated_files = []

    for column in NUMERIC_COLUMNS:
        file_name = column.lower().replace(" ", "_").replace("(", "").replace(")", "")
        file_name = file_name.replace("$", "dollars").replace("-", "_")
        output_path = output_dir / f"{file_name}_distribution.png"

        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, color="#D1495B", bins=20)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        generated_files.append(output_path)

    return generated_files


def generate_eda_plots(data, output_dir: Path) -> list[Path]:
    """Generate all EDA plots and return their file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = [
        save_histograms(data, output_dir),
        save_boxplots(data, output_dir),
        save_pairplot(data, output_dir),
        save_correlation_heatmap(data, output_dir),
        save_gender_distribution(data, output_dir),
    ]
    generated_files.extend(save_individual_distributions(data, output_dir))
    return generated_files
