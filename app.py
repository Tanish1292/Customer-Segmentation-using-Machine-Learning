"""Streamlit app for Customer Segmentation using K-Means Clustering."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

from src.config import DATA_FILE, FIGURE_DIR, MODEL_FILE, TRAINED_DATA_FILE
from src.data_loader import clean_dataset, load_dataset
from src.model_training import run_training_pipeline


st.set_page_config(
    page_title="Customer Segmentation ML",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .metric-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        background: #FFFFFF;
    }
    .segment-result {
        border-left: 5px solid #2E86AB;
        padding: 1rem;
        background: #F8FAFC;
        border-radius: 6px;
        margin-top: 1rem;
    }
</style>
"""


@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load and clean dataset for display."""
    data = load_dataset(DATA_FILE)
    return clean_dataset(data)


@st.cache_resource
def get_model_package() -> dict:
    """Load trained model package, training it if the file is missing."""
    if not MODEL_FILE.exists():
        run_training_pipeline()
    return joblib.load(MODEL_FILE)


def validate_inputs(age: int, income: int, spending_score: int) -> list[str]:
    """Return validation errors for user input values."""
    errors = []

    if age < 18 or age > 100:
        errors.append("Age should be between 18 and 100.")
    if income < 1 or income > 300:
        errors.append("Annual income should be between 1 and 300 k$.")
    if spending_score < 1 or spending_score > 100:
        errors.append("Spending score should be between 1 and 100.")

    return errors


def predict_segment(
    gender: str,
    age: int,
    income: int,
    spending_score: int,
    model_package: dict,
) -> tuple[int, str]:
    """Predict the customer segment for one customer."""
    gender_encoded = 1 if gender == "Male" else 0
    input_data = pd.DataFrame(
        [
            {
                "Gender_Encoded": gender_encoded,
                "Age": age,
                "Annual Income (k$)": income,
                "Spending Score (1-100)": spending_score,
            }
        ]
    )

    scaler = model_package["scaler"]
    model = model_package["model"]
    scaled_input = scaler.transform(input_data[model_package["feature_columns"]])
    cluster = int(model.predict(scaled_input)[0])
    description = model_package["cluster_descriptions"][cluster]

    return cluster, description


def prediction_to_csv(prediction: dict) -> bytes:
    """Convert a single prediction result to downloadable CSV bytes."""
    output = io.StringIO()
    pd.DataFrame([prediction]).to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def show_figures() -> None:
    """Display all saved EDA and clustering figures."""
    figure_files = sorted(FIGURE_DIR.glob("*.png"))

    if not figure_files:
        st.info("Graphs are being generated. Please run the training pipeline once.")
        run_training_pipeline()
        figure_files = sorted(FIGURE_DIR.glob("*.png"))

    for index in range(0, len(figure_files), 2):
        columns = st.columns(2)
        for column, figure_path in zip(columns, figure_files[index : index + 2]):
            readable_title = figure_path.stem.replace("_", " ").title()
            column.image(str(figure_path), caption=readable_title, use_container_width=True)


def main() -> None:
    """Build the Streamlit user interface."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Customer Segmentation using Machine Learning")
    st.caption("K-Means clustering project using the Mall Customers dataset")

    try:
        data = get_dataset()
        model_package = get_model_package()
    except Exception as error:
        st.error(f"Unable to load project resources: {error}")
        st.stop()

    metrics = model_package["metrics"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Best K", int(metrics["Best Cluster Count"]))
    metric_columns[1].metric("Silhouette Score", f"{metrics['Silhouette Score']:.3f}")
    metric_columns[2].metric("Inertia", f"{metrics['Inertia']:.2f}")
    metric_columns[3].metric(
        "Davies-Bouldin Index",
        f"{metrics['Davies-Bouldin Index']:.3f}",
    )

    tab_predict, tab_dataset, tab_eda, tab_clusters = st.tabs(
        ["Predict Segment", "Dataset", "EDA Graphs", "Cluster Results"]
    )

    with tab_predict:
        left_column, right_column = st.columns([1, 1])

        with left_column:
            st.subheader("Enter Customer Details")
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input(
                "Annual Income (k$)",
                min_value=1,
                max_value=300,
                value=60,
            )
            spending_score = st.number_input(
                "Spending Score (1-100)",
                min_value=1,
                max_value=100,
                value=50,
            )

            predict_button = st.button("Predict Customer Segment", type="primary")

        with right_column:
            st.subheader("Prediction Result")
            if predict_button:
                errors = validate_inputs(age, income, spending_score)
                if errors:
                    for error in errors:
                        st.warning(error)
                else:
                    cluster, description = predict_segment(
                        gender,
                        age,
                        income,
                        spending_score,
                        model_package,
                    )
                    segment_name, segment_meaning = description.split(": ", 1)
                    prediction = {
                        "Gender": gender,
                        "Age": age,
                        "Annual Income (k$)": income,
                        "Spending Score (1-100)": spending_score,
                        "Cluster": cluster,
                        "Segment": segment_name,
                        "Description": segment_meaning,
                    }

                    st.markdown(
                        f"""
                        <div class="segment-result">
                            <h3>{segment_name}</h3>
                            <p>{segment_meaning}</p>
                            <p><strong>Cluster:</strong> {cluster}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "Download Prediction as CSV",
                        data=prediction_to_csv(prediction),
                        file_name="customer_segment_prediction.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Fill the form and click the prediction button.")

    with tab_dataset:
        st.subheader("Mall Customers Dataset")
        st.dataframe(data, use_container_width=True)
        st.write("Dataset shape:", data.shape)
        st.write("Missing values:", data.isnull().sum())
        st.write("Data types:", data.dtypes.astype(str))

    with tab_eda:
        st.subheader("Exploratory Data Analysis")
        show_figures()

    with tab_clusters:
        st.subheader("Final Customer Clusters")
        cluster_plot = FIGURE_DIR / "final_customer_clusters.png"
        if cluster_plot.exists():
            st.image(str(cluster_plot), use_container_width=True)

        if TRAINED_DATA_FILE.exists():
            clustered_data = pd.read_csv(TRAINED_DATA_FILE)
            st.dataframe(clustered_data, use_container_width=True)
        else:
            st.info("Clustered output is not available yet. Running training now.")
            run_training_pipeline()
            st.rerun()


if __name__ == "__main__":
    main()
