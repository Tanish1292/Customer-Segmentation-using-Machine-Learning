# Customer Segmentation using Machine Learning

This project builds a complete customer segmentation application using the
Mall Customers dataset and the K-Means Clustering algorithm. It is designed as
a clean and understandable 4th-year B.Tech AI/ML project with modular Python
code, visual analysis, model training, evaluation, and a Streamlit web app.

## Project Objectives

- Group mall customers into meaningful business segments.
- Use purchasing behavior features such as annual income and spending score.
- Apply preprocessing, label encoding, and StandardScaler feature scaling.
- Find a suitable cluster count using the Elbow Method and Silhouette Score.
- Evaluate clustering quality using Silhouette Score, Inertia, and
  Davies-Bouldin Index.
- Build a Streamlit app for customer segment prediction and visual analysis.

## Features

- Automatic dataset loading from `data/Mall_Customers.csv`.
- Public GitHub download fallback if the dataset is missing.
- Synthetic Mall-Customers-like dataset generation if download also fails.
- Missing value handling, duplicate removal, and data type checking.
- Gender label encoding and feature scaling using StandardScaler.
- EDA charts including histograms, box plots, pair plot, heatmap, gender chart,
  and individual feature distributions.
- K-Means model training with saved Joblib model package.
- Final cluster visualization with centroids, legends, labels, and grid lines.
- Streamlit web app with input validation, segment descriptions, dataset view,
  graph view, clustered results, and CSV download for predictions.
- Jupyter Notebook explaining the full workflow.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset Information

Dataset: Mall Customers Dataset

Columns:

- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

The included dataset contains 200 customer records. If the CSV file is deleted,
the project will try to download it automatically. If downloading is not
possible, it will create a similar sample dataset with at least 200 records.

## Folder Structure

```text
customer_segmentation_ml_project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── Mall_Customers.csv
├── docs/
│   └── project_report.md
├── models/
│   └── kmeans_customer_segmentation.joblib
├── notebooks/
│   └── Customer_Segmentation_KMeans.ipynb
├── reports/
│   └── figures/
├── sample_outputs/
│   ├── clustered_customers.csv
│   └── model_metrics.csv
└── src/
    ├── config.py
    ├── data_loader.py
    ├── eda.py
    ├── model_training.py
    ├── preprocessing.py
    └── visualization.py
```

## Installation

Create and activate a virtual environment if desired, then install the required
libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Train the model and regenerate all outputs:

```bash
python -m src.model_training
```

Run the Streamlit web app:

```bash
streamlit run app.py
```

## Model Results

The project uses K-Means clustering. The selected cluster count is based on the
Elbow Method, while Silhouette Scores are also calculated for comparison.

Current trained model metrics:

- Best cluster count selected by elbow method: 6
- Silhouette Score: 0.3311
- Inertia: 276.4118
- Davies-Bouldin Index: 1.0177
- Best silhouette-only K observed in the tested range: 10

## Screenshots

Add screenshots after running the Streamlit app:

- Home and prediction screen
- Dataset screen
- EDA graphs screen
- Cluster visualization screen

## Sample Outputs

- `sample_outputs/clustered_customers.csv`
- `sample_outputs/model_metrics.csv`
- `reports/figures/final_customer_clusters.png`
- `reports/figures/elbow_method.png`
- `reports/figures/silhouette_scores.png`

## Future Improvements

- Add DBSCAN or Agglomerative Clustering for comparison.
- Add PCA-based 2D visualization.
- Add user authentication for business users.
- Store predictions in a database.
- Add automated unit tests and CI checks.

## Conclusion

This project demonstrates how unsupervised machine learning can be used to group
customers into actionable segments. The final Streamlit application makes the
model easy to use and suitable for college submission, demonstrations, and a
GitHub portfolio.
