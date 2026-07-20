# Project Report: Customer Segmentation using K-Means

## Problem Statement

Retail businesses often serve customers with different income levels, spending
patterns, and demographic characteristics. Customer segmentation helps identify
these groups so businesses can plan better marketing campaigns and improve
customer experience.

## Dataset

The project uses the Mall Customers dataset with 200 records and five columns:
CustomerID, Gender, Age, Annual Income, and Spending Score.

## Methodology

1. Load the dataset from the `data` folder.
2. If the dataset is missing, download it from a public source or generate a
   similar sample dataset.
3. Clean the data by removing duplicates, handling missing values, and checking
   data types.
4. Encode `Gender` as a numeric feature.
5. Scale model features using StandardScaler.
6. Generate EDA charts to understand distributions and relationships.
7. Train K-Means models for K values from 2 to 10.
8. Use the Elbow Method and Silhouette Score to compare cluster options.
9. Save the final model package using Joblib.
10. Build a Streamlit interface for prediction and visualization.

## Evaluation Metrics

- Silhouette Score measures how well each point fits within its cluster.
- Inertia measures the compactness of clusters.
- Davies-Bouldin Index measures average cluster similarity, where lower values
  are usually better.

## Final Result

The final trained model uses 6 clusters selected from the elbow point of the
inertia curve. The app displays business-friendly segment descriptions such as
Premium Customers, Careful Customers, Average Customers, Young High Spenders,
Budget Enthusiasts, and Low Spending Customers.

## Conclusion

The project successfully demonstrates an end-to-end unsupervised machine
learning workflow. It includes data preprocessing, EDA, model training,
evaluation, visualization, and deployment through Streamlit.
