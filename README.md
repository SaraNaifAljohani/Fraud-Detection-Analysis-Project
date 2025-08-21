
# Fraud Detection Analysis Project

## 📌 Overview

This project explores techniques for **detecting fraudulent transactions** using unsupervised learning and anomaly detection methods. The goal is not to build a production-ready fraud detection system, but to **practice data analysis, preprocessing, clustering, and anomaly detection techniques** on financial transaction data.

**Goal:** To explore a bank transaction dataset using clustering and anomaly detection to identify potential fraudulent patterns, and understand what characterizes risky behavior.

**Data Soruce:** The dataset used in this project comes from [Kaggle: Fraud Detection Dataset](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection).

The data file is stored in the `data` folder as `bank.transaction.data.csv`.

---

## 🎯 Objectives

* Load and explore transaction dataset using **EDA tools and methods**
* Clean and preprocess data (scaling, removing unnecessary columns)
* Apply **dimensionality reduction (PCA)** for visualization and noise reduction
* Perform **clustering (KMeans)** and evaluate with silhouette score
* Detect potential anomalies using **Isolation Forest**
* Visualize insights with Matplotlib, Seaborn, and Plotly
* Summarize findings on transaction patterns and anomalies

---

## 🔧 Methods & Tools

* **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `scipy`
* **Data Preprocessing**: Standardization, feature scaling
* **Dimensionality Reduction**: PCA
* **Clustering**: KMeans (optimized with silhouette score)
* **Anomaly Detection**: Isolation Forest
* **Visualization**: Correlation heatmaps, PCA scatter plots, bar plots, box plots, Choropleth Map, cluster visualizations, anomaly highlights, line graph

---

## 📊 Key Results

* **Optimal K (clusters)** was selected using silhouette scores.
* PCA reduced dimensionality while preserving variance, making clusters easier to visualize.
* **Isolation Forest** flagged a subset of transactions as potential anomalies, useful for fraud risk analysis.
* Visualizations revealed differences in transaction behavior between normal and anomalous data points.

*More insights and conclusion are in the `insights` file.*

---

## ⚠️ Limitations 

* **Synthetic Data:** The dataset used here is synthetic, and unfortunately, it isn’t very well-synthesized. Some transaction patterns appear unrealistic and would normally require discussion with stakeholders to clarify before deeper analysis.
* **Missing Context:** Key business rules and domain knowledge (e.g., transaction approval processes, fraud thresholds, customer behavior patterns) are unavailable in this dataset.
* In a real-world project, this would require **iterative analysis:** presenting findings, validating with stakeholders, refining assumptions, and re-analyzing.
* **Student Project Scope:** This project is primarily a learning exercise to apply clustering, dimensionality reduction, and anomaly detection techniques. Therefore, the results should be viewed as a first-level exploratory analysis, not a finalized fraud detection system.

---

## 📌 Next Steps

* Try other clustering methods (DBSCAN, Hierarchical Clustering).
* Experiment with supervised approaches (Logistic Regression, Random Forest) if labeled data is available.
* Add model evaluation metrics (precision, recall, F1-score) for fraud detection tasks.

