# Online Retail II — Product Sales Forecasting Pipeline

## Overview

This project forecasts daily product sales for a UK-based online gift retailer using the Online Retail II dataset from the UCI Machine Learning Repository. The dataset contains transactional data from December 2009 to December 2011. The pipeline clusters products by purchasing pattern and trains a separate forecasting model per cluster.

---

## Dataset

- Source: UCI Machine Learning Repository — Online Retail II
- Two sheets: Year 2009-2010 and Year 2010-2011
- Columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
- Total transactions after cleaning: ~800,000 rows

---

## Pipeline Structure

### 1. Data Loading & Preprocessing
- Load both sheets and concatenate into a single dataframe.
- Remove cancelled invoices (Invoice starting with C).
- Remove rows with negative or zero Quantity and Price.
- Normalize product descriptions — strip whitespace and uppercase.
- Filter products below the 10th percentile of transaction count to remove rarely purchased items.

### 2. Daily Aggregation & Product Feature Matrix
- Aggregate transactions to daily level per StockCode.
- Metrics: units_sold, revenue, avg_price.
- Fill missing dates with zero units_sold and forward-fill avg_price.
- Build product feature matrix of shape (n_products x n_days).
- Standardize using StandardScaler for clustering only — raw values preserved for forecasting.

### 3. Product Clustering
- Cluster products by their daily sales time series patterns.
- Algorithms evaluated: KMeans, DBSCAN, HDBSCAN.
- Best configuration: KMeans with k=12 (Silhouette Score evaluated for each k).
- Cluster labels merged back into product-level dataframes.

### 4. Cluster Dataframes & Feature Engineering
- Create separate dataframe per cluster at StockCode x Date granularity.
- Calendar features: Month, DayOfWeek, WeekOfYear, Quarter, is_weekend, is_saturday.
- Holiday features: is_uk_holiday (via holidays library), is_black_friday, is_cyber_monday.
- Seasonal flags: is_christmas_period, is_year_end, is_january.
- Lag features per StockCode: lag_1, lag_7, lag_14.
- Rolling features per StockCode: rolling_mean_7, rolling_mean_14, rolling_std_7.

### 5. Statistical Analysis per Cluster
- Descriptive statistics: mean, median, std, skewness, kurtosis, Shapiro-Wilk test.
- Kruskal-Wallis and pairwise Mann-Whitney U tests to confirm clusters are significantly different.
- Stationarity: ADF and KPSS tests per cluster.
- Seasonal decomposition: weekly (period=7) and monthly (period=30) additive decomposition.
- ACF and PACF plots up to 60 lags.
- CV% and volatility summary table.

### 6. Outlier Treatment
- Kurtosis above 5 triggers outlier treatment.
- Z-score threshold of 3 — values above replaced with cluster median.
- Applied at StockCode level to preserve time series continuity.

### 7. Model Definitions
Five models defined as factory functions:
- **PatchTST** — Patch-based Transformer. No exogenous support.
- **TiDE** — Dense encoder-decoder. Supports future and historical exogenous.
- **NHITS** — Hierarchical interpolation. Supports future and historical exogenous.
- **NBEATS** — Basis expansion. No exogenous support.
- **TimesNet** — 2D time series representation. Supports future exogenous only.

All models: horizon=60, input_size=180, max_steps=1000, learning_rate=1e-4, MAE loss.

### 8. Model Training & Evaluation
- Naive weekly baseline computed for all clusters.
- Multiple models trained per cluster based on CV%, stationarity and seasonality analysis.
- Aggregation from StockCode level to daily cluster total before training.
- Train-test split: last 60 days held out as test set.
- Metrics: SMAPE (primary), MAE, RMSE, MASE.

### 9. Hyperparameter Tuning
- Best model per cluster tuned across 5 configurations.
- Parameters varied: input_size (120, 180, 240, 360), max_steps (1000, 2000, 3000), learning_rate (1e-3, 5e-4, 1e-4).

### 10. Prophet Forecasting
- Prophet applied to all 12 clusters as an additional model.
- Diagnostics computed per cluster: CV%, zero count, seasonality strength, changepoints.
- Parameters tuned per cluster based on diagnostics.
- Logistic growth with floor=0 applied to clusters with above 40% zero-sales days.
- UK public holidays, Black Friday, Cyber Monday and monthly seasonality added.

### 11. Feature Selection via Spearman Correlation
- Spearman rank correlation computed between each feature and units_sold per cluster.
- Only features with p-value below 0.05 and absolute correlation above 0.05 retained.
- Models retrained using only significant features per cluster.

### 12. Forecasting Across Multiple Horizons
- Best model per cluster evaluated at horizons: 30, 60, 90, 180 days.
- Same model and feature set used across all horizons for fair comparison.

### 13. Rolling Window Evaluation
- Four seasonal windows evaluated: Late 2010, Early 2011, Mid 2011, Late 2011.
- Within each window, 30-day sub-windows slid forward in 14-day steps.
- Generates 4-5 SMAPE values per window enabling meaningful boxplot distributions.
- Models retrained from scratch for each sub-window — true walk-forward evaluation.

---

## Final Best Models

| Cluster | Model | SMAPE |
|---------|-------|-------|
| Cluster_0 | NBEATS | 49.65% |
| Cluster_1 | NHITS_SigFeatures | 82.98% |
| Cluster_2 | PatchTST | 189.04% |
| Cluster_3 | NHITS | 89.76% |
| Cluster_4 | NHITS_SigFeatures | 99.40% |
| Cluster_5 | Prophet_Tuned | 183.62% |
| Cluster_6 | PatchTST | 138.18% |
| Cluster_7 | Prophet_Tuned_SigFeatures | 57.75% |
| Cluster_8 | NHITS_SigFeatures | 92.92% |
| Cluster_9 | Prophet_Tuned_SigFeatures | 86.41% |
| Cluster_10 | Prophet_Tuned_SigFeatures | 53.56% |
| Cluster_11 | Prophet_Tuned_SigFeatures | 83.82% |

---

## Evaluation Metric — SMAPE

SMAPE = (1/n) x sum[ 2 x |actual - predicted| / (|actual| + |predicted|) ] x 100

Chosen over MAPE because it handles zero actual values without producing infinite errors.
Bounded between 0% and 200% — lower is better.
High SMAPE values across clusters are expected due to high CV% (83% to 721%) and significant zero-inflation.

---

## Requirements
pandas
numpy
scikit-learn
hdbscan
neuralforecast
prophet
holidays
matplotlib
scipy
statsmodels

---

## How to Run

1. Upload online_retail_II.xlsx to /content/ in Google Colab.
2. Run cells in order from Section 1 to Section 13.
3. All cluster dataframes, models and results are available after each section completes.
4. Saved models are stored in /content/saved_models/.

---

## Notes

- Clustering is performed on raw units_sold time series standardized for shape comparison only.
- Raw values are never modified for forecasting — standardization is clustering-only.
- Cluster 8 is non-stationary (both ADF and KPSS confirm) — flagged for differencing during modelling.
- Clusters 2 and 5 have extreme sparsity (70% and 89% zero days) making them inherently difficult to forecast.
- All lag and rolling features are computed per StockCode before aggregating to cluster level to preserve product-specific momentum.
"""
