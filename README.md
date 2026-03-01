# ⚡ Electricity Consumption Forecasting at Scale

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-Facebook-0866FF?style=for-the-badge&logo=meta&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-189AB4?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

**A cluster-level machine learning framework for industrial electricity demand forecasting**  
*UCI LD2011–2014 · 370 Clients · 15-Minute Resolution · 52M+ Observations*

---

**Afreen Sorathiya · Anamika Kumari Mishra · Bhuvi Ghosh**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Pipeline Flow](#-pipeline-flow)
- [Cluster Segmentation](#-cluster-segmentation)
- [Model Selection Logic](#-model-selection-logic)
- [Project Timeline](#-project-timeline)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results Deep-Dive](#-results-deep-dive)
- [Challenges & Future Work](#-challenges--future-work)
- [Project Structure](#-project-structure)
- [Authors](#-authors)

---

## 🔍 Overview

This project addresses the challenge of forecasting electricity consumption for **370 industrial and commercial clients** measured at **15-minute intervals** over 3 years. Rather than applying a one-size-fits-all model, we:

1. **Segment** clients into 6 behaviorally homogeneous clusters via PCA + K-Means
2. **Route** each cluster to its statistically optimal model family
3. **Forecast** 30-day-ahead demand with 95% prediction intervals

The framework reduces average MAPE from an industry-default 10–15% to **1.74%–8.47%** across deployed clusters, and cuts model maintenance overhead by **98%** (370 models → 6 cluster models).

---

## 🏆 Key Results

| Cluster | Clients | Model | MAE (kWh) | RMSE (kWh) | MAPE | Status |
|:-------:|:-------:|:-----:|:---------:|:----------:|:----:|:------:|
| C0 | 317 | Prophet (cp=0.03) | 4.60 | 5.84 | 6.87% | ✅ Deploy |
| C1 | 1 | SARIMA (1,1,1)(1,0,1,24) | 645.48 | 855.43 | **4.57%** | ✅ Deploy |
| C1 | 1 | Prophet (fallback) | 1,105.69 | 1,405.36 | 8.17% | ⬇️ Fallback |
| C2 | 1 | Prophet | 3,484.77 | 4,762.39 | 12.07% | 🔄 Retry |
| C3 | 9 | Prophet (cp=0.05) | 234.41 | 302.24 | 7.56% | ✅ Deploy |
| C4 | 40 | **LSTM (lookback=96)** | **12.85** | **17.46** | **1.74%** | ✅ Deploy |
| C4 | 40 | Prophet (fallback) | 64.56 | 80.79 | 7.65% | ⬇️ Fallback |
| C5 | 2 | Prophet (cp=0.08) | 675.12 | 857.79 | 8.47% | 🔄 Retry |

> 💡 **LSTM on Cluster 4 achieves 1.74% MAPE** — a **5× improvement** over Prophet's 7.65% on the same cluster. Best-in-class for 15-min industrial energy forecasting.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[("📂 Raw CSV\nUCI LD2011–2014\n370 clients · 15-min")] --> B

    subgraph CLEAN ["🧹 Stage 1 · Data Cleaning"]
        B["Parse & Index\nEuropean decimal · datetime index"]
        B --> C["Impute Missing Values\nForward-fill · reindex gaps"]
        C --> D["Remove Corrupt Rows\nAll-zero meter records"]
        D --> E["Outlier Treatment\nIQR 3× fence · Z-score capping"]
    end

    E --> F

    subgraph FEAT ["⚙️ Stage 2 · Feature Engineering"]
        F["Calendar Features\nhour · dow · is_weekend"]
        F --> G["Lag Features\nlag_96 · lag_672"]
        G --> H["Rolling Stats\nrolling_mean_96"]
    end

    H --> I

    subgraph SEG ["🔬 Stage 3 · Client Segmentation"]
        I["StandardScaler\n(370 × n_timesteps)"]
        I --> J["PCA · 10 Components\n~85% variance explained"]
        J --> K["K-Means · K=6\nElbow method · n_init=10"]
    end

    K --> L

    subgraph SERIES ["📊 Stage 4 · Cluster Series"]
        L["Cluster Mean Series\n6 aggregated time series"]
        L --> M["Time Series Diagnostics\nADF · KPSS · Ljung-Box · CV"]
    end

    M --> N

    subgraph ROUTER ["🧠 Stage 5 · Model Router"]
        N{{"Cluster\nProfile"}}
        N -->|"C0 · 317 clients"| P1["Prophet\ncp=0.03 · Multiplicative"]
        N -->|"C1 · 1 client\nLeft skew −0.64"| P2["SARIMA\n(1,1,1)(1,0,1,24)"]
        N -->|"C2 · 1 client\nCV=0.306"| P3["XGBoost\nlag_96 · lag_672"]
        N -->|"C3 · 9 clients"| P4["Prophet\ncp=0.05 · Multiplicative"]
        N -->|"C4 · 40 clients"| P5["LSTM\nlookback=96 · 64→32 units"]
        N -->|"C5 · 2 clients\nWeekly CV=0.0097"| P6["XGBoost\ndow · lag_672"]
    end

    P1 --> R["📈 30-Day Forecasts\n95% Prediction Intervals"]
    P2 --> R
    P3 --> R
    P4 --> R
    P5 --> R
    P6 --> R

    style CLEAN fill:#1e3a5f,stroke:#c8922a,color:#fff
    style FEAT fill:#1c3f6e,stroke:#c8922a,color:#fff
    style SEG fill:#1a4a7a,stroke:#c8922a,color:#fff
    style SERIES fill:#1b3d6b,stroke:#c8922a,color:#fff
    style ROUTER fill:#0d1b2a,stroke:#c8922a,color:#fff
    style R fill:#1a6840,stroke:#c8922a,color:#fff
```

---

## 🔄 Pipeline Flow

```mermaid
sequenceDiagram
    participant DS as 📂 Data Source
    participant CP as 🧹 Cleaning Pipeline
    participant FE as ⚙️ Feature Eng.
    participant CL as 🔬 Clustering
    participant MR as 🧠 Model Router
    participant MD as 📐 Models
    participant EV as 📊 Evaluation

    DS->>CP: Raw CSV (52M rows, European decimal)
    CP->>CP: ffill NaN · remove all-zero rows
    CP->>CP: IQR + Z-score outlier capping
    CP->>FE: Clean DataFrame (n_timesteps × 370)
    FE->>FE: Calendar · lag_96 · lag_672 · rolling_mean_96
    FE->>CL: Feature matrix
    CL->>CL: StandardScaler → PCA(10) → KMeans(K=6)
    CL->>MR: 6 cluster labels + mean series
    MR->>MD: C0 → Prophet(cp=0.03)
    MR->>MD: C1 → SARIMA(1,1,1)(1,0,1,24)
    MR->>MD: C2 → XGBoost(lag features)
    MR->>MD: C3 → Prophet(cp=0.05)
    MR->>MD: C4 → LSTM(lookback=96)
    MR->>MD: C5 → XGBoost(dow + lag_672)
    MD->>EV: Forecasts on 10% test split
    EV->>EV: MAE · RMSE · MAPE · MAE/RMSE ratio
    EV-->>MR: ✅ Deploy or 🔄 Retry signal
```

---

## 🔬 Cluster Segmentation

```mermaid
graph LR
    subgraph INPUT ["Input: 370 Clients × n_timesteps"]
        A1["Client 1"]
        A2["Client 2"]
        A3["..."]
        A4["Client 370"]
    end

    subgraph PCA ["PCA — 10 Components (~85% var)"]
        B["Standardised\nClient Matrix\n370 × n_timesteps"]
        B --> C["PC₁  PC₂  ...  PC₁₀\n370 × 10 matrix"]
    end

    subgraph KMEANS ["K-Means — K=6 (Elbow Method)"]
        C --> D0["C0\n317 clients\nMean ~59 kWh"]
        C --> D1["C1\n1 client\nMean ~13,707 kWh"]
        C --> D2["C2\n1 client\nMean ~23,778 kWh"]
        C --> D3["C3\n9 clients\nMean ~3,099 kWh"]
        C --> D4["C4\n40 clients\nMean ~749 kWh"]
        C --> D5["C5\n2 clients\nMean ~6,780 kWh"]
    end

    A1 & A2 & A3 & A4 --> B

    style D0 fill:#1c3f6e,stroke:#c8922a,color:#fff
    style D1 fill:#1a6840,stroke:#c8922a,color:#fff
    style D2 fill:#b03a2e,stroke:#c8922a,color:#fff
    style D3 fill:#1c3f6e,stroke:#c8922a,color:#fff
    style D4 fill:#1a6840,stroke:#c8922a,color:#fff
    style D5 fill:#a0522d,stroke:#c8922a,color:#fff
```

### Cluster Diagnostics

| Cluster | Clients | Mean (kWh) | Skewness | Intraday CV | Weekly CV | ADF | KPSS | Ljung-Box |
|:-------:|:-------:|:----------:|:--------:|:-----------:|:---------:|:---:|:----:|:---------:|
| C0 | 317 | ~59.2 | −0.43 | 0.265 | 0.0071 | ✅ | ✅ | Sig. ✅ |
| C1 | 1 | ~13,707 | −0.64 | 0.284 | 0.0085 | ✅ | ✅ | Sig. ✅ |
| C2 | 1 | ~23,778 | −0.55 | **0.306** | 0.0082 | ✅ | ✅ | Sig. ✅ |
| C3 | 9 | ~3,099 | −0.20 | 0.232 | 0.0076 | ✅ | ✅ | Sig. ✅ |
| C4 | 40 | ~749 | −0.44 | 0.269 | 0.0079 | ✅ | ✅ | Sig. ✅ |
| C5 | 2 | ~6,780 | −0.23 | 0.283 | **0.0097** | ✅ | ✅ | Sig. ✅ |

> All clusters are stationary (ADF p < 0.05) with significant autocorrelation at both 1-day (lag 96) and 7-day (lag 672) horizons.

---

## 🧠 Model Selection Logic

```mermaid
flowchart TD
    START(["Cluster\nCharacteristics"]) --> Q1

    Q1{"n_clients\n> 20?"}
    Q1 -->|Yes| Q2{"Signal dense\nenough for LSTM?"}
    Q1 -->|No| Q3{"n_clients\n== 1?"}

    Q2 -->|Yes · C4: 40 clients| LSTM["🧠 LSTM\nlookback=96\n64→32 units\nMAPE: 1.74%"]
    Q2 -->|No · C0: 317 clients| PROPHET_LG["📈 Prophet\ncp=0.03 · Multiplicative\nMAPE: 6.87%"]

    Q3 -->|Yes| Q4{"High intraday CV\n& left skew?"}
    Q3 -->|No| Q5{"High weekly CV\n> 0.009?"}

    Q4 -->|Yes · C1: skew −0.64| SARIMA["📉 SARIMA\n(1,1,1)(1,0,1,24)\nHourly · D=0\nMAPE: 4.57%"]
    Q4 -->|Very high CV · C2: CV=0.306| XGBOOST_A["🌲 XGBoost\nlag_96 · rolling_mean_96\nMAPE: TBD"]

    Q5 -->|Yes · C5: CV=0.0097| XGBOOST_B["🌲 XGBoost\ndow · lag_672\nMAPE: TBD"]
    Q5 -->|No · C3: 9 clients| PROPHET_SM["📈 Prophet\ncp=0.05 · Multiplicative\nMAPE: 7.56%"]

    style LSTM fill:#1a6840,stroke:#c8922a,color:#fff
    style PROPHET_LG fill:#1c3f6e,stroke:#c8922a,color:#fff
    style PROPHET_SM fill:#1c3f6e,stroke:#c8922a,color:#fff
    style SARIMA fill:#1a6b9a,stroke:#c8922a,color:#fff
    style XGBOOST_A fill:#a0522d,stroke:#c8922a,color:#fff
    style XGBOOST_B fill:#a0522d,stroke:#c8922a,color:#fff
```

---

## 📅 Project Timeline

```mermaid
gantt
    title Project Milestones — Electricity Forecasting Pipeline
    dateFormat  YYYY-MM-DD
    axisFormat  Week %W

    section Phase 1 · Ingestion
    Data Ingestion & Validation       :done,    p1a, 2024-01-01, 2w
    Data Cleaning Pipeline            :done,    p1b, 2024-01-15, 1w

    section Phase 2 · EDA
    Correlation & EDA                 :done,    p2a, 2024-01-22, 1w
    PCA + Clustering                  :done,    p2b, 2024-01-29, 1w

    section Phase 3 · Diagnostics
    Cluster Diagnostics (ADF/KPSS)    :done,    p3a, 2024-02-05, 1w
    Feature Engineering               :done,    p3b, 2024-02-12, 1w

    section Phase 4 · Modelling
    Baseline Prophet (all clusters)   :done,    p4a, 2024-02-19, 2w
    SARIMA (Cluster 1)                :done,    p4b, 2024-03-04, 1w
    LSTM (Cluster 4)                  :done,    p4c, 2024-03-11, 2w

    section Phase 5 · Evaluation
    Model Comparison & Selection      :done,    p5a, 2024-03-25, 1w

    section Phase 6 · Deployment
    XGBoost (Clusters 2 & 5)         :active,  p6a, 2024-04-01, 2w
    MLOps Deployment Pipeline         :         p6b, 2024-04-15, 2w
```

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository — LD2011_2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) |
| **Clients** | 370 industrial & commercial |
| **Resolution** | 15-minute intervals (96 slots/day) |
| **Date Range** | January 2011 – December 2014 |
| **Total Observations** | ~52 million |
| **Format** | CSV, semicolon-delimited, European decimal (comma separator) |
| **Units** | kWh per 15-minute slot |

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/electricity-forecasting.git
cd electricity-forecasting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
statsmodels>=0.13.0
prophet>=1.1.0
tensorflow>=2.10.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### 1 · Data Cleaning

```python
import pandas as pd

# Load with European decimal handling
df = pd.read_csv(
    'data/LD2011_2014.txt',
    sep=';',
    decimal=',',
    engine='python',
    on_bad_lines='skip',
    index_col=0,
    parse_dates=True
)

# Forward-fill missing values
df = df.ffill()

# Remove all-zero corrupt rows
df = df[~(df == 0).all(axis=1)]

print(f"Shape after cleaning: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
```

### 2 · PCA + Clustering

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Transpose: (n_clients, n_timesteps)
X = df.T.values

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA — 10 components
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# K-Means — K=6
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

# Cluster mean series
cluster_series = {}
for k in range(6):
    members = df.columns[labels == k]
    cluster_series[k] = df[members].mean(axis=1)
    print(f"Cluster {k}: {len(members)} clients")
```

### 3 · Prophet Forecast

```python
from prophet import Prophet
import pandas as pd

def forecast_prophet(series, cluster_id, changepoint_prior=0.03, horizon=2880):
    """Fit Prophet and generate 30-day forecast (2880 slots at 15-min)."""
    df_prophet = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    }).reset_index(drop=True)

    model = Prophet(
        changepoint_prior_scale=changepoint_prior,
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        daily_seasonality=True,
        yearly_seasonality=False    # Only 6 months available
    )
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon, freq='15min')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Example: Cluster 0 (317 clients)
forecast_c0 = forecast_prophet(cluster_series[0], cluster_id=0, changepoint_prior=0.03)
```

### 4 · SARIMA Forecast (Cluster 1)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_sarima(series, order=(1,1,1), seasonal_order=(1,0,1,24)):
    """Hourly-resampled SARIMA forecast."""
    series_hourly = series.resample('1h').mean()

    train = series_hourly.iloc[:int(len(series_hourly) * 0.9)]
    test  = series_hourly.iloc[int(len(series_hourly) * 0.9):]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False
    )
    fitted = model.fit(disp=False)

    forecast = fitted.get_forecast(steps=len(test))
    return forecast.predicted_mean, forecast.conf_int()

pred, ci = forecast_sarima(cluster_series[1])
```

### 5 · LSTM Forecast (Cluster 4)

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def build_sequences(series, lookback=96):
    """Build (X, y) sequence pairs for LSTM."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def forecast_lstm(series, lookback=96, epochs=30, batch_size=128):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    split = int(len(scaled) * 0.9)
    train, test = scaled[:split], scaled[split:]

    X_train, y_train = build_sequences(train, lookback)
    X_test,  y_test  = build_sequences(test,  lookback)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(*X_train.shape, 1)
    X_test  = X_test.reshape(*X_test.shape,   1)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size, verbose=0,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
    )

    preds_scaled = model.predict(X_test).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    return preds, actuals

preds, actuals = forecast_lstm(cluster_series[4])
```

### 6 · Evaluation

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(actuals, predictions, label="Model"):
    mae  = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    ratio = mae / rmse
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"  MAE   : {mae:.2f} kWh")
    print(f"  RMSE  : {rmse:.2f} kWh")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"  Ratio : {ratio:.2f}")
    return dict(mae=mae, rmse=rmse, mape=mape, ratio=ratio)
```

---

## 📊 Results Deep-Dive

### Why LSTM wins on Cluster 4

```mermaid
xychart-beta
    title "MAPE Comparison: Prophet vs Best Model per Cluster"
    x-axis ["C0 (317)", "C1 (1)", "C2 (1)", "C3 (9)", "C4 (40)", "C5 (2)"]
    y-axis "MAPE (%)" 0 --> 14
    bar [6.87, 8.17, 12.07, 7.56, 7.65, 8.47]
    line [6.87, 4.57, 12.07, 7.56, 1.74, 8.47]
```

> 📌 **Bar = Prophet baseline · Line = Best deployed model**  
> C1 shows SARIMA's 42% MAE reduction; C4 shows LSTM's 5× improvement.

### Aggregation Size vs MAPE

```mermaid
quadrantChart
    title Cluster Size vs Forecast Accuracy
    x-axis "Few Clients" --> "Many Clients"
    y-axis "High MAPE (worse)" --> "Low MAPE (better)"
    quadrant-1 "Large & Accurate"
    quadrant-2 "Small & Accurate"
    quadrant-3 "Small & Inaccurate"
    quadrant-4 "Large & Inaccurate"
    C0 (317 clients): [0.90, 0.50]
    C4 (40 clients): [0.70, 0.90]
    C3 (9 clients): [0.40, 0.45]
    C5 (2 clients): [0.15, 0.40]
    C1 (1 client): [0.05, 0.60]
    C2 (1 client): [0.05, 0.20]
```

---

## 🚧 Challenges & Future Work

### Current Challenges

```mermaid
mindmap
  root((Challenges))
    Data Depth
      Only 6 months usable
      No yearly seasonality
      SARIMA capped at m=24 hourly
    Heterogeneity
      Scale: 10 kWh to 25000 kWh
      No universal model
      Cluster governance overhead
    LSTM Limits
      Needs dense signal
      Only C4 qualifies
      GPU required for training
    Concept Drift
      Behaviour shifts over time
      No automated retraining yet
      Silent accuracy degradation
```

### Roadmap

```mermaid
timeline
    title Future Development Roadmap
    section Immediate
        XGBoost for C2 & C5 : lag_96 · lag_672 · rolling features
                             : Beat Prophet MAPE 12.07% and 8.47%
    section Near-Term
        Extended Data Horizon : 3+ years enables yearly seasonality
                              : SARIMA at native m=96 resolution
        Online Retraining     : ADWIN drift detection
                              : Automated retraining triggers
    section Medium-Term
        Temporal Fusion Transformer : Multi-horizon probabilistic forecasts
                                    : Variable selection · quantile outputs
        Hierarchical Reconciliation : MinT or Bottom-Up
                                    : Cluster ↔ Global coherence
    section Long-Term
        Weather Integration : ERA5 temperature regressors
                            : HVAC and seasonal demand modelling
```

---

## 📁 Project Structure

```
electricity-forecasting/
│
├── 📂 data/
│   ├── raw/                    # LD2011_2014.txt (not tracked by git)
│   └── processed/              # Cleaned cluster mean series
│
├── 📂 notebooks/
│   ├── 01_eda_cleaning.ipynb   # Data exploration & cleaning
│   ├── 02_clustering.ipynb     # PCA + K-Means + diagnostics
│   ├── 03_prophet.ipynb        # Prophet models (all clusters)
│   ├── 04_sarima.ipynb         # SARIMA model (Cluster 1)
│   ├── 05_lstm.ipynb           # LSTM model (Cluster 4)
│   └── 06_evaluation.ipynb     # Cross-cluster comparison
│
├── 📂 src/
│   ├── cleaning.py             # Data cleaning pipeline
│   ├── clustering.py           # PCA + K-Means segmentation
│   ├── features.py             # Feature engineering
│   ├── models/
│   │   ├── prophet_model.py
│   │   ├── sarima_model.py
│   │   ├── lstm_model.py
│   │   └── xgboost_model.py
│   ├── evaluate.py             # MAE, RMSE, MAPE computation
│   └── router.py               # Model selection logic
│
├── 📂 outputs/
│   ├── forecasts/              # Generated forecast CSVs
│   ├── plots/                  # Forecast visualisations
│   └── metrics/                # Evaluation results
│
├── 📂 reports/
│   ├── ElectricityForecasting_TechnicalReport.docx
│   └── ElectricityForecasting_Formal.pptx
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👥 Authors

<div align="center">

| | Name | Role |
|:-:|------|------|
| 👩‍💻 | **Afreen Sorathiya** | Data Pipeline · Clustering · Prophet Modelling |
| 👩‍💻 | **Anamika Kumari Mishra** | SARIMA · LSTM · Model Evaluation |
| 👩‍💻 | **Bhuvi Ghosh** | Feature Engineering · XGBoost · Reporting |

</div>

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Dataset:** [UCI LD2011–2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) · **Framework:** Python · scikit-learn · Prophet · TensorFlow · statsmodels

*Made with ⚡ by Afreen, Anamika & Bhuvi*

</div>
