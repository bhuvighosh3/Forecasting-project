import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet
from prophet.serialize import model_to_json
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

def train():
    print("Loading data (678MB)... this might take a moment")
    df = pd.read_csv(
        'LD2011_2014 2.txt',
        sep=';',
        decimal=',',
        engine='python',
        on_bad_lines='skip'
    )

    print("Preprocessing...")
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()
    df = df.ffill()
    df = df[(df != 0).any(axis=1)]

    # Clustering Logic from Notebook
    client_cols = [col for col in df.columns if col != 'global_mean']
    scaler = StandardScaler()
    # Transpose for clustering clients (rows) by their time series (features)
    X = scaler.fit_transform(df[client_cols].T)

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)

    K_BEST = 6
    km = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)

    client_clusters = pd.Series(labels, index=client_cols, name='cluster')
    
    # Save clustering models
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(pca, 'models/pca.joblib')
    joblib.dump(km, 'models/kmeans.joblib')
    
    # Save client mapping
    client_mapping = client_clusters.to_dict()
    with open('models/client_mapping.json', 'w') as f:
        json.dump(client_mapping, f, indent=4)

    cluster_profiles = []
    
    print("Training cluster models...")
    for c in range(K_BEST):
        members = client_clusters[client_clusters == c].index.tolist()
        mean_series = df[members].mean(axis=1)
        
        # Profile info for LLM context
        profile = {
            "cluster_id": c,
            "num_clients": len(members),
            "mean_consumption": float(mean_series.mean()),
            "std_consumption": float(mean_series.std()),
            "description": "" 
        }
        
        # Heuristics from notebook analysis
        if c == 0:
            profile["description"] = "Large aggregate (317 clients). Very stable trend with clear daily and weekly patterns. Minimal noise."
            model_type = "prophet"
        elif c == 1:
            profile["description"] = "Single client (1 client). High base load with sharp frequent dips. Best modeled with SARIMA."
            model_type = "sarima"
        elif c == 2:
            profile["description"] = "Single client (1 client). Highest variability in the dataset. Profile changes substantially day-to-day."
            model_type = "prophet" # Notebook says XGBoost/SARIMA but Prophet was evaluated
        elif c == 3:
            profile["description"] = "Moderate aggregate (9 clients). Balanced profile with symmetric distribution and clear seasonality."
            model_type = "prophet"
        elif c == 4:
            profile["description"] = "Dense aggregate (40 clients). Clean signal with non-linear temporal patterns. Best modeled with LSTM."
            model_type = "lstm"
        elif c == 5:
            profile["description"] = "Small aggregate (2 clients). Strong day-of-week variation."
            model_type = "prophet"

        cluster_profiles.append(profile)

        # Truncate training data for speed in this environment (last 30 days)
        ts_data = mean_series.tail(96 * 30)
        
        if model_type == "prophet":
            ts = ts_data.reset_index()
            ts.columns = ['ds', 'y']
            if ts['ds'].dt.tz is not None:
                ts['ds'] = ts['ds'].dt.tz_localize(None)
            ts['is_weekend'] = (ts['ds'].dt.dayofweek >= 5).astype(int)
            
            m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, seasonality_mode='multiplicative')
            m.add_regressor('is_weekend')
            m.fit(ts)
            
            with open(f'models/predictor_cluster_{c}.json', 'w') as f:
                f.write(model_to_json(m))
            print(f"Cluster {c} model (Prophet) saved.")

        elif model_type == "sarima":
            # SARIMA on hourly data for speed
            ts_hourly = ts_data.resample('h').mean()
            model = SARIMAX(ts_hourly, order=(1,1,1), seasonal_order=(1,0,1,24))
            results = model.fit(disp=False)
            results.save(f'models/predictor_cluster_{c}.statsmodels')
            print(f"Cluster {c} model (SARIMA) saved.")

        elif model_type == "lstm":
            # Small LSTM
            LOOKBACK = 96
            scaler_lstm = MinMaxScaler()
            scaled_data = scaler_lstm.fit_transform(ts_data.values.reshape(-1, 1))
            
            X_lstm, Y_lstm = [], []
            for i in range(LOOKBACK, len(scaled_data)):
                X_lstm.append(scaled_data[i-LOOKBACK:i, 0])
                Y_lstm.append(scaled_data[i, 0])
            X_lstm, Y_lstm = np.array(X_lstm), np.array(Y_lstm)
            X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
            
            model = Sequential([
                LSTM(32, input_shape=(LOOKBACK, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_lstm, Y_lstm, epochs=5, batch_size=64, verbose=0)
            
            model.save(f'models/predictor_cluster_{c}.keras')
            joblib.dump(scaler_lstm, f'models/scaler_lstm_cluster_{c}.joblib')
            print(f"Cluster {c} model (LSTM) saved.")

    with open('models/cluster_profiles.json', 'w') as f:
        json.dump(cluster_profiles, f, indent=4)

    print("All models successfully trained and serialized.")

if __name__ == "__main__":
    train()
