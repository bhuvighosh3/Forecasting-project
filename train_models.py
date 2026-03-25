import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet
from prophet.serialize import model_to_json
import warnings

warnings.filterwarnings('ignore')

def train():
    print("Loading data...")
    # Load data
    df = pd.read_csv(
        'LD2011_2014 2.txt',
        sep=';',
        decimal=',',
        engine='python',
        on_bad_lines='skip'
    )

    print("Preprocessing...")
    # Handle the date column more robustly
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()
    df = df.ffill()
    df = df[(df != 0).any(axis=1)]

    # Clustering
    client_cols = [col for col in df.columns if col != 'global_mean']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[client_cols].T)  # (n_clients, n_timesteps)

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)

    K_BEST = 6
    km = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)

    client_clusters = pd.Series(labels, index=client_cols, name='cluster')
    
    # Save clustering models
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(pca, 'models/pca.joblib')
    joblib.dump(km, 'models/kmeans.joblib')

    cluster_profiles = []
    
    print("Training cluster models...")
    for c in range(K_BEST):
        members = client_clusters[client_clusters == c].index.tolist()
        mean_series = df[members].mean(axis=1)
        
        # Profile info for LLM
        profile = {
            "cluster_id": c,
            "num_clients": len(members),
            "mean_consumption": float(mean_series.mean()),
            "std_consumption": float(mean_series.std()),
            "description": "" # Will be filled later or based on stats
        }
        
        # Simple description heuristic
        if len(members) > 100:
            profile["description"] = "Large aggregate of clients with highly stable and predictable consumption patterns."
        elif len(members) == 1:
            profile["description"] = "Single client with potentially high variability or unique sharp consumption peaks/dips."
        elif mean_series.mean() > 20000:
            profile["description"] = "High-load industrial or commercial cluster with significant variations."
        else:
            profile["description"] = "Moderate aggregate cluster with typical daily seasonality."
            
        cluster_profiles.append(profile)

        # Train Prophet for each cluster (simpler for this demo)
        ts = mean_series.reset_index()
        ts.columns = ['ds', 'y']
        if ts['ds'].dt.tz is not None:
            ts['ds'] = ts['ds'].dt.tz_localize(None)
        
        # Add weekend regressor
        ts['is_weekend'] = (ts['ds'].dt.dayofweek >= 5).astype(int)

        # Use a very small subset of data to speed up training for the demo
        ts_train = ts.tail(96 * 7) 

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        m.add_regressor('is_weekend')
        m.fit(ts_train)

        # Save model
        with open(f'models/prophet_cluster_{c}.json', 'w') as f:
            f.write(model_to_json(m))
            
        print(f"Cluster {c} model saved.")

    with open('models/cluster_profiles.json', 'w') as f:
        json.dump(cluster_profiles, f, indent=4)

    print("All models trained and saved.")

if __name__ == "__main__":
    train()
