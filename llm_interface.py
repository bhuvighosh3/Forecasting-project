import os
import json
import joblib
import pandas as pd
import numpy as np
import re
from anthropic import Anthropic
from prophet.serialize import model_from_json
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

# Anthropic API Key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

class ForecastingInterface:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Load cluster profiles
        with open(os.path.join(models_dir, 'cluster_profiles.json'), 'r') as f:
            self.profiles = json.load(f)
            
        # Load client mapping
        with open(os.path.join(models_dir, 'client_mapping.json'), 'r') as f:
            self.client_mapping = json.load(f)
        
        # Load shared models
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        self.pca = joblib.load(os.path.join(models_dir, 'pca.joblib'))
        self.kmeans = joblib.load(os.path.join(models_dir, 'kmeans.joblib'))

    def get_query_structure(self, query):
        """Use Anthropic to extract (client_id, cluster_description, periods) from query."""
        print(f"Using Anthropic to analyze query: '{query}'")
        
        system_prompt = f"""
        You are an expert assistant for an electricity forecasting system.
        Extract the following information from the user's query:
        1. client_id: The client identifier if found (e.g., MT_001, MT_322).
        2. cluster_description: A brief summary of the type of consumer if no client_id is given (e.g., "stable industrial", "residential with dips").
        3. forecast_period_hours: The TOTAL number of hours requested for the forecast. 
           Convert phrases like "7 days" to 168, "1 week" to 168, "next 48 hours" to 48, etc.
           Default is 24 if not specified.
        4. reasoning: A short explanation of why you chose this client/cluster/period.
        
        Known Clients examples: {list(self.client_mapping.keys())[:10]}...
        
        Return your answer ONLY as a JSON object with these keys.
        """
        
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        # Extract JSON from response text
        content = response.content[0].text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {}
        
        client_id = result.get("client_id")
        cluster_desc = result.get("cluster_description")
        hours = result.get("forecast_period_hours", 24)
        periods = hours * 4 # 15-min intervals
        
        # Map to cluster_id
        cluster_id = None
        mapping_source = ""
        
        if client_id and client_id in self.client_mapping:
            cluster_id = int(self.client_mapping[client_id])
            mapping_source = f"Direct mapping from Client {client_id}"
        elif cluster_desc:
            # Fallback to keyword matching for cluster_desc
            desc_lower = cluster_desc.lower()
            if "stable" in desc_lower or "large" in desc_lower or "predictable" in desc_lower:
                cluster_id = 0
            elif "single" in desc_lower or "dips" in desc_lower:
                cluster_id = 1
            elif "variable" in desc_lower or "changing" in desc_lower:
                cluster_id = 2
            elif "balanced" in desc_lower or "moderate" in desc_lower:
                cluster_id = 3
            elif "dense" in desc_lower or "non-linear" in desc_lower:
                cluster_id = 4
            elif "weekly" in desc_lower or "day-of-week" in desc_lower:
                cluster_id = 5
            mapping_source = "Derived from query description"
        
        if cluster_id is None:
            cluster_id = 0 # Default to stable aggregate
            mapping_source = "Default (Cluster 0)"
            
        return {
            "client_id": client_id,
            "cluster_id": cluster_id,
            "periods": periods,
            "mapping_source": mapping_source,
            "original_result": result
        }

    def predict(self, cluster_id, periods=96):
        """Generate forecast for given cluster_id."""
        print(f"Generating forecast for Cluster {cluster_id} for {periods} slots...")
        
        if cluster_id in [0, 2, 3, 5]:
            path = os.path.join(self.models_dir, f'predictor_cluster_{cluster_id}.json')
            with open(path, 'r') as f:
                m = model_from_json(f.read())
            future = m.make_future_dataframe(periods=periods, freq='15min')
            future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
            forecast = m.predict(future)
            return forecast[['ds', 'yhat']].tail(periods)

        elif cluster_id == 1:
            res = SARIMAXResults.load(os.path.join(self.models_dir, f'predictor_cluster_{cluster_id}.statsmodels'))
            forecast = res.forecast(steps=periods)
            # Convert series to generic dataframe format
            return pd.DataFrame({"ds": forecast.index, "yhat": forecast.values})

        elif cluster_id == 4:
            # Simplified LSTM prediction (placeholder as before)
            dates = pd.date_range(pd.Timestamp.now(), periods=periods, freq='15min')
            return pd.DataFrame({"ds": dates, "yhat": [np.random.normal(8722, 500) for _ in range(periods)]})

if __name__ == "__main__":
    iface = ForecastingInterface()
    res = iface.get_query_structure("Predict for MT_001 for 2 days")
    print(res)
