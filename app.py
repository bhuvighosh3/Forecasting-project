from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from llm_interface import ForecastingInterface
import pandas as pd

app = FastAPI(title="VoltForecast AI Backend")

# Initialize the interface
iface = ForecastingInterface()

class ForecastRequest(BaseModel):
    query: str

class ForecastResponse(BaseModel):
    cluster_id: int
    description: str
    mean_consumption: float
    periods: int
    data: list

@app.post("/api/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    try:
        # 1. Map query to cluster AND period
        cluster_id, periods = iface.get_cluster_from_query(request.query)
        profile = iface.profiles[cluster_id]
        
        # 2. Get prediction with requested periods
        prediction = iface.predict(cluster_id, periods=periods)
        
        # Format prediction for JSON
        if isinstance(prediction, pd.DataFrame):
            # Convert timestamp to string for JSON
            prediction['ds'] = prediction['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
            result_data = prediction.to_dict(orient='records')
        elif isinstance(prediction, pd.Series):
            # SARIMA Series
            result_data = [{"ds": str(ts), "yhat": float(val)} for ts, val in prediction.items()]
        else:
            # Placeholder for LSTM or error
            result_data = [{"ds": "N/A", "yhat": 0}]

        return ForecastResponse(
            cluster_id=cluster_id,
            description=profile['description'],
            mean_consumption=profile['mean_consumption'],
            periods=periods,
            data=result_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
