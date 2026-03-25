import typer
from llm_interface import ForecastingInterface
import os

app = typer.Typer()

@app.command()
def forecast(query: str):
    """
    Generate a forecast based on a natural language query.
    Example: python main.py "Predict for a stable industrial site"
    """
    if not os.path.exists('models/cluster_profiles.json'):
        print("Error: Models not found. Please run 'uv run python train_final.py' first.")
        return

    iface = ForecastingInterface()
    
    # 1. Map query to cluster AND period
    cluster_id, periods = iface.get_cluster_from_query(query)
    profile = iface.profiles[cluster_id]
    
    print(f"\n[+] Mapped Query to Cluster {cluster_id}")
    print(f"    Description: {profile['description']}")
    print(f"    Avg Consumption: {profile['mean_consumption']:.2f} kWh")
    print(f"    Requested Period: {periods} slots (~{periods/96:.1f} days)")
    
    # 2. Generate Prediction
    print("\n[+] Generating Prediction...")
    prediction = iface.predict(cluster_id, periods=periods)
    
    print("\n[+] Forecast Results (next 24 hours):")
    if isinstance(prediction, str):
        print(prediction)
    else:
        print(prediction.head(10))
        print("...")

if __name__ == "__main__":
    app()
