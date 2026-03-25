import streamlit as st
import pandas as pd
from llm_interface import ForecastingInterface
import os

# Page Config
st.set_page_config(page_title="VoltForecast AI", page_icon="⚡", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Main App Background (Deep Blue) */
    .stApp {
        background-color: #0b192c;
        color: #e2e8f0;
    }
    /* Input Field */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #2563eb;
        background-color: #1e3a8a;
        color: white;
    }
    .stTextInput > div > div > input:focus {
        border-color: #60a5fa;
        box-shadow: 0 0 10px rgba(96, 165, 250, 0.5);
    }
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        background-color: #2563eb;
        color: white;
        width: 100%;
        font-weight: bold;
        border: 1px solid #3b82f6;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
        border-color: #93c5fd;
    }
    /* Metric Cards */
    .metric-card {
        background-color: #172554;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #1e3a8a;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .metric-card h4 {
        color: #bfdbfe;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("⚡ VoltForecast AI Dashboard")
st.markdown("### Next-Gen Electricity Analytics with Anthropic & Specialized Models")

# Initialize Backend
@st.cache_resource
def get_interface():
    return ForecastingInterface()

iface = get_interface()

# User Input
query = st.text_input("What would you like to forecast?", placeholder="e.g., Predict for MT_001 for 3 days")

if query:
    with st.spinner("Analyzing query with Anthropic..."):
        try:
            # 1. Map query
            structure = iface.get_query_structure(query)
            cluster_id = structure['cluster_id']
            periods = structure['periods']
            profile = iface.profiles[cluster_id]
            
            # 2. Layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Selection Metadata")
                st.info(f"**Mapping Source:** {structure['mapping_source']}")
                st.markdown(f"**Reasoning:** *{structure['original_result'].get('reasoning', 'N/A')}*")
                
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#94a3b8; font-size:0.8rem; margin-bottom:5px;">CLUSTER {cluster_id}</p>
                        <h4 style="margin-bottom:10px;">{profile['description']}</h3>
                        <p style="font-size:1.2rem; font-weight:bold;">Avg: {profile['mean_consumption']:.2f} kWh</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                st.success(f"Forecast Duration: {periods//4} hours ({periods//96} days)")

            with col2:
                # 3. Predict
                with st.spinner("Generating prediction..."):
                    forecast_df = iface.predict(cluster_id, periods=periods)
                    
                    st.markdown("#### Consumption Forecast")
                    # Use Area Chart for premium feel
                    chart_df = forecast_df.rename(columns={'ds': 'Time', 'yhat': 'Consumption (kWh)'}).set_index('Time')
                    st.area_chart(chart_df, color="#3b82f6")
                    
                    with st.expander("Show Raw Data"):
                        st.dataframe(forecast_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("💡 Pro-tip: Try specifying a client ID like 'MT_001' or a duration like '2 days'.")

# Footer
st.markdown("---")
st.caption("Powered by Anthropic Claude 3 & Specialized Forecasting Models (Prophet, SARIMA, LSTM)")
