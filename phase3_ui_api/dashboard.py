import os
import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="IOCL Desalter AI Control System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    div.stButton > button:first-child {
        background-color: #0284c7;
        color: white;
        border: None;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #0369a1;
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🛢️ IOCL Desalter AI Control System")
st.markdown("##### Real-time Digital Twin Setpoint Optimization Dashboard")
st.markdown("---")

# API Configuration using Environment Variables
API_URL = os.getenv("API_URL", "http://localhost:8000/optimize")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/refinery.png", width=90)
st.sidebar.title("Configuration")
st.sidebar.markdown("### Incoming Crude Profile")

api_gravity = st.sidebar.slider(
    "API Gravity (°API)", 
    min_value=20.0, 
    max_value=45.0, 
    value=32.0, 
    step=0.1,
    help="Measures how heavy or light the crude is. Heavier crudes have lower API values."
)

inlet_bsw = st.sidebar.slider(
    "Inlet BSW (%)", 
    min_value=0.1, 
    max_value=2.5, 
    value=1.0, 
    step=0.01,
    help="Basic Sediment and Water percentage in the incoming crude."
)

inlet_salt_ptb = st.sidebar.slider(
    "Inlet Salt (PTB)",
    min_value=10.0,
    max_value=60.0,
    value=30.0,
    step=0.1,
    help="Inlet salt content in Pounds per Thousand Barrels (PTB)."
)

st.sidebar.markdown("---")
st.sidebar.info(f"📡 **Backend API Endpoint:**\n`{API_URL}`")

# Dashboard Grid Layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("📥 Operational Profile Inputs")
    
    # Render table summarizing current inputs
    input_data = pd.DataFrame({
        "Variable": ["API Gravity", "Inlet BSW", "Inlet Salt"],
        "Value": [f"{api_gravity:.1f} °API", f"{inlet_bsw:.2f} %", f"{inlet_salt_ptb:.1f} PTB"]
    })
    st.table(input_data)
    
    # Run Optimizer Button
    run_btn = st.button("Run AI Optimizer", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Optimization Results")
    
    if run_btn:
        # Check for heavy/wet crude emulsion risk alert
        if api_gravity < 28.0 or inlet_bsw > 1.2:
            st.error("High Emulsion Risk Detected: Heavy/Wet Crude Profile.")
        else:
            st.success("✅ Normal Operating Conditions Profile.")

        with st.spinner("Sending request to Digital Twin API..."):
            try:
                payload = {
                    "API_Gravity": api_gravity,
                    "Inlet_BSW": inlet_bsw,
                    "Inlet_Salt_PTB": inlet_salt_ptb
                }
                
                # Make HTTP POST request to FastAPI /optimize endpoint
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "success":
                    optimal = result.get("optimal_setpoints", {})
                    pred_thickness = result.get("predicted_emulsion_thickness_mm", 0.0)
                    metadata = result.get("optimization_metadata", {})
                    
                    st.markdown("#### Recommended Setpoint Values")
                    
                    # Columns to hold st.metric cards
                    m_col1, m_col2, m_col3 = st.columns(3)
                    
                    with m_col1:
                        st.metric(
                            label="Recommended Temperature",
                            value=f"{optimal.get('Temperature_C', 0.0):.1f} °C",
                            help="Optimal process heater outlet temperature setpoint."
                        )
                    with m_col2:
                        st.metric(
                            label="Recommended Wash Water",
                            value=f"{optimal.get('Wash_Water_Percent', 0.0):.1f} %",
                            help="Optimal wash water flow percentage setpoint."
                        )
                    with m_col3:
                        st.metric(
                            label="Predicted Emulsion Layer",
                            value=f"{pred_thickness:.2f} mm",
                            help="Expected thickness of the emulsion layer using optimal setpoints."
                        )
                    
                    # Collapsible metadata logs
                    with st.expander("🛠️ Optimization Execution Logs"):
                        st.json({
                            "API Response Status": result.get("status"),
                            "Optimization Method": metadata.get("method_used"),
                            "Raw Grid Min Emulsion": f"{metadata.get('grid_search_raw_min'):.4f} mm",
                            "Powell Success Status": metadata.get("powell_success"),
                            "Powell Iterations": metadata.get("powell_iterations")
                        })
                else:
                    st.error(f"API Error: {result.get('message', 'Optimization failed')}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Connection Error: Could not connect to the API server at `{API_URL}`. Verify the FastAPI backend is running.")
            except Exception as e:
                st.error(f"❌ Error occurred during optimization: {e}")
    else:
        st.info("👈 Set the crude parameters in the sidebar and click **Run AI Optimizer** to execute prescriptive control calculations.")
