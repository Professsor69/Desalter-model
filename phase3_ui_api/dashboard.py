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
API_URL_WARNING = API_URL.replace("/optimize", "/predict-early-warning")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/refinery.png", width=90)
st.sidebar.title("Configuration")

# Tabs Layout
tab1, tab2 = st.tabs(["Static Optimizer", "Live Early Warning Monitor"])

with tab1:
    st.sidebar.markdown("### Incoming Crude Profile (Optimizer)")
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
    st.sidebar.info(f"📡 **Optimizer Endpoint:**\n`{API_URL}`\n\n📡 **Early Warning Endpoint:**\n`{API_URL_WARNING}`")

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

with tab2:
    st.subheader("📡 Live SCADA Stream & Early Warning Monitor")
    st.markdown("Predicting electrostatic desalter grid trips 60 minutes before they happen based on streaming sensor variables.")

    # Locate and read timeseries dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(root_dir, "timeseries_engine", "desalter_timeseries.csv")

    if not os.path.exists(csv_path):
        st.error(f"⚠️ **SCADA Timeseries Dataset not found.** Please verify the file is generated at: `{csv_path}`")
    else:
        # Load timeseries data into dataframe
        df_sim = pd.read_csv(csv_path)

        # Control States
        if 'sim_index' not in st.session_state:
            st.session_state.sim_index = 60
        if 'sim_active' not in st.session_state:
            st.session_state.sim_active = False

        # Controls Row
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("▶️ Start Live Stream", use_container_width=True):
                st.session_state.sim_active = True
        with c2:
            if st.button("⏸️ Pause Live Stream", use_container_width=True):
                st.session_state.sim_active = False
        with c3:
            st.session_state.sim_index = st.slider(
                "Historical Simulation Position (Minutes)", 
                min_value=60, 
                max_value=len(df_sim) - 1, 
                value=st.session_state.sim_index
            )

        idx = st.session_state.sim_index
        window = df_sim.iloc[idx - 59:idx + 1]

        # Prepare streaming payload
        readings = []
        for _, row in window.iterrows():
            readings.append({
                "API_Gravity": float(row["API_Gravity"]),
                "Inlet_Temperature": float(row["Inlet_Temperature"]),
                "Wash_Water_Rate": float(row["Wash_Water_Rate"]),
                "Inlet_Salt_PTB": float(row["Inlet_Salt_PTB"])
            })

        payload = {"readings": readings}

        # Query Early Warning API Endpoint
        warning_status = "Unknown"
        connection_ok = False
        try:
            resp = requests.post(API_URL_WARNING, json=payload, timeout=5)
            if resp.status_code == 200:
                warning_status = resp.json().get("warning_status", "Safe")
                connection_ok = True
            else:
                st.warning(f"📡 API returned error code {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Connection Error: Could not connect to early warning API at `{API_URL_WARNING}`. Ensure the FastAPI backend is running.")
        except Exception as e:
            st.error(f"❌ Error communicating with backend: {e}")

        # Render Alert Banner
        if connection_ok:
            if warning_status == "Imminent Trip":
                st.markdown(
                    """
                    <div style="background-color: #dc2626; padding: 22px; border-radius: 8px; border-left: 10px solid #991b1b; margin-bottom: 24px;">
                        <h2 style="color: white; margin: 0; font-size: 2.1rem; text-align: center; font-weight: 800; animation: blinker 1.2s linear infinite;">
                            🚨 EVACUATE / TRIP IN 60 MIN
                        </h2>
                        <p style="color: #fca5a5; margin: 6px 0 0 0; text-align: center; font-size: 1.15rem; font-weight: 600;">
                            Grid Voltage degradation detected! Action required: initiate desalter unit emergency procedures.
                        </p>
                    </div>
                    <style>
                        @keyframes blinker {
                            50% { opacity: 0.15; }
                        }
                    </style>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.success("✅ **SYSTEM OPERATION STABLE** — Grid Voltage nominal. No trip warnings predicted.")

        # Render Live Sensor Values
        current_reading = df_sim.iloc[idx]
        st.markdown("#### Real-time SCADA Sensor Readings")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        m_col1.metric(
            label="API Gravity",
            value=f"{current_reading['API_Gravity']:.1f} °API"
        )
        m_col2.metric(
            label="Inlet Temperature",
            value=f"{current_reading['Inlet_Temperature']:.1f} °C"
        )
        m_col3.metric(
            label="Wash Water Rate",
            value=f"{current_reading['Wash_Water_Rate']:.2f} %"
        )
        m_col4.metric(
            label="Inlet Salt",
            value=f"{current_reading['Inlet_Salt_PTB']:.1f} PTB"
        )
        
        grid_volts = current_reading['Grid_Voltage']
        if grid_volts == 0.0:
            m_col5.metric(
                label="Grid Voltage",
                value=f"{grid_volts:.1f} kV",
                delta="TRIPPED",
                delta_color="inverse"
            )
        else:
            m_col5.metric(
                label="Grid Voltage",
                value=f"{grid_volts:.1f} kV"
            )

        # Plot Sensor Trends
        st.markdown("---")
        st.markdown("#### 📈 Sensor Trends (Last 60 Minutes)")
        plot_df = window.copy()
        plot_df['Time'] = [t.split(" ")[1][:5] for t in plot_df['Timestamp']]
        plot_df = plot_df.set_index('Time')
        
        # Plot parameters
        st.line_chart(plot_df[['Inlet_Temperature', 'Grid_Voltage', 'Wash_Water_Rate']])

        # Handle Stream Loop Animation
        if st.session_state.sim_active:
            import time
            time.sleep(0.3)
            if st.session_state.sim_index < len(df_sim) - 1:
                st.session_state.sim_index += 1
            else:
                st.session_state.sim_active = False
            st.rerun()
