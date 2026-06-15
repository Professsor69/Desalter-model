import os
import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
import sys

# Establish path to the repository root so we can import from phase2_optimizer
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from phase2_optimizer.prescriptive_optimizer import load_digital_twin
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..')))
    from phase2_optimizer.prescriptive_optimizer import load_digital_twin

@st.cache_resource
def get_local_digital_twin():
    return load_digital_twin()

def render_custom_metric_card(label, value, delta=None, warning=False, help=""):
    border_color = "#ef4444" if warning else "#334155"
    bg_color = "#451a03" if warning else "var(--secondary-background-color)" # dark red if warning, else native secondary background
    text_color = "#ef4444" if warning else "#38bdf8"
    
    delta_html = ""
    if delta:
        d_color = "#ef4444" if "penalty" in delta or "+" in delta else "#10b981"
        if warning:
            d_color = "#ef4444"
        delta_html = f"<div style='font-size: 0.95rem; color: {d_color}; margin-top: 4px; font-weight: 600;'>{delta}</div>"
        
    st.markdown(f"""
        <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 8px; padding: 16px; min-height: 100px;">
            <div style="font-size: 0.9rem; color: #94a3b8; font-weight: 500;">{label}</div>
            <div style="font-size: 1.8rem; color: {text_color}; font-weight: 800; margin-top: 4px;">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

def reset_optimizer_state():
    if st.session_state.get("optimizer_run_done", False):
        st.session_state.optimizer_run_done = False
        st.session_state.ai_results = {}
        st.session_state.comparison_run_done = False
        st.session_state.crude_modified = True

# Page Configuration
st.set_page_config(
    page_title="IOCL Desalter AI Control System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/refinery.png", width=90)
st.sidebar.title("Configuration")

# Custom CSS for rich dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
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



# Tabs Layout
tab1, tab2 = st.tabs(["Static Optimizer", "Live Early Warning Monitor"])

with tab1:
    st.sidebar.markdown("### Incoming Crude Profile (Optimizer)")
    crude_blend = st.sidebar.selectbox(
        "Crude Blend", 
        ['Basrah Heavy', 'Arab Light', 'Ural', 'Bonny Light'],
        help="Select the raw crude blend type.",
        on_change=reset_optimizer_state
    )
    api_gravity = st.sidebar.slider(
        "API Gravity (°API)", 
        min_value=20.0, 
        max_value=45.0, 
        value=32.0, 
        step=0.1,
        help="Measures how heavy or light the crude is. Heavier crudes have lower API values.",
        on_change=reset_optimizer_state
    )

    inlet_bsw = st.sidebar.slider(
        "Inlet BSW (%)", 
        min_value=0.1, 
        max_value=2.5, 
        value=1.0, 
        step=0.01,
        help="Basic Sediment and Water percentage in the incoming crude.",
        on_change=reset_optimizer_state
    )

    inlet_salt_ptb = st.sidebar.slider(
        "Inlet Salt (PTB)",
        min_value=10.0,
        max_value=60.0,
        value=30.0,
        step=0.1,
        help="Inlet salt content in Pounds per Thousand Barrels (PTB).",
        on_change=reset_optimizer_state
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"📡 **Optimizer Endpoint:**\n`{API_URL}`\n\n📡 **Early Warning Endpoint:**\n`{API_URL_WARNING}`")

    # Dashboard Grid Layout
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("📥 Operational Profile Inputs")
        
        # Render table summarizing current inputs
        input_data = pd.DataFrame({
            "Variable": ["Crude Blend", "API Gravity", "Inlet BSW", "Inlet Salt"],
            "Value": [crude_blend, f"{api_gravity:.1f} °API", f"{inlet_bsw:.2f} %", f"{inlet_salt_ptb:.1f} PTB"]
        })
        st.table(input_data)
        
        # Run Optimizer Button
        run_btn = st.button("Run AI Optimizer", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎯 Optimization Results")
        
        # Ensure session state variables for tab1 optimizer exist
        if "optimizer_run_done" not in st.session_state:
            st.session_state.optimizer_run_done = False
            st.session_state.ai_results = {}
            st.session_state.comparison_run_done = False
            st.session_state.op_thickness = 0.0
            st.session_state.crude_modified = False

        if run_btn:
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
                        st.session_state.optimizer_run_done = True
                        st.session_state.ai_results = result
                        # Reset comparison when a new AI optimization runs
                        st.session_state.comparison_run_done = False
                        st.session_state.crude_modified = False
                        
                        # Set default values for operator overrides to match the newly generated recommendations
                        optimal = result.get("optimal_setpoints", {})
                        st.session_state.op_temp = float(optimal.get('Temperature_C', 130.0))
                        st.session_state.op_water = float(optimal.get('Wash_Water_Percent', 5.0))
                    else:
                        st.error(f"API Error: {result.get('message', 'Optimization failed')}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Connection Error: Could not connect to the API server at `{API_URL}`. Verify the FastAPI backend is running.")
                except Exception as e:
                    st.error(f"❌ Error occurred during optimization: {e}")

        if st.session_state.optimizer_run_done:
            result = st.session_state.ai_results
            
            # Render Emulsion Risk Prediction Banner
            predicted_risk = result.get("predicted_risk", "Unknown")
            if predicted_risk == "High":
                st.error("🚨 **HIGH RISK**: This crude batch is predicted to be high risk for emulsion.")
            elif predicted_risk == "Medium":
                st.warning("⚠️ **MEDIUM RISK**: Moderate processing difficulty expected.")
            elif predicted_risk == "Low":
                st.success("✅ **LOW RISK**: This crude batch is predicted to process easily.")
            else:
                st.info(f"**Predicted Emulsion Risk Class:** {predicted_risk}")

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

            # --- Operator Override vs. AI Optimizer Module ---
            st.markdown("---")
            st.markdown("### 🎛️ Operator Override vs. AI Optimizer")
            
            # 1. Operator Input Panel Card
            with st.container(border=True):
                st.markdown("#### 🛠️ Simulate Manual Setpoints")
                
                # Initialize slider session state keys if they are not already set
                if "op_temp" not in st.session_state:
                    st.session_state.op_temp = float(optimal.get('Temperature_C', 130.0))
                if "op_water" not in st.session_state:
                    st.session_state.op_water = float(optimal.get('Wash_Water_Percent', 5.0))
                
                col_inputs1, col_inputs2 = st.columns(2)
                with col_inputs1:
                    op_temp_val = st.slider(
                        "Operator Temperature (°C)",
                        min_value=110.0,
                        max_value=150.0,
                        key="op_temp",
                        step=0.5,
                        help="Simulated operator manual temperature setpoint."
                    )
                with col_inputs2:
                    op_water_val = st.slider(
                        "Operator Wash Water (%)",
                        min_value=2.0,
                        max_value=8.0,
                        key="op_water",
                        step=0.1,
                        help="Simulated operator manual wash water percentage setpoint."
                    )

            # 2. Compute Operator Prediction and run shadow logging in the background
            try:
                payload_predict = {
                    "API_Gravity": api_gravity,
                    "Inlet_BSW": inlet_bsw,
                    "Inlet_Salt_PTB": inlet_salt_ptb,
                    "Temperature_C": op_temp_val,
                    "Wash_Water_Percent": op_water_val
                }
                PREDICT_URL = API_URL.replace("/optimize", "/predict")
                resp_predict = requests.post(PREDICT_URL, json=payload_predict, timeout=10)
                resp_predict.raise_for_status()
                result_predict = resp_predict.json()
                
                if result_predict.get("status") == "success":
                    op_thickness = result_predict.get("predicted_emulsion_thickness_mm", 0.0)
                else:
                    twin_model = get_local_digital_twin()
                    df_single = pd.DataFrame([{
                        'API_Gravity': api_gravity,
                        'Inlet_BSW': inlet_bsw,
                        'Inlet_Salt_PTB': inlet_salt_ptb,
                        'Temperature_C': op_temp_val,
                        'Wash_Water_Percent': op_water_val
                    }])
                    op_thickness = float(twin_model.predict(df_single)[0])
            except Exception as e:
                twin_model = get_local_digital_twin()
                df_single = pd.DataFrame([{
                    'API_Gravity': api_gravity,
                    'Inlet_BSW': inlet_bsw,
                    'Inlet_Salt_PTB': inlet_salt_ptb,
                    'Temperature_C': op_temp_val,
                    'Wash_Water_Percent': op_water_val
                }])
                op_thickness = float(twin_model.predict(df_single)[0])

            # 3. Metric Delta Cards (Top of Module)
            st.markdown("#### 📊 Real-time Performance Comparison")
            
            ai_thickness = pred_thickness
            delta_thickness = op_thickness - ai_thickness
            
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                render_custom_metric_card(
                    label="AI Optimized Emulsion Thickness",
                    value=f"{ai_thickness:.2f} mm",
                    help="Predicted thickness of the emulsion layer using optimal AI setpoints."
                )
            with c_m2:
                warning_state = op_thickness > 45.0
                if delta_thickness > 0:
                    delta_str = f"+{delta_thickness:.2f} mm penalty"
                else:
                    delta_str = f"{delta_thickness:.2f} mm benefit"
                    
                render_custom_metric_card(
                    label="Operator's Predicted Emulsion Thickness",
                    value=f"{op_thickness:.2f} mm",
                    delta=delta_str,
                    warning=warning_state,
                    help="Predicted thickness of the emulsion layer using operator's simulated setpoints."
                )
                
            # 4. Generate 2D Contour Heatmap Grid Data
            temps_lin = np.linspace(110.0, 150.0, 50)
            wws_lin = np.linspace(2.0, 8.0, 50)
            temp_grid, ww_grid = np.meshgrid(temps_lin, wws_lin)
            temp_flat = temp_grid.flatten()
            ww_flat = ww_grid.flatten()

            grid_df = pd.DataFrame({
                'API_Gravity': [api_gravity] * len(temp_flat),
                'Inlet_BSW': [inlet_bsw] * len(temp_flat),
                'Inlet_Salt_PTB': [inlet_salt_ptb] * len(temp_flat),
                'Temperature_C': temp_flat,
                'Wash_Water_Percent': ww_flat
            })

            twin_model = get_local_digital_twin()
            z_flat = twin_model.predict(grid_df)
            z_grid = z_flat.reshape(temp_grid.shape)

            # 5. Render Plotly Contour Map
            st.markdown("##### 🗺️ Topographical Emulsion Performance Map")
            
            fig = go.Figure()
            
            # Contour layer
            fig.add_trace(go.Contour(
                x=temps_lin,
                y=wws_lin,
                z=z_grid,
                colorscale='RdYlBu_r',  # Blue is low (valley/safe), Red is high (peak/dangerous)
                reversescale=False,
                colorbar=dict(
                    title=dict(
                        text='Emulsion (mm)',
                        side='right',
                        font=dict(color='#94a3b8')
                    ),
                    tickfont=dict(color='#94a3b8')
                ),
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=10, color='black')
                ),
                hoverinfo='x+y+z'
            ))
            
            # AI static marker (green star)
            ai_temp = optimal.get('Temperature_C', 130.0)
            ai_water = optimal.get('Wash_Water_Percent', 5.0)
            fig.add_trace(go.Scatter(
                x=[ai_temp],
                y=[ai_water],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=16,
                    color='#10b981',
                    line=dict(color='#ffffff', width=2)
                ),
                name='AI Optimized Setpoint',
                showlegend=True
            ))
            
            # Operator dynamic marker (white circle with crosshair)
            fig.add_trace(go.Scatter(
                x=[op_temp_val],
                y=[op_water_val],
                mode='markers',
                marker=dict(
                    symbol='circle-cross-open',
                    size=14,
                    color='#ffffff',
                    line=dict(width=3)
                ),
                name='Operator Setpoint Override',
                showlegend=True
            ))
            
            fig.update_layout(
                plot_bgcolor='#0f172a',
                paper_bgcolor='#0f172a',
                margin=dict(l=40, r=40, t=40, b=40),
                font=dict(color='#94a3b8'),
                xaxis=dict(
                    title='Temperature (°C)',
                    gridcolor='#1e293b',
                    zeroline=False,
                    range=[110, 150]
                ),
                yaxis=dict(
                    title='Wash Water (%)',
                    gridcolor='#1e293b',
                    zeroline=False,
                    range=[2, 8]
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    font=dict(color='#f8fafc')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

        else:
            if st.session_state.get("crude_modified", False):
                st.warning("Incoming crude profile modified. Please click 'Run AI Optimizer' to generate new baseline setpoints.")
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
        csv_path = os.path.join(os.path.dirname(root_dir), "timeseries_engine", "desalter_timeseries.csv")

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
            st.session_state.sim_index = st.number_input(
                "Historical Simulation Position (Minutes)", 
                min_value=60, 
                max_value=len(df_sim) - 1, 
                value=st.session_state.sim_index,
                step=1
            )

        idx = st.session_state.sim_index
        window = df_sim.iloc[idx - 60:idx + 1].copy()
        current_reading = df_sim.iloc[idx]

        # Initialize session state variables if they do not exist
        if 'ov_temp' not in st.session_state:
            st.session_state.ov_temp = float(current_reading['Inlet_Temperature'])
            st.session_state.ov_water = float(current_reading['Wash_Water_Rate'])

        # Operator Intervention Sliders
        st.markdown("---")
        st.markdown("#### 🛠️ Operator Manual Intervention (What-If Analysis)")
        st.markdown("Use these controls to simulate adjusting desalter settings during an alert and see if your intervention clears the trip warning.")
        override_active = st.checkbox("Enable Operator Manual Overrides", value=False)
        
        # If overrides are NOT active, continuously sync with original SCADA readings
        if not override_active:
            st.session_state.ov_temp = float(current_reading['Inlet_Temperature'])
            st.session_state.ov_water = float(current_reading['Wash_Water_Rate'])
            st.session_state.prev_sim_index = idx
        
        if override_active:
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                st.session_state.ov_temp = st.slider(
                    "Manual Inlet Temperature Override (°C)", 
                    min_value=100.0, 
                    max_value=160.0, 
                    value=st.session_state.ov_temp,
                    step=0.5
                )
            with col_o2:
                st.session_state.ov_water = st.slider(
                    "Manual Wash Water Rate Override (%)", 
                    min_value=0.0, 
                    max_value=12.0, 
                    value=st.session_state.ov_water,
                    step=0.1
                )
            # Apply overrides to the current reading (the latest row in our 60-minute buffer)
            window.iloc[-1, window.columns.get_loc('Inlet_Temperature')] = st.session_state.ov_temp
            window.iloc[-1, window.columns.get_loc('Wash_Water_Rate')] = st.session_state.ov_water
            
            ov_temp = st.session_state.ov_temp
            ov_water = st.session_state.ov_water
        else:
            ov_temp = float(current_reading['Inlet_Temperature'])
            ov_water = float(current_reading['Wash_Water_Rate'])

        # Prepare streaming payload
        readings = []
        for _, row in window.iterrows():
            readings.append({
                "API_Gravity": float(row["API_Gravity"]),
                "Inlet_Temperature": float(row["Inlet_Temperature"]),
                "Wash_Water_Rate": float(row["Wash_Water_Rate"]),
                "Inlet_Salt_PTB": float(row["Inlet_Salt_PTB"]),
                "Emulsion_Layer_Thickness": float(row["Emulsion_Layer_Thickness"])
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
        grid_volts = float(current_reading['Grid_Voltage'])

        if connection_ok:
            if grid_volts == 0.0:
                st.markdown(
                    """
                    <div style="background-color: #78350f; padding: 22px; border-radius: 8px; border-left: 10px solid #d97706; margin-bottom: 24px;">
                        <h2 style="color: white; margin: 0; font-size: 2.1rem; text-align: center; font-weight: 800;">
                            ⚠️ SYSTEM TRIPPED — DESALTER OFFLINE
                        </h2>
                        <p style="color: #fde68a; margin: 6px 0 0 0; text-align: center; font-size: 1.15rem; font-weight: 600;">
                            Grid Voltage at 0.0 kV. Emergency shutdown in progress.
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif warning_status == "Imminent Trip":
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
        st.markdown("#### Real-time SCADA Sensor Readings")
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        
        m_col1.metric(
            label="API Gravity",
            value=f"{current_reading['API_Gravity']:.1f} °API"
        )
        m_col2.metric(
            label="Inlet Temperature",
            value=f"{current_reading['Inlet_Temperature']:.1f} °C" if not override_active else f"{ov_temp:.1f} °C (Manual)",
            delta=f"{ov_temp - current_reading['Inlet_Temperature']:.1f} °C" if override_active else None
        )
        m_col3.metric(
            label="Wash Water Rate",
            value=f"{current_reading['Wash_Water_Rate']:.2f} %" if not override_active else f"{ov_water:.2f} % (Manual)",
            delta=f"{ov_water - current_reading['Wash_Water_Rate']:.2f} %" if override_active else None
        )
        m_col4.metric(
            label="Inlet Salt",
            value=f"{current_reading['Inlet_Salt_PTB']:.1f} PTB"
        )
        m_col5.metric(
            label="Emulsion Layer",
            value=f"{current_reading['Emulsion_Layer_Thickness']:.2f} mm"
        )
        
        if grid_volts == 0.0:
            m_col6.metric(
                label="Grid Voltage",
                value=f"{grid_volts:.1f} kV",
                delta="TRIPPED",
                delta_color="inverse"
            )
        else:
            m_col6.metric(
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
        st.line_chart(plot_df[['Inlet_Temperature', 'Emulsion_Layer_Thickness', 'Grid_Voltage']])

        # Handle Stream Loop Animation
        if st.session_state.sim_active:
            import time
            time.sleep(0.3)
            if st.session_state.sim_index < len(df_sim) - 1:
                st.session_state.sim_index += 1
            else:
                st.session_state.sim_active = False
            st.rerun()
