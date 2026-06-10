import streamlit as st
import requests
import os

# 1. Configuration & Networking
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# 3. Build the UI Layout
st.set_page_config(page_title="IOCL Panipat: Desalter Crude Risk Profiling", layout="wide")

st.title("IOCL Panipat: Desalter Crude Risk Profiling")
st.markdown("### Synthetic Crude Data Risk Analyzer")
st.markdown("Enter the crude parameters in the sidebar to predict the processing risk in the Desalter unit.")

st.sidebar.header("Incoming Crude Parameters")

crude_blend = st.sidebar.selectbox("Crude Blend", ['Basrah Heavy', 'Arab Light', 'Ural', 'Bonny Light'])
api_gravity = st.sidebar.slider("API Gravity (°API)", min_value=20.0, max_value=45.0, value=32.0, step=0.1)
inlet_bsw = st.sidebar.slider("Inlet BSW (%)", min_value=0.1, max_value=2.5, value=1.0, step=0.01)
inlet_salt_ptb = st.sidebar.slider("Inlet Salt (PTB)", min_value=10.0, max_value=60.0, value=30.0, step=0.1)

# 4. API Integration
if st.sidebar.button("Run Risk Analysis", type="primary", use_container_width=True):
    payload = {
        "Crude_Blend": crude_blend,
        "API_Gravity": api_gravity,
        "Inlet_BSW": inlet_bsw,
        "Inlet_Salt_PTB": inlet_salt_ptb
    }
    
    with st.spinner("Connecting to FastAPI backend..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            
            result = response.json()
            risk_class = result.get("predicted_risk", "Unknown")
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # 5. Visual Output
            if risk_class == "High":
                st.error("🚨 **HIGH RISK**: This crude batch is predicted to be difficult to process.")
            elif risk_class == "Medium":
                st.warning("⚠️ **MEDIUM RISK**: Moderate processing difficulty expected.")
            elif risk_class == "Low":
                st.success("✅ **LOW RISK**: This crude batch should process easily.")
            else:
                st.info(f"**Predicted Risk:** {risk_class}")
                
        except requests.exceptions.ConnectionError:
            st.error(f"❌ **Connection Error:** Could not reach the FastAPI backend at {API_URL}. Make sure it is running.")
        except Exception as e:
            st.error(f"❌ **An error occurred:** {e}")
