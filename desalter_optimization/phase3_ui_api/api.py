import os
import sys
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Establish path to the repository root so we can import from phase2_optimizer
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import the twin loading and optimize logic
try:
    from phase2_optimizer.prescriptive_optimizer import optimize_setpoints, load_digital_twin
except ImportError as e:
    # Fallback to local import if run under a different directory structure
    sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..')))
    from phase2_optimizer.prescriptive_optimizer import optimize_setpoints, load_digital_twin

app = FastAPI(
    title="IOCL Desalter Prescriptive Optimizer API",
    description="Provides optimal Temperature and Wash Water setpoint recommendations and early grid trip warnings.",
    version="1.1.0"
)

# Configure CORS Middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
timeseries_model_dict = None
risk_model_dict = None

@app.on_event("startup")
def startup_event():
    global model, timeseries_model_dict
    # Load digital twin optimizer model
    try:
        print("Loading digital twin model during API startup...")
        model = load_digital_twin()
        print("Digital twin model loaded successfully.")
    except Exception as e:
        print(f"Error loading digital twin model during startup: {e}")
        
    # Load timeseries early warning model
    try:
        timeseries_model_path = os.path.join(root_dir, "timeseries_engine", "timeseries_warning_model.pkl")
        if not os.path.exists(timeseries_model_path):
            timeseries_model_path = os.path.join(os.path.dirname(root_dir), "timeseries_engine", "timeseries_warning_model.pkl")
        timeseries_model_dict = joblib.load(timeseries_model_path)
        print("Timeseries early warning model loaded successfully.")
    except Exception as e:
        print(f"Error loading timeseries early warning model: {e}")

    # Load crude risk profiler model
    global risk_model_dict
    try:
        risk_model_path = os.path.join(root_dir, "phase1_profiler", "crude_profiler.pkl")
        if not os.path.exists(risk_model_path):
            risk_model_path = os.path.join(os.path.dirname(root_dir), "phase1_profiler", "crude_profiler.pkl")
        import pickle
        with open(risk_model_path, 'rb') as f:
            risk_model_dict = pickle.load(f)
        print("Crude risk profiler model loaded successfully.")
    except Exception as e:
        print(f"Error loading crude risk profiler model: {e}")

class CrudeConditions(BaseModel):
    API_Gravity: float = Field(..., ge=20.0, le=45.0, description="API Gravity of incoming crude batch (20.0 to 45.0)")
    Inlet_BSW: float = Field(..., ge=0.1, le=2.5, description="Inlet BSW % of incoming crude batch (0.1 to 2.5)")
    Inlet_Salt_PTB: float = Field(..., ge=10.0, le=60.0, description="Inlet Salt PTB of incoming crude batch (10.0 to 60.0)")

    class Config:
        json_schema_extra = {
            "example": {
                "API_Gravity": 25.0,
                "Inlet_BSW": 1.8,
                "Inlet_Salt_PTB": 30.0
            }
        }

class PredictConditions(BaseModel):
    API_Gravity: float = Field(..., ge=20.0, le=45.0, description="API Gravity of incoming crude batch (20.0 to 45.0)")
    Inlet_BSW: float = Field(..., ge=0.1, le=2.5, description="Inlet BSW % of incoming crude batch (0.1 to 2.5)")
    Inlet_Salt_PTB: float = Field(..., ge=10.0, le=60.0, description="Inlet Salt PTB of incoming crude batch (10.0 to 60.0)")
    Temperature_C: float = Field(..., ge=110.0, le=150.0, description="Simulated Operator Temperature setpoint (110.0 to 150.0)")
    Wash_Water_Percent: float = Field(..., ge=2.0, le=8.0, description="Simulated Operator Wash Water percentage setpoint (2.0 to 8.0)")

    class Config:
        json_schema_extra = {
            "example": {
                "API_Gravity": 25.0,
                "Inlet_BSW": 1.8,
                "Inlet_Salt_PTB": 30.0,
                "Temperature_C": 130.0,
                "Wash_Water_Percent": 5.0
            }
        }

# schemas for time-series early warning monitor
class TimeSeriesReading(BaseModel):
    API_Gravity: float = Field(..., description="API Gravity")
    Inlet_Temperature: float = Field(..., description="Inlet Temperature in °C")
    Wash_Water_Rate: float = Field(..., description="Wash Water flow rate in %")
    Inlet_Salt_PTB: float = Field(..., description="Inlet Salt PTB")
    Emulsion_Layer_Thickness: float = Field(..., description="Emulsion Layer Thickness in mm")

class EarlyWarningInput(BaseModel):
    readings: list[TimeSeriesReading] = Field(..., description="List of last 60 minutes of SCADA readings (from oldest to newest)")

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "IOCL Desalter API is operational"
    }

@app.post("/optimize")
def get_optimized_setpoints(conditions: CrudeConditions):
    global model, risk_model_dict
    if model is None:
        try:
            model = load_digital_twin()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
            
    try:
        result = optimize_setpoints(model, conditions.API_Gravity, conditions.Inlet_BSW, conditions.Inlet_Salt_PTB)
        
        # Predict risk class
        predicted_risk = "Unknown"
        if risk_model_dict is not None:
            clf = risk_model_dict['model']
            le = risk_model_dict['label_encoder']
            input_df = pd.DataFrame([{
                'API_Gravity': conditions.API_Gravity,
                'Inlet_BSW': conditions.Inlet_BSW,
                'Inlet_Salt_PTB': conditions.Inlet_Salt_PTB
            }])
            y_pred = clf.predict(input_df)
            predicted_risk = le.inverse_transform(y_pred)[0]
            
        result["predicted_risk"] = predicted_risk
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

def log_anomaly_to_db(api_gravity, bsw, salt, ai_temp, ai_water, ai_thickness, op_temp, op_water, op_thickness):
    db_path = os.getenv("DATABASE_PATH", os.path.join(current_dir, "optimizer_anomalies.sqlite"))
    try:
        # Ensure parent directory of database exists if customized
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                api_gravity REAL NOT NULL,
                inlet_bsw REAL NOT NULL,
                inlet_salt_ptb REAL NOT NULL,
                ai_temp REAL NOT NULL,
                ai_water REAL NOT NULL,
                ai_thickness REAL NOT NULL,
                operator_temp REAL NOT NULL,
                operator_water REAL NOT NULL,
                operator_thickness REAL NOT NULL
            )
        """)
        cursor.execute("""
            INSERT INTO anomalies (
                timestamp, api_gravity, inlet_bsw, inlet_salt_ptb, 
                ai_temp, ai_water, ai_thickness, 
                operator_temp, operator_water, operator_thickness
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            api_gravity,
            bsw,
            salt,
            ai_temp,
            ai_water,
            ai_thickness,
            op_temp,
            op_water,
            op_thickness
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging anomaly to SQLite: {e}", file=sys.stderr)

@app.post("/predict")
def predict_emulsion_thickness(conditions: PredictConditions):
    global model
    if model is None:
        try:
            model = load_digital_twin()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
            
    try:
        # Evaluate operator manual setpoints
        df_single = pd.DataFrame([{
            'API_Gravity': conditions.API_Gravity,
            'Inlet_BSW': conditions.Inlet_BSW,
            'Inlet_Salt_PTB': conditions.Inlet_Salt_PTB,
            'Temperature_C': conditions.Temperature_C,
            'Wash_Water_Percent': conditions.Wash_Water_Percent
        }])
        op_thickness = float(model.predict(df_single)[0])
        
        # Compare with AI Optimized recommendations
        ai_res = optimize_setpoints(model, conditions.API_Gravity, conditions.Inlet_BSW, conditions.Inlet_Salt_PTB)
        ai_opt = ai_res.get("optimal_setpoints", {})
        ai_temp = ai_opt.get("Temperature_C", 0.0)
        ai_water = ai_opt.get("Wash_Water_Percent", 0.0)
        ai_thickness = ai_res.get("predicted_emulsion_thickness_mm", 0.0)
        
        # If Operator outperforms AI, log it silently (Shadow Logging)
        if op_thickness < ai_thickness:
            log_anomaly_to_db(
                conditions.API_Gravity,
                conditions.Inlet_BSW,
                conditions.Inlet_Salt_PTB,
                ai_temp,
                ai_water,
                ai_thickness,
                conditions.Temperature_C,
                conditions.Wash_Water_Percent,
                op_thickness
            )
            
        return {
            "status": "success",
            "predicted_emulsion_thickness_mm": round(op_thickness, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-early-warning")
def predict_early_warning(payload: EarlyWarningInput):
    global timeseries_model_dict
    if timeseries_model_dict is None:
        raise HTTPException(status_code=500, detail="Timeseries early warning model is not loaded.")
        
    if len(payload.readings) < 61:
        raise HTTPException(status_code=400, detail=f"Need at least 61 readings (60 minutes lag + current). Received: {len(payload.readings)}")

    try:
        # Convert list of Pydantic models to a pandas DataFrame
        data_list = [reading.model_dump() if hasattr(reading, 'model_dump') else reading.dict() for reading in payload.readings]
        df = pd.DataFrame(data_list)
        
        # Calculate identical time-series rolling features
        df['Temp_roll_mean_15'] = df['Inlet_Temperature'].rolling(window=15).mean()
        df['Temp_roll_mean_60'] = df['Inlet_Temperature'].rolling(window=60).mean()
        df['Water_roll_mean_15'] = df['Wash_Water_Rate'].rolling(window=15).mean()
        df['Water_roll_mean_60'] = df['Wash_Water_Rate'].rolling(window=60).mean()
        df['Emulsion_roll_mean_15'] = df['Emulsion_Layer_Thickness'].rolling(window=15).mean()
        df['Emulsion_roll_mean_60'] = df['Emulsion_Layer_Thickness'].rolling(window=60).mean()

        df['Temp_slope_15'] = df['Inlet_Temperature'] - df['Inlet_Temperature'].shift(15)
        df['Temp_slope_60'] = df['Inlet_Temperature'] - df['Inlet_Temperature'].shift(60)
        df['Water_slope_15'] = df['Wash_Water_Rate'] - df['Wash_Water_Rate'].shift(15)
        df['Water_slope_60'] = df['Wash_Water_Rate'] - df['Wash_Water_Rate'].shift(60)
        df['Emulsion_slope_15'] = df['Emulsion_Layer_Thickness'] - df['Emulsion_Layer_Thickness'].shift(15)
        df['Emulsion_slope_60'] = df['Emulsion_Layer_Thickness'] - df['Emulsion_Layer_Thickness'].shift(60)

        # Get latest row features and format for prediction
        features_list = timeseries_model_dict['features']
        latest_row = df.iloc[-1][features_list].to_frame().T
        
        # Predict warning status
        model = timeseries_model_dict['model']
        prediction = int(model.predict(latest_row)[0])
        
        warning_status = "Imminent Trip" if prediction == 1 else "Safe"
        return {"warning_status": warning_status}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
