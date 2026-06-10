import os
import sys
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
    description="Provides optimal Temperature and Wash Water setpoint recommendations based on crude profile inputs.",
    version="1.0.0"
)

# Configure CORS Middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

@app.on_event("startup")
def startup_event():
    global model
    try:
        print("Loading digital twin model during API startup...")
        model = load_digital_twin()
        print("Digital twin model loaded successfully.")
    except Exception as e:
        print(f"Error loading digital twin model during startup: {e}")

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

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "IOCL Desalter API is operational"
    }

@app.post("/optimize")
def get_optimized_setpoints(conditions: CrudeConditions):
    global model
    if model is None:
        try:
            model = load_digital_twin()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
            
    try:
        result = optimize_setpoints(model, conditions.API_Gravity, conditions.Inlet_BSW, conditions.Inlet_Salt_PTB)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
