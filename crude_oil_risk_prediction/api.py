from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Desalter Risk API")

# Define the custom function so joblib can unpickle the pipeline
def add_emulsion_risk(X):
    X_new = X.copy()
    X_new['Emulsion_Risk_Factor'] = (X_new['Inlet_BSW'] * X_new['Inlet_Salt_PTB']) / X_new['API_Gravity']
    return X_new

# 4. Load the Model globally
print("Loading model pipeline...")
try:
    model_dict = joblib.load("desalter_risk_model.pkl")
    pipeline = model_dict['pipeline']
    label_encoder = model_dict['label_encoder']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Define the Schema
class CrudeBatchInput(BaseModel):
    Crude_Blend: str = Field(..., examples=["Basrah Heavy"])
    API_Gravity: float = Field(..., examples=[27.5])
    Inlet_BSW: float = Field(..., examples=[1.5])
    Inlet_Salt_PTB: float = Field(..., examples=[45.0])

# 5. Create Endpoints
@app.get("/")
def health_check():
    return {"status": "Desalter Risk API is live"}

@app.post("/predict")
def predict_risk(batch: CrudeBatchInput):
    # Convert input to DataFrame (using model_dump for Pydantic V2 compatibility)
    data = batch.model_dump() if hasattr(batch, 'model_dump') else batch.dict()
    df = pd.DataFrame([data])
    
    # Predict using the loaded pipeline
    y_pred_encoded = pipeline.predict(df)
    
    # Decode the prediction
    predicted_class = label_encoder.inverse_transform(y_pred_encoded)[0]
    
    return {"predicted_risk": predicted_class}
