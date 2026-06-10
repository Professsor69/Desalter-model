import os
import pickle
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Define global operational bounds
TEMP_MIN, TEMP_MAX = 110.0, 150.0
WW_MIN, WW_MAX = 2.0, 8.0

def load_digital_twin():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'desalter_twin.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Digital twin model not found at {model_path}. Please run train_twin.py first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def optimize_setpoints(model, api_gravity, inlet_bsw):
    """
    Finds the optimal Temperature_C and Wash_Water_Percent for the given API_Gravity and Inlet_BSW.
    Uses a hybrid approach:
    1. Grid search over the operating space to find a good global starting point and avoid local minima.
    2. SciPy minimize (Powell method) initialized at the grid winner for fine local optimization.
    """
    # Step 1: Grid Search
    # Create a dense grid of 100 points for temperature and 60 points for wash water percent
    temps = np.linspace(TEMP_MIN, TEMP_MAX, 100)
    wws = np.linspace(WW_MIN, WW_MAX, 60)
    
    # Create coordinate matrices
    temp_grid, ww_grid = np.meshgrid(temps, wws)
    temp_flat = temp_grid.flatten()
    ww_flat = ww_grid.flatten()
    
    # Create evaluation DataFrame for batch inference (XGBoost handles this instantly)
    eval_df = pd.DataFrame({
        'API_Gravity': [api_gravity] * len(temp_flat),
        'Inlet_BSW': [inlet_bsw] * len(temp_flat),
        'Temperature_C': temp_flat,
        'Wash_Water_Percent': ww_flat
    })
    
    # Run batch prediction
    predictions = model.predict(eval_df)
    
    # Find grid search winner
    best_idx = np.argmin(predictions)
    grid_best_temp = temp_flat[best_idx]
    grid_best_ww = ww_flat[best_idx]
    grid_min_emulsion = predictions[best_idx]
    
    # Step 2: Local refinement using Powell optimizer (gradient-free, bounds-aware)
    def objective_fn(x):
        # x[0] is Temperature_C, x[1] is Wash_Water_Percent
        temp_val = np.clip(x[0], TEMP_MIN, TEMP_MAX)
        ww_val = np.clip(x[1], WW_MIN, WW_MAX)
        
        df_single = pd.DataFrame([{
            'API_Gravity': api_gravity,
            'Inlet_BSW': inlet_bsw,
            'Temperature_C': temp_val,
            'Wash_Water_Percent': ww_val
        }])
        return float(model.predict(df_single)[0])
    
    # Bounds for the optimizer
    bounds = [(TEMP_MIN, TEMP_MAX), (WW_MIN, WW_MAX)]
    
    # Run optimization initialized at grid search winner
    res = minimize(
        objective_fn,
        x0=[grid_best_temp, grid_best_ww],
        method='Powell',
        bounds=bounds,
        options={'xtol': 1e-4, 'ftol': 1e-4}
    )
    
    # Choose the best result between grid search and optimizer refinement
    if res.success and res.fun < grid_min_emulsion:
        opt_temp = np.clip(res.x[0], TEMP_MIN, TEMP_MAX)
        opt_ww = np.clip(res.x[1], WW_MIN, WW_MAX)
        min_emulsion = res.fun
        optimization_method = "Powell Local Refinement"
    else:
        opt_temp = grid_best_temp
        opt_ww = grid_best_ww
        min_emulsion = grid_min_emulsion
        optimization_method = "Grid Search Global Minimum"
        
    return {
        "status": "success",
        "crude_conditions": {
            "API_Gravity": float(api_gravity),
            "Inlet_BSW": float(inlet_bsw)
        },
        "optimal_setpoints": {
            "Temperature_C": round(float(opt_temp), 2),
            "Wash_Water_Percent": round(float(opt_ww), 2)
        },
        "predicted_emulsion_thickness_mm": round(float(min_emulsion), 4),
        "optimization_metadata": {
            "method_used": optimization_method,
            "grid_search_raw_min": round(float(grid_min_emulsion), 4),
            "powell_success": bool(res.success),
            "powell_iterations": int(res.nit) if hasattr(res, 'nit') else 0
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Desalter Prescriptive Optimizer")
    parser.add_argument("--api", type=float, default=25.0, help="API Gravity of incoming crude batch (20.0 to 45.0)")
    parser.add_argument("--bsw", type=float, default=1.8, help="Inlet BSW % of incoming crude batch (0.1 to 2.5)")
    args = parser.parse_args()
    
    # Input validation
    if not (20.0 <= args.api <= 45.0):
        print(f"Warning: API Gravity {args.api} is outside the normal bounds [20.0, 45.0]")
    if not (0.1 <= args.bsw <= 2.5):
        print(f"Warning: Inlet BSW {args.bsw} is outside the normal bounds [0.1, 2.5]")
        
    # Load model and run optimization
    try:
        model = load_digital_twin()
        result = optimize_setpoints(model, args.api, args.bsw)
        # Output JSON payload
        print(json.dumps(result, indent=2))
    except Exception as e:
        error_payload = {
            "status": "error",
            "message": str(e)
        }
        print(json.dumps(error_payload, indent=2))

if __name__ == '__main__':
    main()
