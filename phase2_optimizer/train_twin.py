import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def main():
    print("Loading historian data from SQLite database...")
    # Determine paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'historian_data.sqlite')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}. Please run generate_historian.py first.")
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM historian", conn)
    conn.close()
    
    print(f"Loaded {len(df)} rows.")
    
    # Define features and target
    feature_cols = ['API_Gravity', 'Inlet_BSW', 'Temperature_C', 'Wash_Water_Percent']
    target_col = 'Emulsion_Thickness_mm'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split sizes - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Instantiate and train XGBoost Regressor optimized for CPU
    print("\nTraining XGBoost Regressor (Digital Twin)...")
    # tree_method='hist' and n_jobs=-1 are used for CPU multi-core performance
    twin_model = XGBRegressor(
        tree_method='hist',
        n_jobs=-1,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    
    twin_model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Evaluate performance
    print("\nEvaluating model performance on test set...")
    y_pred = twin_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.4f} mm")
    print(f"Test R^2: {r2:.4f}")
    
    # Save model as pickle
    model_path = os.path.join(script_dir, 'desalter_twin.pkl')
    print(f"\nSaving model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(twin_model, f)
    
    # Verify that we can load and run inference with the saved model
    print("Verifying saved model file...")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    test_infer = loaded_model.predict(X_test.iloc[[0]])
    print(f"Verification inference successful! Predicted: {test_infer[0]:.4f} vs Actual: {y_test.iloc[0]:.4f}")

if __name__ == '__main__':
    main()
