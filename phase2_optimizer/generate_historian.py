import os
import sqlite3
import numpy as np
import pandas as pd

def generate_data(num_rows=10000, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate uncontrollable external variables
    # API Gravity: Float (20 to 45) - Heavier crudes have lower API gravity
    api_gravity = np.random.uniform(20.0, 45.0, size=num_rows)
    
    # Inlet BSW: Float (0.1 to 2.5) - Basic Sediment & Water percentage
    inlet_bsw = np.random.uniform(0.1, 2.5, size=num_rows)
    
    # Generate controllable operational variables
    # Temperature: Float (110 to 150 °C)
    temperature_c = np.random.uniform(110.0, 150.0, size=num_rows)
    
    # Wash Water Percent: Float (2.0 to 8.0 %)
    wash_water_percent = np.random.uniform(2.0, 8.0, size=num_rows)
    
    # Normalize features to [0, 1] for formulating physics-informed relations
    api_norm = (api_gravity - 20.0) / (45.0 - 20.0)
    bsw_norm = (inlet_bsw - 0.1) / (2.5 - 0.1)
    temp_norm = (temperature_c - 110.0) / (150.0 - 110.0)
    ww_norm = (wash_water_percent - 2.0) / (8.0 - 2.0)
    
    # Heuristics:
    # 1. Base emulsion thickness of 10.0 mm
    # 2. Low API + Low Temp = massive emulsion stability and thickness.
    #    Term: 50.0 * (1.0 - api_norm) * (1.0 - temp_norm)
    # 3. High Inlet BSW increases emulsion thickness, with greater effect on heavier crudes.
    #    Term: 30.0 * bsw_norm * (1.5 - api_norm)
    # 4. High Wash Water + High Temp helps resolve/minimize emulsion.
    #    Term representing lack of resolution: 20.0 * (1.0 - ww_norm) * (1.0 - temp_norm)
    emulsion_base = (
        10.0 +
        50.0 * (1.0 - api_norm) * (1.0 - temp_norm) +
        30.0 * bsw_norm * (1.5 - api_norm) +
        20.0 * (1.0 - ww_norm) * (1.0 - temp_norm)
    )
    
    # Add Gaussian noise to prevent perfect correlation (sigma = 2.0)
    noise = np.random.normal(0.0, 2.0, size=num_rows)
    emulsion_thickness_mm = emulsion_base + noise
    
    # Clip to ensure physical realism (minimum thickness is 2.0 mm)
    emulsion_thickness_mm = np.clip(emulsion_thickness_mm, 2.0, None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'API_Gravity': api_gravity,
        'Inlet_BSW': inlet_bsw,
        'Temperature_C': temperature_c,
        'Wash_Water_Percent': wash_water_percent,
        'Emulsion_Thickness_mm': emulsion_thickness_mm
    })
    
    return df

def main():
    print("Generating synthetic historian dataset...")
    df = generate_data()
    
    # Print statistics
    print("\nDataset Shape:", df.shape)
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Verify heuristics at extremes
    print("\nVerifying Heuristics:")
    # 1. Low API (< 25) + Low Temp (< 120)
    extreme_heavy = df[(df['API_Gravity'] < 25.0) & (df['Temperature_C'] < 120.0)]
    print(f"Low API & Low Temp - Mean Emulsion Thickness: {extreme_heavy['Emulsion_Thickness_mm'].mean():.2f} mm (expected to be high)")
    
    # 2. High Wash Water (> 7) + High Temp (> 140)
    extreme_optimal = df[(df['Wash_Water_Percent'] > 7.0) & (df['Temperature_C'] > 140.0)]
    print(f"High Wash Water & High Temp - Mean Emulsion Thickness: {extreme_optimal['Emulsion_Thickness_mm'].mean():.2f} mm (expected to be low)")
    
    # Determine save path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'historian_data.sqlite')
    
    print(f"\nSaving to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    df.to_sql('historian', conn, if_exists='replace', index=False)
    conn.close()
    print("Database save complete!")

if __name__ == '__main__':
    main()
