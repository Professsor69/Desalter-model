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
    
    # Inlet Salt: Float (10.0 to 60.0 PTB) - Inlet salt load (Pounds per Thousand Barrels)
    inlet_salt_ptb = np.random.uniform(10.0, 60.0, size=num_rows)
    
    # Generate controllable operational variables
    # Temperature: Float (110 to 150 °C)
    temperature_c = np.random.uniform(110.0, 150.0, size=num_rows)
    
    # Wash Water Percent: Float (2.0 to 8.0 %)
    wash_water_percent = np.random.uniform(2.0, 8.0, size=num_rows)
    
    # Normalize features to [0, 1] for physics modeling
    api_norm = (api_gravity - 20.0) / (45.0 - 20.0)
    bsw_norm = (inlet_bsw - 0.1) / (2.5 - 0.1)
    salt_norm = (inlet_salt_ptb - 10.0) / (60.0 - 10.0)
    
    # Define parabolic optimum targets depending on crude properties:
    # 1. Optimal Temperature:
    #    Lighter crude (higher api_norm) requires slightly less temperature.
    #    Higher BSW (higher bsw_norm) requires slightly less temperature (to prevent foaming).
    #    Optimum ranges between ~130 °C and ~145 °C.
    opt_temp = 145.0 - 5.0 * api_norm - 10.0 * bsw_norm
    
    # 2. Optimal Wash Water:
    #    Higher BSW requires more wash water.
    #    Heavier crude (lower api_norm) requires more wash water.
    #    Optimum ranges between ~4.0 % and ~7.0 %.
    opt_ww = 4.0 + 2.0 * bsw_norm + 1.0 * (1.0 - api_norm)
    
    # Emulsion thickness increases quadratically as operational variables deviate from optimal targets:
    temp_dev = 15.0 * ((temperature_c - opt_temp) / 20.0) ** 2
    ww_dev = 10.0 * ((wash_water_percent - opt_ww) / 3.0) ** 2
    
    # Higher salt load increases emulsion stability/thickness
    salt_effect = 8.0 * salt_norm
    
    # Base emulsion thickness logic incorporating gravity, moisture, salt and deviations
    emulsion_base = (
        10.0 +
        temp_dev +
        ww_dev +
        salt_effect +
        25.0 * (1.0 - api_norm) * bsw_norm
    )
    
    # Add Gaussian noise (sigma = 2.0)
    noise = np.random.normal(0.0, 2.0, size=num_rows)
    emulsion_thickness_mm = emulsion_base + noise
    
    # Clip to physically realistic minimum thickness (2.0 mm)
    emulsion_thickness_mm = np.clip(emulsion_thickness_mm, 2.0, None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'API_Gravity': api_gravity,
        'Inlet_BSW': inlet_bsw,
        'Inlet_Salt_PTB': inlet_salt_ptb,
        'Temperature_C': temperature_c,
        'Wash_Water_Percent': wash_water_percent,
        'Emulsion_Thickness_mm': emulsion_thickness_mm
    })
    
    return df

def main():
    print("Generating updated synthetic historian dataset...")
    df = generate_data()
    
    print("\nDataset Shape:", df.shape)
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Determine save path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'historian_data.sqlite')
    
    print(f"\nSaving to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    df.to_sql('historian', conn, if_exists='replace', index=False)
    conn.close()
    print("Database update complete!")

if __name__ == '__main__':
    main()
