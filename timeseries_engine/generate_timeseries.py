import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
DAYS = 30
MINUTES_PER_DAY = 24 * 60
TOTAL_MINUTES = DAYS * MINUTES_PER_DAY  # 43,200 minutes

# 1. Base timestamps starting from 2026-06-01 00:00:00
start_time = datetime(2026, 6, 1, 0, 0, 0)
timestamps = [start_time + timedelta(minutes=i) for i in range(TOTAL_MINUTES)]

# 2. Setup sensors with mean-reverting Ornstein-Uhlenbeck processes
# x[t] = x[t-1] + theta * (mu - x[t-1]) + noise
def generate_mean_reverting_walk(n_steps, init_val, mu, theta, sigma, val_min, val_max):
    values = np.empty(n_steps)
    values[0] = init_val
    for t in range(1, n_steps):
        drift = theta * (mu - values[t-1])
        noise = np.random.normal(0, sigma)
        values[t] = np.clip(values[t-1] + drift + noise, val_min, val_max)
    return values

print("Generating normal operating condition sensor random walks...")
API_Gravity = generate_mean_reverting_walk(TOTAL_MINUTES, 32.0, 31.0, 0.001, 0.03, 20.0, 40.0)
Inlet_Temperature = generate_mean_reverting_walk(TOTAL_MINUTES, 135.0, 135.0, 0.005, 0.15, 115.0, 150.0)
Wash_Water_Rate = generate_mean_reverting_walk(TOTAL_MINUTES, 5.0, 5.0, 0.005, 0.05, 1.0, 10.0)
Inlet_Salt_PTB = generate_mean_reverting_walk(TOTAL_MINUTES, 30.0, 30.0, 0.003, 0.15, 10.0, 60.0)
Emulsion_Layer_Thickness = generate_mean_reverting_walk(TOTAL_MINUTES, 10.0, 10.0, 0.005, 0.1, 5.0, 15.0)
Grid_Voltage = np.random.normal(15.0, 0.05, TOTAL_MINUTES)  # normal voltage is ~15 kV

# 3. Inject failures
# Determine number of failures (8 to 12)
num_failures = np.random.randint(8, 13)
print(f"Injecting {num_failures} failure events...")

# Choose failure times (separated by at least 2500 minutes to avoid overlaps)
failure_times = []
min_spacing = 2500
attempts = 0
while len(failure_times) < num_failures and attempts < 1000:
    candidate = np.random.randint(5000, TOTAL_MINUTES - 120)
    # Check spacing
    if all(abs(candidate - f) >= min_spacing for f in failure_times):
        failure_times.append(candidate)
    attempts += 1

failure_times.sort()
print(f"Failure trip indices: {failure_times}")

# Initialize warnings
Trip_Warning_60m = np.zeros(TOTAL_MINUTES, dtype=int)

# For each failure event:
# - Precursor begins 120 minutes before the trip (T - 120)
# - Inlet_Temperature begins to drop slowly
# - API_Gravity drifts down
# - Grid_Voltage decays from 15.0 kV to 0.0 kV over the 90 minutes preceding the trip
# - Grid_Voltage hits exactly 0.0 kV at T
# - Grid_Voltage stays 0.0 kV for 30 minutes (repair phase)
for T in failure_times:
    # Set the warning label for 60 minutes preceding the trip
    Trip_Warning_60m[T-60:T] = 1
    
    # Precursor phase (T-120 to T)
    precursor_len = 120
    for i in range(precursor_len):
        t_idx = T - precursor_len + i
        fraction = i / precursor_len  # 0 to 1
        
        # Temp drops by up to 25°C at the trip point
        Inlet_Temperature[t_idx] -= 20.0 * fraction + np.random.normal(0, 0.1)
        
        # API Gravity shifts down towards heavier, more emulsion-prone values
        API_Gravity[t_idx] -= 6.0 * fraction + np.random.normal(0, 0.02)
        
        # Salt PTB fluctuates and spikes up due to poor separation
        Inlet_Salt_PTB[t_idx] += 15.0 * fraction + np.random.normal(0, 0.2)
        
        # Wash water rate fluctuates wildly
        Wash_Water_Rate[t_idx] += np.random.normal(0, 0.2)
        
        # Emulsion Layer Thickness grows exponentially towards the shorting point (crosses 25.0mm around T)
        Emulsion_Layer_Thickness[t_idx] += 16.0 * (fraction ** 2) + np.random.normal(0, 0.1)
        
    # Grid Voltage decay starts at T - 90 down to T
    decay_start = T - 90
    for i in range(90):
        t_idx = decay_start + i
        fraction = i / 90.0
        # Smooth decay with some electrical noise
        Grid_Voltage[t_idx] = 15.0 * (1.0 - fraction) + np.random.normal(0, 0.1)
        Grid_Voltage[t_idx] = max(0.0, Grid_Voltage[t_idx])
        
    # At index T, it trips to exactly 0.0 kV
    Grid_Voltage[T] = 0.0
    
    # Stay down at 0.0 kV for 30 minutes (T to T+30)
    recovery_len = 30
    for i in range(recovery_len):
        t_idx = T + i
        if t_idx < TOTAL_MINUTES:
            Grid_Voltage[t_idx] = 0.0
            # Sensors show static / shut down values
            Inlet_Temperature[t_idx] = max(115.0, Inlet_Temperature[T-1] - (i * 0.5))
            API_Gravity[t_idx] = API_Gravity[T-1]
            Wash_Water_Rate[t_idx] = 0.0
            Inlet_Salt_PTB[t_idx] = Inlet_Salt_PTB[T-1]
            Emulsion_Layer_Thickness[t_idx] = Emulsion_Layer_Thickness[T-1]

# Post-processing clips
API_Gravity = np.clip(API_Gravity, 20.0, 40.0).round(2)
Inlet_Temperature = Inlet_Temperature.round(2)
Wash_Water_Rate = np.clip(Wash_Water_Rate, 0.0, 10.0).round(2)
Inlet_Salt_PTB = np.clip(Inlet_Salt_PTB, 10.0, 60.0).round(2)
Emulsion_Layer_Thickness = np.clip(Emulsion_Layer_Thickness, 0.0, 40.0).round(2)
Grid_Voltage = np.clip(Grid_Voltage, 0.0, 16.0).round(2)

# Create DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'API_Gravity': API_Gravity,
    'Inlet_Temperature': Inlet_Temperature,
    'Wash_Water_Rate': Wash_Water_Rate,
    'Inlet_Salt_PTB': Inlet_Salt_PTB,
    'Emulsion_Layer_Thickness': Emulsion_Layer_Thickness,
    'Grid_Voltage': Grid_Voltage,
    'Trip_Warning_60m': Trip_Warning_60m
})

# Save to CSV
csv_path = 'timeseries_engine/desalter_timeseries.csv'
df.to_csv(csv_path, index=False)
print(f"Data generation complete. Saved to: {csv_path}")
print(f"Dataset shape: {df.shape}")
print(f"Warning label distribution:\n{df['Trip_Warning_60m'].value_counts(normalize=True)}")
