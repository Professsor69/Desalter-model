import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# 1. Load data
csv_path = 'timeseries_engine/desalter_timeseries.csv'
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# 2. Feature Engineering
print("Engineering rolling features and slopes...")
# Rolling averages
df['Temp_roll_mean_15'] = df['Inlet_Temperature'].rolling(window=15).mean()
df['Temp_roll_mean_60'] = df['Inlet_Temperature'].rolling(window=60).mean()
df['Water_roll_mean_15'] = df['Wash_Water_Rate'].rolling(window=15).mean()
df['Water_roll_mean_60'] = df['Wash_Water_Rate'].rolling(window=60).mean()
df['Emulsion_roll_mean_15'] = df['Emulsion_Layer_Thickness'].rolling(window=15).mean()
df['Emulsion_roll_mean_60'] = df['Emulsion_Layer_Thickness'].rolling(window=60).mean()

# Slopes (difference between current value and value N steps ago)
df['Temp_slope_15'] = df['Inlet_Temperature'] - df['Inlet_Temperature'].shift(15)
df['Temp_slope_60'] = df['Inlet_Temperature'] - df['Inlet_Temperature'].shift(60)
df['Water_slope_15'] = df['Wash_Water_Rate'] - df['Wash_Water_Rate'].shift(15)
df['Water_slope_60'] = df['Wash_Water_Rate'] - df['Wash_Water_Rate'].shift(60)
df['Emulsion_slope_15'] = df['Emulsion_Layer_Thickness'] - df['Emulsion_Layer_Thickness'].shift(15)
df['Emulsion_slope_60'] = df['Emulsion_Layer_Thickness'] - df['Emulsion_Layer_Thickness'].shift(60)

# Drop rows with NaN values resulting from rolling/shifting
df_clean = df.dropna().reset_index(drop=True)

# Features and target
# Note: Grid_Voltage is excluded to prevent data leakage (since it drops during failures)
feature_cols = [
    'API_Gravity', 'Inlet_Temperature', 'Wash_Water_Rate', 'Inlet_Salt_PTB', 'Emulsion_Layer_Thickness',
    'Temp_roll_mean_15', 'Temp_roll_mean_60',
    'Temp_slope_15', 'Temp_slope_60',
    'Water_roll_mean_15', 'Water_roll_mean_60',
    'Water_slope_15', 'Water_slope_60',
    'Emulsion_roll_mean_15', 'Emulsion_roll_mean_60',
    'Emulsion_slope_15', 'Emulsion_slope_60'
]
target_col = 'Trip_Warning_60m'

X = df_clean[feature_cols]
y = df_clean[target_col]

# 3. Chronological Split (No random shuffling to avoid data leakage)
split_idx = int(len(df_clean) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_val.shape[0]} rows")
print(f"Train class balance: {np.bincount(y_train)}")
print(f"Test class balance: {np.bincount(y_val)}")

# 4. Model Training
print("\nTraining XGBoost Classifier...")
# Scale positive weight to handle class imbalance
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# 6. Serialization
model_path = 'timeseries_engine/timeseries_warning_model.pkl'
model_dict = {
    'model': model,
    'features': feature_cols
}
joblib.dump(model_dict, model_path)
print(f"\nModel and feature configuration serialized successfully to: {model_path}")
