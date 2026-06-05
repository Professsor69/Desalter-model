import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
import xgboost as xgb
import optuna
import shap
import joblib

print("Starting Setup & Data Loading...")
# 3. Data Loading & Preprocessing
df = pd.read_csv("crude_profile_data.csv")

X = df.drop("Target_Risk_Class", axis=1)
y = df["Target_Risk_Class"]

# LabelEncode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocessor for One-Hot Encoding 'Crude_Blend'
categorical_cols = ["Crude_Blend"]
numeric_cols = ["API_Gravity", "Inlet_BSW", "Inlet_Salt_PTB"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Hyperparameter Tuning
print("\nStarting Hyperparameter Tuning with Optuna...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'objective': 'multi:softmax',
        'num_class': len(label_encoder.classes_),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Preprocess within trial to prevent data leakage, though applying globally to X_train is also fine.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_processed, y_train)
    
    y_pred = model.predict(X_test_processed)
    return f1_score(y_test, y_pred, average='macro')

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

print("\n--- BEST OPTUNA PARAMETERS ---")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

# 5. Training & Evaluation
print("\nTraining Final Model...")
best_params = study.best_params
best_params['objective'] = 'multi:softmax'
best_params['num_class'] = len(label_encoder.classes_)
best_params['random_state'] = 42
best_params['n_jobs'] = -1

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(**best_params))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n--- MODEL PERFORMANCE ---")
print(f"Final Model Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 6. Explainability
print("\nRunning SHAP Explainability...")
xgb_model = pipeline.named_steps['classifier']
X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)

cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_cols)
feature_names = numeric_cols + list(cat_features)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_processed)

# Extract global feature importance across all classes
if isinstance(shap_values, list):
    mean_abs_shap = np.zeros(X_test_processed.shape[1])
    for class_shap in shap_values:
        mean_abs_shap += np.abs(class_shap).mean(axis=0)
    mean_abs_shap /= len(shap_values)
elif len(shap_values.shape) == 3:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

importances = list(zip(feature_names, mean_abs_shap))
importances.sort(key=lambda x: x[1], reverse=True)

print("\n--- SHAP FEATURE IMPORTANCES ---")
for feat, imp in importances:
    print(f"{feat}: {imp:.4f}")

# 7. Serialization
print("\nSerializing Model pipeline...")
full_model = {
    'pipeline': pipeline,
    'label_encoder': label_encoder
}
joblib.dump(full_model, 'desalter_risk_model.pkl')
print("\nPipeline saved as 'desalter_risk_model.pkl'")
