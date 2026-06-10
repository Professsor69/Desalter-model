import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def main():
    print("Loading crude profile data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve the path to the Stage 1 crude profile data
    csv_path = os.path.join(os.path.dirname(script_dir), 'crude_oil_risk_prediction', 'crude_profile_data.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find crude_profile_data.csv at {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    # Define features and target columns
    feature_cols = ['API_Gravity', 'Inlet_BSW', 'Inlet_Salt_PTB']
    target_col = 'Target_Risk_Class'
    
    X = df[feature_cols]
    y_raw = df[target_col]
    
    # Encode categorical target variables to integers (Low=0, Medium=1, High=2, etc.)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Print the mapping representation
    class_mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    print("Category to integer mapping:", class_mapping)
    
    # Train/Test Split (80/20) with random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Initialize XGBClassifier optimized for multi-core CPU
    print("\nTraining XGBoost Classifier (Crude Profiler)...")
    clf = XGBClassifier(
        tree_method='hist',
        n_jobs=-1,
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print("Model training complete.")
    
    # Run evaluation checks
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save the pipeline dictionary containing the model and the encoder
    model_path = os.path.join(script_dir, 'crude_profiler.pkl')
    print(f"\nSaving model and encoder to: {model_path}")
    save_dict = {
        'model': clf,
        'label_encoder': le
    }
    with open(model_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    # Verification test
    print("\nVerifying model serialization...")
    with open(model_path, 'rb') as f:
        loaded = pickle.load(f)
    loaded_clf = loaded['model']
    loaded_le = loaded['label_encoder']
    
    test_pred_encoded = loaded_clf.predict(X_test.iloc[[0]])
    test_pred_label = loaded_le.inverse_transform(test_pred_encoded)[0]
    print(f"Verification check successful! Predicted class: '{test_pred_label}' vs Actual class: '{le.inverse_transform([y_test[0]])[0]}'")

if __name__ == '__main__':
    main()
