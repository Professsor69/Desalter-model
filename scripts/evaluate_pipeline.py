import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

def format_confusion_matrix(cm, classes):
    """Formats a confusion matrix into a clean ASCII table."""
    header = f"{'Actual \\ Predicted':<20} " + " ".join([f"{cls:>10}" for cls in classes])
    lines = [header, "-" * len(header)]
    for idx, cls in enumerate(classes):
        row = f"{cls:<20} " + " ".join([f"{val:>10d}" for val in cm[idx]])
        lines.append(row)
    return "\n".join(lines)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    print("Initializing Pipeline Global Evaluation...")
    
    # ----------------------------------------------------
    # 1. Evaluate Crude Profiler Classifier
    # ----------------------------------------------------
    classifier_path = os.path.join(root_dir, 'phase1_profiler', 'crude_profiler.pkl')
    if not os.path.exists(classifier_path):
        classifier_path = os.path.join(root_dir, 'desalter_optimization', 'phase1_profiler', 'crude_profiler.pkl')
    csv_path = os.path.join(root_dir, 'crude_oil_risk_prediction', 'crude_profile_data.csv')
    
    print("\n[1/2] Evaluating Crude Profiler Classifier...")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier model not found at: {classifier_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
        
    with open(classifier_path, 'rb') as f:
        clf_dict = pickle.load(f)
    clf = clf_dict['model']
    le = clf_dict['label_encoder']
    
    df_clf = pd.read_csv(csv_path)
    X_clf = df_clf[['API_Gravity', 'Inlet_BSW', 'Inlet_Salt_PTB']]
    y_clf = le.transform(df_clf['Target_Risk_Class'])
    
    _, X_clf_test, _, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    y_clf_pred = clf.predict(X_clf_test)
    
    clf_acc = accuracy_score(y_clf_test, y_clf_pred)
    clf_f1 = f1_score(y_clf_test, y_clf_pred, average='macro')
    clf_cm = confusion_matrix(y_clf_test, y_clf_pred)
    
    # ----------------------------------------------------
    # 2. Evaluate Digital Twin Regressor
    # ----------------------------------------------------
    regressor_path = os.path.join(root_dir, 'phase2_optimizer', 'desalter_twin.pkl')
    if not os.path.exists(regressor_path):
        regressor_path = os.path.join(root_dir, 'desalter_optimization', 'phase2_optimizer', 'desalter_twin.pkl')
    db_path = os.path.join(root_dir, 'phase2_optimizer', 'historian_data.sqlite')
    if not os.path.exists(db_path):
        db_path = os.path.join(root_dir, 'desalter_optimization', 'phase2_optimizer', 'historian_data.sqlite')
    
    print("[2/2] Evaluating Digital Twin Regressor...")
    if not os.path.exists(regressor_path):
        raise FileNotFoundError(f"Regressor model not found at: {regressor_path}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
        
    with open(regressor_path, 'rb') as f:
        reg = pickle.load(f)
        
    conn = sqlite3.connect(db_path)
    df_reg = pd.read_sql_query("SELECT * FROM historian", conn)
    conn.close()
    
    X_reg = df_reg[['API_Gravity', 'Inlet_BSW', 'Inlet_Salt_PTB', 'Temperature_C', 'Wash_Water_Percent']]
    y_reg = df_reg['Emulsion_Thickness_mm']
    
    _, X_reg_test, _, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    y_reg_pred = reg.predict(X_reg_test)
    
    reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
    reg_r2 = r2_score(y_reg_test, y_reg_pred)
    
    # ----------------------------------------------------
    # 3. Format and Output ASCII Report
    # ----------------------------------------------------
    report_title = "=== IOCL DESALTER AI PIPELINE EVALUATION REPORT ==="
    border = "=" * len(report_title)
    
    report = f"""
{border}
{report_title}
{border}

SECTION 1: CRUDE RISK PROFILER (XGBoost Classifier)
----------------------------------------------------
Evaluation Dataset: crude_profile_data.csv (Test Split: 20%)
Total Test Samples: {len(y_clf_test)}

Metrics:
- Model Accuracy : {clf_acc * 100:.2f}%
- F1-Score (Macro): {clf_f1 * 100:.2f}%

Confusion Matrix:
{format_confusion_matrix(clf_cm, le.classes_)}


SECTION 2: DESALTER DIGITAL TWIN (XGBoost Regressor)
----------------------------------------------------
Evaluation Dataset: historian_data.sqlite [historian table] (Test Split: 20%)
Total Test Samples: {len(y_reg_test)}

Metrics:
- Root Mean Squared Error (RMSE) : {reg_rmse:.4f} mm
- Mean Absolute Error (MAE)     : {reg_mae:.4f} mm
- Coefficient of Determination (R^2): {reg_r2:.4f} ({reg_r2 * 100:.2f}%)

{border}
Report Generated Successfully.
{border}
"""
    print(report)

if __name__ == '__main__':
    main()
