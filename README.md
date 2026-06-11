# 🛢️ IOCL Desalter AI Control System & Predictive Engine

![Project Status: Operational](https://img.shields.io/badge/Status-In%20Operations-success?style=for-the-badge)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![MLOps Engine](https://img.shields.io/badge/MLOps-Phase%203-orange?style=for-the-badge)

An end-to-end intelligent MLOps control system built for the **IOCL Panipat Refinery Desalter Unit**. The system is composed of three interconnected parts designed to classify raw crude processing risk, prescriptively optimize heater and wash water setpoints using a physics-informed digital twin, and monitor real-time SCADA streams to predict electrostatic grid trips 60 minutes before they occur.

---

## 🛠️ Complete Tech Stack

The project utilizes a modern MLOps stack for machine learning, API development, interactive visualization, and containerized deployment:

* **Machine Learning & Analytics:**
  * **XGBoost:** Core ML algorithm for classification (risk profiling, trip warning) and regression (digital twin emulator).
  * **Scikit-learn:** Data preprocessing, train-test splitting, and classification metrics.
  * **SciPy:** Hybrid optimization (Powell gradient-free optimizer combined with Grid Search).
  * **Pandas & NumPy:** Time-series rolling features, feature engineering, and data manipulation.
  * **Optuna:** Bayesian hyperparameter optimization.
* **API Development & Serving:**
  * **FastAPI:** High-performance ASGI framework for serving inference endpoints.
  * **Uvicorn:** Light-weight ASGI server.
  * **Pydantic:** Strictly enforced request payload validation schemas.
  * **Requests:** Client HTTP communication.
* **Frontend Visualization:**
  * **Streamlit:** Highly interactive reactive dashboards with custom CSS styles.
* **Database & Storage:**
  * **SQLite3:** Process historian simulation database.
  * **Joblib / Pickle:** Serialization and deserialization of trained pipelines.
* **DevOps & Containerization:**
  * **Docker & Docker Compose:** Lightweight environments for independent service deployment.

---

## 📂 Repository Architecture

The project is structured into three distinct modular components:

```bash
Desalter Project/
├── .gitignore                         # Excludes cache files and bytecode
├── README.md                          # Main documentation (this file)
├── crude_oil_risk_prediction/         # PART 1: Crude Oil Risk Classification
│   ├── api.py                         # FastAPI service for risk prediction
│   ├── app.py                         # Streamlit UI for risk profiling
│   ├── crude_profile_data.csv         # Generated synthetic dataset
│   ├── desalter_risk_model.pkl        # Stored classification pipeline
│   ├── Dockerfile.api                 # Docker environment for API
│   ├── Dockerfile.ui                  # Docker environment for Streamlit UI
│   ├── docker-compose.yml             # Local compose stack
│   └── (training scripts...)
├── desalter_optimization/             # PART 2: Identification & Best Settings System
│   ├── Dockerfile.api                 # Builds FastAPI optimizer backend
│   ├── Dockerfile.ui                  # Builds Streamlit optimizer dashboard
│   ├── docker-compose.yml             # Orchestrates the optimizer stack
│   ├── phase1_profiler/               # Phase 1: Crude Risk Profiler classifier
│   │   ├── crude_profiler.pkl         # Trained XGBoost classifier
│   │   └── train_profiler.py          # Profiler training script
│   ├── phase2_optimizer/              # Phase 2: Prescriptive Digital Twin & SQLite Historian
│   │   ├── desalter_twin.pkl          # Digital Twin XGBoost regressor
│   │   ├── historian_data.sqlite      # SQLite DB simulating refinery historian
│   │   ├── train_twin.py              # Regressor training script
│   │   └── prescriptive_optimizer.py  # SciPy Powell optimizer script
│   └── phase3_ui_api/                 # Phase 3: Combined UI/API Serving
│       ├── api.py                     # API exposing setpoint & trip predictions
│       └── dashboard.py               # Combined Streamlit control panel
├── timeseries_engine/                 # PART 3: Time-Series Early Warning Engine
│   ├── desalter_timeseries.csv        # 30-day SCADA timeseries dataset
│   ├── generate_timeseries.py         # Time-series random walk data generator
│   ├── train_timeseries.py            # Training pipeline for warnings
│   └── timeseries_warning_model.pkl   # Serialized early warning model
└── scripts/
    └── evaluate_pipeline.py           # Global AI pipeline evaluation script
```

---

## 📖 Component Overview

### Part 1: Crude Oil Risk Classifier (`crude_oil_risk_prediction/`)
Predicts the processing risk class (**Low**, **Medium**, or **High**) of an incoming crude oil batch before it enters the desalter. 
* **Core Drivers:** API Gravity, Inlet BSW (Basic Sediment & Water %), and Inlet Salt (PTB).
* **Model:** XGBoost Classifier trained on 5,000 synthetic crude profiles.
* **Accuracy:** **94.20%** (F1-Score: **92.66%**).

### Part 2: Prescriptive Digital Twin Optimizer (`desalter_optimization/`)
Recommends optimal setpoints for **Process Temperature (°C)** and **Wash Water Flow Rate (%)** to minimize the desalter emulsion layer thickness, preventing downstream corrosion and instrument fouling.
* **Heuristic Engine:** Utilizes SciPy's Powell algorithm initialized via global Grid Search over the operating bounds (Temperature: 110-150°C, Wash Water: 2-8%) against an XGBoost Digital Twin model.
* **Data Source:** Process historian records stored in SQLite (`historian_data.sqlite`).
* **Digital Twin Performance:** $R^2$ Score of **96.10%** (RMSE: **2.2675 mm**).

### Part 3: Time-Series Predictive Maintenance Engine (`timeseries_engine/`)
Monitors high-frequency SCADA streams to predict electrostatic grid voltage trips (failures) **60 minutes before they happen**, allowing operators to intervene proactively.
* **Engineered Features:** 15-minute and 60-minute rolling means and slopes (gradients) of temperature and wash water.
* **Model:** Weighted `XGBClassifier` trained on a 30-day simulated continuous run (43,200 rows) containing 10 trip failure events.
* **Early Warning Resolution:** Formatted with a 61-row sliding history window to compute valid rolling differences and slopes, avoiding `NaN` values and ensuring reliable warning alarms.

---

## 🚀 Installation & Running the Project

### Prerequisites
* Python 3.11+
* Docker & Docker Compose (Desktop)

### Option A: Running Containerized (Recommended)

To run any of the components inside Docker containers, navigate to their respective directory and run Docker Compose:

#### 1. Running the Setpoint Optimizer & Early Warning System (Parts 2 & 3)
This builds and serves the consolidated desalter control dashboard on port 8501 and the API on port 8000.
```bash
# Navigate to the optimization directory
cd desalter_optimization

# Start the stack
docker compose up -d --build
```
* **FastAPI documentation:** `http://localhost:8000/docs`
* **Streamlit Control Panel:** `http://localhost:8501` (includes "Static Optimizer" and "Live Early Warning Monitor" with what-if override analysis).

#### 2. Running the Crude Oil Risk Classifier (Part 1)
```bash
# Navigate to the risk classifier directory
cd crude_oil_risk_prediction

# Start the stack
docker compose up -d --build
```
* **API Documentation:** `http://localhost:8000/docs`
* **Streamlit App:** `http://localhost:8501`

---

### Option B: Running Locally (Bare Metal)

#### 1. Setup Virtual Environment & Install Dependencies
```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required python packages
pip install pandas numpy scikit-learn xgboost scipy joblib fastapi uvicorn pydantic streamlit requests
```

#### 2. Running the Combined Optimizer Dashboard
```bash
# Start FastAPI backend
python desalter_optimization/phase3_ui_api/api.py

# In a new terminal, launch the Streamlit frontend
streamlit run desalter_optimization/phase3_ui_api/dashboard.py
```

#### 3. Run Pipeline Evaluation Script
To check the performance metrics of the trained models locally:
```bash
python scripts/evaluate_pipeline.py
```

---

## 📈 Model Performance & Evaluation Report

Run the evaluation script to generate verification reports:
```text
===================================================
=== IOCL DESALTER AI PIPELINE EVALUATION REPORT ===
===================================================

SECTION 1: CRUDE RISK PROFILER (XGBoost Classifier)
----------------------------------------------------
Evaluation Dataset: crude_profile_data.csv (Test Split: 20%)
Total Test Samples: 1000

Metrics:
- Model Accuracy : 94.20%
- F1-Score (Macro): 92.66%

SECTION 2: DESALTER DIGITAL TWIN (XGBoost Regressor)
----------------------------------------------------
Evaluation Dataset: historian_data.sqlite [historian table] (Test Split: 20%)
Total Test Samples: 2000

Metrics:
- Root Mean Squared Error (RMSE) : 2.2675 mm
- Mean Absolute Error (MAE)     : 1.7954 mm
- Coefficient of Determination (R^2): 0.9610 (96.10%)
===================================================
```
