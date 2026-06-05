# 🛢️ Desalter-model: Crude Oil Risk Classification

![Status: In Operations](https://img.shields.io/badge/Status-In%20Operations-success?style=for-the-badge)
![Python 3.x](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![MLOps](https://img.shields.io/badge/MLOps-Stage%201-orange?style=for-the-badge)

**IOCL Panipat Refinery — Desalter Unit Optimization (Stage 1)**

Predicting whether a crude batch will be high or low risk before it enters the desalter unit. This project is actively in operations, generating synthetic data profiles to support the training of predictive Machine Learning models.

---

## 📖 Domain Context

A **Desalter unit** removes salt and water (Basic Sediment & Water — BSW) from incoming crude oil before it enters the Atmospheric Distillation Unit (ADU). Processing difficulty (risk) is primarily driven by three key operational parameters:

- **API Gravity (°API):** Heavier crudes form tighter emulsions, making them harder to separate.
- **Inlet BSW (%):** More free water means it is harder to coalesce and separate effectively.
- **Inlet Salt (PTB):** High salt load increases the corrosion risk in downstream units.

This project implements physics-informed risk logic based on industry heuristics (IP-77 / ASTM D4007 standards, Petro-Canada, and Saudi Aramco desalter operating manuals) to simulate these real-world operational challenges.

## ⚙️ How It Works (Data Generation)

This repository contains the logic to generate a highly realistic **synthetic dataset** (`crude_profile_data.csv`) of 5,000 crude oil batches. 

The `generate_crude_profile.py` script builds data profiles representing typical crude blends (e.g., Basrah Heavy, Arab Light, Ural, Bonny Light) processed in refineries. To make the dataset robust for ML models, **noise** is strategically injected into the data:

1. **Feature-level Gaussian noise:** Simulates temperature swings, upstream blending variations, and measurement drift.
2. **Score-level Gaussian noise:** Blurs decision boundaries for the risk score, forcing ML models to generalize rather than memorize hard thresholds.
3. **Random Class Flips (~5%):** Randomly reassigns labels to simulate mislabeling, instrument faults, or unpredictable chemical variations.

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `Crude_Blend` | The type of crude oil (e.g., Basrah Heavy, Arab Light). |
| `API_Gravity` | Measures how heavy the crude is (°API). |
| `Inlet_BSW` | Basic Sediment & Water (% vol). |
| `Inlet_Salt_PTB` | Salt content in pounds per thousand barrels (PTB). |
| `Target_Risk_Class` | The processing difficulty target variable: **Low**, **Medium**, or **High**. |

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Professsor69/Desalter-model.git
cd Desalter-model
```

### 2. Install Dependencies
Ensure you have `pandas` and `numpy` installed:
```bash
pip install pandas numpy
```

### 3. Generate the Dataset
To regenerate the dataset or run the pipeline in your operations environment, execute:
```bash
python generate_crude_profile.py
```
This will run the generation logic and save a new batch of records to `crude_profile_data.csv`.

## 🤝 Contributing
For internal operations teams contributing to the risk logic:
1. Update heuristics inside the `generate_crude_profile.py` parameters section.
2. Ensure you maintain the noise strategy to prevent model overfitting.
3. Submit a Pull Request for review by the MLOps architect.

---
*Maintained by the Operations & MLOps Team*
