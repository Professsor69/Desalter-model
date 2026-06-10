"""
Regenerated Crude Oil Batch Dataset — Aggressively Reduced Noise V2
IOCL Panipat Refinery — Desalter Unit Optimization
Goal: Generate clean data to achieve 85-92% model accuracy.
Noise reduction strategy:
  - Feature-level noise sigma reduced by ~70%
  - Score-level noise sigma: 0.45 -> 0.10
  - Label flip probability: 5% -> 0%
"""

import numpy as np
import pandas as pd
import sys

SEED = 42
rng  = np.random.default_rng(SEED)
N    = 5_000

# ── 1. CRUDE BLEND ────────────────────────────────────────────────
blends = ['Basrah Heavy', 'Arab Light', 'Ural', 'Bonny Light']
blend_weights = [0.35, 0.30, 0.20, 0.15]
crude_blend = rng.choice(blends, size=N, p=blend_weights)

# ── 2. API GRAVITY ────────────────────────────────────────────────
api_params = {
    'Basrah Heavy': (2.0, 3.5, 20.0, 34.0),
    'Arab Light'  : (3.0, 2.5, 28.0, 40.0),
    'Ural'        : (2.5, 2.5, 26.0, 38.0),
    'Bonny Light' : (3.5, 2.0, 34.0, 45.0),
}
api_raw = np.empty(N)
for blend, (alpha, beta_, lo, hi) in api_params.items():
    mask = crude_blend == blend
    n_blend = mask.sum()
    api_raw[mask] = lo + (hi - lo) * rng.beta(alpha, beta_, size=n_blend)

# REDUCED: feature-level noise sigma 0.5 -> 0.15
API_Gravity = np.clip(api_raw + rng.normal(0.0, 0.15, N), 20.0, 45.0).round(2)

# ── 3. INLET BSW ──────────────────────────────────────────────────
api_norm = (API_Gravity - 20.0) / (45.0 - 20.0)
bsw_base = 2.0 - 1.7 * api_norm

# REDUCED: log-normal sigma 0.35 -> 0.10
bsw_noise = rng.lognormal(mean=0.0, sigma=0.10, size=N)
bsw_noise /= np.exp(0.10**2 / 2)
Inlet_BSW = np.clip(bsw_base * bsw_noise, 0.1, 2.5).round(3)

# ── 4. INLET SALT ─────────────────────────────────────────────────
salt_mean  = 10.0 + 20.0 * (Inlet_BSW / 2.5)
salt_std   = 2.0   # REDUCED: from 8.0 -> 2.0
salt_shape = (salt_mean**2) / (salt_std**2)
salt_scale = (salt_std**2)  / salt_mean
Inlet_Salt_PTB = np.clip(
    rng.gamma(shape=salt_shape, scale=salt_scale), 10.0, 60.0
).round(2)

# ── 5. RISK SCORE & TARGET CLASS ──────────────────────────────────
S_api  = 1.5 / (1.0 + np.exp(0.5 * (API_Gravity - 28.0)))
S_bsw  = 1.5 / (1.0 + np.exp(-3.0 * (Inlet_BSW - 1.2)))
S_salt = np.clip((Inlet_Salt_PTB - 40.0) / 20.0, 0.0, 1.0) * 0.8

risk_score_true = S_api + S_bsw + S_salt

# REDUCED: score noise sigma 0.45 -> 0.10
score_noise = rng.normal(0.0, 0.10, N)
risk_score_noisy = risk_score_true + score_noise

HIGH_THRESH   = 2.20
MEDIUM_THRESH = 1.10

conditions = [
    risk_score_noisy >= HIGH_THRESH,
    (risk_score_noisy >= MEDIUM_THRESH) & (risk_score_noisy < HIGH_THRESH),
    risk_score_noisy < MEDIUM_THRESH,
]
choices = ['High', 'Medium', 'Low']
Target_Risk_Class = np.select(conditions, choices, default='Low')

# REMOVED: label flips — physics-pure labels for max separability
# Target_Risk_Class is now the direct binned result with no random flips

# ── 6. ASSEMBLE ───────────────────────────────────────────────────
df = pd.DataFrame({
    'Crude_Blend'      : crude_blend,
    'API_Gravity'      : API_Gravity,
    'Inlet_BSW'        : Inlet_BSW,
    'Inlet_Salt_PTB'   : Inlet_Salt_PTB,
    'Target_Risk_Class': Target_Risk_Class,
})

print(f"Shape            : {df.shape}")
print(f"Null values      : {df.isnull().sum().sum()}")
print("Target Class Distribution:")
class_counts = df['Target_Risk_Class'].value_counts()
for cls, cnt in class_counts.items():
    print(f"  {cls}: {cnt} ({100*cnt/N:.1f}%)")

df.to_csv("crude_profile_data.csv", index=False)
print("Dataset saved -> crude_profile_data.csv")
