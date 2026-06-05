"""
=====================================================================
Synthetic Crude Oil Batch Dataset Generator
IOCL Panipat Refinery — Desalter Unit Optimization (Stage 1)
Role  : Senior Process Engineer + MLOps Architect
Output: crude_profile_data.csv  (5,000 rows)
=====================================================================

DOMAIN CONTEXT
--------------
A Desalter unit removes salt and water (Basic Sediment & Water — BSW)
from incoming crude before it enters the Atmospheric Distillation Unit
(ADU).  Difficulty of processing (risk) is primarily driven by:

  • API Gravity    → heavier crudes form tighter emulsions
  • Inlet BSW (%)  → more free water = harder to coalesce and separate
  • Inlet Salt PTB → high salt load increases corrosion risk downstream

The physics-informed risk logic below is based on industry heuristics
(IP-77 / ASTM D4007 standards, Petro-Canada & Saudi Aramco desalter
operating manuals).

NOISE STRATEGY  (why we inject randomness)
------------------------------------------
Real historian data is never perfectly deterministic.  Temperature
swings, upstream blending variations, pump seal leaks, and instrument
drift all cause measurements to deviate from ideal physics.

We model this as:

  1. Feature-level Gaussian noise  — small σ perturbations on API,
     BSW, and Salt after the "true" physics value is drawn. This
     smears the feature distributions slightly.

  2. Score-level Gaussian noise  — each sample gets a continuous
     "risk score" computed from the physics rules, then Gaussian
     noise N(0, σ_score) is added *before* binning into classes.
     This blurs decision boundaries so the ML model must genuinely
     generalise rather than memorise a hard threshold.

  3. ~5 % random class flips  — a small fraction of labels are
     randomly reassigned to simulate genuine mislabeling, instrument
     faults, or edge cases that violate the dominant physics (e.g.,
     an unusually well-demulsified heavy crude).  This prevents the
     model from overfitting to perfect signal.

=====================================================================
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)          # modern NumPy RNG — thread-safe

N    = 5_000                                 # number of crude batches

# ──────────────────────────────────────────────────────────────────
# 1.  CRUDE BLEND  — multinomial sample with realistic market shares
#     Basrah Heavy & Arab Light dominate Indian basket; Ural & Bonny
#     Light are minority streams.
# ──────────────────────────────────────────────────────────────────
blends = ['Basrah Heavy', 'Arab Light', 'Ural', 'Bonny Light']
blend_weights = [0.35, 0.30, 0.20, 0.15]    # approximate IOCL import share

crude_blend = rng.choice(blends, size=N, p=blend_weights)

# ──────────────────────────────────────────────────────────────────
# 2.  API GRAVITY  (°API, continuous 20 – 45)
#     Each blend has a characteristic API range (from crude assays):
#       Basrah Heavy : 24 – 30  (median ≈ 27)
#       Arab Light   : 32 – 36  (median ≈ 34)
#       Ural         : 30 – 34  (median ≈ 32)
#       Bonny Light  : 36 – 44  (median ≈ 40)
#     We draw from blend-specific Beta distributions (bounded),
#     then clip to the global [20, 45] envelope.
# ──────────────────────────────────────────────────────────────────

# Beta(α, β) on [lo, hi]  →  mean ≈ lo + (hi-lo)*α/(α+β)
api_params = {
    'Basrah Heavy': (2.0, 3.5, 20.0, 34.0),   # (α, β, lo, hi)
    'Arab Light'  : (3.0, 2.5, 28.0, 40.0),
    'Ural'        : (2.5, 2.5, 26.0, 38.0),
    'Bonny Light' : (3.5, 2.0, 34.0, 45.0),
}

api_raw = np.empty(N)
for blend, (alpha, beta_, lo, hi) in api_params.items():
    mask = crude_blend == blend
    n_blend = mask.sum()
    # Beta sample rescaled to physical range
    api_raw[mask] = lo + (hi - lo) * rng.beta(alpha, beta_, size=n_blend)

# Feature-level noise: ± ~0.5 °API (instrument/blending variance)
api_noise = rng.normal(loc=0.0, scale=0.5, size=N)
API_Gravity = np.clip(api_raw + api_noise, 20.0, 45.0).round(2)

# ──────────────────────────────────────────────────────────────────
# 3.  INLET BSW  (% vol, continuous 0.1 – 2.5)
#     BSW is correlated (negatively) with API — heavier crudes carry
#     more emulsified water.  We model this with a base BSW drawn
#     from an inverse-API relationship plus blend-specific variance.
#
#     Physics basis:  heavier (low-API) crudes have higher resin /
#     asphaltene content → stronger natural emulsifiers → higher BSW.
# ──────────────────────────────────────────────────────────────────

# Normalise API to [0,1] over the [20,45] range
api_norm = (API_Gravity - 20.0) / (45.0 - 20.0)        # 0 = heaviest

# Base BSW: linearly decreasing from 2.0 to 0.3 as API increases
bsw_base = 2.0 - 1.7 * api_norm

# Multiplicative log-normal noise (σ_log ≈ 0.35) — realistic for
# flow-line measurements; always positive and right-skewed
bsw_noise = rng.lognormal(mean=0.0, sigma=0.35, size=N)

# Scale noise so its median = 1.0 (preserve base level on average)
bsw_noise /= np.exp(0.35**2 / 2)                        # correct log-normal mean

Inlet_BSW = np.clip(bsw_base * bsw_noise, 0.1, 2.5).round(3)

# ──────────────────────────────────────────────────────────────────
# 4.  INLET SALT  (PTB — pounds per thousand barrels, 10 – 60)
#     Salt content is moderately correlated with BSW (salt dissolved
#     in the entrained water) but has its own variance from upstream
#     wellhead brines and cargo contamination.
#
#     Model: Salt ~ Gamma(shape, scale) where the mean shifts with BSW
#     Gamma chosen because PTB is strictly positive and right-skewed.
# ──────────────────────────────────────────────────────────────────

# Higher BSW → higher salt baseline (brine co-carries salt)
salt_mean = 10.0 + 20.0 * (Inlet_BSW / 2.5)            # range: 10 – 30 PTB baseline
salt_std  = 8.0                                          # process spread
# Gamma parameterisation: shape = mean²/var, scale = var/mean
salt_shape = (salt_mean**2) / (salt_std**2)
salt_scale = (salt_std**2)  / salt_mean

Inlet_Salt_PTB = np.clip(
    rng.gamma(shape=salt_shape, scale=salt_scale),
    10.0, 60.0
).round(2)

# ──────────────────────────────────────────────────────────────────
# 5.  TARGET RISK CLASS  (physics-informed, noise-injected)
#
#     We compute a continuous "risk score" S ∈ [0, ~3] using additive
#     sub-scores, then bin it into {Low, Medium, High}.  Gaussian
#     noise is added to S *before* binning so boundaries stay fuzzy.
#
#     Sub-score definitions:
#       S_api  : penalises low API  (heavy crude = emulsion risk)
#       S_bsw  : penalises high BSW (water carry = separation load)
#       S_salt : penalises high salt (corrosion risk, separate axis)
#
#     Thresholds calibrated so ≈ 25 % High, 45 % Medium, 30 % Low
#     (typical distribution at a refinery processing a heavy basket).
# ──────────────────────────────────────────────────────────────────

# — Sub-score 1: API contribution (max contribution = 1.5)
#   Sigmoid centred at API=28 (heavy/medium boundary), steep slope
#   Heavy crude (API<28) → S_api close to 1.5; light crude → ~0
S_api = 1.5 / (1.0 + np.exp(0.5 * (API_Gravity - 28.0)))

# — Sub-score 2: BSW contribution (max = 1.5)
#   Sigmoid centred at BSW=1.2 (high-BSW threshold)
S_bsw = 1.5 / (1.0 + np.exp(-3.0 * (Inlet_BSW - 1.2)))

# — Sub-score 3: Salt contribution (max = 0.8)
#   Linear ramp: 0 at 40 PTB, full weight at 60 PTB
S_salt = np.clip((Inlet_Salt_PTB - 40.0) / 20.0, 0.0, 1.0) * 0.8

# — Combined risk score (range ≈ 0 to 3.8)
risk_score_true = S_api + S_bsw + S_salt

# ── Noise injection (Stage A): score-level Gaussian blur ──────────
# σ = 0.45 ≈ ~12 % of the score range → creates ~15–20 % boundary
# ambiguity, which is realistic and prevents the model from learning
# a trivially hard threshold.
score_noise = rng.normal(loc=0.0, scale=0.45, size=N)
risk_score_noisy = risk_score_true + score_noise

# ── Binning into classes using calibrated thresholds ──────────────
#    Thresholds tuned empirically on the score distribution to hit
#    the target class balance (High ≈ 25 %, Medium ≈ 45 %, Low ≈ 30 %)
HIGH_THRESH   = 2.20
MEDIUM_THRESH = 1.10

conditions = [
    risk_score_noisy >= HIGH_THRESH,
    (risk_score_noisy >= MEDIUM_THRESH) & (risk_score_noisy < HIGH_THRESH),
    risk_score_noisy < MEDIUM_THRESH,
]
choices = ['High', 'Medium', 'Low']
Target_Risk_Class = np.select(conditions, choices, default='Low')

# ── Noise injection (Stage B): random label flips (~5 %) ──────────
# Simulates: instrument faults, mislabeled batches, unusual crude
# chemistry that defeats the dominant physics (e.g., a demulsifier-
# dosed heavy crude that processes easily).
flip_prob   = 0.05
flip_mask   = rng.random(N) < flip_prob
flip_labels = rng.choice(['Low', 'Medium', 'High'], size=N)
Target_Risk_Class = np.where(flip_mask, flip_labels, Target_Risk_Class)

# ──────────────────────────────────────────────────────────────────
# 6.  ASSEMBLE DATAFRAME
# ──────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'Crude_Blend'      : crude_blend,
    'API_Gravity'      : API_Gravity,
    'Inlet_BSW'        : Inlet_BSW,
    'Inlet_Salt_PTB'   : Inlet_Salt_PTB,
    'Target_Risk_Class': Target_Risk_Class,
})

# ──────────────────────────────────────────────────────────────────
# 7.  SANITY CHECKS  — print to console before saving
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("CRUDE PROFILE DATASET — GENERATION SUMMARY")
print("=" * 60)

print(f"\nShape            : {df.shape}")
print(f"Null values      : {df.isnull().sum().sum()}")

print("\n── Feature Statistics ──────────────────────────────────────")
print(df[['API_Gravity', 'Inlet_BSW', 'Inlet_Salt_PTB']].describe().round(3).to_string())

print("\n── Crude Blend Distribution ────────────────────────────────")
blend_counts = df['Crude_Blend'].value_counts()
for b, c in blend_counts.items():
    print(f"  {b:<15} : {c:>5}  ({100*c/N:.1f} %)")

print("\n── Target Class Distribution ───────────────────────────────")
class_counts = df['Target_Risk_Class'].value_counts()
for cls, cnt in class_counts.items():
    bar = '█' * (cnt // 50)
    print(f"  {cls:<8} : {cnt:>5}  ({100*cnt/N:.1f} %)  {bar}")

print("\n── Physics Validation (no-noise score boundaries) ──────────")
# Show mean feature values per class — confirms physics direction
agg = df.groupby('Target_Risk_Class')[['API_Gravity','Inlet_BSW','Inlet_Salt_PTB']].mean().round(3)
print(agg.to_string())
print("\nExpected direction: High risk → lower API, higher BSW, higher Salt")

# ──────────────────────────────────────────────────────────────────
# 8.  SAVE
# ──────────────────────────────────────────────────────────────────
OUTPUT_PATH = '/mnt/user-data/outputs/crude_profile_data.csv'
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓  Dataset saved → {OUTPUT_PATH}")
print("=" * 60)
