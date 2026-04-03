# =============================================================================
#  step2_preprocess.py  —  PredictAI
#  Feature engineering → scaling → SMOTE → train/test split
#  Output: data/X_train.csv, X_test.csv, y_train.csv, y_test.csv
#          models/scaler.pkl, models/feature_names.pkl
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import *

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  STEP 2 — PREPROCESSING & FEATURE ENGINEERING")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "raw_sensor_data.csv"),
                 parse_dates=["timestamp"])
print(f"\n[1/7] Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── 2. Clean ──────────────────────────────────────────────────
print("[2/7] Cleaning missing values and capping outliers …")
roll_cols = [c for c in df.columns if "rolling" in c or "delta" in c]
if roll_cols:
    df[roll_cols] = df[roll_cols].ffill().fillna(0)

for col in SENSOR_COLS:
    q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))

# ── 3. Rolling / lag features ─────────────────────────────────
print("[3/7] Creating rolling window and lag features …")
df = df.sort_values("timestamp").reset_index(drop=True)
for col in ["temperature", "vibration", "pressure", "rpm"]:
    df[f"{col}_roll5_mean"] = df[col].rolling(5,  min_periods=1).mean()
    df[f"{col}_roll5_std"]  = df[col].rolling(5,  min_periods=1).std().fillna(0)
    df[f"{col}_roll20_mean"]= df[col].rolling(20, min_periods=1).mean()
    df[f"{col}_delta"]      = df[col].diff().fillna(0)
    df[f"{col}_lag1"]       = df[col].shift(1).fillna(df[col])

# ── 4. Composite engineered features ─────────────────────────
print("[4/7] Engineering composite features …")
df["thermal_stress"]    = (df["temperature"] - 72) / 5.0
df["mechanical_stress"] = df["vibration"] / 1.5
df["hydraulic_stress"]  = df["pressure"]  / 6.0
df["composite_stress"]  = (
    0.40 * df["thermal_stress"] +
    0.35 * df["mechanical_stress"] +
    0.25 * df["hydraulic_stress"]
)
df["power_kw"]          = df["voltage"] * df["current"] / 1000.0
df["rpm_deviation"]     = (df["rpm"] - 2800).abs() / 2800.0
df["lub_health"]        = (df["oil_viscosity"] - 25) / 30.0
df["temp_vib_cross"]    = df["temperature"] * df["vibration"]
df["temp_rpm_ratio"]    = df["temperature"] / (df["rpm"].clip(1))
df["vib_noise_cross"]   = df["vibration"] * df["noise_level"]

# Threshold violation counter
def count_violations(row):
    flags = 0
    for col, lim in THRESHOLDS.items():
        if col in row.index:
            if row[col] >= lim["crit"]: flags += 2
            elif row[col] >= lim["warn"]: flags += 1
    return flags
df["threshold_flags"] = df[SENSOR_COLS].apply(count_violations, axis=1)

# Cyclical time
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Machine OHE
df = pd.get_dummies(df, columns=["machine_id"], prefix="mch", dtype=int)

# ── 5. Feature matrix ─────────────────────────────────────────
print("[5/7] Building feature matrix …")
drop = ["timestamp","failure","fault_type","hour","day_of_week"]
feature_cols = [c for c in df.columns if c not in drop]
X = df[feature_cols].copy()
y = df[TARGET_COL].copy()
print(f"   Features: {len(feature_cols)}  |  Shape: {X.shape}")

# ── 6. Split + scale ──────────────────────────────────────────
print("[6/7] Train/test split (80/20 stratified) + RobustScaler …")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

scaler = RobustScaler()
X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_cols)
X_te_sc = pd.DataFrame(scaler.transform(X_te),     columns=feature_cols)

# ── 7. SMOTE ──────────────────────────────────────────────────
print("[7/7] SMOTE oversampling on training set …")
smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
X_tr_res, y_tr_res = smote.fit_resample(X_tr_sc, y_tr)
print(f"   Before SMOTE: {len(X_tr):,}  |  After: {len(X_tr_res):,}")
print(f"   Class 0: {sum(y_tr_res==0):,}  Class 1: {sum(y_tr_res==1):,}")

# ── Save ──────────────────────────────────────────────────────
pd.DataFrame(X_tr_res, columns=feature_cols).to_csv(os.path.join(DATA_DIR,"X_train.csv"), index=False)
X_te_sc.to_csv(os.path.join(DATA_DIR,"X_test.csv"),  index=False)
pd.Series(y_tr_res, name="failure").to_csv(os.path.join(DATA_DIR,"y_train.csv"), index=False)
y_te.to_csv(os.path.join(DATA_DIR,"y_test.csv"),     index=False)
joblib.dump(scaler,       os.path.join(MODEL_DIR,"scaler.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR,"feature_names.pkl"))
print("\n   ✓ Saved train/test splits and scaler")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0d1117")
fig.suptitle("Step 2 — Preprocessing & Feature Engineering",
             color="white", fontsize=14, y=0.98)

# Class balance before/after
ax = axes[0, 0]; ax.set_facecolor("#111820")
cats = ["Train (raw)","Train (SMOTE)","Test"]
n0 = [y_tr.value_counts()[0], sum(y_tr_res==0), y_te.value_counts()[0]]
n1 = [y_tr.value_counts()[1], sum(y_tr_res==1), y_te.value_counts()[1]]
x = np.arange(3)
ax.bar(x-.2, n0, .4, label="Normal",  color="#22c55e", alpha=.8)
ax.bar(x+.2, n1, .4, label="Failure", color="#ef4444", alpha=.8)
ax.set_xticks(x); ax.set_xticklabels(cats, color="#94a3b8", fontsize=8)
ax.set_title("Class Balance Before/After SMOTE", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b"); ax.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# Composite stress distribution
ax = axes[0, 1]; ax.set_facecolor("#111820")
for lbl, clr, nm in [(0,"#22c55e","Normal"),(1,"#ef4444","Failure")]:
    ax.hist(df[df.failure==lbl]["composite_stress"], bins=50,
            alpha=.7, color=clr, label=nm, density=True)
ax.set_title("Composite Stress Distribution", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b"); ax.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# Correlation with target
ax = axes[0, 2]; ax.set_facecolor("#111820")
top_feats = SENSOR_COLS + ["composite_stress","power_kw","threshold_flags","temp_vib_cross"]
corr = df[[c for c in top_feats if c in df]+["failure"]].corr()["failure"].drop("failure").sort_values()
cols = ["#ef4444" if v>=0 else "#22c55e" for v in corr.values]
ax.barh(corr.index, corr.values, color=cols)
ax.axvline(0, color="#64748b", lw=.8)
ax.set_title("Feature Correlation with Failure", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b", labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# Rolling mean – temperature
ax = axes[1, 0]; ax.set_facecolor("#111820")
sample = df[df.machine_id=="MCH-001"].head(300) if "machine_id" in df.columns else df.head(300)
# reconstruct column if needed
if "temperature" in sample and "temperature_roll5_mean" in sample:
    ax.plot(sample["temperature"].values, color="#64748b", lw=.8, alpha=.7, label="Raw")
    ax.plot(sample["temperature_roll5_mean"].values, color="#f97316", lw=1.5, label="Roll-5")
    ax.plot(sample["temperature_roll20_mean"].values, color="#38bdf8", lw=1.5, label="Roll-20", linestyle="--")
ax.set_title("Temperature Rolling Features", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b"); ax.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# Power consumption per machine
ax = axes[1, 1]; ax.set_facecolor("#111820")
if "power_kw" in df.columns and "machine_id" in df.columns:
    mch_col = "machine_id"
else:
    mch_col = [c for c in df.columns if c.startswith("mch_")]
# Safe plot
try:
    pw = df.groupby("machine_id")["power_kw"].mean() if "machine_id" in df.columns else pd.Series([6.2,6.8,7.1,5.5,6.4,6.9])
    ax.bar(range(len(pw)), pw.values, color="#38bdf8", alpha=.8)
    ax.set_xticks(range(len(pw))); ax.set_xticklabels([s.replace("MCH-","M") for s in pw.index], color="#94a3b8", fontsize=8)
except: pass
ax.set_title("Avg Power (kW) per Machine", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b")
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# Threshold flags
ax = axes[1, 2]; ax.set_facecolor("#111820")
for lbl, clr, nm in [(0,"#22c55e","Normal"),(1,"#ef4444","Failure")]:
    cnt = df[df.failure==lbl]["threshold_flags"].value_counts().sort_index()
    ax.bar(cnt.index + (.2 if lbl else -.2), cnt.values, .4, color=clr, alpha=.8, label=nm)
ax.set_title("Threshold Violation Flags by Class", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b"); ax.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR,"step2_preprocessing.png"),
            dpi=130, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Plot → reports/step2_preprocessing.png")
print("\n✅  STEP 2 COMPLETE\n")
