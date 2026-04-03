# =============================================================================
#  step1_generate_data.py  —  PredictAI
#  Generates 12,000 realistic sensor readings with 5 fault types.
#  Output: data/raw_sensor_data.csv
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
from config import *

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  STEP 1 — REALISTIC DATA GENERATION")
print("=" * 60)

# ── Helpers ───────────────────────────────────────────────────
def rnd(a, b): return np.random.uniform(a, b)

# ── Generate base readings with realistic autocorrelation ─────
def generate_sensor_stream(n, mean, std, autocorr=0.85):
    """AR(1) process — realistic sensor drift rather than pure noise"""
    vals = [np.random.normal(mean, std)]
    for _ in range(n - 1):
        noise = np.random.normal(0, std * (1 - autocorr))
        vals.append(autocorr * vals[-1] + (1 - autocorr) * mean + noise)
    return np.array(vals)

print(f"\n[1/5] Generating {N_SAMPLES:,} samples with AR(1) autocorrelation …")
data = {}
for col, (mean, std) in SENSOR_NORMAL.items():
    data[col] = generate_sensor_stream(N_SAMPLES, mean, std)
df = pd.DataFrame(data)

# ── Realistic time stamps (10-min intervals) ──────────────────
start = datetime(2024, 1, 1, 6, 0)
df["timestamp"] = [start + timedelta(minutes=10*i) for i in range(N_SAMPLES)]
df["hour"]      = pd.to_datetime(df["timestamp"]).dt.hour
df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

# Machine assignment (cyclic)
machine_ids = [m["id"] for m in MACHINES]
df["machine_id"] = [machine_ids[i % len(machine_ids)] for i in range(N_SAMPLES)]

# ── Shift baseline per machine (each machine runs differently) ─
print("[2/5] Applying per-machine operating baselines …")
machine_shifts = {
    "MCH-001": {"temperature": -2, "vibration": -0.1, "rpm": +50},
    "MCH-002": {"temperature": -7, "vibration": +0.3, "pressure": +6.2},
    "MCH-003": {"temperature": +6, "vibration": +0.2, "rpm": +800},
    "MCH-004": {"temperature":-17, "vibration": -0.6, "pressure": -3.8, "rpm": -2350},
    "MCH-005": {"temperature": -4, "vibration": -0.1, "pressure": +0.5, "rpm": -1000},
    "MCH-006": {"temperature": +8, "vibration": +0.1, "rpm": +200},
}
for mch_id, shifts in machine_shifts.items():
    mask = df["machine_id"] == mch_id
    for col, delta in shifts.items():
        df.loc[mask, col] += delta

# ── Inject realistic daily cycles (temperature rises during day) ─
print("[3/5] Adding realistic daily load cycles …")
df["temperature"] += 3 * np.sin(2 * np.pi * (df["hour"] - 6) / 16).clip(0)
df["current"]     += 1.5 * np.sin(2 * np.pi * (df["hour"] - 8) / 10).clip(0)
df["rpm"]         += 60  * np.sin(2 * np.pi * (df["hour"] - 7) / 12).clip(0)

# ── Inject faults ─────────────────────────────────────────────
print(f"[4/5] Injecting faults ({FAILURE_RATE*100:.0f}% failure rate) …")
n_fail = int(N_SAMPLES * FAILURE_RATE)
fail_idx = np.random.choice(df.index, n_fail, replace=False)
df["failure"]    = 0
df["fault_type"] = "none"
fault_names = list(FAULT_PROFILES.keys())

for k, idx in enumerate(fail_idx):
    fault = fault_names[k % len(fault_names)]
    for sensor, delta in FAULT_PROFILES[fault].items():
        df.at[idx, sensor] += delta + np.random.normal(0, abs(delta) * 0.1)
    df.at[idx, "failure"]    = 1
    df.at[idx, "fault_type"] = fault

# ── Clip to physical bounds ────────────────────────────────────
bounds = {
    "temperature":(20,200),"vibration":(0,15),"pressure":(0,20),
    "rpm":(50,5000),"current":(1,40),"voltage":(300,480),
    "oil_viscosity":(5,100),"noise_level":(30,120)
}
for col, (lo, hi) in bounds.items():
    df[col] = df[col].clip(lo, hi)

# ── Save ──────────────────────────────────────────────────────
print("[5/5] Saving dataset …")
out = os.path.join(DATA_DIR, "raw_sensor_data.csv")
df.to_csv(out, index=False)

n_fail_actual = df["failure"].sum()
print(f"\n  Total samples : {len(df):,}")
print(f"  Normal (0)    : {len(df)-n_fail_actual:,} ({(len(df)-n_fail_actual)/len(df)*100:.1f}%)")
print(f"  Failure (1)   : {n_fail_actual:,} ({n_fail_actual/len(df)*100:.1f}%)")
print(f"  Saved → {out}")

# ── Plot ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
fig.suptitle("Step 1 — Sensor Data Generation Analysis", color="white", fontsize=15, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

for i, col in enumerate(SENSOR_COLS):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.set_facecolor("#111820")
    for lbl, clr, nm in [(0,"#22c55e","Normal"),(1,"#ef4444","Failure")]:
        ax.hist(df[df.failure==lbl][col], bins=50, alpha=0.65,
                color=clr, label=nm, density=True)
    ax.set_title(col.replace("_"," ").title(), color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#64748b", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#0d1117", framealpha=0.3)

# Class balance
ax = fig.add_subplot(gs[2, 0])
ax.set_facecolor("#111820")
ax.pie([len(df)-n_fail_actual, n_fail_actual],
       labels=["Normal","Failure"], colors=["#22c55e","#ef4444"],
       autopct="%1.1f%%", textprops={"color":"white","fontsize":9})
ax.set_title("Class Balance", color="white", fontsize=10, fontweight="bold")

# Fault breakdown
ax = fig.add_subplot(gs[2, 1])
ax.set_facecolor("#111820")
fc = df[df.failure==1]["fault_type"].value_counts()
ax.barh(fc.index, fc.values,
        color=["#f97316","#ef4444","#eab308","#38bdf8","#a78bfa"])
ax.set_title("Fault Distribution", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b", labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# Temp over time for one machine
ax = fig.add_subplot(gs[2, 2])
ax.set_facecolor("#111820")
m1 = df[df.machine_id=="MCH-001"].head(200)
ax.plot(m1["temperature"].values, color="#f97316", lw=1)
fail_pts = m1[m1.failure==1]
ax.scatter(fail_pts.index - m1.index[0], fail_pts["temperature"],
           color="#ef4444", s=20, zorder=5, label="Fault")
ax.set_title("MCH-001 Temp w/ Faults", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b", labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")
ax.legend(fontsize=7, labelcolor="white", facecolor="#0d1117", framealpha=0.3)

plt.savefig(os.path.join(REPORT_DIR, "step1_data_overview.png"),
            dpi=130, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Plot → reports/step1_data_overview.png")
print("\n✅  STEP 1 COMPLETE\n")
