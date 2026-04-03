# =============================================================================
#  step3_train_models.py  —  PredictAI
#  Trains Random Forest · Isolation Forest · SVM  (no TF required)
#  Output: models/random_forest.pkl, isolation_forest.pkl, svm_model.pkl
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib, time, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from config import *

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  STEP 3 — MODEL TRAINING")
print("  Random Forest | Isolation Forest | SVM | Gradient Boosting")
print("=" * 60)

X_train = pd.read_csv(os.path.join(DATA_DIR,"X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR,"y_train.csv"))["failure"]
X_test  = pd.read_csv(os.path.join(DATA_DIR,"X_test.csv"))
y_test  = pd.read_csv(os.path.join(DATA_DIR,"y_test.csv"))["failure"]
feature_cols = joblib.load(os.path.join(MODEL_DIR,"feature_names.pkl"))

print(f"\n[DATA] Train {X_train.shape}  Test {X_test.shape}")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
log = {}

# ── MODEL 1: Random Forest ────────────────────────────────────
print("\n" + "─"*55)
print("  MODEL 1/4  ·  Random Forest Classifier")
print("─"*55)
t0 = time.time()
rf = RandomForestClassifier(**RF_PARAMS)
cv = cross_val_score(rf, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1)
print(f"  CV F1 : {[f'{v:.4f}' for v in cv]}")
print(f"  Mean  : {cv.mean():.4f} ± {cv.std():.4f}")
rf.fit(X_train, y_train)
t_rf = time.time() - t0
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"  Time  : {t_rf:.1f}s")
print("  Top features:")
for feat, imp in fi.head(5).items():
    print(f"    {feat:<40} {imp:.4f}")
joblib.dump(rf, os.path.join(MODEL_DIR,"random_forest.pkl"))
log["Random Forest"] = {"cv":cv.tolist(),"mean_f1":cv.mean(),"time":t_rf}
print("  ✓ Saved → models/random_forest.pkl")

# ── MODEL 2: Isolation Forest ─────────────────────────────────
print("\n" + "─"*55)
print("  MODEL 2/4  ·  Isolation Forest (Anomaly Detection)")
print("─"*55)
t0 = time.time()
iso = IsolationForest(**IF_PARAMS)
X_normal = X_train[y_train == 0]
iso.fit(X_normal)
t_iso = time.time() - t0
iso_pred = np.where(iso.predict(X_test) == -1, 1, 0)
iso_acc  = (iso_pred == y_test).mean()
print(f"  Trained on {len(X_normal):,} normal samples (unsupervised)")
print(f"  Quick accuracy on test : {iso_acc:.4f}")
print(f"  Time : {t_iso:.1f}s")
joblib.dump(iso, os.path.join(MODEL_DIR,"isolation_forest.pkl"))
log["Isolation Forest"] = {"acc":iso_acc,"time":t_iso}
print("  ✓ Saved → models/isolation_forest.pkl")

# ── MODEL 3: SVM ──────────────────────────────────────────────
print("\n" + "─"*55)
print("  MODEL 3/4  ·  Support Vector Machine (RBF)")
print("─"*55)
SUB = 6000
idx = np.random.choice(len(X_train), min(SUB, len(X_train)), replace=False)
X_svm = X_train.iloc[idx]; y_svm = y_train.iloc[idx]
print(f"  Subsample: {len(X_svm):,} rows")
t0 = time.time()
svm = SVC(**SVM_PARAMS)
cv_svm = cross_val_score(svm, X_svm, y_svm, cv=5, scoring="f1", n_jobs=-1)
print(f"  CV F1 : {[f'{v:.4f}' for v in cv_svm]}")
print(f"  Mean  : {cv_svm.mean():.4f} ± {cv_svm.std():.4f}")
svm.fit(X_svm, y_svm)
t_svm = time.time() - t0
print(f"  Time  : {t_svm:.1f}s")
joblib.dump(svm, os.path.join(MODEL_DIR,"svm_model.pkl"))
log["SVM"] = {"cv":cv_svm.tolist(),"mean_f1":cv_svm.mean(),"time":t_svm}
print("  ✓ Saved → models/svm_model.pkl")

# ── MODEL 4: Gradient Boosting ────────────────────────────────
print("\n" + "─"*55)
print("  MODEL 4/4  ·  Gradient Boosting (XGBoost-style)")
print("─"*55)
t0 = time.time()
gb = GradientBoostingClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.08,
    subsample=0.8, random_state=RANDOM_SEED)
cv_gb = cross_val_score(gb, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1)
print(f"  CV F1 : {[f'{v:.4f}' for v in cv_gb]}")
print(f"  Mean  : {cv_gb.mean():.4f} ± {cv_gb.std():.4f}")
gb.fit(X_train, y_train)
t_gb = time.time() - t0
print(f"  Time  : {t_gb:.1f}s")
joblib.dump(gb, os.path.join(MODEL_DIR,"gradient_boost.pkl"))
log["Gradient Boosting"] = {"cv":cv_gb.tolist(),"mean_f1":cv_gb.mean(),"time":t_gb}
print("  ✓ Saved → models/gradient_boost.pkl")

joblib.dump(log, os.path.join(MODEL_DIR,"training_log.pkl"))

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*55)
print("  TRAINING SUMMARY")
print("="*55)
for name, info in log.items():
    f1 = info.get("mean_f1", info.get("acc", 0))
    print(f"  {name:<25} F1/Acc={f1:.4f}  Time={info['time']:.1f}s")

# ── Plots ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12), facecolor="#0d1117")
fig.suptitle("Step 3 — Model Training Analysis", color="white", fontsize=14, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Feature importance (RF)
ax = fig.add_subplot(gs[0, 0:2])
ax.set_facecolor("#111820")
top = fi.head(15)
clrs = ["#f97316"]*5 + ["#38bdf8"]*5 + ["#64748b"]*5
ax.barh(top.index[::-1], top.values[::-1], color=clrs[::-1])
ax.set_title("Random Forest — Top 15 Feature Importances", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b", labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# CV comparison
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor("#111820")
models_cv = {"RF": log["Random Forest"]["cv"], "SVM": log["SVM"]["cv"], "GB": log["Gradient Boosting"]["cv"]}
bp = ax.boxplot(list(models_cv.values()), labels=list(models_cv.keys()),
                patch_artist=True,
                boxprops=dict(facecolor="#1e2d3d",color="#38bdf8"),
                medianprops=dict(color="#f97316",linewidth=2),
                whiskerprops=dict(color="#64748b"),
                capprops=dict(color="#64748b"))
ax.set_title("CV F1 Distribution", color="white", fontsize=10, fontweight="bold")
ax.set_ylabel("F1 Score", color="#94a3b8")
ax.tick_params(colors="#64748b")
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# Learning curve (RF)
ax = fig.add_subplot(gs[1, 0:2])
ax.set_facecolor("#111820")
sizes = np.linspace(0.1, 1.0, 6)
train_sizes_abs, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1),
    X_train, y_train, train_sizes=sizes, cv=3, scoring="f1", n_jobs=-1)
ax.plot(train_sizes_abs, train_scores.mean(axis=1), "o-", color="#f97316", lw=2, label="Train F1")
ax.fill_between(train_sizes_abs, train_scores.mean(1)-train_scores.std(1),
                train_scores.mean(1)+train_scores.std(1), alpha=.15, color="#f97316")
ax.plot(train_sizes_abs, val_scores.mean(axis=1), "o--", color="#38bdf8", lw=2, label="Val F1")
ax.fill_between(train_sizes_abs, val_scores.mean(1)-val_scores.std(1),
                val_scores.mean(1)+val_scores.std(1), alpha=.15, color="#38bdf8")
ax.set_title("RF Learning Curve", color="white", fontsize=10, fontweight="bold")
ax.set_xlabel("Training Samples", color="#94a3b8")
ax.set_ylabel("F1 Score", color="#94a3b8")
ax.tick_params(colors="#64748b")
ax.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3)
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")

# Training time comparison
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor("#111820")
names = list(log.keys())
times = [log[n]["time"] for n in names]
ax.barh(names, times, color=["#f97316","#38bdf8","#a78bfa","#22c55e"])
ax.set_title("Training Time (seconds)", color="white", fontsize=10, fontweight="bold")
ax.tick_params(colors="#64748b")
for sp in ax.spines.values(): sp.set_edgecolor("#1e2d3d")
for i, t in enumerate(times):
    ax.text(t+0.5, i, f"{t:.1f}s", va="center", color="white", fontsize=9)

plt.savefig(os.path.join(REPORT_DIR,"step3_training.png"),
            dpi=130, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("\n  Plot → reports/step3_training.png")
print("\n✅  STEP 3 COMPLETE\n")
