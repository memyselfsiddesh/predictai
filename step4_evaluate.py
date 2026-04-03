# =============================================================================
#  step4_evaluate.py  —  PredictAI
#  Full evaluation: accuracy, F1, AUC-ROC, confusion matrices, ROC curves
#  Output: reports/step4_evaluation_report.txt, step4_metrics.csv
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
import joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
from config import *

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  STEP 4 — COMPREHENSIVE MODEL EVALUATION")
print("=" * 60)

X_test  = pd.read_csv(os.path.join(DATA_DIR,"X_test.csv"))
y_test  = pd.read_csv(os.path.join(DATA_DIR,"y_test.csv"))["failure"]
print(f"\n[DATA] Test set: {X_test.shape[0]:,} samples  "
      f"(0={sum(y_test==0):,}  1={sum(y_test==1):,})")

rf  = joblib.load(os.path.join(MODEL_DIR,"random_forest.pkl"))
iso = joblib.load(os.path.join(MODEL_DIR,"isolation_forest.pkl"))
svm = joblib.load(os.path.join(MODEL_DIR,"svm_model.pkl"))
gb  = joblib.load(os.path.join(MODEL_DIR,"gradient_boost.pkl"))
print("[MODELS] ✓ 4 models loaded")

# ── Predictions ───────────────────────────────────────────────
rf_prob  = rf.predict_proba(X_test)[:,1]
rf_pred  = (rf_prob >= FAILURE_PROB_THRESHOLD).astype(int)

iso_score = -iso.decision_function(X_test)
iso_pred  = np.where(iso.predict(X_test)==-1, 1, 0)
iso_prob  = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9)

svm_prob = svm.predict_proba(X_test)[:,1]
svm_pred = (svm_prob >= FAILURE_PROB_THRESHOLD).astype(int)

gb_prob  = gb.predict_proba(X_test)[:,1]
gb_pred  = (gb_prob >= FAILURE_PROB_THRESHOLD).astype(int)

# Ensemble (weighted)
ens_prob = 0.40*rf_prob + 0.25*gb_prob + 0.20*svm_prob + 0.15*iso_prob
ens_pred = (ens_prob >= FAILURE_PROB_THRESHOLD).astype(int)

# ── Metrics ───────────────────────────────────────────────────
def metrics(yt, yp, yprob, name):
    cm = confusion_matrix(yt, yp); tn,fp,fn,tp = cm.ravel()
    return {
        "Model":        name,
        "Accuracy":     round(accuracy_score(yt,yp),4),
        "Precision":    round(precision_score(yt,yp,zero_division=0),4),
        "Recall":       round(recall_score(yt,yp,zero_division=0),4),
        "F1-Score":     round(f1_score(yt,yp,zero_division=0),4),
        "AUC-ROC":      round(roc_auc_score(yt,yprob),4),
        "Avg Precision":round(average_precision_score(yt,yprob),4),
        "MCC":          round(matthews_corrcoef(yt,yp),4),
        "Specificity":  round(tn/(tn+fp),4) if (tn+fp)>0 else 0,
        "TP":tp,"TN":tn,"FP":fp,"FN":fn,
    }

all_m = [
    metrics(y_test, rf_pred,  rf_prob,  "Random Forest"),
    metrics(y_test, iso_pred, iso_prob, "Isolation Forest"),
    metrics(y_test, svm_pred, svm_prob, "SVM (RBF)"),
    metrics(y_test, gb_pred,  gb_prob,  "Gradient Boosting"),
    metrics(y_test, ens_pred, ens_prob, "Ensemble"),
]
df_m = pd.DataFrame(all_m).set_index("Model")

# ── Console report ────────────────────────────────────────────
lines = []
def rpt(s=""):
    print(s); lines.append(s)

rpt("\n" + "="*60)
rpt("  PREDICTIVE MAINTENANCE — EVALUATION REPORT")
rpt("="*60)
rpt(f"\n  {'Model':<22}{'Accuracy':<12}{'Precision':<12}{'Recall':<10}{'F1':<10}{'AUC-ROC'}")
rpt("  " + "─"*70)
for idx,row in df_m[["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]].iterrows():
    star = " ★" if idx=="Ensemble" else "  "
    rpt(f"  {star}{idx:<20}" + "".join(f"{v:<12.4f}" for v in row.values))

best = df_m["F1-Score"].idxmax()
rpt(f"\n  ★  BEST MODEL: {best}  (F1={df_m.loc[best,'F1-Score']:.4f}  AUC={df_m.loc[best,'AUC-ROC']:.4f})")

for name, yt, yp in [("Random Forest",y_test,rf_pred),("SVM (RBF)",y_test,svm_pred),
                      ("Gradient Boosting",y_test,gb_pred),("Ensemble",y_test,ens_pred)]:
    rpt(f"\n  ── {name} — Classification Report ──")
    cr = classification_report(yt, yp, target_names=["Normal","Failure"], digits=4)
    for l in cr.split("\n"): rpt("  "+l)

rpt("="*60)

with open(os.path.join(REPORT_DIR,"step4_evaluation_report.txt"),"w", encoding="utf-8") as f:
    f.write("\n".join(lines))
df_m.to_csv(os.path.join(REPORT_DIR,"step4_metrics.csv"))

# ── Save metrics JSON for Flask API ──────────────────────────
import json
metrics_export = {}
for idx, row in df_m.iterrows():
    metrics_export[idx] = {k: float(v) if isinstance(v,(int,float,np.number)) else v
                           for k,v in row.items()}
with open(os.path.join(MODEL_DIR,"eval_metrics.json"),"w") as f:
    json.dump(metrics_export, f, indent=2)
print("\n  ✓ Saved metrics → models/eval_metrics.json (used by Flask API)")

# ── Plots ─────────────────────────────────────────────────────
MODEL_COLORS = {"Random Forest":"#f97316","Isolation Forest":"#38bdf8",
                "SVM (RBF)":"#a78bfa","Gradient Boosting":"#22c55e","Ensemble":"#fbbf24"}

fig = plt.figure(figsize=(24, 18), facecolor="#0d1117")
fig.suptitle("Step 4 — Full Model Evaluation Report", color="white", fontsize=16, y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Confusion matrices (top row — 4 models)
for col, (name, yt, yp) in enumerate([
    ("Random Forest",    y_test, rf_pred),
    ("SVM (RBF)",        y_test, svm_pred),
    ("Gradient Boosting",y_test, gb_pred),
    ("Ensemble",         y_test, ens_pred),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor("#111820")
    cm = confusion_matrix(yt, yp)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="YlOrRd",
                linewidths=1, linecolor="#0d1117", cbar=False,
                annot_kws={"size":13,"weight":"bold","color":"white"},
                xticklabels=["Normal","Failure"],
                yticklabels=["Normal","Failure"])
    f1v = df_m.loc[name,"F1-Score"] if name in df_m.index else 0
    ax.set_title(f"{name}\nF1={f1v:.4f}",
                 color=MODEL_COLORS.get(name,"white"), fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Actual",    color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)

# ROC curves
ax_roc = fig.add_subplot(gs[1, 0:2])
ax_roc.set_facecolor("#111820")
for name, yt, yp in [("Random Forest",y_test,rf_prob),
                      ("SVM (RBF)",y_test,svm_prob),
                      ("Gradient Boosting",y_test,gb_prob),
                      ("Ensemble",y_test,ens_prob)]:
    fpr, tpr, _ = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)
    ax_roc.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                label=f"{name} (AUC={auc:.4f})")
ax_roc.plot([0,1],[0,1],"--",color="#475569",lw=1,label="Random")
ax_roc.set_title("ROC Curves — All Models", color="white", fontsize=11, fontweight="bold")
ax_roc.set_xlabel("FPR", color="#94a3b8"); ax_roc.set_ylabel("TPR", color="#94a3b8")
ax_roc.tick_params(colors="#64748b")
ax_roc.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3, loc="lower right")
for sp in ax_roc.spines.values(): sp.set_edgecolor("#1e2d3d")

# Precision-Recall curves
ax_pr = fig.add_subplot(gs[1, 2:4])
ax_pr.set_facecolor("#111820")
for name, yt, yp in [("Random Forest",y_test,rf_prob),
                      ("SVM (RBF)",y_test,svm_prob),
                      ("Gradient Boosting",y_test,gb_prob),
                      ("Ensemble",y_test,ens_prob)]:
    p, r, _ = precision_recall_curve(yt, yp)
    ap = average_precision_score(yt, yp)
    ax_pr.plot(r, p, color=MODEL_COLORS[name], lw=2, label=f"{name} (AP={ap:.4f})")
ax_pr.axhline(y_test.mean(), color="#475569", lw=1, linestyle="--")
ax_pr.set_title("Precision-Recall Curves", color="white", fontsize=11, fontweight="bold")
ax_pr.set_xlabel("Recall", color="#94a3b8"); ax_pr.set_ylabel("Precision", color="#94a3b8")
ax_pr.tick_params(colors="#64748b")
ax_pr.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3)
for sp in ax_pr.spines.values(): sp.set_edgecolor("#1e2d3d")

# Metric bar comparison
ax_bar = fig.add_subplot(gs[2, 0:3])
ax_bar.set_facecolor("#111820")
mkeys = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
mnames = [n for n in df_m.index if n != "Isolation Forest"]
x = np.arange(len(mkeys)); w = 0.18
offsets = np.linspace(-0.27, 0.27, len(mnames))
for mn, off in zip(mnames, offsets):
    vals = df_m.loc[mn, mkeys].values
    ax_bar.bar(x + off, vals, w, color=MODEL_COLORS.get(mn,"#64748b"), alpha=.85, label=mn)
ax_bar.set_xticks(x); ax_bar.set_xticklabels(mkeys, color="#94a3b8", fontsize=9)
ax_bar.set_ylim(0, 1.1)
ax_bar.set_title("Metric Comparison — All Models", color="white", fontsize=11, fontweight="bold")
ax_bar.tick_params(colors="#64748b")
ax_bar.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3)
for sp in ax_bar.spines.values(): sp.set_edgecolor("#1e2d3d")

# Probability distribution (Ensemble)
ax_pb = fig.add_subplot(gs[2, 3])
ax_pb.set_facecolor("#111820")
for lbl, clr, nm in [(0,"#22c55e","Normal"),(1,"#ef4444","Failure")]:
    ax_pb.hist(ens_prob[y_test==lbl], bins=50, alpha=.7, color=clr, label=nm, density=True)
ax_pb.axvline(FAILURE_PROB_THRESHOLD, color="#f97316", lw=2, linestyle="--")
ax_pb.set_title("Ensemble Prob. Distribution", color="white", fontsize=10, fontweight="bold")
ax_pb.tick_params(colors="#64748b")
ax_pb.legend(fontsize=8, labelcolor="white", facecolor="#0d1117", framealpha=.3)
for sp in ax_pb.spines.values(): sp.set_edgecolor("#1e2d3d")

plt.savefig(os.path.join(REPORT_DIR,"step4_evaluation.png"),
            dpi=130, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Plot → reports/step4_evaluation.png")
print("\n✅  STEP 4 COMPLETE\n")