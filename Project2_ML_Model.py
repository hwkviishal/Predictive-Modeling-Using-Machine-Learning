# ============================================================
#   Project 2 — Predictive Modeling Using Machine Learning
#   Algorithms : Linear Regression, Decision Tree, Random Forest
#   Evaluation : Accuracy, Confusion Matrix, ROC Curve
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# ── Colour palette ─────────────────────────────────────────
TEAL   = "#0F6E56"
BLUE   = "#185FA5"
CORAL  = "#D85A30"
AMBER  = "#BA7517"
PURPLE = "#534AB7"
GRAY   = "#5F5E5A"
LIGHT  = "#F1EFE8"
WHITE  = "#FFFFFF"
ALGO_COLORS = [TEAL, BLUE, CORAL, AMBER, PURPLE]

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.facecolor"  : WHITE,
    "figure.facecolor": WHITE,
    "axes.grid"       : True,
    "grid.color"      : "#E8E6DF",
    "grid.linewidth"  : 0.6,
})

# ══════════════════════════════════════════════════════════
#  1.  DATASET
# ══════════════════════════════════════════════════════════
print("=" * 58)
print("  PROJECT 2 — Predictive Modeling Using Machine Learning")
print("=" * 58)

data   = load_breast_cancer()
X      = pd.DataFrame(data.data,   columns=data.feature_names)
y      = pd.Series(data.target,    name="diagnosis")   # 0=malignant 1=benign

print(f"\n  Dataset : Breast Cancer Wisconsin (UCI)")
print(f"  Samples : {X.shape[0]}   |   Features : {X.shape[1]}")
print(f"  Classes : Malignant={sum(y==0)}  Benign={sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler  = StandardScaler()
Xtr_sc  = scaler.fit_transform(X_train)
Xte_sc  = scaler.transform(X_test)

print(f"\n  Train : {len(X_train)} samples  |  Test : {len(X_test)} samples")

# ══════════════════════════════════════════════════════════
#  2.  MODELS
# ══════════════════════════════════════════════════════════
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    "k-NN"                : KNeighborsClassifier(n_neighbors=5),
    "SVM"                 : SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}

results   = {}
cms       = {}
roc_data  = {}

print("\n  Training models …\n")

for name, model in models.items():
    model.fit(Xtr_sc, y_train)
    y_pred  = model.predict(Xte_sc)
    y_prob  = model.predict_proba(Xte_sc)[:, 1]

    acc   = accuracy_score(y_test,  y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test,    y_pred)
    f1    = f1_score(y_test,        y_pred)
    cv    = cross_val_score(model, Xtr_sc, y_train, cv=5, scoring="accuracy").mean()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)

    results[name]  = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, cv=cv, auc=roc_auc)
    cms[name]      = confusion_matrix(y_test, y_pred)
    roc_data[name] = (fpr, tpr, roc_auc)

    print(f"  [{name:22s}]  Acc={acc:.3f}  F1={f1:.3f}  AUC={roc_auc:.3f}")

df_res = pd.DataFrame(results).T.round(3)
print("\n" + df_res.to_string())

# ══════════════════════════════════════════════════════════
#  3.  VISUALISATIONS
# ══════════════════════════════════════════════════════════
names  = list(models.keys())
short  = ["Log.Reg.", "Dec.Tree", "Rnd.Forest", "k-NN", "SVM"]
colors = ALGO_COLORS

fig = plt.figure(figsize=(20, 26), facecolor=WHITE)
fig.suptitle(
    "Project 2 — Predictive Modeling Using Machine Learning",
    fontsize=18, fontweight="bold", color="#2C2C2A", y=0.98
)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1 : Accuracy bar chart ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
accs = [results[n]["accuracy"] for n in names]
bars = ax1.bar(short, accs, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=0.8)
ax1.set_ylim(0.88, 1.00)
ax1.set_title("Test Accuracy", fontweight="bold", color="#2C2C2A")
ax1.set_ylabel("Accuracy")
ax1.tick_params(axis="x", labelsize=8)
for b, v in zip(bars, accs):
    ax1.text(b.get_x()+b.get_width()/2, v+0.001, f"{v:.3f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2C2C2A")

# ── Panel 2 : F1 Score ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
f1s = [results[n]["f1"] for n in names]
bars2 = ax2.bar(short, f1s, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=0.8)
ax2.set_ylim(0.88, 1.00)
ax2.set_title("F1 Score", fontweight="bold", color="#2C2C2A")
ax2.set_ylabel("F1")
ax2.tick_params(axis="x", labelsize=8)
for b, v in zip(bars2, f1s):
    ax2.text(b.get_x()+b.get_width()/2, v+0.001, f"{v:.3f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2C2C2A")

# ── Panel 3 : AUC bar chart ────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
aucs = [results[n]["auc"] for n in names]
bars3 = ax3.bar(short, aucs, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=0.8)
ax3.set_ylim(0.94, 1.002)
ax3.set_title("ROC-AUC Score", fontweight="bold", color="#2C2C2A")
ax3.set_ylabel("AUC")
ax3.tick_params(axis="x", labelsize=8)
for b, v in zip(bars3, aucs):
    ax3.text(b.get_x()+b.get_width()/2, v+0.0005, f"{v:.3f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2C2C2A")

# ── Panel 4 : ROC Curves (all models) ─────────────────────
ax4 = fig.add_subplot(gs[1, :2])
ax4.plot([0,1],[0,1],"--", color="#CCCCCC", lw=1, label="Random (AUC=0.500)")
for i, name in enumerate(names):
    fpr, tpr, ra = roc_data[name]
    ax4.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name}  (AUC={ra:.3f})")
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curves — All Models", fontweight="bold", color="#2C2C2A")
ax4.legend(fontsize=8, loc="lower right")
ax4.set_xlim([0,1]); ax4.set_ylim([0,1.01])

# ── Panel 5 : Multi-metric grouped bar ────────────────────
ax5 = fig.add_subplot(gs[1, 2])
metrics   = ["accuracy","precision","recall","f1"]
m_labels  = ["Accuracy","Precision","Recall","F1"]
x         = np.arange(len(metrics))
width     = 0.15
for i, name in enumerate(names):
    vals = [results[name][m] for m in metrics]
    ax5.bar(x + i*width, vals, width, label=short[i], color=colors[i], zorder=3, edgecolor="white")
ax5.set_xticks(x + width*2)
ax5.set_xticklabels(m_labels, fontsize=8)
ax5.set_ylim(0.88, 1.01)
ax5.set_title("Metrics Comparison", fontweight="bold", color="#2C2C2A")
ax5.legend(fontsize=7, ncol=1)

# ── Panels 6-10 : Confusion Matrices ──────────────────────
cm_axes = [
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    fig.add_subplot(gs[2, 2]),
    fig.add_subplot(gs[3, 0]),
    fig.add_subplot(gs[3, 1]),
]
for ax, name, col in zip(cm_axes, names, colors):
    cm = cms[name]
    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax,
        cmap=sns.light_palette(col, as_cmap=True),
        linewidths=0.5, linecolor="white",
        xticklabels=["Malignant","Benign"],
        yticklabels=["Malignant","Benign"],
        cbar=False, annot_kws={"size":12,"weight":"bold"}
    )
    acc = results[name]["accuracy"]
    ax.set_title(f"{name}\nAcc={acc:.3f}", fontweight="bold", color="#2C2C2A", fontsize=9)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual",    fontsize=8)
    ax.tick_params(labelsize=8)

# ── Panel 11 : Feature Importance (Random Forest) ─────────
ax_fi = fig.add_subplot(gs[3, 2])
rf        = models["Random Forest"]
imp       = rf.feature_importances_
top_idx   = np.argsort(imp)[-10:]
top_names = [data.feature_names[i] for i in top_idx]
top_vals  = imp[top_idx]
ax_fi.barh(top_names, top_vals, color=TEAL, edgecolor="white", zorder=3)
ax_fi.set_title("Top 10 Feature Importance\n(Random Forest)", fontweight="bold", color="#2C2C2A", fontsize=9)
ax_fi.set_xlabel("Importance Score", fontsize=8)
ax_fi.tick_params(labelsize=7)
for i, v in enumerate(top_vals):
    ax_fi.text(v+0.002, i, f"{v:.3f}", va="center", fontsize=7, color="#2C2C2A")

plt.savefig("/mnt/user-data/outputs/Project2_ML_Results.png",
            dpi=180, bbox_inches="tight", facecolor=WHITE)
print("\n  Chart saved → Project2_ML_Results.png")

# ══════════════════════════════════════════════════════════
#  4.  BEST MODEL DETAIL
# ══════════════════════════════════════════════════════════
best_name = max(results, key=lambda n: results[n]["accuracy"])
best_model = models[best_name]
y_pred_best = best_model.predict(Xte_sc)

print("\n" + "="*58)
print(f"  BEST MODEL : {best_name}")
print("="*58)
print(classification_report(y_test, y_pred_best,
      target_names=["Malignant","Benign"]))
print("  Done. All outputs saved.")
