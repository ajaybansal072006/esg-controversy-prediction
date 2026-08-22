"""
============================================================
ESG Controversy Prediction — SVM + SHAP
============================================================
Dataset : esg_controversy_features.csv

Why SVM instead of XGBoost?
  • SVM finds the maximum-margin hyperplane — excellent for
    structured financial data with moderate dimensionality.
  • RBF kernel captures non-linear feature interactions
    without explicit feature engineering.
  • More interpretable decision boundary than ensemble trees.
  • SHAP (KernelSHAP) works on any black-box model.

⚠️  SVM does NOT natively support SHAP TreeExplainer.
    We use shap.KernelExplainer (model-agnostic) which is
    slower but correct. We sample 100 background + 150 test
    points to keep runtime reasonable.

Steps:
  1.  Load engineered dataset
  2.  Drop identifier columns
  3.  Separate target
  4.  Impute NaN values
  5.  Walk-forward temporal train/test split
  6.  Scale features (mandatory for SVM)
  7.  Handle class imbalance with class_weight='balanced'
  8.  Train SVM (RBF kernel) with GridSearchCV
  9.  5-Fold Stratified Cross-Validation
 10.  Evaluation metrics + dashboard plot
 11.  SHAP analysis  (KernelSHAP — beeswarm, bar, waterfall)
 12.  Company-level risk driver CSV (SHAP)
 13.  Save model + metrics CSV
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR1 = os.path.join(BASE_DIR, "output")
# Create output folder inside it
OUTPUT_DIR = os.path.join(BASE_DIR1, "svm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import sys
sys.stdout.reconfigure(encoding="utf-8")

import shap
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)

# ── Colour palette ────────────────────────────────────────────
NAVY      = "#1B3A6B"
TEAL      = "#0D7377"
RED       = "#B91C1C"
GREEN     = "#15803D"
AMBER     = "#D97706"
BLUE      = "#2563EB"
BG        = "#F1F5F9"
SVM_COLOR = "#7C3AED"

# ── SHAP sample sizes ──────────────────────────────────────────
N_SHAP_BG      = 100
N_SHAP_EXPLAIN = 150

print("=" * 60)
print("  ESG Controversy Prediction — SVM + SHAP")
print("=" * 60)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/10] Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "esg_controversy_features.csv"))
print(f"       Shape    : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"       Years    : {df['Year'].min()} – {df['Year'].max()}")
print(f"       Companies: {df['CompanyID'].nunique():,}")


# ============================================================
# STEP 2: DROP IDENTIFIER COLUMNS
# ============================================================
print("\n[2/10] Dropping identifier columns (CompanyID, Year)...")
ID_COLS  = ["CompanyID", "Year"]
df_model = df.drop(columns=ID_COLS)


# ============================================================
# STEP 3: SEPARATE TARGET
# ============================================================
print("\n[3/10] Separating target column...")
TARGET = "ESG_Controversy"
y = df_model[TARGET]
X = df_model.drop(columns=[TARGET])

bool_cols = X.select_dtypes(include="bool").columns.tolist()
if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)

feature_names = X.columns.tolist()
print(f"       Feature count : {len(feature_names)}")
print(f"       Class 0 (No Controversy) : {(y==0).sum():,}  ({(y==0).mean():.1%})")
print(f"       Class 1 (Controversy)    : {(y==1).sum():,}  ({(y==1).mean():.1%})")


# ============================================================
# STEP 4: IMPUTE MISSING VALUES
# ============================================================
print("\n[4/10] Imputing missing values...")
zero_impute_cols = [
    "Revenue_YoY", "Revenue_2Y_Growth", "Margin_YoY_Change",
    "MarketCap_YoY", "GrowthRate_YoY_Change", "Margin_Volatility_3Y",
    "Revenue_Volatility_3Y", "MCap_Volatility_3Y",
    "Margin_Deterioration", "Revenue_Surprise",
]
for col in zero_impute_cols:
    if col in X.columns:
        n = X[col].isnull().sum()
        if n > 0:
            X[col] = X[col].fillna(0)
            print(f"       → {col:<30} filled {n:,} NaNs with 0")

for roll_col, src_col in [("Avg_Margin_3Y", "ProfitMargin"), ("Avg_Revenue_3Y", "Revenue")]:
    if roll_col in X.columns and src_col in X.columns:
        n = X[roll_col].isnull().sum()
        if n > 0:
            X[roll_col] = X[roll_col].fillna(X[src_col])
            print(f"       → {roll_col:<30} filled {n:,} NaNs from {src_col}")

if "GrowthRate" in X.columns:
    n = X["GrowthRate"].isnull().sum()
    if n > 0:
        X["GrowthRate"] = X["GrowthRate"].fillna(X["GrowthRate"].median())
if "Growth_vs_Industry" in X.columns:
    X["Growth_vs_Industry"] = X["Growth_vs_Industry"].fillna(0)

print(f"       Total NaNs remaining : {X.isnull().sum().sum()}")


# ============================================================
# STEP 5: WALK-FORWARD TEMPORAL TRAIN / TEST SPLIT
# ============================================================
print("\n[5/10] Walk-forward temporal train/test split...")
SPLIT_YEAR = 2023
train_mask = df["Year"] < SPLIT_YEAR
test_mask  = df["Year"] >= SPLIT_YEAR

X_train_raw, y_train = X[train_mask].copy(), y[train_mask].copy()
X_test_raw,  y_test  = X[test_mask].copy(),  y[test_mask].copy()

X_test_raw = X_test_raw.reset_index(drop=True)
y_test      = y_test.reset_index(drop=True)
df_test     = df[test_mask].reset_index(drop=True)

print(f"       Train : {X_train_raw.shape[0]:,} rows | "
      f"controversy rate: {y_train.mean():.1%}")
print(f"       Test  : {X_test_raw.shape[0]:,} rows  | "
      f"controversy rate: {y_test.mean():.1%}")


# ============================================================
# STEP 6: FEATURE SCALING (mandatory for SVM)
# ============================================================
print("\n[6/10] Scaling features (StandardScaler)...")
scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train_raw)
X_test   = scaler.transform(X_test_raw)
X_scaled_all = scaler.transform(X)

print(f"       Scaler fitted on {X_train.shape[0]:,} training rows")


# ============================================================
# STEP 7: GRIDSEARCH — FIND BEST SVM HYPERPARAMETERS
# ============================================================
print("\n[7/10] GridSearchCV for SVM hyperparameters...")
print("       (searching C and gamma — this may take 1-2 minutes)")

param_grid = {
    "C":     [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.1],
}

svm_base = SVC(
    kernel       = "rbf",
    class_weight = "balanced",
    probability  = True,
    random_state = 42,
)

cv_gs = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    svm_base, param_grid,
    cv      = cv_gs,
    scoring = "roc_auc",
    n_jobs  = -1,
    verbose = 0,
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"       Best params  : {best_params}")
print(f"       Best CV AUC  : {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_


# ============================================================
# STEP 8: 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================
print("\n[8/10] 5-Fold Stratified Cross-Validation...")
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_roc = cross_val_score(model, X_scaled_all, y, cv=cv5,
                         scoring="roc_auc",  n_jobs=-1)
cv_f1  = cross_val_score(model, X_scaled_all, y, cv=cv5,
                         scoring="f1",       n_jobs=-1)
cv_acc = cross_val_score(model, X_scaled_all, y, cv=cv5,
                         scoring="accuracy", n_jobs=-1)

print(f"       ROC-AUC  : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print(f"       F1 Score : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")
print(f"       Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")


# ============================================================
# STEP 9: EVALUATION METRICS
# ============================================================
print("\n[9/10] Evaluation metrics on held-out test set...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)

print(f"\n  {'Metric':<25} {'Value':>10}")
print(f"  {'─'*37}")
print(f"  {'Accuracy':<25} {acc:>10.4f}")
print(f"  {'Precision':<25} {prec:>10.4f}")
print(f"  {'Recall':<25} {rec:>10.4f}")
print(f"  {'F1 Score':<25} {f1:>10.4f}")
print(f"  {'ROC-AUC':<25} {roc_auc:>10.4f}")
print(f"  {'PR-AUC':<25} {pr_auc:>10.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Controversy','Controversy'])}")

metrics = {
    "Model": "SVM-RBF",
    "Best_C": best_params["C"],
    "Best_gamma": str(best_params["gamma"]),
    "Accuracy": acc, "Precision": prec, "Recall": rec,
    "F1": f1, "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
    "CV_ROC_AUC_Mean": cv_roc.mean(), "CV_ROC_AUC_Std": cv_roc.std(),
}


# ── Human-readable display names ──────────────────────────────
name_map = {
    "Revenue": "Revenue", "ProfitMargin": "Profit Margin (%)",
    "MarketCap": "Market Cap", "GrowthRate": "Growth Rate (%)",
    "Price_to_Sales": "Price-to-Sales", "Log_Revenue": "Log(Revenue)",
    "Log_MarketCap": "Log(MarketCap)", "Revenue_to_MarketCap": "Revenue / MarketCap",
    "EBIT_proxy": "EBIT Proxy", "Return_on_Sales": "Return on Sales",
    "Profitability_Score": "Profitability Score",
    "Revenue_YoY": "Revenue YoY Growth (%)",
    "Revenue_2Y_Growth": "Revenue 2-Year Growth (%)",
    "Margin_YoY_Change": "Margin YoY Change (pp)",
    "MarketCap_YoY": "MarketCap YoY Change (%)",
    "GrowthRate_YoY_Change": "Growth Rate Acceleration",
    "Margin_Volatility_3Y": "Margin Volatility (3Y)",
    "Revenue_Volatility_3Y": "Revenue Volatility CoV (3Y)",
    "MCap_Volatility_3Y": "MCap Volatility CoV (3Y)",
    "Revenue_vs_Industry": "Revenue vs Industry Median",
    "Margin_vs_Industry": "Margin vs Industry Median (pp)",
    "MCap_vs_Industry": "MCap vs Industry Median",
    "Growth_vs_Industry": "Growth vs Industry Median (pp)",
    "Revenue_Pctile_Industry": "Revenue Percentile (Industry)",
    "Margin_Pctile_Industry": "Margin Percentile (Industry)",
    "Negative_Margin_Flag": "Flag: Negative Margin",
    "Declining_Revenue_Flag": "Flag: Declining Revenue",
    "MCap_Collapse_Flag": "Flag: MCap Collapse >20%",
    "Low_Growth_Low_Margin": "Flag: Low Growth + Low Margin",
    "Valuation_Excess": "Valuation Excess (log)",
    "PS_Margin_Ratio": "P/S ÷ Margin Ratio",
    "Revenue_Efficiency": "Revenue Efficiency",
    "Avg_Margin_3Y": "Avg Margin (3Y Rolling)",
    "Margin_Deterioration": "Margin Deterioration vs 3Y",
    "Avg_Revenue_3Y": "Avg Revenue (3Y Rolling)",
    "Revenue_Surprise": "Revenue Surprise vs 3Y",
    "Size_Tier": "Size Tier (Quartile)",
}
for col in feature_names:
    if col.startswith("Industry_"):
        name_map[col] = "Ind: " + col.replace("Industry_", "")
    elif col.startswith("Region_"):
        name_map[col] = "Reg: " + col.replace("Region_", "")

display_names = [name_map.get(f, f) for f in feature_names]

# ── Feature group map ─────────────────────────────────────────
group_map = {
    "Revenue": "A-B: Valuation & Profitability",
    "ProfitMargin": "A-B: Valuation & Profitability",
    "MarketCap": "A-B: Valuation & Profitability",
    "GrowthRate": "A-B: Valuation & Profitability",
    "Price_to_Sales": "A-B: Valuation & Profitability",
    "Log_Revenue": "A-B: Valuation & Profitability",
    "Log_MarketCap": "A-B: Valuation & Profitability",
    "Revenue_to_MarketCap": "A-B: Valuation & Profitability",
    "EBIT_proxy": "A-B: Valuation & Profitability",
    "Return_on_Sales": "A-B: Valuation & Profitability",
    "Profitability_Score": "A-B: Valuation & Profitability",
    "Revenue_YoY": "C: Momentum & Growth",
    "Revenue_2Y_Growth": "C: Momentum & Growth",
    "Margin_YoY_Change": "C: Momentum & Growth",
    "MarketCap_YoY": "C: Momentum & Growth",
    "GrowthRate_YoY_Change": "C: Momentum & Growth",
    "Margin_Volatility_3Y": "D: Volatility",
    "Revenue_Volatility_3Y": "D: Volatility",
    "MCap_Volatility_3Y": "D: Volatility",
    "Revenue_vs_Industry": "E: Peer-Relative",
    "Margin_vs_Industry": "E: Peer-Relative",
    "MCap_vs_Industry": "E: Peer-Relative",
    "Growth_vs_Industry": "E: Peer-Relative",
    "Revenue_Pctile_Industry": "E: Peer-Relative",
    "Margin_Pctile_Industry": "E: Peer-Relative",
    "Negative_Margin_Flag": "F: Distress Flags",
    "Declining_Revenue_Flag": "F: Distress Flags",
    "MCap_Collapse_Flag": "F: Distress Flags",
    "Low_Growth_Low_Margin": "F: Distress Flags",
    "Valuation_Excess": "G: Valuation Anomaly",
    "PS_Margin_Ratio": "G: Valuation Anomaly",
    "Revenue_Efficiency": "G: Valuation Anomaly",
    "Avg_Margin_3Y": "H: Rolling Trends",
    "Margin_Deterioration": "H: Rolling Trends",
    "Avg_Revenue_3Y": "H: Rolling Trends",
    "Revenue_Surprise": "H: Rolling Trends",
    "Size_Tier": "I: Size Tier",
}
group_colors = {
    "A-B: Valuation & Profitability": TEAL,
    "C: Momentum & Growth":           BLUE,
    "D: Volatility":                  NAVY,
    "E: Peer-Relative":               GREEN,
    "F: Distress Flags":              RED,
    "G: Valuation Anomaly":           SVM_COLOR,
    "H: Rolling Trends":              "#0891B2",
    "I: Size Tier":                   "#BE185D",
    "J: Industry":                    AMBER,
    "J: Region":                      "#B45309",
}

def bar_color(feat_name):
    if any(k in feat_name for k in ["Margin", "Profit", "Return on"]):
        return TEAL
    if any(k in feat_name for k in ["Revenue", "MCap", "Growth", "Size"]):
        return BLUE
    if any(k in feat_name for k in ["Flag:", "Collapse", "Negative", "Low Growth"]):
        return RED
    if any(k in feat_name for k in ["Ind:", "Reg:"]):
        return AMBER
    if any(k in feat_name for k in ["Percentile", "vs Industry"]):
        return GREEN
    return NAVY


# ============================================================
# STEP 10: PLOTS — EVALUATION DASHBOARD
# ============================================================
print("\n[10/10] Generating evaluation + SHAP plots...")

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"ESG Controversy Prediction — SVM (RBF, C={best_params['C']}, "
    f"γ={best_params['gamma']}) Evaluation Dashboard\n"
    f"Train: 2015–{SPLIT_YEAR-1}  |  Test: {SPLIT_YEAR}–2025  |  "
    f"Features: {len(feature_names)}  |  Companies: 1,000",
    fontsize=15, fontweight="bold", color=NAVY, y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Metric bar
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
m_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
m_values = [acc, prec, rec, f1, roc_auc, pr_auc]
m_colors = [NAVY, TEAL, BLUE, GREEN, AMBER, RED]
bars = ax1.barh(m_labels, m_values, color=m_colors, edgecolor="white", height=0.55)
ax1.set_xlim(0, 1.15)
ax1.set_title("Test Set Metrics", fontweight="bold", fontsize=12, color=NAVY)
ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
for bar, val in zip(bars, m_values):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=10, fontweight="bold", color=NAVY)
ax1.grid(axis="x", alpha=0.2)

# Confusion matrix
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Controversy", "Controversy"])
disp.plot(ax=ax2, cmap="Purples", colorbar=False)
ax2.set_title("Confusion Matrix", fontweight="bold", fontsize=12, color=NAVY)

# ROC curve
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax3.plot(fpr, tpr, color=SVM_COLOR, lw=2.5, label=f"SVM-RBF (AUC = {roc_auc:.4f})")
ax3.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.5, label="Random Baseline")
ax3.fill_between(fpr, tpr, alpha=0.08, color=SVM_COLOR)
ax3.set_xlabel("False Positive Rate", fontsize=10)
ax3.set_ylabel("True Positive Rate", fontsize=10)
ax3.set_title("ROC Curve", fontweight="bold", fontsize=12, color=NAVY)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.25)

# PR curve
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("white")
prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
ax4.plot(rec_c, prec_c, color=TEAL, lw=2.5, label=f"SVM-RBF (PR-AUC = {pr_auc:.4f})")
ax4.axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.5, lw=1.5,
            label=f"Baseline ({y_test.mean():.2f})")
ax4.fill_between(rec_c, prec_c, alpha=0.08, color=TEAL)
ax4.set_xlabel("Recall", fontsize=10)
ax4.set_ylabel("Precision", fontsize=10)
ax4.set_title("Precision-Recall Curve", fontweight="bold", fontsize=12, color=NAVY)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.25)

# CV distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("white")
bp = ax5.boxplot([cv_roc, cv_f1, cv_acc], positions=[1, 2, 3], patch_artist=True,
                 widths=0.45, medianprops=dict(color="white", linewidth=2.5))
for patch, color in zip(bp["boxes"], [SVM_COLOR, TEAL, GREEN]):
    patch.set_facecolor(color); patch.set_alpha(0.85)
ax5.set_xticks([1, 2, 3])
ax5.set_xticklabels(["ROC-AUC", "F1", "Accuracy"], fontsize=10)
ax5.set_title("5-Fold CV Distribution", fontweight="bold", fontsize=12, color=NAVY)
ax5.set_ylim(0.4, 1.0)
ax5.grid(axis="y", alpha=0.3)

# Probability distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("white")
ax6.hist(y_proba[y_test.values == 0], bins=40, alpha=0.65, color=GREEN,
         label="Actual: No Controversy", density=True)
ax6.hist(y_proba[y_test.values == 1], bins=40, alpha=0.65, color=RED,
         label="Actual: Controversy", density=True)
ax6.axvline(0.5, color=NAVY, linestyle="--", lw=2, label="Threshold (0.5)")
ax6.set_xlabel("Predicted Probability (Controversy)", fontsize=10)
ax6.set_ylabel("Density", fontsize=10)
ax6.set_title("Prediction Probability Distribution", fontweight="bold", fontsize=12, color=NAVY)
ax6.legend(fontsize=8)
ax6.grid(alpha=0.25)

plt.savefig(os.path.join(OUTPUT_DIR, "S1_evaluation_dashboard.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: S1_evaluation_dashboard.png")


# ============================================================
# SHAP — KernelExplainer (model-agnostic, works with SVM)
# ============================================================
print(f"\n      Computing KernelSHAP values...")
print(f"      Background: {N_SHAP_BG} samples | Explain: {N_SHAP_EXPLAIN} test samples")
print("      (KernelSHAP is slower than TreeSHAP — estimated ~2-3 minutes)")

np.random.seed(42)
bg_idx   = np.random.choice(len(X_train), size=N_SHAP_BG, replace=False)
X_bg     = X_train[bg_idx]

# KernelExplainer wraps predict_proba — explains class=1 probability
explainer_shap = shap.KernelExplainer(
    model.predict_proba, X_bg, link="logit"
)

shap_idx    = np.random.choice(len(X_test), size=min(N_SHAP_EXPLAIN, len(X_test)), replace=False)
X_shap_test = X_test[shap_idx]

shap_values_raw = explainer_shap.shap_values(X_shap_test, nsamples=200)

# ── FIX: Robustly extract class-1 SHAP values regardless of SHAP version ──
# Older SHAP  : returns list  [class0_array(n,f), class1_array(n,f)]
# Newer SHAP  : may return a single ndarray of shape (n, f, n_classes)
#               or (n, f) when only one class is returned.
# We always want a 2-D array of shape (n_samples, n_features) for class=1.
if isinstance(shap_values_raw, list):
    # List format — pick index 1 (class=1)
    shap_values = np.array(shap_values_raw[1])
else:
    shap_values = np.array(shap_values_raw)

# If 3-D (n_samples, n_features, n_classes) — slice class=1
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# Final safety: must be exactly 2-D (n_samples, n_features)
assert shap_values.ndim == 2, (
    f"Unexpected shap_values shape after extraction: {shap_values.shape}"
)

mean_abs_shap = np.abs(shap_values).mean(axis=0)   # guaranteed 1-D
assert mean_abs_shap.ndim == 1, "mean_abs_shap must be 1-D"
print("      ✅ SHAP computation complete.")

# ── PLOT S2: SHAP Beeswarm ────────────────────────────────────
X_shap_df = pd.DataFrame(X_shap_test, columns=feature_names)

fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor(BG)
shap.summary_plot(
    shap_values, X_shap_df,
    feature_names=display_names, show=False, plot_size=None,
    max_display=20, color_bar_label="Feature Value (low → high)",
)
plt.title(
    "SHAP (KernelSHAP) Beeswarm — Feature Impact on ESG Controversy Prediction\n"
    "(SVM-RBF  |  Each dot = one test observation  |  color = feature value magnitude)",
    fontsize=13, fontweight="bold", color=NAVY, pad=15
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "S2_shap_beeswarm.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: S2_shap_beeswarm.png")

# ── PLOT S3: SHAP Feature Importance Bar ──────────────────────
fi_df = pd.DataFrame({"feature": display_names, "importance": mean_abs_shap})
fi_df = fi_df.sort_values("importance", ascending=True).tail(20)

legend_patches = [
    mpatches.Patch(color=TEAL,  label="Profitability"),
    mpatches.Patch(color=BLUE,  label="Revenue / Size / Growth"),
    mpatches.Patch(color=RED,   label="Distress Flags"),
    mpatches.Patch(color=GREEN, label="Peer-Relative"),
    mpatches.Patch(color=AMBER, label="Industry / Region"),
    mpatches.Patch(color=NAVY,  label="Other"),
]

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor("white")
colors_ = [bar_color(f) for f in fi_df["feature"]]
bars_ = ax.barh(fi_df["feature"], fi_df["importance"], color=colors_,
                edgecolor="white", height=0.65)
for bar_ in bars_:
    ax.text(bar_.get_width() + 0.0001,
            bar_.get_y() + bar_.get_height() / 2,
            f"{bar_.get_width():.4f}",
            va="center", ha="left", fontsize=9, color=NAVY)
ax.set_title(
    "SHAP Feature Importance — Mean |SHAP Value| (Top 20)\n"
    f"SVM-RBF  |  Computed via KernelSHAP over {N_SHAP_EXPLAIN} test samples",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.grid(axis="x", alpha=0.2)
ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "S3_shap_feature_importance.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: S3_shap_feature_importance.png")

# ── PLOT S4: SHAP Waterfall — Two individual predictions ──────
y_pred_shap = model.predict(X_shap_test)
y_test_shap = y_test.values[shap_idx]

idx_cont   = np.where((y_test_shap == 1) & (y_pred_shap == 1))[0]
idx_nocont = np.where((y_test_shap == 0) & (y_pred_shap == 0))[0]

if len(idx_cont) > 0 and len(idx_nocont) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "SHAP Waterfall — Individual Prediction Explanations (SVM-RBF)\n"
        "Left: Correctly Predicted Controversy  |  Right: Correctly Predicted Non-Controversy",
        fontsize=13, fontweight="bold", color=NAVY
    )
    # base_value: handle list or scalar returned by KernelExplainer
    base_val = explainer_shap.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).ravel()[1])
    else:
        base_val = float(base_val)

    for ax_i, (i, label, lcolor) in enumerate([
        (idx_cont[0],   "Predicted: CONTROVERSY ✅",    RED),
        (idx_nocont[0], "Predicted: NO CONTROVERSY ✅", GREEN),
    ]):
        plt.sca(axes[ax_i])
        exp = shap.Explanation(
            values        = shap_values[i],
            base_values   = base_val,
            data          = X_shap_test[i],
            feature_names = display_names,
        )
        shap.waterfall_plot(exp, max_display=15, show=False)
        axes[ax_i].set_title(label, fontsize=11, fontweight="bold",
                             color=lcolor, pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "S4_shap_waterfall.png"), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("      ✅ Saved: S4_shap_waterfall.png")
else:
    print("      ⚠️  Skipped S4: not enough correct predictions of both classes for waterfall.")

# ── PLOT S5: SHAP Feature Group Importance ────────────────────
group_imp_shap = {}
for feat, imp in zip(feature_names, mean_abs_shap):
    if feat.startswith("Industry_"):
        grp = "J: Industry"
    elif feat.startswith("Region_"):
        grp = "J: Region"
    else:
        grp = group_map.get(feat, "Other")
    group_imp_shap[grp] = group_imp_shap.get(grp, 0) + imp

grp_shap_df = pd.DataFrame(
    list(group_imp_shap.items()), columns=["Group", "Total SHAP"]
).sort_values("Total SHAP", ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor("white")
colors_ = [group_colors.get(g, NAVY) for g in grp_shap_df["Group"]]
bars_ = ax.barh(grp_shap_df["Group"], grp_shap_df["Total SHAP"],
                color=colors_, edgecolor="white", height=0.6)
for bar_ in bars_:
    ax.text(bar_.get_width() + 0.0001,
            bar_.get_y() + bar_.get_height() / 2,
            f"{bar_.get_width():.4f}",
            va="center", ha="left", fontsize=10, color=NAVY)
ax.set_title(
    "SHAP Feature Group Importance — Summed Mean |SHAP Value| (SVM-RBF)",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Total Mean |SHAP Value|", fontsize=11)
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "S5_shap_group_importance.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: S5_shap_group_importance.png")


# ── Company-level risk driver CSV ─────────────────────────────
print("      Building company-level risk driver explanations...")
risk_prob = model.predict_proba(X_test)[:, 1]

shap_drivers_rows = []
for test_i in shap_idx[:50]:
    sv = shap_values[np.where(shap_idx == test_i)[0][0]] if test_i in shap_idx else None
    if sv is None:
        continue
    top_feat_idx = np.argsort(np.abs(sv))[::-1][:4]
    drivers = [feature_names[i] for i in top_feat_idx]
    shap_drivers_rows.append({
        "CompanyID":        df_test.loc[test_i, "CompanyID"],
        "Year":             df_test.loc[test_i, "Year"],
        "Risk_Probability": round(risk_prob[test_i] * 100, 2),
        "SHAP_Driver_1":    drivers[0] if len(drivers) > 0 else None,
        "SHAP_Driver_2":    drivers[1] if len(drivers) > 1 else None,
        "SHAP_Driver_3":    drivers[2] if len(drivers) > 2 else None,
        "SHAP_Driver_4":    drivers[3] if len(drivers) > 3 else None,
    })

pd.DataFrame(shap_drivers_rows).to_csv(os.path.join(OUTPUT_DIR, "company_svm_risk_explanations.csv"), index=False)
print("      ✅ Saved: company_svm_risk_explanations.csv")


# ── Save model + scaler + metrics ─────────────────────────────
joblib.dump({"model": model, "scaler": scaler}, os.path.join(OUTPUT_DIR, "svm_esg_model.pkl"))
metrics_df = pd.DataFrame([metrics])
for fold_i, score in enumerate(cv_roc, 1):
    metrics_df[f"CV_ROC_AUC_Fold{fold_i}"] = score
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "svm_model_metrics.csv"), index=False)
print("      ✅ Saved: svm_esg_model.pkl")
print("      ✅ Saved: svm_model_metrics.csv")

print("\n" + "=" * 60)
print("  SUMMARY — SVM-RBF")
print("=" * 60)
print(f"  Best C        : {best_params['C']}")
print(f"  Best gamma    : {best_params['gamma']}")
print(f"  Accuracy      : {acc:.4f}")
print(f"  Precision     : {prec:.4f}")
print(f"  Recall        : {rec:.4f}")
print(f"  F1 Score      : {f1:.4f}")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  PR-AUC        : {pr_auc:.4f}")
print(f"  CV ROC-AUC    : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print("=" * 60)
print("\nOutput files:")
print("  S1_evaluation_dashboard.png    — metrics, confusion matrix, ROC, PR")
print("  S2_shap_beeswarm.png           — KernelSHAP beeswarm")
print("  S3_shap_feature_importance.png — mean |SHAP value| bar")
print("  S4_shap_waterfall.png          — individual SHAP explanations")
print("  S5_shap_group_importance.png   — SHAP by feature group")
print("  company_svm_risk_explanations.csv")
print("  svm_model_metrics.csv")
print("  svm_esg_model.pkl              — SVM + scaler saved together")
print("\n✅ All outputs saved.")