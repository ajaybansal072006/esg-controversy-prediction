"""
============================================================
ESG Controversy Prediction — Random Forest + SHAP
============================================================
Dataset : esg_controversy_features.csv
          (output of esg_feature_engineering.py)

Steps:
  1. Load engineered dataset
  2. Drop non-feature identifier columns (CompanyID, Year)
  3. Separate target column (ESG_Controversy)
  4. Impute NaN values (time-series lag features)
  5. Train/Test split — walk-forward temporal split
  6. Handle class imbalance with class_weight="balanced"
  7. Train Random Forest
  8. 5-Fold Cross-Validation
  9. Evaluation metrics (Accuracy, Precision, Recall, F1,
     ROC-AUC, PR-AUC, Confusion Matrix, Classification Report)
 10. SHAP analysis (summary beeswarm, bar, waterfall, dependence)
 11. Save all plots + model + metrics CSV
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR1 = os.path.join(BASE_DIR, "output")
# Create output folder inside it
OUTPUT_DIR = os.path.join(BASE_DIR1, "random_forest")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
import joblib

# ── Colour palette ───────────────────────────────────────────
NAVY  = "#1B3A6B"
TEAL  = "#0D7377"
RED   = "#B91C1C"
GREEN = "#15803D"
AMBER = "#D97706"
BLUE  = "#2563EB"
GRAY  = "#F8FAFC"
BG    = "#F1F5F9"

print("=" * 60)
print("  ESG Controversy Prediction — Random Forest + SHAP")
print("=" * 60)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/9] Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "esg_controversy_features.csv"))
print(f"      Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"      Years : {df['Year'].min()} – {df['Year'].max()}")
print(f"      Companies : {df['CompanyID'].nunique():,}")


# ============================================================
# STEP 2: DROP IDENTIFIER COLUMNS (NOT FEATURES)
# ============================================================
print("\n[2/9] Dropping identifier columns...")

# ── WHY THESE ARE DROPPED ────────────────────────────────────
#
#  CompanyID  — An arbitrary integer key that identifies a
#      company.  It carries no financial signal; keeping it
#      would allow the model to memorise company identities
#      rather than learn generalizable financial patterns.
#      In deployment the company key changes (new firms,
#      re-indexing) so it is meaningless out-of-sample.
#
#  Year       — The calendar year is used as the time-series
#      sorting key and for walk-forward split logic.  Adding
#      it as a raw feature would cause the model to learn
#      spurious year-level effects (e.g. 2020 = COVID dip)
#      that do not generalise to future years.  The temporal
#      signal is already encoded in YoY and rolling features.
#
#  Note: ALL ESG score columns (ESG_Overall, ESG_Environmental,
#  ESG_Social, ESG_Governance, CarbonEmissions, WaterUsage,
#  EnergyConsumption) were already removed during the feature
#  engineering step (esg_feature_engineering.py) because they
#  are direct inputs to the ESG_Controversy target variable.
#  Retaining any of them would cause catastrophic target
#  leakage — see the Feature Engineering Report for details.
# ─────────────────────────────────────────────────────────────

ID_COLS = [
    "CompanyID", "Year",
    "Region_Africa", "Region_Asia", "Region_Europe",
    "Region_Latin America", "Region_Middle East",
    "Region_North America", "Region_Oceania",
]
for col in ID_COLS:
    print(f"      ❌ Dropped: {col}")

df_model = df.drop(columns=ID_COLS)
print(f"\n      Remaining columns : {df_model.shape[1]}")


# ============================================================
# STEP 3: SEPARATE TARGET COLUMN
# ============================================================
print("\n[3/9] Separating target column...")

TARGET = "ESG_Controversy"
y = df_model[TARGET]
X = df_model.drop(columns=[TARGET])

print(f"      X shape : {X.shape}")
print(f"      y shape : {y.shape}")
print(f"      Class 0 (No Controversy) : {(y==0).sum():,}  ({(y==0).mean():.1%})")
print(f"      Class 1 (Controversy)    : {(y==1).sum():,}  ({(y==1).mean():.1%})")

# Cast any bool columns (from pd.get_dummies) to int
bool_cols = X.select_dtypes(include="bool").columns.tolist()
if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)

feature_names = X.columns.tolist()
print(f"\n      Feature count : {len(feature_names)}")


# ============================================================
# STEP 4: IMPUTE MISSING VALUES
# ============================================================
print("\n[4/9] Imputing missing values...")

# ── WHY IMPUTATION IS NEEDED HERE ───────────────────────────
#
#  Time-series lag features (Revenue_YoY, Margin_YoY_Change,
#  rolling windows etc.) are undefined for a company's first
#  1–2 years because there is no prior year to diff/roll over.
#  This creates exactly 1,000 NaN values per 1-year lag and
#  2,000 per 2-year lag (one per company).
#
#  Unlike XGBoost, Random Forest in scikit-learn does NOT
#  handle NaN natively — it will raise a ValueError if any
#  NaN remains in X at fit time.  Imputation is therefore
#  mandatory for this model.
#
#  Strategy per feature class:
#   • YoY / diff features   → 0  (no change = safe neutral)
#   • Rolling std / CoV     → 0  (single obs has zero variance)
#   • Rolling mean / trend  → same as current value (no history)
#   • GrowthRate (raw)      → column median
# ─────────────────────────────────────────────────────────────

# Features where 0 is the correct neutral imputation
zero_impute_cols = [
    "Revenue_YoY", "Revenue_2Y_Growth",
    "Margin_YoY_Change", "MarketCap_YoY", "GrowthRate_YoY_Change",
    "Margin_Volatility_3Y", "Revenue_Volatility_3Y", "MCap_Volatility_3Y",
    "Margin_Deterioration", "Revenue_Surprise",
]

for col in zero_impute_cols:
    if col in X.columns:
        n_null = X[col].isnull().sum()
        if n_null > 0:
            X[col] = X[col].fillna(0)
            print(f"      → {col:<30} filled {n_null:,} NaNs with 0")

# Rolling mean features: fill with own current value (no history available)
rolling_mean_map = {
    "Avg_Margin_3Y":  "ProfitMargin",
    "Avg_Revenue_3Y": "Revenue",
}
for roll_col, src_col in rolling_mean_map.items():
    if roll_col in X.columns and src_col in X.columns:
        n_null = X[roll_col].isnull().sum()
        if n_null > 0:
            X[roll_col] = X[roll_col].fillna(X[src_col])
            print(f"      → {roll_col:<30} filled {n_null:,} NaNs from {src_col}")

# GrowthRate: fill with column median
if "GrowthRate" in X.columns:
    n_null = X["GrowthRate"].isnull().sum()
    if n_null > 0:
        X["GrowthRate"] = X["GrowthRate"].fillna(X["GrowthRate"].median())
        print(f"      → {'GrowthRate':<30} filled {n_null:,} NaNs with median")

# Growth_vs_Industry propagates from GrowthRate → fill with 0
if "Growth_vs_Industry" in X.columns:
    n_null = X["Growth_vs_Industry"].isnull().sum()
    if n_null > 0:
        X["Growth_vs_Industry"] = X["Growth_vs_Industry"].fillna(0)
        print(f"      → {'Growth_vs_Industry':<30} filled {n_null:,} NaNs with 0")

# Safety net: fill any remaining NaNs with column median
remaining_nulls = X.isnull().sum().sum()
if remaining_nulls > 0:
    print(f"      ⚠  {remaining_nulls} NaNs still present — applying median fallback")
    X = X.fillna(X.median(numeric_only=True))

remaining_nulls = X.isnull().sum().sum()
print(f"\n      Total NaNs remaining : {remaining_nulls}")


# ============================================================
# STEP 5: WALK-FORWARD TEMPORAL TRAIN / TEST SPLIT
# ============================================================
print("\n[5/9] Walk-forward temporal train/test split...")

# ── WHY TEMPORAL SPLIT (NOT RANDOM) ─────────────────────────
#
#  This dataset has a panel structure: 1,000 companies × 11
#  years.  Standard random split leaks future data into
#  training — a company's 2023 row can end up in the training
#  set alongside its 2020 row in the test set, allowing the
#  model to "see" future financial trajectories implicitly.
#
#  Walk-forward split: train on 2015–2022 (8 years, ~80%)
#  and test on 2023–2025 (3 years, ~20%).  This mirrors the
#  real deployment scenario: predict future controversy from
#  historical financial data only.
# ─────────────────────────────────────────────────────────────

SPLIT_YEAR = 2023    # first year of the test set

train_mask = df["Year"] < SPLIT_YEAR
test_mask  = df["Year"] >= SPLIT_YEAR

X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
X_test,  y_test  = X[test_mask].copy(),  y[test_mask].copy()

print(f"      Train : years 2015–{SPLIT_YEAR-1}  →  "
      f"{X_train.shape[0]:,} rows  (controversy rate: {y_train.mean():.1%})")
print(f"      Test  : years {SPLIT_YEAR}–2025    →  "
      f"{X_test.shape[0]:,} rows  (controversy rate: {y_test.mean():.1%})")


# ============================================================
# STEP 6: CLASS IMBALANCE + TRAIN RANDOM FOREST
# ============================================================
print("\n[6/9] Training Random Forest model...")

# ── HOW RANDOM FOREST HANDLES CLASS IMBALANCE ───────────────
#
#  Random Forest uses class_weight="balanced" which
#  automatically computes weights inversely proportional to
#  class frequencies:
#      w_k = n_samples / (n_classes × n_samples_k)
#
#  This is equivalent to XGBoost's scale_pos_weight but
#  applied per-tree at each node split.  Each tree in the
#  ensemble is built on a bootstrap sample that respects
#  these weights, so the minority controversy class receives
#  proportionally higher vote weight across all 500 trees.
# ─────────────────────────────────────────────────────────────

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
print(f"      Class distribution — 0: {neg:,}  |  1: {pos:,}  "
      f"(ratio {neg/pos:.2f}:1)")
print(f"      Using class_weight='balanced' to compensate")

model = RandomForestClassifier(
    n_estimators      = 500,        # 500 trees (matches XGBoost n_estimators)
    max_depth         = 10,         # Moderate depth to prevent overfitting
    min_samples_split = 10,         # Require ≥10 samples to split a node
    min_samples_leaf  = 5,          # Require ≥5 samples at each leaf
    max_features      = "sqrt",     # √n_features per split (standard for clf)
    class_weight      = "balanced", # Handles ~75/25 class imbalance
    bootstrap         = True,       # Bagging with replacement
    oob_score         = True,       # Out-of-bag estimate (free validation)
    random_state      = 42,
    n_jobs            = -1,
)

model.fit(X_train, y_train)

print(f"      OOB Score (train estimate) : {model.oob_score_:.4f}")
print(f"      Trees built                : {len(model.estimators_)}")


# ============================================================
# STEP 7: 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================
print("\n[7/9] 5-Fold Stratified Cross-Validation...")

# Note: CV uses the full dataset (X, y) with stratification
# to give an overall performance estimate across all years.
cv_model = RandomForestClassifier(
    n_estimators      = 500,
    max_depth         = 10,
    min_samples_split = 10,
    min_samples_leaf  = 5,
    max_features      = "sqrt",
    class_weight      = "balanced",
    bootstrap         = True,
    random_state      = 42,
    n_jobs            = -1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_roc = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc",  n_jobs=-1)
cv_f1  = cross_val_score(cv_model, X, y, cv=cv, scoring="f1",       n_jobs=-1)
cv_acc = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"      ROC-AUC  : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print(f"      F1 Score : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")
print(f"      Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")


# ============================================================
# STEP 8: EVALUATION METRICS
# ============================================================
print("\n[8/9] Evaluation metrics on held-out test set...")

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
print(f"\n  Classification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=["No Controversy", "Controversy"]
))

metrics = {
    "Accuracy":        acc,
    "Precision":       prec,
    "Recall":          rec,
    "F1":              f1,
    "ROC-AUC":         roc_auc,
    "PR-AUC":          pr_auc,
    "CV_ROC_AUC_Mean": cv_roc.mean(),
    "CV_ROC_AUC_Std":  cv_roc.std(),
    "OOB_Score":       model.oob_score_,
}


# ============================================================
# STEP 9: PLOTS
# ============================================================
print("\n[9/9] Generating evaluation + SHAP plots...")

# ── PLOT 1: Evaluation Dashboard ────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "ESG Controversy Prediction — Random Forest Evaluation Dashboard\n"
    f"Train: 2015–{SPLIT_YEAR-1}  |  Test: {SPLIT_YEAR}–2025  |  "
    f"Features: {len(feature_names)}  |  Companies: 1,000",
    fontsize=16, fontweight="bold", color=NAVY, y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1a: Metric Bar Chart ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
metric_values = [acc, prec, rec, f1, roc_auc, pr_auc]
bar_colors    = [NAVY, TEAL, BLUE, GREEN, AMBER, RED]
bars = ax1.barh(metric_labels, metric_values,
                color=bar_colors, edgecolor="white", height=0.55)
ax1.set_xlim(0, 1.15)
ax1.set_title("Test Set Metrics", fontweight="bold", fontsize=12, color=NAVY)
ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
for bar, val in zip(bars, metric_values):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=10,
             fontweight="bold", color=NAVY)
ax1.set_xlabel("Score", fontsize=10)
ax1.grid(axis="x", alpha=0.2)
ax1.tick_params(labelsize=10)

# ── 1b: Confusion Matrix ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Controversy", "Controversy"]
)
disp.plot(ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title("Confusion Matrix", fontweight="bold", fontsize=12, color=NAVY)
for text in disp.text_.ravel():
    text.set_fontsize(14)
    text.set_fontweight("bold")
ax2.tick_params(labelsize=9)

# ── 1c: ROC Curve ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax3.plot(fpr, tpr, color=NAVY, lw=2.5,
         label=f"Random Forest (AUC = {roc_auc:.4f})")
ax3.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.5, label="Random Baseline")
ax3.fill_between(fpr, tpr, alpha=0.08, color=NAVY)
ax3.set_xlabel("False Positive Rate", fontsize=10)
ax3.set_ylabel("True Positive Rate", fontsize=10)
ax3.set_title("ROC Curve", fontweight="bold", fontsize=12, color=NAVY)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.25)

# ── 1d: Precision-Recall Curve ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("white")
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
ax4.plot(rec_curve, prec_curve, color=TEAL, lw=2.5,
         label=f"Random Forest (PR-AUC = {pr_auc:.4f})")
ax4.axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.5, lw=1.5,
            label=f"Baseline ({y_test.mean():.2f})")
ax4.fill_between(rec_curve, prec_curve, alpha=0.08, color=TEAL)
ax4.set_xlabel("Recall", fontsize=10)
ax4.set_ylabel("Precision", fontsize=10)
ax4.set_title("Precision-Recall Curve", fontweight="bold", fontsize=12, color=NAVY)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.25)

# ── 1e: CV Score Distribution ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("white")
positions  = [1, 2, 3]
bp_colors  = [NAVY, TEAL, GREEN]
bp = ax5.boxplot(
    [cv_roc, cv_f1, cv_acc],
    positions=positions,
    patch_artist=True,
    widths=0.45,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=5),
)
for patch, color in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
for i, (scores, color) in enumerate(zip([cv_roc, cv_f1, cv_acc], bp_colors)):
    ax5.scatter([positions[i]] * len(scores), scores,
                color=color, zorder=5, s=60, edgecolors="white", linewidth=1)
ax5.set_xticks(positions)
ax5.set_xticklabels(["ROC-AUC", "F1 Score", "Accuracy"], fontsize=10)
ax5.set_title("5-Fold CV Score Distribution", fontweight="bold",
              fontsize=12, color=NAVY)
ax5.set_ylim(0.4, 1.0)
ax5.grid(axis="y", alpha=0.3)

# ── 1f: Prediction Probability Distribution ───────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("white")
proba_0 = y_proba[y_test.values == 0]
proba_1 = y_proba[y_test.values == 1]
ax6.hist(proba_0, bins=40, alpha=0.65, color=GREEN,
         label="Actual: No Controversy", density=True)
ax6.hist(proba_1, bins=40, alpha=0.65, color=RED,
         label="Actual: Controversy",    density=True)
ax6.axvline(0.5, color=NAVY, linestyle="--", lw=2,
            label="Decision threshold (0.5)")
ax6.set_xlabel("Predicted Probability (Controversy)", fontsize=10)
ax6.set_ylabel("Density", fontsize=10)
ax6.set_title("Prediction Probability Distribution", fontweight="bold",
              fontsize=12, color=NAVY)
ax6.legend(fontsize=8)
ax6.grid(alpha=0.25)

plt.savefig(os.path.join(OUTPUT_DIR,  "01_evaluation_dashboard.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 01_evaluation_dashboard.png")


# ── PLOT 2: SHAP Analysis ─────────────────────────────────────
print("      Computing SHAP values (TreeExplainer)...")

# ── WHY TreeExplainer WORKS FOR RANDOM FOREST ───────────────
#
#  shap.TreeExplainer supports any tree-based model in
#  scikit-learn, including RandomForestClassifier.  It
#  computes exact Shapley values in polynomial time by
#  exploiting the tree structure — no approximation needed.
#
#  For Random Forest, SHAP averages contributions across all
#  500 trees, giving a more stable importance estimate than
#  single-tree SHAP values.
#
#  predict_output="probability" ensures SHAP values are in
#  probability space (log-odds for binary clf), making the
#  waterfall base value interpretable as the mean predicted
#  probability across the background dataset.
# ─────────────────────────────────────────────────────────────

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap_values shape varies by shap version:
#   Older shap (<0.42): list of two 2D arrays [class_0, class_1]
#   Newer shap (>=0.42): single 3D array of shape (n_samples, n_features, n_classes)
# In both cases we extract class 1 (Controversy).
if isinstance(shap_values, list):
    # Older format: list[class_0_array, class_1_array]
    shap_values_pos = shap_values[1]
    base_value_pos  = (explainer.expected_value[1]
                       if hasattr(explainer.expected_value, "__len__")
                       else explainer.expected_value)
elif shap_values.ndim == 3:
    # Newer format: (n_samples, n_features, n_classes) → slice class 1
    shap_values_pos = shap_values[:, :, 1]
    base_value_pos  = (explainer.expected_value[1]
                       if hasattr(explainer.expected_value, "__len__")
                       else explainer.expected_value)
else:
    # Already 2D (edge case)
    shap_values_pos = shap_values
    base_value_pos  = explainer.expected_value

print(f"      SHAP array shape used : {shap_values_pos.shape}  (samples × features)")

shap_df = pd.DataFrame(shap_values_pos, columns=feature_names)

# ── Human-readable display names ─────────────────────────────
name_map = {
    # Raw financials
    "Revenue":                  "Revenue",
    "ProfitMargin":             "Profit Margin (%)",
    "MarketCap":                "Market Cap",
    "GrowthRate":               "Growth Rate (%)",
    # Group A: Valuation & Size
    "Price_to_Sales":           "Price-to-Sales",
    "Log_Revenue":              "Log(Revenue)",
    "Log_MarketCap":            "Log(MarketCap)",
    "Revenue_to_MarketCap":     "Revenue / MarketCap",
    # Group B: Profitability
    "EBIT_proxy":               "EBIT Proxy",
    "Return_on_Sales":          "Return on Sales",
    "Profitability_Score":      "Profitability Score",
    # Group C: Momentum
    "Revenue_YoY":              "Revenue YoY Growth (%)",
    "Revenue_2Y_Growth":        "Revenue 2-Year Growth (%)",
    "Margin_YoY_Change":        "Margin YoY Change (pp)",
    "MarketCap_YoY":            "MarketCap YoY Change (%)",
    "GrowthRate_YoY_Change":    "Growth Rate Acceleration",
    # Group D: Volatility
    "Margin_Volatility_3Y":     "Margin Volatility (3Y)",
    "Revenue_Volatility_3Y":    "Revenue Volatility CoV (3Y)",
    "MCap_Volatility_3Y":       "MCap Volatility CoV (3Y)",
    # Group E: Peer-relative
    "Revenue_vs_Industry":      "Revenue vs Industry Median",
    "Margin_vs_Industry":       "Margin vs Industry Median (pp)",
    "MCap_vs_Industry":         "MCap vs Industry Median",
    "Growth_vs_Industry":       "Growth vs Industry Median (pp)",
    "Revenue_Pctile_Industry":  "Revenue Percentile (Industry)",
    "Margin_Pctile_Industry":   "Margin Percentile (Industry)",
    # Group F: Distress flags
    "Negative_Margin_Flag":     "Flag: Negative Margin",
    "Declining_Revenue_Flag":   "Flag: Declining Revenue",
    "MCap_Collapse_Flag":       "Flag: MCap Collapse >20%",
    "Low_Growth_Low_Margin":    "Flag: Low Growth + Low Margin",
    # Group G: Valuation anomaly
    "Valuation_Excess":         "Valuation Excess (log)",
    "PS_Margin_Ratio":          "P/S ÷ Margin Ratio",
    "Revenue_Efficiency":       "Revenue Efficiency",
    # Group H: Rolling trends
    "Avg_Margin_3Y":            "Avg Margin (3Y Rolling)",
    "Margin_Deterioration":     "Margin Deterioration vs 3Y",
    "Avg_Revenue_3Y":           "Avg Revenue (3Y Rolling)",
    "Revenue_Surprise":         "Revenue Surprise vs 3Y",
    # Group I: Size
    "Size_Tier":                "Size Tier (Quartile)",
}
# Industry and Region dummies
for col in feature_names:
    if col.startswith("Industry_"):
        name_map[col] = "Ind: " + col.replace("Industry_", "")
    elif col.startswith("Region_"):
        name_map[col] = "Reg: " + col.replace("Region_", "")

display_names = [name_map.get(f, f) for f in feature_names]

# ── PLOT 2a: SHAP Beeswarm Summary ───────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor(BG)
shap.summary_plot(
    shap_values_pos, X_test,
    feature_names=display_names,
    show=False,
    plot_size=None,
    max_display=20,
    color_bar_label="Feature Value (low → high)",
)
plt.title(
    "SHAP Beeswarm — Feature Impact on ESG Controversy Prediction\n"
    "(Each dot = one test observation  |  color = feature value magnitude)",
    fontsize=13, fontweight="bold", color=NAVY, pad=15
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"02_shap_beeswarm.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 02_shap_beeswarm.png")

# ── PLOT 2b: SHAP Bar (Mean |SHAP|) ──────────────────────────
mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
fi_df = pd.DataFrame({
    "feature":    display_names,
    "importance": mean_abs_shap,
}).sort_values("importance", ascending=True).tail(20)

# Colour by feature group
def bar_color(feat_name):
    if any(k in feat_name for k in ["Margin", "Profit", "Return on"]):
        return TEAL
    if any(k in feat_name for k in ["Revenue", "MCap", "Growth", "Size"]):
        return BLUE
    if any(k in feat_name for k in ["Flag:", "Collapse", "Negative", "Low Growth"]):
        return RED
    if any(k in feat_name for k in ["Ind:", "Reg:"]):
        return AMBER
    if any(k in feat_name for k in ["Percentile", "vs Industry", "vs Peer"]):
        return GREEN
    return NAVY

bar_colors_shap = [bar_color(f) for f in fi_df["feature"]]

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor("white")
bars = ax.barh(fi_df["feature"], fi_df["importance"],
               color=bar_colors_shap, edgecolor="white", height=0.65)
for bar in bars:
    ax.text(bar.get_width() + 0.0003,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center", ha="left", fontsize=9, color=NAVY)
ax.set_title(
    "SHAP Feature Importance — Mean |SHAP Value| (Top 20)\n"
    "Higher value = stronger influence on ESG Controversy prediction",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.grid(axis="x", alpha=0.2)
ax.tick_params(labelsize=10)
legend_patches = [
    mpatches.Patch(color=TEAL,  label="Profitability Features"),
    mpatches.Patch(color=BLUE,  label="Revenue / Size / Growth"),
    mpatches.Patch(color=RED,   label="Distress Flags"),
    mpatches.Patch(color=GREEN, label="Peer-Relative Features"),
    mpatches.Patch(color=AMBER, label="Industry / Region"),
    mpatches.Patch(color=NAVY,  label="Other Financial Features"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_shap_feature_importance.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 03_shap_feature_importance.png")

# ── PLOT 2c: SHAP Waterfall — Two individual predictions ──────
idx_controversy    = np.where((y_test.values == 1) & (y_pred == 1))[0]
idx_no_controversy = np.where((y_test.values == 0) & (y_pred == 0))[0]

if len(idx_controversy) == 0 or len(idx_no_controversy) == 0:
    print("      ⚠️  Skipped waterfall: no correct predictions found for one class")
else:
    i_cont   = idx_controversy[0]
    i_nocont = idx_no_controversy[0]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "SHAP Waterfall — Individual Prediction Explanations\n"
        "Left: Correctly Predicted Controversy  |  "
        "Right: Correctly Predicted Non-Controversy",
        fontsize=13, fontweight="bold", color=NAVY
    )
    for ax_idx, (idx, label, lcolor) in enumerate([
        (i_cont,   "Predicted: CONTROVERSY ✅",     RED),
        (i_nocont, "Predicted: NO CONTROVERSY ✅",  GREEN),
    ]):
        plt.sca(axes[ax_idx])
        exp = shap.Explanation(
            values        = shap_values_pos[idx],
            base_values   = base_value_pos,
            data          = X_test.iloc[idx].values,
            feature_names = display_names,
        )
        shap.waterfall_plot(exp, max_display=15, show=False)
        axes[ax_idx].set_title(label, fontsize=11, fontweight="bold",
                               color=lcolor, pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"04_shap_waterfall.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("      ✅ Saved: 04_shap_waterfall.png")

# ── PLOT 2d: SHAP Dependence — Top 4 Features ────────────────
top4_idx     = np.argsort(mean_abs_shap)[::-1][:4]
top4_names   = [feature_names[i] for i in top4_idx]
top4_display = [display_names[i] for i in top4_idx]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "SHAP Dependence Plots — Top 4 Most Important Features\n"
    "(X = feature value  |  Y = SHAP contribution  |  "
    "Color = interaction with most correlated feature)",
    fontsize=13, fontweight="bold", color=NAVY
)
for i, (feat_name, feat_display) in enumerate(zip(top4_names, top4_display)):
    ax = axes[i // 2][i % 2]
    ax.set_facecolor("white")
    shap.dependence_plot(
        feat_name,
        shap_values_pos,
        X_test,
        feature_names=feature_names,
        ax=ax,
        show=False,
        dot_size=12,
        alpha=0.5,
    )
    ax.set_title(f"Dependence: {feat_display}", fontsize=11,
                 fontweight="bold", color=NAVY)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_shap_dependence.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 05_shap_dependence.png")


# ── PLOT 2e: Feature Group Contribution ──────────────────────
# Aggregate mean |SHAP| by feature group for a high-level view
group_map = {
    "Revenue":              "A-B: Valuation & Profitability",
    "ProfitMargin":         "A-B: Valuation & Profitability",
    "MarketCap":            "A-B: Valuation & Profitability",
    "GrowthRate":           "A-B: Valuation & Profitability",
    "Price_to_Sales":       "A-B: Valuation & Profitability",
    "Log_Revenue":          "A-B: Valuation & Profitability",
    "Log_MarketCap":        "A-B: Valuation & Profitability",
    "Revenue_to_MarketCap": "A-B: Valuation & Profitability",
    "EBIT_proxy":           "A-B: Valuation & Profitability",
    "Return_on_Sales":      "A-B: Valuation & Profitability",
    "Profitability_Score":  "A-B: Valuation & Profitability",
    "Revenue_YoY":          "C: Momentum & Growth",
    "Revenue_2Y_Growth":    "C: Momentum & Growth",
    "Margin_YoY_Change":    "C: Momentum & Growth",
    "MarketCap_YoY":        "C: Momentum & Growth",
    "GrowthRate_YoY_Change":"C: Momentum & Growth",
    "Margin_Volatility_3Y": "D: Volatility",
    "Revenue_Volatility_3Y":"D: Volatility",
    "MCap_Volatility_3Y":   "D: Volatility",
    "Revenue_vs_Industry":  "E: Peer-Relative",
    "Margin_vs_Industry":   "E: Peer-Relative",
    "MCap_vs_Industry":     "E: Peer-Relative",
    "Growth_vs_Industry":   "E: Peer-Relative",
    "Revenue_Pctile_Industry":"E: Peer-Relative",
    "Margin_Pctile_Industry":"E: Peer-Relative",
    "Negative_Margin_Flag": "F: Distress Flags",
    "Declining_Revenue_Flag":"F: Distress Flags",
    "MCap_Collapse_Flag":   "F: Distress Flags",
    "Low_Growth_Low_Margin":"F: Distress Flags",
    "Valuation_Excess":     "G: Valuation Anomaly",
    "PS_Margin_Ratio":      "G: Valuation Anomaly",
    "Revenue_Efficiency":   "G: Valuation Anomaly",
    "Avg_Margin_3Y":        "H: Rolling Trends",
    "Margin_Deterioration": "H: Rolling Trends",
    "Avg_Revenue_3Y":       "H: Rolling Trends",
    "Revenue_Surprise":     "H: Rolling Trends",
    "Size_Tier":            "I: Size Tier",
}

group_importance = {}
for feat, imp in zip(feature_names, mean_abs_shap):
    if feat.startswith("Industry_"):
        group = "J: Industry"
    elif feat.startswith("Region_"):
        group = "J: Region"
    else:
        group = group_map.get(feat, "Other")
    group_importance[group] = group_importance.get(group, 0) + imp

grp_df = pd.DataFrame(
    list(group_importance.items()), columns=["Group", "Total SHAP"]
).sort_values("Total SHAP", ascending=True)

group_colors = {
    "A-B: Valuation & Profitability": TEAL,
    "C: Momentum & Growth":           BLUE,
    "D: Volatility":                  NAVY,
    "E: Peer-Relative":               GREEN,
    "F: Distress Flags":              RED,
    "G: Valuation Anomaly":           "#7C3AED",
    "H: Rolling Trends":              "#0891B2",
    "I: Size Tier":                   "#BE185D",
    "J: Industry":                    AMBER,
    "J: Region":                      "#B45309",
}

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor("white")
colors = [group_colors.get(g, NAVY) for g in grp_df["Group"]]
bars = ax.barh(grp_df["Group"], grp_df["Total SHAP"],
               color=colors, edgecolor="white", height=0.6)
for bar in bars:
    ax.text(bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center", ha="left", fontsize=10, color=NAVY)
ax.set_title(
    "SHAP Feature Group Importance — Summed Mean |SHAP Value|\n"
    "Which category of financial features drives ESG controversy prediction most?",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Total Mean |SHAP Value| (summed across group)", fontsize=11)
ax.grid(axis="x", alpha=0.2)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_shap_group_importance.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 06_shap_group_importance.png")


# ── PLOT 3: RF-specific — Feature Importance Comparison ──────
# Compare sklearn's built-in MDI importance vs SHAP importance
mdi_importance = model.feature_importances_
top20_idx   = np.argsort(mean_abs_shap)[::-1][:20]
top20_names = [display_names[i] for i in top20_idx]
top20_shap  = mean_abs_shap[top20_idx]
top20_mdi   = mdi_importance[top20_idx]

# Normalise MDI to same scale as SHAP for visual comparison
top20_mdi_norm = top20_mdi / top20_mdi.sum() * top20_shap.sum()

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor("white")
y_pos = np.arange(len(top20_names))
ax.barh(y_pos - 0.2, top20_shap[::-1], height=0.35,
        color=NAVY,  label="SHAP Importance", alpha=0.85)
ax.barh(y_pos + 0.2, top20_mdi_norm[::-1], height=0.35,
        color=TEAL,  label="MDI Importance (normalised)", alpha=0.75)
ax.set_yticks(y_pos)
ax.set_yticklabels(top20_names[::-1], fontsize=9)
ax.set_title(
    "Random Forest: SHAP vs MDI Feature Importance (Top 20)\n"
    "MDI (Mean Decrease in Impurity) normalised to SHAP scale for comparison",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Importance Score", fontsize=11)
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="x", alpha=0.2)
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_shap_vs_mdi_importance.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 07_shap_vs_mdi_importance.png  (RF bonus: SHAP vs MDI)")


# ============================================================
# SAVE MODEL + METRICS
# ============================================================
joblib.dump(model, os.path.join(OUTPUT_DIR, "randomforest_esg_model.pkl"))
print("      ✅ Saved: randomforest_esg_model.pkl")

metrics_df = pd.DataFrame([metrics])
for fold_i, score in enumerate(cv_roc, 1):
    metrics_df[f"CV_ROC_AUC_Fold{fold_i}"] = score
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
print("      ✅ Saved: model_metrics.csv")

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  PR-AUC    : {pr_auc:.4f}")
print(f"  OOB Score : {model.oob_score_:.4f}")
print(f"  CV ROC-AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print("=" * 60)
print("\n✅ All outputs saved.")