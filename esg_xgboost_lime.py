"""
============================================================
ESG Controversy Prediction — XGBoost + LIME
============================================================
Companion to esg_xgboost_shap.py — mirrors the same pipeline
so LIME and SHAP outputs are directly comparable.

LIME explains individual predictions by fitting a local linear
model around each test observation using perturbed samples.

Steps:
  1-8.  Identical to SHAP script (data → train → evaluate)
  9.    LIME global feature importance  (mean |weight| over N samples)
 10.    LIME beeswarm-style scatter     (weight distribution per feature)
 11.    LIME waterfall — two individual predictions
 12.    LIME group importance           (same groups as SHAP script)
 13.    Company-level risk driver CSV   (same schema as SHAP script)
 14.    Combined SHAP vs LIME comparison bar chart
 15.    Save model + metrics CSV
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR1 = os.path.join(BASE_DIR, "output")
# Create output folder inside it
OUTPUT_DIR = os.path.join(BASE_DIR1, "xg_boost")
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

import xgboost as xgb
import shap          # still needed for SHAP comparison plot
import joblib

from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)

# ── Colour palette (same as SHAP script) ──────────────────────
NAVY  = "#1B3A6B"
TEAL  = "#0D7377"
RED   = "#B91C1C"
GREEN = "#15803D"
AMBER = "#D97706"
BLUE  = "#2563EB"
GRAY  = "#F8FAFC"
BG    = "#F1F5F9"
LIME_COLOR = "#65A30D"   # distinct accent for LIME visuals

print("=" * 60)
print("  ESG Controversy Prediction — XGBoost + LIME")
print("=" * 60)

# Number of test samples to explain with LIME
# (full test set is ~3000 rows; 300 gives stable global importance fast)
N_LIME_SAMPLES = 300
LIME_NUM_FEATURES = 15   # features per individual explanation


# ============================================================
# STEPS 1-5: DATA LOADING, PREP, SPLIT (identical to SHAP script)
# ============================================================
print("\n[1/9] Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "esg_controversy_features.csv"))
print(f"      Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\n[2/9] Dropping identifier columns...")
ID_COLS = ["CompanyID", "Year"]
df_model = df.drop(columns=ID_COLS)

print("\n[3/9] Separating target column...")
TARGET = "ESG_Controversy"
y = df_model[TARGET]
X = df_model.drop(columns=[TARGET])

bool_cols = X.select_dtypes(include="bool").columns.tolist()
if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)

feature_names = X.columns.tolist()
print(f"      Feature count : {len(feature_names)}")

print("\n[4/9] Imputing missing values...")
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

rolling_mean_map = {"Avg_Margin_3Y": "ProfitMargin", "Avg_Revenue_3Y": "Revenue"}
for roll_col, src_col in rolling_mean_map.items():
    if roll_col in X.columns and src_col in X.columns:
        X[roll_col] = X[roll_col].fillna(X[src_col])

if "GrowthRate" in X.columns:
    X["GrowthRate"] = X["GrowthRate"].fillna(X["GrowthRate"].median())
if "Growth_vs_Industry" in X.columns:
    X["Growth_vs_Industry"] = X["Growth_vs_Industry"].fillna(0)

print(f"      Total NaNs remaining : {X.isnull().sum().sum()}")

print("\n[5/9] Walk-forward temporal train/test split...")
SPLIT_YEAR = 2023
train_mask = df["Year"] < SPLIT_YEAR
test_mask  = df["Year"] >= SPLIT_YEAR

X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
X_test,  y_test  = X[test_mask].copy(),  y[test_mask].copy()

print(f"      Train : {X_train.shape[0]:,} rows | Test : {X_test.shape[0]:,} rows")

print("\n[6/9] Training XGBoost model...")
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos

model = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=spw, eval_metric="logloss",
    random_state=42, n_jobs=-1, early_stopping_rounds=30,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
best_iter = model.best_iteration
print(f"      Best iteration : {best_iter}")

print("\n[7/9] 5-Fold Stratified Cross-Validation...")
cv_model = xgb.XGBClassifier(
    n_estimators=best_iter, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=spw, eval_metric="logloss",
    random_state=42, n_jobs=-1,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_roc = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc",  n_jobs=-1)
cv_f1  = cross_val_score(cv_model, X, y, cv=cv, scoring="f1",       n_jobs=-1)
cv_acc = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"      ROC-AUC : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

print("\n[8/9] Evaluation metrics on held-out test set...")
X_test  = X_test.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)
df_test = df[test_mask].reset_index(drop=True)

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

metrics = {
    "Accuracy": acc, "Precision": prec, "Recall": rec,
    "F1": f1, "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
    "CV_ROC_AUC_Mean": cv_roc.mean(), "CV_ROC_AUC_Std": cv_roc.std(),
}

# ── Human-readable display names (same as SHAP script) ────────
name_map = {
    "Revenue": "Revenue", "ProfitMargin": "Profit Margin (%)",
    "MarketCap": "Market Cap", "GrowthRate": "Growth Rate (%)",
    "Price_to_Sales": "Price-to-Sales", "Log_Revenue": "Log(Revenue)",
    "Log_MarketCap": "Log(MarketCap)", "Revenue_to_MarketCap": "Revenue / MarketCap",
    "EBIT_proxy": "EBIT Proxy", "Return_on_Sales": "Return on Sales",
    "Profitability_Score": "Profitability Score",
    "Revenue_YoY": "Revenue YoY Growth (%)", "Revenue_2Y_Growth": "Revenue 2-Year Growth (%)",
    "Margin_YoY_Change": "Margin YoY Change (pp)", "MarketCap_YoY": "MarketCap YoY Change (%)",
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

# ── Feature group map (same as SHAP script) ───────────────────
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


# ============================================================
# STEP 9: LIME ANALYSIS
# ============================================================
print("\n[9/9] Generating LIME explanations and plots...")

# ── Build LIME Explainer ─────────────────────────────────────
# LIME needs to know which features are categorical.
# Binary flag columns and dummies are treated as categorical.
categorical_features = []
categorical_names = {}

for i, feat in enumerate(feature_names):
    unique_vals = X_train[feat].nunique()
    if unique_vals <= 2:
        categorical_features.append(i)
        categorical_names[i] = ["0", "1"]

explainer_lime = LimeTabularExplainer(
    training_data       = X_train.values,
    feature_names       = display_names,
    class_names         = ["No Controversy", "Controversy"],
    categorical_features= categorical_features,
    categorical_names   = categorical_names,
    mode                = "classification",
    random_state        = 42,
    discretize_continuous= True,    # bins continuous features for interpretability
)

# ── Compute LIME weights for N_LIME_SAMPLES test observations ─
# We sample proportionally to class distribution so both
# classes are well represented in the global importance.
np.random.seed(42)
n_test = len(X_test)
sample_idx = np.random.choice(n_test, size=min(N_LIME_SAMPLES, n_test), replace=False)

print(f"      Computing LIME explanations for {len(sample_idx)} test samples...")
print("      (This takes ~30 seconds — LIME fits a local model per sample)")

lime_weights_matrix = np.zeros((len(sample_idx), len(feature_names)))

for iter_i, test_i in enumerate(sample_idx):
    if iter_i % 50 == 0:
        print(f"      ... {iter_i}/{len(sample_idx)}")

    exp = explainer_lime.explain_instance(
        data_row          = X_test.iloc[test_i].values,
        predict_fn        = model.predict_proba,
        num_features      = len(feature_names),  # get all features
        num_samples       = 1000,
        labels            = (1,),    # explain class=1 (Controversy)
    )

    # Map LIME's feature names back to column indices
    # LIME returns (binned_feature_name, weight) tuples
    weight_dict = dict(exp.as_list(label=1))

    for feat_idx, disp_name in enumerate(display_names):
        # LIME bins continuous features: "Revenue YoY Growth (%) > 0.50"
        # We match by checking if the display name appears in the key
        matched = False
        for lime_feat_key, weight in weight_dict.items():
            if disp_name in lime_feat_key or lime_feat_key in disp_name:
                lime_weights_matrix[iter_i, feat_idx] += weight
                matched = True
                break
        # If no partial match, try exact match on raw feature name
        if not matched:
            raw_name = feature_names[feat_idx]
            for lime_feat_key, weight in weight_dict.items():
                if raw_name in lime_feat_key:
                    lime_weights_matrix[iter_i, feat_idx] += weight
                    break

print("      ✅ LIME computation complete.")

mean_abs_lime = np.abs(lime_weights_matrix).mean(axis=0)

# ── PLOT L1: Evaluation Dashboard (same as SHAP script) ───────
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "ESG Controversy Prediction — XGBoost Evaluation Dashboard\n"
    f"Train: 2015–{SPLIT_YEAR-1}  |  Test: {SPLIT_YEAR}–2025  |  "
    f"Features: {len(feature_names)}  |  Companies: 1,000",
    fontsize=16, fontweight="bold", color=NAVY, y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
metric_values = [acc, prec, rec, f1, roc_auc, pr_auc]
bar_colors    = [NAVY, TEAL, BLUE, GREEN, AMBER, RED]
bars = ax1.barh(metric_labels, metric_values, color=bar_colors, edgecolor="white", height=0.55)
ax1.set_xlim(0, 1.15)
ax1.set_title("Test Set Metrics", fontweight="bold", fontsize=12, color=NAVY)
ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
for bar, val in zip(bars, metric_values):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=10, fontweight="bold", color=NAVY)
ax1.set_xlabel("Score", fontsize=10)
ax1.grid(axis="x", alpha=0.2)

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Controversy", "Controversy"])
disp.plot(ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title("Confusion Matrix", fontweight="bold", fontsize=12, color=NAVY)

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax3.plot(fpr, tpr, color=NAVY, lw=2.5, label=f"XGBoost (AUC = {roc_auc:.4f})")
ax3.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.5, label="Random Baseline")
ax3.fill_between(fpr, tpr, alpha=0.08, color=NAVY)
ax3.set_xlabel("False Positive Rate", fontsize=10)
ax3.set_ylabel("True Positive Rate", fontsize=10)
ax3.set_title("ROC Curve", fontweight="bold", fontsize=12, color=NAVY)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.25)

ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("white")
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
ax4.plot(rec_curve, prec_curve, color=TEAL, lw=2.5, label=f"XGBoost (PR-AUC = {pr_auc:.4f})")
ax4.axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.5, lw=1.5,
            label=f"Baseline ({y_test.mean():.2f})")
ax4.fill_between(rec_curve, prec_curve, alpha=0.08, color=TEAL)
ax4.set_xlabel("Recall", fontsize=10)
ax4.set_ylabel("Precision", fontsize=10)
ax4.set_title("Precision-Recall Curve", fontweight="bold", fontsize=12, color=NAVY)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.25)

ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("white")
positions  = [1, 2, 3]
bp_colors_list = [NAVY, TEAL, GREEN]
bp = ax5.boxplot([cv_roc, cv_f1, cv_acc], positions=positions, patch_artist=True,
                 widths=0.45, medianprops=dict(color="white", linewidth=2.5))
for patch, color in zip(bp["boxes"], bp_colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax5.set_xticks(positions)
ax5.set_xticklabels(["ROC-AUC", "F1 Score", "Accuracy"], fontsize=10)
ax5.set_title("5-Fold CV Score Distribution", fontweight="bold", fontsize=12, color=NAVY)
ax5.set_ylim(0.4, 1.0)
ax5.grid(axis="y", alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("white")
proba_0 = y_proba[y_test.values == 0]
proba_1 = y_proba[y_test.values == 1]
ax6.hist(proba_0, bins=40, alpha=0.65, color=GREEN, label="Actual: No Controversy", density=True)
ax6.hist(proba_1, bins=40, alpha=0.65, color=RED,   label="Actual: Controversy",    density=True)
ax6.axvline(0.5, color=NAVY, linestyle="--", lw=2, label="Decision threshold (0.5)")
ax6.set_xlabel("Predicted Probability (Controversy)", fontsize=10)
ax6.set_ylabel("Density", fontsize=10)
ax6.set_title("Prediction Probability Distribution", fontweight="bold", fontsize=12, color=NAVY)
ax6.legend(fontsize=8)
ax6.grid(alpha=0.25)

plt.savefig(os.path.join(OUTPUT_DIR,  "L1_evaluation_dashboard.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: L1_evaluation_dashboard.png")


# ── PLOT L2: LIME Global Feature Importance ───────────────────
lime_fi_df = pd.DataFrame({
    "feature":    display_names,
    "importance": mean_abs_lime,
}).sort_values("importance", ascending=True).tail(20)

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

bar_colors_lime = [bar_color(f) for f in lime_fi_df["feature"]]

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor("white")
bars = ax.barh(lime_fi_df["feature"], lime_fi_df["importance"],
               color=bar_colors_lime, edgecolor="white", height=0.65)
for bar in bars:
    ax.text(bar.get_width() + 0.00005,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.5f}",
            va="center", ha="left", fontsize=9, color=NAVY)
ax.set_title(
    "LIME Feature Importance — Mean |LIME Weight| (Top 20)\n"
    f"Averaged over {len(sample_idx)} test samples  |  "
    "Higher value = stronger local influence on Controversy prediction",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Mean |LIME Weight|", fontsize=11)
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
plt.savefig(os.path.join(OUTPUT_DIR,  "L2_lime_feature_importance.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: L2_lime_feature_importance.png")


# ── PLOT L3: LIME Beeswarm-style Weight Distribution ──────────
# Shows weight distribution per feature (analogous to SHAP beeswarm)
top20_idx = np.argsort(mean_abs_lime)[::-1][:20]
top20_names  = [display_names[i]   for i in top20_idx]
top20_matrix = lime_weights_matrix[:, top20_idx]

fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor("white")

# Normalise feature values for coloring (same concept as SHAP beeswarm)
feature_vals_norm = np.zeros_like(top20_matrix)
for j, feat_idx in enumerate(top20_idx):
    col_vals = X_test.iloc[sample_idx, feat_idx].values.astype(float)
    col_min, col_max = col_vals.min(), col_vals.max()
    if col_max > col_min:
        feature_vals_norm[:, j] = (col_vals - col_min) / (col_max - col_min)
    else:
        feature_vals_norm[:, j] = 0.5

cmap = plt.cm.RdBu_r

for j in range(len(top20_names)):
    y_pos = len(top20_names) - 1 - j   # top feature at top
    weights = top20_matrix[:, j]
    colors  = cmap(feature_vals_norm[:, j])
    jitter  = np.random.uniform(-0.25, 0.25, size=len(weights))
    ax.scatter(weights, y_pos + jitter, c=colors, alpha=0.5, s=12, linewidths=0)

ax.set_yticks(range(len(top20_names)))
ax.set_yticklabels(list(reversed(top20_names)), fontsize=9)
ax.axvline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.6)
ax.set_xlabel("LIME Weight (positive → pushes toward Controversy)", fontsize=11)
ax.set_title(
    "LIME Beeswarm — Local Weight Distribution per Feature (Top 20)\n"
    "(Each dot = one test observation  |  color = feature value: blue=low, red=high)",
    fontsize=13, fontweight="bold", color=NAVY, pad=15
)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.4, aspect=15, pad=0.02)
cbar.set_label("Feature Value (low → high)", fontsize=9)
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,  "L3_lime_beeswarm.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: L3_lime_beeswarm.png")


# ── PLOT L4: LIME Waterfall — Two Individual Predictions ──────
idx_controversy    = np.where((y_test.values == 1) & (y_pred == 1))[0]
idx_no_controversy = np.where((y_test.values == 0) & (y_pred == 0))[0]

def lime_waterfall(ax, test_idx, label, label_color):
    """Plot a LIME waterfall for one test observation."""
    exp = explainer_lime.explain_instance(
        data_row = X_test.iloc[test_idx].values,
        predict_fn = model.predict_proba,
        num_features = LIME_NUM_FEATURES,
        num_samples  = 2000,
        labels       = (1,),
    )
    items = exp.as_list(label=1)                   # list of (feature_bin, weight)
    items_sorted = sorted(items, key=lambda x: x[1])  # sort ascending for waterfall

    feat_labels = [it[0][:35] for it in items_sorted]  # truncate long bin strings
    weights     = [it[1]      for it in items_sorted]
    colors      = [RED if w > 0 else BLUE for w in weights]

    # Waterfall: cumulative bars starting from base value
    base_val = model.predict_proba(X_test.iloc[[test_idx]].values)[0][1]
    ax.set_facecolor("white")
    bars = ax.barh(feat_labels, weights, color=colors, edgecolor="white", height=0.65, alpha=0.85)
    for bar, w in zip(bars, weights):
        ax.text(
            w + (0.001 if w >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{w:+.4f}", va="center",
            ha="left" if w >= 0 else "right",
            fontsize=8, color=NAVY
        )
    ax.axvline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_title(
        f"{label}\n(Predicted Controversy Prob: {base_val:.3f})",
        fontsize=11, fontweight="bold", color=label_color, pad=8
    )
    ax.set_xlabel("LIME Weight", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.2)

if len(idx_controversy) > 0 and len(idx_no_controversy) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "LIME Waterfall — Individual Prediction Explanations\n"
        "Left: Correctly Predicted Controversy  |  "
        "Right: Correctly Predicted Non-Controversy",
        fontsize=13, fontweight="bold", color=NAVY
    )
    lime_waterfall(axes[0], idx_controversy[0],    "Predicted: CONTROVERSY ✅",    RED)
    lime_waterfall(axes[1], idx_no_controversy[0], "Predicted: NO CONTROVERSY ✅", GREEN)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,  "L4_lime_waterfall.png"), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("      ✅ Saved: L4_lime_waterfall.png")
else:
    print("      ⚠️  Skipped waterfall: insufficient correct predictions")


# ── PLOT L5: LIME Feature Group Importance ────────────────────
group_importance_lime = {}
for feat, imp in zip(feature_names, mean_abs_lime):
    if feat.startswith("Industry_"):
        group = "J: Industry"
    elif feat.startswith("Region_"):
        group = "J: Region"
    else:
        group = group_map.get(feat, "Other")
    group_importance_lime[group] = group_importance_lime.get(group, 0) + imp

grp_lime_df = pd.DataFrame(
    list(group_importance_lime.items()), columns=["Group", "Total LIME Weight"]
).sort_values("Total LIME Weight", ascending=True)

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
colors = [group_colors.get(g, NAVY) for g in grp_lime_df["Group"]]
bars = ax.barh(grp_lime_df["Group"], grp_lime_df["Total LIME Weight"],
               color=colors, edgecolor="white", height=0.6)
for bar in bars:
    ax.text(bar.get_width() + 0.0001,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center", ha="left", fontsize=10, color=NAVY)
ax.set_title(
    "LIME Feature Group Importance — Summed Mean |LIME Weight|\n"
    "Which category of financial features drives ESG controversy prediction most?",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Total Mean |LIME Weight| (summed across group)", fontsize=11)
ax.grid(axis="x", alpha=0.2)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,  "L5_lime_group_importance.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: L5_lime_group_importance.png")


# ── PLOT L6: SHAP vs LIME Comparison Bar ─────────────────────
# Load pre-computed SHAP values if available; else compute them
print("      Computing SHAP values for comparison plot...")
shap_explainer = shap.TreeExplainer(model)
shap_values    = shap_explainer.shap_values(X_test)
mean_abs_shap  = np.abs(shap_values).mean(axis=0)

# Normalise both to [0,1] for fair visual comparison
shap_norm = mean_abs_shap / mean_abs_shap.max()
lime_norm = mean_abs_lime / mean_abs_lime.max() if mean_abs_lime.max() > 0 else mean_abs_lime

# Top-15 by SHAP rank
top15_shap_idx   = np.argsort(mean_abs_shap)[::-1][:15]
compare_names    = [display_names[i] for i in top15_shap_idx]
shap_vals_plot   = shap_norm[top15_shap_idx]
lime_vals_plot   = lime_norm[top15_shap_idx]

y_pos = np.arange(len(compare_names))
bar_h = 0.38

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor("white")

bars_shap = ax.barh(y_pos + bar_h/2, shap_vals_plot, height=bar_h,
                    color=NAVY, alpha=0.85, label="SHAP (normalised)", edgecolor="white")
bars_lime = ax.barh(y_pos - bar_h/2, lime_vals_plot, height=bar_h,
                    color=LIME_COLOR, alpha=0.85, label="LIME (normalised)", edgecolor="white")

ax.set_yticks(y_pos)
ax.set_yticklabels(compare_names, fontsize=10)
ax.set_xlabel("Normalised Importance Score (0 = min, 1 = max within method)", fontsize=11)
ax.set_title(
    "SHAP vs LIME — Feature Importance Comparison (Top 15 by SHAP)\n"
    "Both methods normalised to [0,1].  Agreement = similar bar lengths.\n"
    "Divergence = LIME captures different local dynamics vs SHAP global attribution.",
    fontsize=13, fontweight="bold", color=NAVY, pad=15
)
ax.legend(fontsize=11, loc="lower right")
ax.axvline(0, color="gray", linewidth=1, alpha=0.4)
ax.grid(axis="x", alpha=0.2)

# Annotate agreement / disagreement
for i, (sv, lv) in enumerate(zip(shap_vals_plot, lime_vals_plot)):
    diff = abs(sv - lv)
    if diff < 0.1:
        ax.text(max(sv, lv) + 0.02, y_pos[i], "✓", fontsize=9,
                color=GREEN, va="center", fontweight="bold")
    elif diff > 0.3:
        ax.text(max(sv, lv) + 0.02, y_pos[i], "!", fontsize=9,
                color=RED, va="center", fontweight="bold")

ax.text(0.99, 0.01, "✓ = Agreement (<10% diff)   ! = Divergence (>30% diff)",
        transform=ax.transAxes, fontsize=8, color=NAVY,
        va="bottom", ha="right", style="italic")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "L6_shap_vs_lime_comparison.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: L6_shap_vs_lime_comparison.png")


# ── COMPANY-LEVEL LIME RISK DRIVER CSV ────────────────────────
print("      Building company-level LIME risk driver explanations...")

feature_label_map = {
    "Declining Revenue": "declining revenue",
    "Margin Volatility": "high margin volatility",
    "Negative Margin":   "negative margin",
    "Margin vs Industry":"below-industry profitability",
    "Revenue vs Industry":"below-industry revenue",
    "MCap Collapse":     "market cap collapse",
    "Low Growth":        "low growth + low margin",
    "Revenue Surprise":  "negative revenue surprise",
    "Profitability Score":"weak profitability",
    "Valuation Excess":  "valuation excess",
}

risk_prob = model.predict_proba(X_test)[:, 1]
explanations = []

for test_i in sample_idx:
    exp = explainer_lime.explain_instance(
        data_row   = X_test.iloc[test_i].values,
        predict_fn = model.predict_proba,
        num_features = 6,
        num_samples  = 500,
        labels       = (1,),
    )
    pos_drivers = [(f, w) for f, w in exp.as_list(label=1) if w > 0]
    pos_drivers.sort(key=lambda x: -x[1])

    def clean_driver(feat_str):
        for key, label in feature_label_map.items():
            if key.lower() in feat_str.lower():
                return label
        # fallback: strip bin notation (e.g., "> 0.50") and clean
        import re
        clean = re.sub(r"[<>= \d\.]+$", "", feat_str).strip()
        return clean.lower().replace("_", " ")

    drivers = [clean_driver(f) for f, _ in pos_drivers[:4]]
    explanations.append({
        "CompanyID":        df_test.loc[test_i, "CompanyID"],
        "Year":             df_test.loc[test_i, "Year"],
        "Risk_Probability": round(risk_prob[test_i] * 100, 2),
        "LIME_Driver_1":    drivers[0] if len(drivers) > 0 else None,
        "LIME_Driver_2":    drivers[1] if len(drivers) > 1 else None,
        "LIME_Driver_3":    drivers[2] if len(drivers) > 2 else None,
        "LIME_Driver_4":    drivers[3] if len(drivers) > 3 else None,
    })

lime_risk_df = pd.DataFrame(explanations)
lime_risk_df.to_csv(os.path.join(OUTPUT_DIR,  "company_lime_risk_explanations.csv"), index=False)
print("      ✅ Saved: company_lime_risk_explanations.csv")


# ── SAVE MODEL + METRICS ──────────────────────────────────────
joblib.dump(model, os.path.join(OUTPUT_DIR,  "xgboost_esg_model_lime.pkl"))

metrics_df = pd.DataFrame([metrics])
for fold_i, score in enumerate(cv_roc, 1):
    metrics_df[f"CV_ROC_AUC_Fold{fold_i}"] = score
metrics_df.to_csv(os.path.join(OUTPUT_DIR,  "model_metrics_lime.csv"), index=False)
print("      ✅ Saved: model_metrics_lime.csv")
print("      ✅ Saved: xgboost_esg_model_lime.pkl")

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  PR-AUC    : {pr_auc:.4f}")
print(f"  CV ROC-AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print("=" * 60)
print("\n✅ All LIME outputs saved.")
print("\nOutput files:")
print("  L1_evaluation_dashboard.png      — same metrics as SHAP script")
print("  L2_lime_feature_importance.png   — analogous to 03_shap_feature_importance.png")
print("  L3_lime_beeswarm.png             — analogous to 02_shap_beeswarm.png")
print("  L4_lime_waterfall.png            — analogous to 04_shap_waterfall.png")
print("  L5_lime_group_importance.png     — analogous to 06_shap_group_importance.png")
print("  L6_shap_vs_lime_comparison.png   — side-by-side SHAP vs LIME comparison")
print("  company_lime_risk_explanations.csv")
print("  model_metrics_lime.csv")