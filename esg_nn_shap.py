"""
============================================================
ESG Controversy Prediction — Neural Network + SHAP
============================================================
Dataset : esg_controversy_features.csv

Backend : PyTorch (CPU-friendly, no AVX/AVX2 requirement)

Why Neural Network?
  • Multi-layer perceptron captures complex non-linear
    feature interactions that linear models miss.
  • Dropout regularisation prevents overfitting on
    moderate-sized financial datasets.
  • BatchNorm stabilises training and speeds convergence.
  • Fully compatible with KernelSHAP (model-agnostic
    explainer), enabling a fair comparison with the SVM run.

Architecture:
  Input → BatchNorm1d → Linear(256, ReLU) → Dropout(0.3)
        → Linear(128, ReLU) → Dropout(0.3)
        → Linear(64,  ReLU) → Dropout(0.2)
        → Linear(1)  [logits → Sigmoid at inference]

Steps:
  1.  Load engineered dataset
  2.  Drop identifier columns
  3.  Separate target
  4.  Impute NaN values
  5.  Walk-forward temporal train/test split
  6.  Scale features (mandatory for NNs)
  7.  Handle class imbalance with pos_weight
  8.  Train Neural Network with Early Stopping
  9.  5-Fold Stratified Cross-Validation
 10.  Evaluation metrics + dashboard plot
 11.  SHAP analysis  (KernelSHAP — beeswarm, bar, waterfall)
 12.  Company-level risk driver CSV
 13.  Save model + metrics CSV
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR1 = os.path.join(BASE_DIR, "output")
# Create output folder inside it
OUTPUT_DIR = os.path.join(BASE_DIR1, "neural_network")
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cpu")

# ── Colour palette ────────────────────────────────────────────
NAVY     = "#1B3A6B"
TEAL     = "#0D7377"
RED      = "#B91C1C"
GREEN    = "#15803D"
AMBER    = "#D97706"
BLUE     = "#2563EB"
BG       = "#F1F5F9"
NN_COLOR = "#DB2777"

# ── SHAP sample sizes ─────────────────────────────────────────
N_SHAP_BG      = 100
N_SHAP_EXPLAIN = 150

print("=" * 60)
print("  ESG Controversy Prediction — Neural Network + SHAP")
print("  Backend: PyTorch (CPU)")
print("=" * 60)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/10] Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "esg_controversy_features.csv"))
print(f"       Shape    : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"       Years    : {df['Year'].min()} - {df['Year'].max()}")
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
            print(f"       -> {col:<30} filled {n:,} NaNs with 0")

for roll_col, src_col in [("Avg_Margin_3Y", "ProfitMargin"), ("Avg_Revenue_3Y", "Revenue")]:
    if roll_col in X.columns and src_col in X.columns:
        n = X[roll_col].isnull().sum()
        if n > 0:
            X[roll_col] = X[roll_col].fillna(X[src_col])
            print(f"       -> {roll_col:<30} filled {n:,} NaNs from {src_col}")

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

print(f"       Train : {X_train_raw.shape[0]:,} rows | controversy rate: {y_train.mean():.1%}")
print(f"       Test  : {X_test_raw.shape[0]:,} rows  | controversy rate: {y_test.mean():.1%}")


# ============================================================
# STEP 6: FEATURE SCALING
# ============================================================
print("\n[6/10] Scaling features (StandardScaler)...")
scaler       = StandardScaler()
X_train      = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test       = scaler.transform(X_test_raw).astype(np.float32)
X_scaled_all = scaler.transform(X).astype(np.float32)
print(f"       Scaler fitted on {X_train.shape[0]:,} training rows")


# ============================================================
# STEP 7: CLASS IMBALANCE
# ============================================================
print("\n[7/10] Computing pos_weight for imbalance handling...")
n_neg          = int((y_train == 0).sum())
n_pos          = int((y_train == 1).sum())
pos_weight_val = n_neg / n_pos
pos_weight     = torch.tensor([pos_weight_val], dtype=torch.float32)
print(f"       neg={n_neg:,}  pos={n_pos:,}  pos_weight={pos_weight_val:.4f}")


# ============================================================
# STEP 8: DEFINE & TRAIN NEURAL NETWORK (PyTorch)
# ============================================================
print("\n[8/10] Building and training Neural Network (PyTorch)...")

N_FEATURES = X_train.shape[1]


class ESG_MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_loader(X_arr, y_arr, batch_size=64, shuffle=True):
    tx = torch.tensor(X_arr, dtype=torch.float32)
    ty = torch.tensor(y_arr, dtype=torch.float32)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


def train_model(X_tr, y_tr, n_features,
                epochs=200, batch_size=64, lr=1e-3,
                val_frac=0.15, patience=15, verbose=True):
    n_val  = max(1, int(len(X_tr) * val_frac))
    idx    = np.random.permutation(len(X_tr))
    val_i, tr_i = idx[:n_val], idx[n_val:]

    tr_loader  = make_loader(X_tr[tr_i],  y_tr[tr_i],  batch_size, shuffle=True)
    val_loader = make_loader(X_tr[val_i], y_tr[val_i], batch_size, shuffle=False)

    net       = ESG_MLP(n_features).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-6
    )

    history        = {"loss": [], "val_loss": [], "val_auc": []}
    best_val_auc   = -1.0
    best_state     = None
    no_improve     = 0

    for epoch in range(1, epochs + 1):
        net.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_i)

        net.eval()
        val_loss  = 0.0
        all_probs = []
        all_true  = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = net(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_true.extend(yb.cpu().numpy())
        val_loss /= len(val_i)

        try:
            val_auc = roc_auc_score(all_true, all_probs)
        except Exception:
            val_auc = 0.5

        scheduler.step(val_auc)
        history["loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"       Epoch {epoch:3d}/{epochs}  "
                  f"loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_AUC={val_auc:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"       Early stop at epoch {epoch} (best val_AUC={best_val_auc:.4f})")
            break

    net.load_state_dict(best_state)
    return net, history


print("       Architecture: Input -> BN -> 256 -> 128 -> 64 -> 1")
model, history = train_model(
    X_train, y_train.values.astype(np.float32),
    n_features=N_FEATURES, epochs=200, batch_size=64,
    lr=1e-3, val_frac=0.15, patience=15, verbose=True
)
print(f"       Training complete.  Epochs run: {len(history['loss'])}")


def nn_predict_proba(X_arr):
    """Return (n, 2) probability array compatible with SHAP."""
    model.eval()
    with torch.no_grad():
        t     = torch.tensor(X_arr.astype(np.float32), dtype=torch.float32)
        prob1 = torch.sigmoid(model(t)).numpy().ravel()
    return np.column_stack([1 - prob1, prob1])


# ============================================================
# STEP 9: 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================
print("\n[9/10] 5-Fold Stratified Cross-Validation...")


class TorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features, epochs=80):
        self.n_features = n_features
        self.epochs     = epochs

    def fit(self, X, y):
        self.model_, _ = train_model(
            X.astype(np.float32), y.astype(np.float32),
            n_features=self.n_features, epochs=self.epochs,
            batch_size=64, lr=1e-3, val_frac=0.15, patience=10, verbose=False
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            t = torch.tensor(X.astype(np.float32), dtype=torch.float32)
            p = torch.sigmoid(self.model_(t)).numpy().ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


cv5      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_roc   = np.zeros(5)
cv_f1    = np.zeros(5)
cv_acc   = np.zeros(5)

for fold, (tr_i, val_i) in enumerate(cv5.split(X_scaled_all, y.values)):
    print(f"       Fold {fold+1}/5 ...")
    w = TorchWrapper(n_features=N_FEATURES, epochs=80)
    w.fit(X_scaled_all[tr_i], y.values[tr_i])
    prob = w.predict_proba(X_scaled_all[val_i])[:, 1]
    pred = (prob >= 0.5).astype(int)
    cv_roc[fold] = roc_auc_score(y.values[val_i], prob)
    cv_f1[fold]  = f1_score(y.values[val_i], pred)
    cv_acc[fold] = accuracy_score(y.values[val_i], pred)

print(f"       ROC-AUC  : {cv_roc.mean():.4f} +/- {cv_roc.std():.4f}")
print(f"       F1 Score : {cv_f1.mean():.4f}  +/- {cv_f1.std():.4f}")
print(f"       Accuracy : {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")


# ============================================================
# EVALUATION METRICS
# ============================================================
print("\n      Evaluation metrics on held-out test set...")
y_proba = nn_predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)

print(f"\n  {'Metric':<25} {'Value':>10}")
print(f"  {'---'*12}")
print(f"  {'Accuracy':<25} {acc:>10.4f}")
print(f"  {'Precision':<25} {prec:>10.4f}")
print(f"  {'Recall':<25} {rec:>10.4f}")
print(f"  {'F1 Score':<25} {f1:>10.4f}")
print(f"  {'ROC-AUC':<25} {roc_auc:>10.4f}")
print(f"  {'PR-AUC':<25} {pr_auc:>10.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Controversy','Controversy'])}")

metrics = {
    "Model": "Neural-Network-PyTorch",
    "Architecture": "256-128-64",
    "Dropout": "0.3/0.3/0.2",
    "Epochs_Run": len(history["loss"]),
    "Accuracy": acc, "Precision": prec, "Recall": rec,
    "F1": f1, "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
    "CV_ROC_AUC_Mean": cv_roc.mean(), "CV_ROC_AUC_Std": cv_roc.std(),
}

# ── Display names & group maps ────────────────────────────────
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
    "PS_Margin_Ratio": "P/S Margin Ratio",
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
    "G: Valuation Anomaly":           NN_COLOR,
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


legend_patches = [
    mpatches.Patch(color=TEAL,  label="Profitability"),
    mpatches.Patch(color=BLUE,  label="Revenue / Size / Growth"),
    mpatches.Patch(color=RED,   label="Distress Flags"),
    mpatches.Patch(color=GREEN, label="Peer-Relative"),
    mpatches.Patch(color=AMBER, label="Industry / Region"),
    mpatches.Patch(color=NAVY,  label="Other"),
]


# ============================================================
# STEP 10: PLOTS
# ============================================================
print("\n[10/10] Generating plots...")

# ── N1: Evaluation Dashboard ─────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"ESG Controversy Prediction — Neural Network (256-128-64, PyTorch) Dashboard\n"
    f"Train: 2015-{SPLIT_YEAR-1}  |  Test: {SPLIT_YEAR}-2025  |  "
    f"Features: {len(feature_names)}  |  Companies: 1,000",
    fontsize=15, fontweight="bold", color=NAVY, y=0.99
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("white")
m_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
m_values = [acc, prec, rec, f1, roc_auc, pr_auc]
m_colors = [NAVY, TEAL, BLUE, GREEN, AMBER, RED]
bars = ax1.barh(m_labels, m_values, color=m_colors, edgecolor="white", height=0.55)
ax1.set_xlim(0, 1.15)
ax1.set_title("Test Set Metrics", fontweight="bold", fontsize=12, color=NAVY)
ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
for bar, val in zip(bars, m_values):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=10, fontweight="bold", color=NAVY)
ax1.grid(axis="x", alpha=0.2)

ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("white")
cm_mat = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm_mat, display_labels=["No Controversy","Controversy"]
                       ).plot(ax=ax2, cmap="RdPu", colorbar=False)
ax2.set_title("Confusion Matrix", fontweight="bold", fontsize=12, color=NAVY)

ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("white")
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax3.plot(fpr, tpr, color=NN_COLOR, lw=2.5, label=f"NN (AUC={roc_auc:.4f})")
ax3.plot([0,1],[0,1], "k--", alpha=0.4, lw=1.5, label="Random")
ax3.fill_between(fpr, tpr, alpha=0.08, color=NN_COLOR)
ax3.set_xlabel("FPR", fontsize=10); ax3.set_ylabel("TPR", fontsize=10)
ax3.set_title("ROC Curve", fontweight="bold", fontsize=12, color=NAVY)
ax3.legend(fontsize=9); ax3.grid(alpha=0.25)

ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor("white")
prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
ax4.plot(rec_c, prec_c, color=TEAL, lw=2.5, label=f"NN (PR-AUC={pr_auc:.4f})")
ax4.axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.5, lw=1.5,
            label=f"Baseline ({y_test.mean():.2f})")
ax4.fill_between(rec_c, prec_c, alpha=0.08, color=TEAL)
ax4.set_xlabel("Recall", fontsize=10); ax4.set_ylabel("Precision", fontsize=10)
ax4.set_title("Precision-Recall Curve", fontweight="bold", fontsize=12, color=NAVY)
ax4.legend(fontsize=9); ax4.grid(alpha=0.25)

ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor("white")
bp = ax5.boxplot([cv_roc, cv_f1, cv_acc], positions=[1,2,3], patch_artist=True,
                 widths=0.45, medianprops=dict(color="white", linewidth=2.5))
for patch, color in zip(bp["boxes"], [NN_COLOR, TEAL, GREEN]):
    patch.set_facecolor(color); patch.set_alpha(0.85)
ax5.set_xticks([1,2,3])
ax5.set_xticklabels(["ROC-AUC","F1","Accuracy"], fontsize=10)
ax5.set_title("5-Fold CV Distribution", fontweight="bold", fontsize=12, color=NAVY)
ax5.set_ylim(0.4, 1.0); ax5.grid(axis="y", alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor("white")
ax6.hist(y_proba[y_test.values==0], bins=40, alpha=0.65, color=GREEN,
         label="Actual: No Controversy", density=True)
ax6.hist(y_proba[y_test.values==1], bins=40, alpha=0.65, color=RED,
         label="Actual: Controversy", density=True)
ax6.axvline(0.5, color=NAVY, linestyle="--", lw=2, label="Threshold 0.5")
ax6.set_xlabel("Predicted Probability", fontsize=10); ax6.set_ylabel("Density", fontsize=10)
ax6.set_title("Probability Distribution", fontweight="bold", fontsize=12, color=NAVY)
ax6.legend(fontsize=8); ax6.grid(alpha=0.25)

ax7 = fig.add_subplot(gs[2, 0]); ax7.set_facecolor("white")
ax7.plot(history["loss"],     color=NN_COLOR, lw=2,              label="Train Loss")
ax7.plot(history["val_loss"], color=NAVY,     lw=2, linestyle="--", label="Val Loss")
ax7.set_xlabel("Epoch", fontsize=10); ax7.set_ylabel("BCE Loss", fontsize=10)
ax7.set_title("Training vs Validation Loss", fontweight="bold", fontsize=12, color=NAVY)
ax7.legend(fontsize=9); ax7.grid(alpha=0.25)

ax8 = fig.add_subplot(gs[2, 1]); ax8.set_facecolor("white")
ax8.plot(history["val_auc"], color=TEAL, lw=2, label="Val AUC")
ax8.axhline(roc_auc, color=NN_COLOR, lw=1.5, linestyle="--",
            label=f"Test AUC={roc_auc:.4f}")
ax8.set_xlabel("Epoch", fontsize=10); ax8.set_ylabel("ROC-AUC", fontsize=10)
ax8.set_title("Validation AUC over Training", fontweight="bold", fontsize=12, color=NAVY)
ax8.legend(fontsize=9); ax8.grid(alpha=0.25)

ax9 = fig.add_subplot(gs[2, 2]); ax9.set_facecolor("white"); ax9.axis("off")
arch_text = (
    "Neural Network Architecture\n\n"
    f"  Input Layer  ({N_FEATURES} features)\n"
    "  BatchNorm1d\n"
    "  Linear 256   ReLU\n"
    "  Dropout 30%\n"
    "  Linear 128   ReLU\n"
    "  Dropout 30%\n"
    "  Linear 64    ReLU\n"
    "  Dropout 20%\n"
    "  Linear 1     Sigmoid\n\n"
    f"  Optimizer : Adam  lr=1e-3\n"
    f"  Loss      : BCEWithLogitsLoss\n"
    f"  pos_weight: {pos_weight_val:.3f}\n"
    f"  Epochs    : {len(history['loss'])} (early stop)\n"
    f"  Batch     : 64  |  Val: 15%\n"
    f"  Backend   : PyTorch (CPU)"
)
ax9.text(0.05, 0.95, arch_text, transform=ax9.transAxes,
         fontsize=10, va="top", fontfamily="monospace", color=NAVY,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#E2E8F0", alpha=0.8))

plt.savefig(os.path.join(OUTPUT_DIR,  "N1_evaluation_dashboard.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      Saved: N1_evaluation_dashboard.png")


# ── SHAP (KernelExplainer) ────────────────────────────────────
print(f"\n      Computing KernelSHAP ({N_SHAP_BG} bg, {N_SHAP_EXPLAIN} explain)...")

np.random.seed(42)
bg_idx         = np.random.choice(len(X_train), size=N_SHAP_BG, replace=False)
X_bg           = X_train[bg_idx]
explainer_shap = shap.KernelExplainer(nn_predict_proba, X_bg, link="logit")
shap_idx       = np.random.choice(len(X_test),
                                  size=min(N_SHAP_EXPLAIN, len(X_test)), replace=False)
X_shap_test    = X_test[shap_idx]
shap_values_raw = explainer_shap.shap_values(X_shap_test, nsamples=200)

if isinstance(shap_values_raw, list):
    shap_values = np.array(shap_values_raw[1])
else:
    shap_values = np.array(shap_values_raw)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]
assert shap_values.ndim == 2

mean_abs_shap = np.abs(shap_values).mean(axis=0)
print("      SHAP computation complete.")

# N2 beeswarm
X_shap_df = pd.DataFrame(X_shap_test, columns=feature_names)
fig, ax = plt.subplots(figsize=(13, 11)); fig.patch.set_facecolor(BG)
shap.summary_plot(shap_values, X_shap_df, feature_names=display_names,
                  show=False, plot_size=None, max_display=20,
                  color_bar_label="Feature Value (low -> high)")
plt.title("SHAP (KernelSHAP) Beeswarm — ESG Controversy Prediction (Neural Network)",
          fontsize=13, fontweight="bold", color=NAVY, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,  "N2_shap_beeswarm.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("      Saved: N2_shap_beeswarm.png")

# N3 bar
fi_df = pd.DataFrame({"feature": display_names, "importance": mean_abs_shap})
fi_df = fi_df.sort_values("importance", ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(13, 10)); fig.patch.set_facecolor(BG); ax.set_facecolor("white")
colors_ = [bar_color(f) for f in fi_df["feature"]]
bars_   = ax.barh(fi_df["feature"], fi_df["importance"],
                  color=colors_, edgecolor="white", height=0.65)
for bar_ in bars_:
    ax.text(bar_.get_width()+0.0001, bar_.get_y()+bar_.get_height()/2,
            f"{bar_.get_width():.4f}", va="center", ha="left", fontsize=9, color=NAVY)
ax.set_title(f"SHAP Feature Importance — Mean |SHAP Value| Top 20 (Neural Network)",
             fontsize=13, fontweight="bold", color=NAVY, pad=12)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11); ax.grid(axis="x", alpha=0.2)
ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,  "N3_shap_feature_importance.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("      Saved: N3_shap_feature_importance.png")

# N4 waterfall
y_pred_shap = (nn_predict_proba(X_shap_test)[:,1] >= 0.5).astype(int)
y_test_shap = y_test.values[shap_idx]
idx_cont    = np.where((y_test_shap==1) & (y_pred_shap==1))[0]
idx_nocont  = np.where((y_test_shap==0) & (y_pred_shap==0))[0]

if len(idx_cont) > 0 and len(idx_nocont) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8)); fig.patch.set_facecolor(BG)
    fig.suptitle("SHAP Waterfall — Individual Prediction Explanations (Neural Network)",
                 fontsize=13, fontweight="bold", color=NAVY)
    base_val = explainer_shap.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).ravel()[1])
    else:
        base_val = float(base_val)
    for ax_i, (i, label, lcolor) in enumerate([
        (idx_cont[0],   "Predicted: CONTROVERSY",    RED),
        (idx_nocont[0], "Predicted: NO CONTROVERSY", GREEN),
    ]):
        plt.sca(axes[ax_i])
        exp = shap.Explanation(values=shap_values[i], base_values=base_val,
                               data=X_shap_test[i], feature_names=display_names)
        shap.waterfall_plot(exp, max_display=15, show=False)
        axes[ax_i].set_title(label, fontsize=11, fontweight="bold", color=lcolor, pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,  "N4_shap_waterfall.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(); print("      Saved: N4_shap_waterfall.png")
else:
    print("      Skipped N4: not enough correct predictions for waterfall.")

# N5 group importance
group_imp_shap = {}
for feat, imp in zip(feature_names, mean_abs_shap):
    grp = ("J: Industry" if feat.startswith("Industry_") else
           "J: Region"   if feat.startswith("Region_")   else
           group_map.get(feat, "Other"))
    group_imp_shap[grp] = group_imp_shap.get(grp, 0) + imp
grp_shap_df = pd.DataFrame(list(group_imp_shap.items()),
                            columns=["Group","Total SHAP"]).sort_values("Total SHAP", ascending=True)
fig, ax = plt.subplots(figsize=(12, 7)); fig.patch.set_facecolor(BG); ax.set_facecolor("white")
colors_ = [group_colors.get(g, NAVY) for g in grp_shap_df["Group"]]
bars_   = ax.barh(grp_shap_df["Group"], grp_shap_df["Total SHAP"],
                  color=colors_, edgecolor="white", height=0.6)
for bar_ in bars_:
    ax.text(bar_.get_width()+0.0001, bar_.get_y()+bar_.get_height()/2,
            f"{bar_.get_width():.4f}", va="center", ha="left", fontsize=10, color=NAVY)
ax.set_title("SHAP Feature Group Importance (Neural Network)",
             fontsize=13, fontweight="bold", color=NAVY, pad=12)
ax.set_xlabel("Total Mean |SHAP Value|", fontsize=11); ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,  "N5_shap_group_importance.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("      Saved: N5_shap_group_importance.png")


# ── Company-level risk CSV ────────────────────────────────────
print("      Building company risk driver CSV...")
risk_prob = nn_predict_proba(X_test)[:, 1]

shap_idx_risk = np.random.choice(len(X_test), size=min(50, len(X_test)), replace=False)
shap_drivers_rows = []
for test_i in shap_idx_risk:
    sv = shap_values[np.where(shap_idx == test_i)[0]]
    if len(sv) == 0:
        continue
    sv = sv[0]
    top_feat_idx = np.argsort(np.abs(sv))[::-1][:4]
    drivers = [display_names[i] for i in top_feat_idx]
    shap_drivers_rows.append({
        "CompanyID":        df_test.loc[test_i, "CompanyID"],
        "Year":             df_test.loc[test_i, "Year"],
        "Risk_Probability": round(risk_prob[test_i] * 100, 2),
        "SHAP_Driver_1":    drivers[0] if len(drivers) > 0 else None,
        "SHAP_Driver_2":    drivers[1] if len(drivers) > 1 else None,
        "SHAP_Driver_3":    drivers[2] if len(drivers) > 2 else None,
        "SHAP_Driver_4":    drivers[3] if len(drivers) > 3 else None,
    })

pd.DataFrame(shap_drivers_rows).to_csv(
    os.path.join(OUTPUT_DIR, "company_nn_risk_explanations.csv"), index=False
)
print("      Saved: company_nn_risk_explanations.csv")

# ── Save model artefacts ──────────────────────────────────────
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "nn_esg_model.pt"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "nn_scaler.pkl"))
metrics_df = pd.DataFrame([metrics])
for fold_i, score in enumerate(cv_roc, 1):
    metrics_df[f"CV_ROC_AUC_Fold{fold_i}"] = score
metrics_df.to_csv(os.path.join(OUTPUT_DIR,  "nn_model_metrics.csv"), index=False)
print("      Saved: nn_esg_model.pt  (PyTorch state_dict)")
print("      Saved: nn_scaler.pkl")
print("      Saved: nn_model_metrics.csv")

print("\n" + "=" * 60)
print("  SUMMARY — Neural Network (256-128-64, PyTorch CPU)")
print("=" * 60)
print(f"  Architecture  : Input -> BN -> 256 -> 128 -> 64 -> 1")
print(f"  Epochs run    : {len(history['loss'])}")
print(f"  Accuracy      : {acc:.4f}")
print(f"  Precision     : {prec:.4f}")
print(f"  Recall        : {rec:.4f}")
print(f"  F1 Score      : {f1:.4f}")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  PR-AUC        : {pr_auc:.4f}")
print(f"  CV ROC-AUC    : {cv_roc.mean():.4f} +/- {cv_roc.std():.4f}")
print("=" * 60)
print("\nOutput files (output/neural_network/):")
print("  N1_evaluation_dashboard.png")
print("  N2_shap_beeswarm.png")
print("  N3_shap_feature_importance.png")
print("  N4_shap_waterfall.png")
print("  N5_shap_group_importance.png")
print("  company_nn_risk_explanations.csv")
print("  nn_model_metrics.csv")
print("  nn_esg_model.pt")
print("  nn_scaler.pkl")
print("\nAll outputs saved.")