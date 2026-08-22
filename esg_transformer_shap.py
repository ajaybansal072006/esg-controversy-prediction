"""
============================================================
ESG Controversy Prediction — Transformer + SHAP
============================================================
Dataset : esg_controversy_features.csv
          (output of esg_feature_engineering.py)

Architecture:
  A tabular Transformer encoder (FT-Transformer style) that
  treats each feature as a separate token, applies multi-head
  self-attention, and produces a binary controversy prediction.

Steps:
  1.  Load engineered dataset
  2.  Drop non-feature identifier columns (CompanyID, Year)
  3.  Separate target column (ESG_Controversy)
  4.  Impute NaN values
  5.  Walk-forward temporal train / test split (2015-2022 / 2023-2025)
  6.  Standard-scale continuous features
  7.  Build & train Transformer (PyTorch)
  8.  5-Fold Cross-Validation
  9.  Evaluation metrics (Accuracy, Precision, Recall, F1,
      ROC-AUC, PR-AUC, Confusion Matrix, Classification Report)
 10.  SHAP analysis via KernelExplainer (model-agnostic,
      beeswarm, bar, waterfall, dependence)
 11.  Save all plots + model (.pt) + metrics CSV

Bank-Controversy Focus
  The same pipeline applies directly to bank ESG-controversy
  prediction.  Banks often show elevated controversy scores
  due to: lending to carbon-heavy sectors, governance scandals,
  fee practices, and regulatory fines.  The peer-relative
  features (Revenue_Pctile_Industry, Margin_vs_Industry etc.)
  are especially informative for banking since banks cluster
  tightly within the Finance industry.
============================================================
"""

import warnings, os, sys
warnings.filterwarnings("ignore")
# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR1 = os.path.join(BASE_DIR, "output")
# Create output folder inside it
OUTPUT_DIR = os.path.join(BASE_DIR1, "transformer")
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay,
)

# ── Colour palette (mirrors XGBoost script) ──────────────────
NAVY  = "#1B3A6B"
TEAL  = "#0D7377"
RED   = "#B91C1C"
GREEN = "#15803D"
AMBER = "#D97706"
BLUE  = "#2563EB"
GRAY  = "#F8FAFC"
BG    = "#F1F5F9"
PURPLE = "#6D28D9"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  ESG Controversy Prediction — Transformer + SHAP")
print("=" * 60)
print(f"  Device : {DEVICE}")


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/9] Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "esg_controversy_features.csv"))
print(f"      Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"      Years      : {df['Year'].min()} – {df['Year'].max()}")
print(f"      Companies  : {df['CompanyID'].nunique():,}")


# ============================================================
# STEP 2: DROP IDENTIFIER COLUMNS
# ============================================================
print("\n[2/9] Dropping identifier columns...")
ID_COLS = ["CompanyID", "Year"]
for col in ID_COLS:
    print(f"      ❌ Dropped: {col}")
df_model = df.drop(columns=ID_COLS)


# ============================================================
# STEP 3: SEPARATE TARGET
# ============================================================
print("\n[3/9] Separating target column...")
TARGET = "ESG_Controversy"
y = df_model[TARGET]
X = df_model.drop(columns=[TARGET])

# cast bool → int
bool_cols = X.select_dtypes(include="bool").columns.tolist()
if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)

feature_names = X.columns.tolist()
print(f"      Feature count : {len(feature_names)}")
print(f"      Class 0 (No Controversy) : {(y==0).sum():,}  ({(y==0).mean():.1%})")
print(f"      Class 1 (Controversy)    : {(y==1).sum():,}  ({(y==1).mean():.1%})")


# ============================================================
# STEP 4: IMPUTE MISSING VALUES
# ============================================================
print("\n[4/9] Imputing missing values...")
zero_impute_cols = [
    "Revenue_YoY", "Revenue_2Y_Growth",
    "Margin_YoY_Change", "MarketCap_YoY", "GrowthRate_YoY_Change",
    "Margin_Volatility_3Y", "Revenue_Volatility_3Y", "MCap_Volatility_3Y",
    "Margin_Deterioration", "Revenue_Surprise",
]
for col in zero_impute_cols:
    if col in X.columns:
        n = X[col].isnull().sum()
        if n > 0:
            X[col] = X[col].fillna(0)
            print(f"      → {col:<30} filled {n:,} NaNs with 0")

rolling_mean_map = {"Avg_Margin_3Y": "ProfitMargin", "Avg_Revenue_3Y": "Revenue"}
for roll_col, src_col in rolling_mean_map.items():
    if roll_col in X.columns and src_col in X.columns:
        n = X[roll_col].isnull().sum()
        if n > 0:
            X[roll_col] = X[roll_col].fillna(X[src_col])
            print(f"      → {roll_col:<30} filled {n:,} NaNs from {src_col}")

for col in ["GrowthRate", "Growth_vs_Industry"]:
    if col in X.columns:
        n = X[col].isnull().sum()
        if n > 0:
            fill_val = X[col].median() if col == "GrowthRate" else 0
            X[col] = X[col].fillna(fill_val)

remaining = X.isnull().sum().sum()
print(f"\n      Total NaNs remaining : {remaining}")
X = X.fillna(0)   # catch any residual


# ============================================================
# STEP 5: WALK-FORWARD TEMPORAL SPLIT
# ============================================================
print("\n[5/9] Walk-forward temporal train/test split...")
SPLIT_YEAR = 2023
train_mask = df["Year"] < SPLIT_YEAR
test_mask  = df["Year"] >= SPLIT_YEAR

X_train_raw, y_train = X[train_mask].copy(), y[train_mask].copy()
X_test_raw,  y_test  = X[test_mask].copy(),  y[test_mask].copy()

print(f"      Train : 2015–{SPLIT_YEAR-1} → {X_train_raw.shape[0]:,} rows")
print(f"      Test  : {SPLIT_YEAR}–2025   → {X_test_raw.shape[0]:,} rows")


# ============================================================
# STEP 6: STANDARD SCALING
# ============================================================
print("\n[6/9] Standard-scaling features for Transformer...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_raw.values).astype(np.float32)
X_test_sc  = scaler.transform(X_test_raw.values).astype(np.float32)
print(f"      Feature range after scaling (train): "
      f"mean≈{X_train_sc.mean():.3f}, std≈{X_train_sc.std():.3f}")


# ============================================================
# TRANSFORMER MODEL DEFINITION
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm architecture (more stable for tabular data)
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(attn_out)
        x = x + self.ff(self.ln2(x))
        return x


class TabularTransformer(nn.Module):
    """
    FT-Transformer style model for tabular binary classification.

    Each of the N_features is projected to a d_model-dim embedding,
    giving a sequence of length N for the self-attention.  A [CLS]
    token aggregates the sequence for the final classification head.

    Architecture highlights
    -----------------------
    • Feature tokenisation  : Linear(1 → d_model) per feature
    • Positional info       : Learnable feature-position embeddings
    • Transformer depth     : n_layers blocks of MHA + FFN (pre-norm)
    • Classification head   : [CLS] → Dropout → Linear → sigmoid
    • Regularisation        : dropout + weight decay in AdamW
    • Class imbalance       : pos_weight in BCEWithLogitsLoss
    """
    def __init__(self, n_features, d_model=64, n_heads=4,
                 n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model

        # Project each scalar feature to d_model
        self.feature_embed = nn.Linear(1, d_model)

        # Learnable positional / feature-identity embeddings
        self.pos_embed = nn.Embedding(n_features + 1, d_model)   # +1 for CLS

        # CLS token (learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head on CLS token
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        """
        x : (batch, n_features) float32
        returns logits : (batch,)
        """
        B = x.size(0)

        # Tokenise: (B, n_features, 1) → (B, n_features, d_model)
        tokens = self.feature_embed(x.unsqueeze(-1))

        # Prepend CLS token → (B, n_features+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        # Add positional embeddings
        pos_ids = torch.arange(seq.size(1), device=x.device)
        seq = seq + self.pos_embed(pos_ids).unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            seq = block(seq)

        # CLS token output → logit
        cls_out = seq[:, 0, :]          # (B, d_model)
        logits  = self.head(cls_out).squeeze(-1)   # (B,)
        return logits


# ============================================================
# STEP 7: TRAIN TRANSFORMER
# ============================================================
print("\n[7/9] Training Transformer model...")

# Hyper-parameters
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 3
D_FF     = 256
DROPOUT  = 0.15
EPOCHS   = 80
BATCH    = 256
LR       = 3e-4
WEIGHT_DECAY = 1e-4

n_features = X_train_sc.shape[1]

# Convert to tensors
X_tr_t = torch.tensor(X_train_sc, dtype=torch.float32)
y_tr_t = torch.tensor(y_train.values, dtype=torch.float32)
X_te_t = torch.tensor(X_test_sc,  dtype=torch.float32)
y_te_t = torch.tensor(y_test.values,  dtype=torch.float32)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)

# Class imbalance: pos_weight = neg/pos
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(DEVICE)
print(f"      pos_weight = {pos_weight.item():.3f}")

model = TabularTransformer(
    n_features=n_features,
    d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ff=D_FF,
    dropout=DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"      Trainable parameters : {total_params:,}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val_auc, best_epoch = 0.0, 0
train_losses, val_aucs   = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    scheduler.step()

    epoch_loss /= len(train_ds)
    train_losses.append(epoch_loss)

    # Validation AUC on full test set
    model.eval()
    with torch.no_grad():
        val_logits = model(X_te_t.to(DEVICE)).cpu().numpy()
        val_proba  = 1 / (1 + np.exp(-val_logits))
    val_auc = roc_auc_score(y_test.values, val_proba)
    val_aucs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc  = val_auc
        best_epoch    = epoch
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,"transformer_esg_model_best.pt"))

    if epoch % 10 == 0:
        print(f"      Epoch {epoch:3d}/{EPOCHS} | loss={epoch_loss:.4f} | "
              f"val ROC-AUC={val_auc:.4f}")

print(f"\n      Best epoch : {best_epoch}  |  Best val ROC-AUC : {best_val_auc:.4f}")

# Load best checkpoint
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "transformer_esg_model_best.pt"),
                                 map_location=DEVICE))
model.eval()


# ============================================================
# STEP 8: 5-FOLD CROSS-VALIDATION
# ============================================================
print("\n[8/9] 5-Fold Stratified Cross-Validation...")

# We run a compact version (30 epochs) for CV speed
def run_fold_model(X_fold_tr, y_fold_tr, X_fold_val, y_fold_val,
                   n_features, pos_w):
    m = TabularTransformer(n_features=n_features,
                           d_model=D_MODEL, n_heads=N_HEADS,
                           n_layers=N_LAYERS, d_ff=D_FF,
                           dropout=DROPOUT).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(DEVICE))
    opt  = optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
    ds   = TensorDataset(torch.tensor(X_fold_tr, dtype=torch.float32),
                         torch.tensor(y_fold_tr, dtype=torch.float32))
    dl   = DataLoader(ds, batch_size=BATCH, shuffle=True)
    for _ in range(30):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(m(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        logits = m(torch.tensor(X_fold_val, dtype=torch.float32).to(DEVICE))
        proba  = torch.sigmoid(logits).cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    return (roc_auc_score(y_fold_val, proba),
            f1_score(y_fold_val, pred),
            accuracy_score(y_fold_val, pred))

X_full_sc = scaler.fit_transform(X.values).astype(np.float32)
y_full    = y.values.astype(np.float32)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_roc, cv_f1, cv_acc = [], [], []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_full_sc, y_full), 1):
    Xtr, Xva = X_full_sc[tr_idx], X_full_sc[va_idx]
    ytr, yva = y_full[tr_idx], y_full[va_idx]
    pw = ((ytr == 0).sum() / (ytr == 1).sum())
    r, f, a = run_fold_model(Xtr, ytr, Xva, yva, n_features, pw)
    cv_roc.append(r);  cv_f1.append(f);  cv_acc.append(a)
    print(f"      Fold {fold}: ROC-AUC={r:.4f}  F1={f:.4f}  Acc={a:.4f}")

cv_roc = np.array(cv_roc)
cv_f1  = np.array(cv_f1)
cv_acc = np.array(cv_acc)
print(f"\n      ROC-AUC  : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print(f"      F1 Score : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")
print(f"      Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")


# ============================================================
# STEP 9: EVALUATION METRICS
# ============================================================
print("\n[9/9] Evaluation metrics on held-out test set...")

model.eval()
with torch.no_grad():
    logits  = model(X_te_t.to(DEVICE)).cpu().numpy()
    y_proba = (1 / (1 + np.exp(-logits))).astype(np.float64)

y_pred = (y_proba >= 0.5).astype(int)
y_test_np = y_test.values

acc     = accuracy_score(y_test_np, y_pred)
prec    = precision_score(y_test_np, y_pred)
rec     = recall_score(y_test_np, y_pred)
f1      = f1_score(y_test_np, y_pred)
roc_auc = roc_auc_score(y_test_np, y_proba)
pr_auc  = average_precision_score(y_test_np, y_proba)

print(f"\n  {'Metric':<25} {'Value':>10}")
print(f"  {'─'*37}")
for name, val in [("Accuracy", acc), ("Precision", prec), ("Recall", rec),
                  ("F1 Score", f1), ("ROC-AUC", roc_auc), ("PR-AUC", pr_auc)]:
    print(f"  {name:<25} {val:>10.4f}")
print(f"\n  Classification Report:\n")
print(classification_report(y_test_np, y_pred,
                             target_names=["No Controversy", "Controversy"]))

metrics = {
    "Model": "Transformer",
    "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
    "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
    "CV_ROC_AUC_Mean": cv_roc.mean(), "CV_ROC_AUC_Std": cv_roc.std(),
}


# ============================================================
# PLOTS
# ============================================================
print("\n[10/9] Generating evaluation + SHAP plots...")

# ── DISPLAY NAMES (same as XGBoost script) ───────────────────
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


# ── PLOT 1: Evaluation Dashboard ────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "ESG Controversy Prediction — Transformer Evaluation Dashboard\n"
    f"Train: 2015–{SPLIT_YEAR-1}  |  Test: {SPLIT_YEAR}–2025  |  "
    f"Features: {len(feature_names)}  |  Companies: 1,000",
    fontsize=16, fontweight="bold", color=NAVY, y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a: Metric bar
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("white")
m_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
m_values = [acc, prec, rec, f1, roc_auc, pr_auc]
m_colors = [NAVY, TEAL, BLUE, GREEN, AMBER, PURPLE]
bars = ax1.barh(m_labels, m_values, color=m_colors, edgecolor="white", height=0.55)
ax1.set_xlim(0, 1.15)
ax1.set_title("Test Set Metrics", fontweight="bold", fontsize=12, color=NAVY)
ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
for bar, val in zip(bars, m_values):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=10, fontweight="bold", color=NAVY)
ax1.set_xlabel("Score", fontsize=10); ax1.grid(axis="x", alpha=0.2)

# 1b: Confusion matrix
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("white")
cm   = confusion_matrix(y_test_np, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Controversy", "Controversy"])
disp.plot(ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title("Confusion Matrix", fontweight="bold", fontsize=12, color=NAVY)
for text in disp.text_.ravel():
    text.set_fontsize(14); text.set_fontweight("bold")

# 1c: ROC Curve
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("white")
fpr, tpr, _ = roc_curve(y_test_np, y_proba)
ax3.plot(fpr, tpr, color=PURPLE, lw=2.5, label=f"Transformer (AUC = {roc_auc:.4f})")
ax3.plot([0,1],[0,1],"k--", alpha=0.4, lw=1.5, label="Random Baseline")
ax3.fill_between(fpr, tpr, alpha=0.08, color=PURPLE)
ax3.set_xlabel("False Positive Rate", fontsize=10); ax3.set_ylabel("True Positive Rate", fontsize=10)
ax3.set_title("ROC Curve", fontweight="bold", fontsize=12, color=NAVY)
ax3.legend(fontsize=9); ax3.grid(alpha=0.25)

# 1d: Precision-Recall
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor("white")
prec_curve, rec_curve, _ = precision_recall_curve(y_test_np, y_proba)
ax4.plot(rec_curve, prec_curve, color=TEAL, lw=2.5, label=f"Transformer (PR-AUC = {pr_auc:.4f})")
ax4.axhline(y_test_np.mean(), color="gray", linestyle="--", alpha=0.5, lw=1.5,
            label=f"Baseline ({y_test_np.mean():.2f})")
ax4.fill_between(rec_curve, prec_curve, alpha=0.08, color=TEAL)
ax4.set_xlabel("Recall", fontsize=10); ax4.set_ylabel("Precision", fontsize=10)
ax4.set_title("Precision-Recall Curve", fontweight="bold", fontsize=12, color=NAVY)
ax4.legend(fontsize=9); ax4.grid(alpha=0.25)

# 1e: CV Distribution
ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor("white")
positions = [1, 2, 3]; bp_colors = [PURPLE, TEAL, GREEN]
bp = ax5.boxplot([cv_roc, cv_f1, cv_acc], positions=positions,
                 patch_artist=True, widths=0.45,
                 medianprops=dict(color="white", linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
                 flierprops=dict(marker="o", markersize=5))
for patch, color in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(color); patch.set_alpha(0.85)
for i, (scores, color) in enumerate(zip([cv_roc, cv_f1, cv_acc], bp_colors)):
    ax5.scatter([positions[i]]*len(scores), scores, color=color,
                zorder=5, s=60, edgecolors="white", linewidth=1)
ax5.set_xticks(positions)
ax5.set_xticklabels(["ROC-AUC", "F1 Score", "Accuracy"], fontsize=10)
ax5.set_title("5-Fold CV Score Distribution", fontweight="bold", fontsize=12, color=NAVY)
ax5.set_ylim(0.4, 1.0); ax5.grid(axis="y", alpha=0.3)

# 1f: Probability distribution
ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor("white")
ax6.hist(y_proba[y_test_np==0], bins=40, alpha=0.65, color=GREEN,
         label="Actual: No Controversy", density=True)
ax6.hist(y_proba[y_test_np==1], bins=40, alpha=0.65, color=RED,
         label="Actual: Controversy", density=True)
ax6.axvline(0.5, color=NAVY, linestyle="--", lw=2, label="Decision threshold (0.5)")
ax6.set_xlabel("Predicted Probability (Controversy)", fontsize=10)
ax6.set_ylabel("Density", fontsize=10)
ax6.set_title("Prediction Probability Distribution", fontweight="bold", fontsize=12, color=NAVY)
ax6.legend(fontsize=8); ax6.grid(alpha=0.25)

plt.savefig(os.path.join(OUTPUT_DIR, "07_transformer_evaluation_dashboard.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 07_transformer_evaluation_dashboard.png")

# ── Training Loss Curve ───────────────────────────────────────
fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
ax_loss.set_facecolor("white"); ax_auc.set_facecolor("white")

ax_loss.plot(range(1, EPOCHS+1), train_losses, color=PURPLE, lw=2)
ax_loss.axvline(best_epoch, color=RED, linestyle="--", lw=1.5,
                label=f"Best epoch: {best_epoch}")
ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("BCE Loss (train)")
ax_loss.set_title("Transformer Training Loss", fontweight="bold", color=NAVY)
ax_loss.legend(); ax_loss.grid(alpha=0.25)

ax_auc.plot(range(1, EPOCHS+1), val_aucs, color=TEAL, lw=2)
ax_auc.axvline(best_epoch, color=RED, linestyle="--", lw=1.5,
               label=f"Best ROC-AUC: {best_val_auc:.4f}")
ax_auc.set_xlabel("Epoch"); ax_auc.set_ylabel("Validation ROC-AUC")
ax_auc.set_title("Transformer Validation ROC-AUC", fontweight="bold", color=NAVY)
ax_auc.legend(); ax_auc.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_transformer_training_curves.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 08_transformer_training_curves.png")


# ── SHAP via KernelExplainer (model-agnostic) ─────────────────
print("      Computing SHAP values (KernelExplainer — may take a few minutes)...")

def transformer_predict(X_np):
    """Wrapper: numpy array → predicted probabilities."""
    X_t = torch.tensor(X_np.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
    return 1 / (1 + np.exp(-logits))

# Use a background summary (k-means 50 centroids) for speed
background = shap.kmeans(X_train_sc, 50)

# Explain a random subset of 300 test samples for SHAP plots
np.random.seed(42)
n_explain = min(300, len(X_test_sc))
explain_idx = np.random.choice(len(X_test_sc), n_explain, replace=False)
X_explain = X_test_sc[explain_idx]

explainer   = shap.KernelExplainer(transformer_predict, background)
shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)
print(f"      SHAP values shape: {shap_values.shape}")

# Create a temporary DataFrame with scaled values for SHAP plots
X_explain_df = pd.DataFrame(X_explain, columns=feature_names)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

# ── PLOT: SHAP Beeswarm ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor(BG)
shap.summary_plot(shap_values, X_explain_df, feature_names=display_names,
                  show=False, plot_size=None, max_display=20,
                  color_bar_label="Feature Value (low → high)")
plt.title(
    "SHAP Beeswarm — Transformer Feature Impact on ESG Controversy\n"
    "(Each dot = one test observation  |  color = feature value magnitude)",
    fontsize=13, fontweight="bold", color=NAVY, pad=15
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_transformer_shap_beeswarm.png"), dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 09_transformer_shap_beeswarm.png")

# ── PLOT: SHAP Feature Importance Bar ────────────────────────
fi_df = pd.DataFrame({
    "feature":    display_names,
    "importance": mean_abs_shap,
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

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor(BG); ax.set_facecolor("white")
colors = [bar_color(f) for f in fi_df["feature"]]
bars = ax.barh(fi_df["feature"], fi_df["importance"],
               color=colors, edgecolor="white", height=0.65)
for bar in bars:
    ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}", va="center", ha="left", fontsize=9, color=NAVY)
ax.set_title(
    "SHAP Feature Importance — Transformer | Mean |SHAP Value| (Top 20)\n"
    "Higher value = stronger influence on ESG Controversy prediction",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.grid(axis="x", alpha=0.2); ax.tick_params(labelsize=10)
legend_patches = [
    mpatches.Patch(color=TEAL,   label="Profitability Features"),
    mpatches.Patch(color=BLUE,   label="Revenue / Size / Growth"),
    mpatches.Patch(color=RED,    label="Distress Flags"),
    mpatches.Patch(color=GREEN,  label="Peer-Relative Features"),
    mpatches.Patch(color=AMBER,  label="Industry / Region"),
    mpatches.Patch(color=NAVY,   label="Other Financial Features"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_transformer_shap_feature_importance.png"), dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 10_transformer_shap_feature_importance.png")

# ── PLOT: SHAP Waterfall ─────────────────────────────────────
y_explain   = y_test_np[explain_idx]
pred_explain = (transformer_predict(X_explain) >= 0.5).astype(int)

idx_cont   = np.where((y_explain == 1) & (pred_explain == 1))[0]
idx_nocont = np.where((y_explain == 0) & (pred_explain == 0))[0]

if len(idx_cont) > 0 and len(idx_nocont) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "SHAP Waterfall — Transformer Individual Prediction Explanations\n"
        "Left: Correctly Predicted Controversy  |  Right: Correctly Predicted Non-Controversy",
        fontsize=13, fontweight="bold", color=NAVY
    )
    base_val = explainer.expected_value
    for ax_idx, (i, label, lcolor) in enumerate([
        (idx_cont[0],   "Predicted: CONTROVERSY ✅",    RED),
        (idx_nocont[0], "Predicted: NO CONTROVERSY ✅", GREEN),
    ]):
        plt.sca(axes[ax_idx])
        exp = shap.Explanation(
            values=shap_values[i], base_values=base_val,
            data=X_explain[i], feature_names=display_names,
        )
        shap.waterfall_plot(exp, max_display=15, show=False)
        axes[ax_idx].set_title(label, fontsize=11, fontweight="bold",
                               color=lcolor, pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_transformer_shap_waterfall.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("      ✅ Saved: 11_transformer_shap_waterfall.png")

# ── PLOT: SHAP Dependence (Top 4) ────────────────────────────
top4_idx     = np.argsort(mean_abs_shap)[::-1][:4]
top4_names   = [feature_names[i] for i in top4_idx]
top4_display = [display_names[i] for i in top4_idx]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "SHAP Dependence Plots — Transformer | Top 4 Most Important Features\n"
    "(X = feature value  |  Y = SHAP contribution  |  "
    "Color = interaction with most correlated feature)",
    fontsize=13, fontweight="bold", color=NAVY
)
for i, (feat_name, feat_display) in enumerate(zip(top4_names, top4_display)):
    ax = axes[i // 2][i % 2]
    ax.set_facecolor("white")
    shap.dependence_plot(feat_name, shap_values, X_explain_df,
                         feature_names=feature_names, ax=ax,
                         show=False, dot_size=12, alpha=0.5)
    ax.set_title(f"Dependence: {feat_display}", fontsize=11,
                 fontweight="bold", color=NAVY)
    ax.grid(alpha=0.2); ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "12_transformer_shap_dependence.png"), dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 12_transformer_shap_dependence.png")

# ── PLOT: Feature Group Importance ───────────────────────────
group_map = {
    "Revenue": "A-B: Valuation & Profitability", "ProfitMargin": "A-B: Valuation & Profitability",
    "MarketCap": "A-B: Valuation & Profitability", "GrowthRate": "A-B: Valuation & Profitability",
    "Price_to_Sales": "A-B: Valuation & Profitability", "Log_Revenue": "A-B: Valuation & Profitability",
    "Log_MarketCap": "A-B: Valuation & Profitability", "Revenue_to_MarketCap": "A-B: Valuation & Profitability",
    "EBIT_proxy": "A-B: Valuation & Profitability", "Return_on_Sales": "A-B: Valuation & Profitability",
    "Profitability_Score": "A-B: Valuation & Profitability",
    "Revenue_YoY": "C: Momentum & Growth", "Revenue_2Y_Growth": "C: Momentum & Growth",
    "Margin_YoY_Change": "C: Momentum & Growth", "MarketCap_YoY": "C: Momentum & Growth",
    "GrowthRate_YoY_Change": "C: Momentum & Growth",
    "Margin_Volatility_3Y": "D: Volatility", "Revenue_Volatility_3Y": "D: Volatility",
    "MCap_Volatility_3Y": "D: Volatility",
    "Revenue_vs_Industry": "E: Peer-Relative", "Margin_vs_Industry": "E: Peer-Relative",
    "MCap_vs_Industry": "E: Peer-Relative", "Growth_vs_Industry": "E: Peer-Relative",
    "Revenue_Pctile_Industry": "E: Peer-Relative", "Margin_Pctile_Industry": "E: Peer-Relative",
    "Negative_Margin_Flag": "F: Distress Flags", "Declining_Revenue_Flag": "F: Distress Flags",
    "MCap_Collapse_Flag": "F: Distress Flags", "Low_Growth_Low_Margin": "F: Distress Flags",
    "Valuation_Excess": "G: Valuation Anomaly", "PS_Margin_Ratio": "G: Valuation Anomaly",
    "Revenue_Efficiency": "G: Valuation Anomaly",
    "Avg_Margin_3Y": "H: Rolling Trends", "Margin_Deterioration": "H: Rolling Trends",
    "Avg_Revenue_3Y": "H: Rolling Trends", "Revenue_Surprise": "H: Rolling Trends",
    "Size_Tier": "I: Size Tier",
}
group_importance = {}
for feat, imp in zip(feature_names, mean_abs_shap):
    group = ("J: Industry" if feat.startswith("Industry_")
             else "J: Region" if feat.startswith("Region_")
             else group_map.get(feat, "Other"))
    group_importance[group] = group_importance.get(group, 0) + imp

grp_df = pd.DataFrame(list(group_importance.items()),
                      columns=["Group", "Total SHAP"]).sort_values("Total SHAP", ascending=True)
group_colors = {
    "A-B: Valuation & Profitability": TEAL, "C: Momentum & Growth": BLUE,
    "D: Volatility": NAVY, "E: Peer-Relative": GREEN,
    "F: Distress Flags": RED, "G: Valuation Anomaly": PURPLE,
    "H: Rolling Trends": "#0891B2", "I: Size Tier": "#BE185D",
    "J: Industry": AMBER, "J: Region": "#B45309",
}

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor("white")
colors = [group_colors.get(g, NAVY) for g in grp_df["Group"]]
bars = ax.barh(grp_df["Group"], grp_df["Total SHAP"],
               color=colors, edgecolor="white", height=0.6)
for bar in bars:
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}", va="center", ha="left", fontsize=10, color=NAVY)
ax.set_title(
    "SHAP Feature Group Importance — Transformer | Summed Mean |SHAP Value|\n"
    "Which category of financial features drives ESG controversy prediction most?",
    fontsize=13, fontweight="bold", color=NAVY, pad=12
)
ax.set_xlabel("Total Mean |SHAP Value| (summed across group)", fontsize=11)
ax.grid(axis="x", alpha=0.2); ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_transformer_shap_group_importance.png"), dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("      ✅ Saved: 13_transformer_shap_group_importance.png")


# ============================================================
# SAVE MODEL + METRICS
# ============================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "transformer_esg_model.pt"))
print("      ✅ Saved: transformer_esg_model.pt")

metrics_df = pd.DataFrame([metrics])
for fold_i, score in enumerate(cv_roc, 1):
    metrics_df[f"CV_ROC_AUC_Fold{fold_i}"] = score
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "transformer_model_metrics.csv"), index=False)
print("      ✅ Saved: transformer_model_metrics.csv")

print("\n" + "=" * 60)
print("  TRANSFORMER SUMMARY")
print("=" * 60)
print(f"  Architecture  : FT-Transformer")
print(f"  d_model       : {D_MODEL}  |  n_heads : {N_HEADS}  |  n_layers : {N_LAYERS}")
print(f"  Parameters    : {total_params:,}")
print(f"  Best epoch    : {best_epoch}/{EPOCHS}")
print(f"  Accuracy      : {acc:.4f}")
print(f"  Precision     : {prec:.4f}")
print(f"  Recall        : {rec:.4f}")
print(f"  F1 Score      : {f1:.4f}")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  PR-AUC        : {pr_auc:.4f}")
print(f"  CV ROC-AUC    : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print("=" * 60)
print("\n✅ All Transformer outputs saved.")