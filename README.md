# ESG Controversy Prediction — Multi-Model ML Pipeline

Predicting ESG (Environmental, Social, and Governance) controversy risk for companies using purely financial data. A company is labeled **controversial** (`ESG_Controversy = 1`) if its `ESG_Overall` score falls at or below the 25th percentile of its industry-year peer group.
Five machine learning models are trained, evaluated, and explained: XGBoost, Random Forest, SVM (RBF), Neural Network, Transformer Each is paired with an explainability layer (SHAP or LIME) to produce per-company risk driver reports alongside prediction dashboards.

Five machine learning models are trained, evaluated, and explained end-to-end:

| Model            | Explainability  |
| ---------------- | --------------- |
| XGBoost          | TreeSHAP + LIME |
| Random Forest    | TreeSHAP        |
| SVM (RBF kernel) | KernelSHAP      |
| Neural Network   | KernelSHAP      |
| Transformer      | KernelSHAP      |

Each model produces per-company risk-driver reports alongside prediction dashboards.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
Quant_project/
├── company_esg_financial_dataset.csv   # Raw input data (companies x years)
├── esg_controversy_features.csv        # Engineered features (Step 1 output)
│
├── esg_feature_engineering.py          # STEP 1 — Feature engineering pipeline
├── esg_xgboost_shap.py                 # STEP 2a — XGBoost + TreeSHAP
├── esg_randomforest_shap.py            # STEP 2b — Random Forest + TreeSHAP
├── esg_svm_shap.py                     # STEP 2c — SVM (RBF) + KernelSHAP
├── esg_nn_shap.py                      # STEP 2d — Neural Network + KernelSHAP
├── esg_transformer_shap.py             # STEP 2e — Transformer + KernelSHAP
├── esg_xgboost_lime.py                 # STEP 2f — XGBoost + LIME
│
├── requirements.txt                    # Pinned Python dependencies
├── README.md                           # This file
│
└── model_output/
    ├── xg_boost/         # Model (.pkl), metrics CSV, SHAP + LIME plots
    ├── random_forest/    # Model (.pkl), metrics CSV, SHAP plots
    ├── svm/              # Model (.pkl), metrics CSV, SHAP plots
    ├── neural_network/   # Model (.pt), scaler (.pkl), metrics CSV, SHAP plots
    └── transformer/      # Model (.pt), metrics CSV, SHAP plots
```

> All scripts resolve file paths relative to their own location (`os.path.dirname(os.path.abspath(__file__))`), so they can be run from any working directory.

---

## Setup & Installation

**1. Verify Python version**

```bash
python --version
# Must print Python 3.10.x
```

If you have multiple Python versions installed, replace `python` with `python3.10` in all commands below.

**2. Clone the repository**

```bash
git clone https://github.com/<ajaybansal072006>/<esg-controversy-prediction>.git
cd <esg-controversy-prediction>
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the virtual environment**

| OS                   | Command                     |
| -------------------- | --------------------------- |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |
| Windows (CMD)        | `venv\Scripts\activate`     |
| macOS / Linux        | `source venv/bin/activate`  |

Your terminal prompt should show `(venv)` once activated.

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

PyTorch installs the CPU-only build by default. For GPU acceleration, install the matching CUDA build from [pytorch.org](https://pytorch.org) instead.

---

## Running the Pipeline

**Step 1 — Feature Engineering** (run first, only once)

```bash
python esg_feature_engineering.py
```

Reads `company_esg_financial_dataset.csv` → writes `esg_controversy_features.csv`

**Step 2 — Model Training + Explainability** (run after Step 1, any order)

```bash
python esg_xgboost_shap.py
python esg_randomforest_shap.py
python esg_svm_shap.py
python esg_nn_shap.py
python esg_transformer_shap.py
python esg_xgboost_lime.py
```

Each script reads `esg_controversy_features.csv` and writes outputs to `model_output/<model_name>/`.

---

## Dependencies

```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.2
scikit-learn==1.5.2
joblib==1.4.2
xgboost==2.1.4
shap==0.49.1
lime==0.2.0.1
torch
tqdm==4.66.4
scipy==1.13.1
```

---

## Troubleshooting

| Issue                                                                             | Fix                                                                                                                 |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError: esg_controversy_features.csv`                                 | Run Step 1 first: `python esg_feature_engineering.py`                                                               |
| `ModuleNotFoundError: No module named 'xgboost'` (or similar)                     | Activate your virtual environment, then `pip install -r requirements.txt`                                           |
| torch version mismatch warning                                                    | Latest stable CPU torch installs by default; for GPU/CUDA, install manually from [pytorch.org](https://pytorch.org) |
| KernelSHAP running for a long time (`esg_svm_shap.py`, `esg_transformer_shap.py`) | Expected on CPU — samples 100 background + 150 test points by design. Allow 10–30 minutes; do not interrupt.        |
