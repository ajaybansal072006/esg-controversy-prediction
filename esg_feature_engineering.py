"""
ESG Controversy Feature Engineering Pipeline
=============================================
Author  : Data Science / Finance Expert
Purpose : Build leak-free, finance-grounded features for predicting
          ESG controversy from a company ESG-financial dataset.

Target  : ESG_Controversy (binary)
          = 1 if company's ESG_Overall <= 25th percentile of its
            industry-year peer group (poor relative ESG performer)

Leakage : All ESG scores and ESG operational columns are used ONLY
          to construct the target, then dropped before any features
          are created.

Usage
-----
    python esg_feature_engineering.py \
        --input  company_esg_financial_dataset.csv \
        --output esg_controversy_features.csv

Dependencies: pandas, numpy
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Columns that directly expose ESG information → must be dropped
ESG_LEAK_COLUMNS = [
    "ESG_Overall",
    "ESG_Environmental",
    "ESG_Social",
    "ESG_Governance",
    "CarbonEmissions",   # direct ESG operational metric
    "WaterUsage",        # direct ESG operational metric
    "EnergyConsumption", # direct ESG operational metric
    "CompanyName",       # non-informative string identifier
]

TARGET_COLUMN     = "ESG_Controversy"
ESG_SOURCE_COLUMN = "ESG_Overall"   # used ONLY to build the target
CONTROVERSY_QUANTILE = 0.25         # bottom 25% of industry-year = controversy


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["CompanyID", "Year"]).reset_index(drop=True)
    print(f"[load]  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"        Companies : {df['CompanyID'].nunique()}")
    print(f"        Year range: {df['Year'].min()} – {df['Year'].max()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD TARGET (uses ESG_Overall, then drops all ESG columns)
# ─────────────────────────────────────────────────────────────────────────────

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    ESG_Controversy = 1 when a company's ESG_Overall score is at or below
    the 25th percentile of companies in the same Industry AND Year.
    This captures genuine underperformers relative to peers, not just
    companies that operate in inherently low-ESG industries.
    """
    q25 = df.groupby(["Industry", "Year"])[ESG_SOURCE_COLUMN].transform(
        lambda x: x.quantile(CONTROVERSY_QUANTILE)
    )
    df[TARGET_COLUMN] = (df[ESG_SOURCE_COLUMN] <= q25).astype(int)

    rate = df[TARGET_COLUMN].mean()
    counts = df[TARGET_COLUMN].value_counts()
    print(f"\n[target] ESG_Controversy created")
    print(f"         Controversy rate : {rate:.1%}")
    print(f"         Class 0 (clean)  : {counts[0]:,}")
    print(f"         Class 1 (flagged): {counts[1]:,}")

    # Drop ALL ESG-leaking columns immediately after target is built
    cols_to_drop = [c for c in ESG_LEAK_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"\n[leak]   Dropped ESG columns: {cols_to_drop}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    g  = df.groupby("CompanyID")
    iy = df.groupby(["Industry", "Year"])

    # ── A. Valuation & Size Ratios ──────────────────────────────────────────
    # Price-to-Sales: how the market values revenue — high values can signal
    # speculative premium or financial engineering risk
    df["Price_to_Sales"]       = df["MarketCap"] / df["Revenue"].replace(0, np.nan)

    # Log transforms stabilise skewed financial distributions
    df["Log_Revenue"]          = np.log1p(df["Revenue"])
    df["Log_MarketCap"]        = np.log1p(df["MarketCap"])

    # Revenue-to-MarketCap: inverse of P/S, captures fundamental backing
    df["Revenue_to_MarketCap"] = df["Revenue"] / df["MarketCap"].replace(0, np.nan)

    # ── B. Profitability ────────────────────────────────────────────────────
    # Absolute earnings proxy (no balance sheet available)
    df["EBIT_proxy"]           = df["Revenue"] * df["ProfitMargin"] / 100

    # Clean ratio form of margin
    df["Return_on_Sales"]      = df["ProfitMargin"] / 100

    # Earnings power relative to market valuation
    df["Profitability_Score"]  = df["EBIT_proxy"] / df["MarketCap"].replace(0, np.nan)

    # ── C. Momentum & Growth Dynamics ──────────────────────────────────────
    # YoY revenue growth (%) — sharp declines often precede controversies
    df["Revenue_YoY"]          = g["Revenue"].pct_change() * 100

    # Two-year cumulative revenue growth — medium-term trajectory
    df["Revenue_2Y_Growth"]    = g["Revenue"].pct_change(2) * 100

    # Margin trend — deteriorating margins signal cost pressure or mismanagement
    df["Margin_YoY_Change"]    = g["ProfitMargin"].diff()

    # Market cap trajectory — large drops can reflect scandal/controversy
    df["MarketCap_YoY"]        = g["MarketCap"].pct_change() * 100

    # Growth acceleration (or deceleration) — sudden spikes can hide issues
    df["GrowthRate_YoY_Change"]= g["GrowthRate"].diff()

    # ── D. Volatility / Operational Stability ───────────────────────────────
    # Rolling 3-year margin volatility — erratic margins suggest instability
    df["Margin_Volatility_3Y"] = g["ProfitMargin"].transform(
        lambda x: x.rolling(3, min_periods=2).std()
    )

    # Coefficient of variation for revenue — normalised revenue instability
    df["Revenue_Volatility_3Y"]= g["Revenue"].transform(
        lambda x: x.rolling(3, min_periods=2).std()
             / x.rolling(3, min_periods=2).mean().replace(0, np.nan)
    )

    # Market cap volatility — stock price instability proxy
    df["MCap_Volatility_3Y"]   = g["MarketCap"].transform(
        lambda x: x.rolling(3, min_periods=2).std()
             / x.rolling(3, min_periods=2).mean().replace(0, np.nan)
    )

    # ── E. Peer-Relative Features (Industry-Year Benchmarking) ──────────────
    # Companies underperforming peers financially are more likely to cut ESG
    df["Revenue_vs_Industry"]  = df["Revenue"]      / iy["Revenue"].transform("median")
    df["Margin_vs_Industry"]   = df["ProfitMargin"] - iy["ProfitMargin"].transform("median")
    df["MCap_vs_Industry"]     = df["MarketCap"]    / iy["MarketCap"].transform("median")
    df["Growth_vs_Industry"]   = df["GrowthRate"]   - iy["GrowthRate"].transform("median")

    # Percentile rank within industry-year — robust to outliers
    df["Revenue_Pctile_Industry"] = iy["Revenue"].transform(
        lambda x: x.rank(pct=True)
    )
    df["Margin_Pctile_Industry"]  = iy["ProfitMargin"].transform(
        lambda x: x.rank(pct=True)
    )

    # ── F. Financial Distress Flags ─────────────────────────────────────────
    # Binary signals for acute financial stress — research links these to
    # governance failures and subsequent ESG controversies
    df["Negative_Margin_Flag"]  = (df["ProfitMargin"] < 0).astype(int)
    df["Declining_Revenue_Flag"]= (df["Revenue_YoY"] < 0).astype(int)
    df["MCap_Collapse_Flag"]    = (df["MarketCap_YoY"] < -20).astype(int)

    # Dual distress: low growth AND thin margins simultaneously
    df["Low_Growth_Low_Margin"] = (
        (df["GrowthRate"] < 0) & (df["ProfitMargin"] < 5)
    ).astype(int)

    # ── G. Valuation Anomaly Features (Governance Proxy) ────────────────────
    # Excess market premium vs revenue fundamentals — can signal financial risk
    df["Valuation_Excess"]     = df["Log_MarketCap"] - df["Log_Revenue"]

    # High P/S with low margin → potential financial engineering
    df["PS_Margin_Ratio"]      = df["Price_to_Sales"] / (df["ProfitMargin"].abs() + 1)

    # Revenue efficiency — low value may indicate misallocation of capital
    df["Revenue_Efficiency"]   = df["Revenue"] / df["MarketCap"].replace(0, np.nan)

    # ── H. Rolling Financial Health Trends ──────────────────────────────────
    # 3-year rolling average margin — contextualises current performance
    df["Avg_Margin_3Y"]        = g["ProfitMargin"].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )
    # Deviation from own 3-year average — catches sudden deteriorations
    df["Margin_Deterioration"] = df["ProfitMargin"] - df["Avg_Margin_3Y"]

    # 3-year rolling average revenue
    df["Avg_Revenue_3Y"]       = g["Revenue"].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )
    # Revenue surprise vs own trend (+ = beat trend, - = missed trend)
    df["Revenue_Surprise"]     = (
        (df["Revenue"] - df["Avg_Revenue_3Y"])
        / df["Avg_Revenue_3Y"].replace(0, np.nan)
    )

    # ── I. Company Size Tier ────────────────────────────────────────────────
    # ESG controversy rates differ structurally across size buckets
    df["Size_Tier"] = pd.qcut(
        df["Log_MarketCap"], q=4, labels=[1, 2, 3, 4]
    ).astype(int)

    # ── J. Categorical Encoding ─────────────────────────────────────────────
    df = pd.get_dummies(df, columns=["Industry", "Region"], drop_first=False, dtype=int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FINALISE & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def finalise(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    # Put identifier + target first, features after
    id_cols     = ["CompanyID", "Year", TARGET_COLUMN]
    feature_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + feature_cols]

    # Summary
    print(f"\n[output] Final shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"         Features     : {len(feature_cols)}")

    null_counts = df[feature_cols].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        print("\n[nulls]  NaN counts (expected for time-series lag features in Year 1):")
        print(null_counts.to_string())
        print("         → Impute with per-company forward-fill or median before training.")
        print("         → Tree-based models (XGBoost, LightGBM) handle NaNs natively.")

    df.to_csv(output_path, index=False)
    print(f"\n[done]   Saved → {output_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CATALOGUE (for documentation / EDA)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_CATALOGUE = {
    # ── Raw Financials ──────────────────────────────────────────────────────
    "Revenue"              : "Annual revenue (raw, in millions)",
    "ProfitMargin"         : "Net profit margin (%)",
    "MarketCap"            : "Market capitalisation (raw, in millions)",
    "GrowthRate"           : "Company-reported top-line growth rate (%)",

    # ── A. Valuation & Size ─────────────────────────────────────────────────
    "Price_to_Sales"       : "MarketCap / Revenue — market premium over sales",
    "Log_Revenue"          : "log(1 + Revenue) — normalises skewed distribution",
    "Log_MarketCap"        : "log(1 + MarketCap) — normalises skewed distribution",
    "Revenue_to_MarketCap" : "Revenue / MarketCap — inverse of P/S ratio",

    # ── B. Profitability ────────────────────────────────────────────────────
    "EBIT_proxy"           : "Revenue × ProfitMargin/100 — absolute earnings estimate",
    "Return_on_Sales"      : "ProfitMargin / 100 — clean ratio form",
    "Profitability_Score"  : "EBIT_proxy / MarketCap — earnings yield proxy",

    # ── C. Momentum ─────────────────────────────────────────────────────────
    "Revenue_YoY"          : "Year-over-year revenue growth (%)",
    "Revenue_2Y_Growth"    : "Two-year cumulative revenue growth (%)",
    "Margin_YoY_Change"    : "Year-over-year change in profit margin (pp)",
    "MarketCap_YoY"        : "Year-over-year change in market cap (%)",
    "GrowthRate_YoY_Change": "Year-over-year change in growth rate — acceleration signal",

    # ── D. Volatility ───────────────────────────────────────────────────────
    "Margin_Volatility_3Y" : "Std dev of profit margin over trailing 3 years",
    "Revenue_Volatility_3Y": "CoV of revenue over trailing 3 years (std/mean)",
    "MCap_Volatility_3Y"   : "CoV of market cap over trailing 3 years (std/mean)",

    # ── E. Peer-Relative ────────────────────────────────────────────────────
    "Revenue_vs_Industry"  : "Revenue / industry-year median revenue",
    "Margin_vs_Industry"   : "ProfitMargin − industry-year median margin (pp delta)",
    "MCap_vs_Industry"     : "MarketCap / industry-year median MarketCap",
    "Growth_vs_Industry"   : "GrowthRate − industry-year median GrowthRate (pp delta)",
    "Revenue_Pctile_Industry": "Percentile rank of revenue within industry-year",
    "Margin_Pctile_Industry" : "Percentile rank of margin within industry-year",

    # ── F. Distress Flags ───────────────────────────────────────────────────
    "Negative_Margin_Flag" : "1 if ProfitMargin < 0",
    "Declining_Revenue_Flag":"1 if Revenue_YoY < 0",
    "MCap_Collapse_Flag"   : "1 if MarketCap_YoY < −20%",
    "Low_Growth_Low_Margin": "1 if GrowthRate < 0 AND ProfitMargin < 5%",

    # ── G. Valuation Anomaly ────────────────────────────────────────────────
    "Valuation_Excess"     : "Log_MarketCap − Log_Revenue — premium above revenue base",
    "PS_Margin_Ratio"      : "Price_to_Sales / (|ProfitMargin| + 1) — P/S adjusted for margin",
    "Revenue_Efficiency"   : "Revenue / MarketCap — sales generated per unit of market value",

    # ── H. Rolling Trends ───────────────────────────────────────────────────
    "Avg_Margin_3Y"        : "3-year rolling average profit margin",
    "Margin_Deterioration" : "ProfitMargin − Avg_Margin_3Y — drop vs own history",
    "Avg_Revenue_3Y"       : "3-year rolling average revenue",
    "Revenue_Surprise"     : "(Revenue − Avg_Revenue_3Y) / Avg_Revenue_3Y — vs own trend",

    # ── I. Size ─────────────────────────────────────────────────────────────
    "Size_Tier"            : "Quartile bucket of Log_MarketCap (1=smallest, 4=largest)",
}


def print_catalogue():
    print("\n" + "=" * 70)
    print("FEATURE CATALOGUE")
    print("=" * 70)
    for feat, desc in FEATURE_CATALOGUE.items():
        print(f"  {feat:<30} {desc}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="ESG Controversy Feature Engineering Pipeline"
    )
    parser.add_argument(
        "--input",  "-i",
        default=os.path.join(BASE_DIR, "company_esg_financial_dataset.csv"),
        help="Path to raw input CSV (default: company_esg_financial_dataset.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(BASE_DIR, "esg_controversy_features.csv"),
        help="Path for output CSV (default: esg_controversy_features.csv)",
    )
    parser.add_argument(
        "--catalogue",
        action="store_true",
        help="Print feature catalogue and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.catalogue:
        print_catalogue()
        sys.exit(0)

    print("=" * 60)
    print("  ESG Controversy Feature Engineering Pipeline")
    print("=" * 60)

    df = load_data(args.input)
    df = build_target(df)
    df = engineer_features(df)
    df = finalise(df, args.output)

    print_catalogue()
    return df


if __name__ == "__main__":
    main()
