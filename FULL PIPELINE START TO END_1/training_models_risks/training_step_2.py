from __future__ import annotations
##############################################################################
#  Cell 1 — Imports & common paths
##############################################################################
import os
import json
import pandas as pd
import numpy as np
import psycopg2
import joblib
import logging
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neighbors import LocalOutlierFactor
import logging
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
from typing import Optional

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set options to show full DataFrame output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import random
from pathlib import Path
from datetime import timedelta
import joblib, json, warnings
import numpy as np
import pandas as pd
import argparse
from typing import Any, Dict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
import sys, os

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"


RAW_PO      = Path("po_output.pkl") #("../Po_Invoice_Data/po_output_tushar.pkl")      # raw PO data (input)
FEAT_PO     = Path("Po_Invoice_Data/po_output_features_df_auto_model.pkl")         # engineered features
MODEL_PKL   = Path("Po_Invoice_Data/po_autoencoder_model.pkl")         # tuned model file
#CV_REPORT   = Path("Po_Invoice_Data/cv_results.csv")            # param grid results
#SCORING_OUT = Path("Po_Invoice_Data/scored_po.pkl")             # predictions file

# -----------------------------------------------------------------------------
# 1) Column standardisation helper
# -----------------------------------------------------------------------------


COL_MAP: Dict[str, str] = {
    # raw_column                          # internal name
    "vendor_or_creditor_acct_no_hpd_po": "vendor_id",
    "material_no_src_po": "material_id",
    "purch_doc_date_hpd_po": "po_date",
    "doc_change_date_src_po": "po_change_date",
    "net_price_doc_curr_src_po": "net_price",
    "gross_val_po_curr_src_po": "gross_val",
    "exchange_rate_hpd_po": "exch_rate",
    "requester_name_src_po": "requester",
    # unit‑conversion numerators / denominators
    "p2o_unit_conv_num_src_po": "p2o_num",
    "p2o_unit_conv_denom_src_po": "p2o_den",
    "o2b_unit_conv_num_src_po": "o2b_num",
    "o2b_unit_conv_denom_src_po": "o2b_den",
}

# -----------------------------------------------------------------------------
# 2) Feature engineering functions (pure, chainable)
# -----------------------------------------------------------------------------

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Rename key columns, parse dates & fill obvious nulls."""
    #df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
    df["purch_doc_date_hpd_po"] = pd.to_datetime(df["purch_doc_date_hpd_po"], errors="coerce")
    df["doc_change_date_src_po"] = pd.to_datetime(df.get("doc_change_date_src_po"), errors="coerce")
    df["vendor_or_creditor_acct_no_hpd_po"] = df.get("vendor_or_creditor_acct_no_hpd_po", "UNKNOWN").fillna("UNKNOWN")
    df["requester_name_src_po"] = df.get("requester_name_src_po", "UNKNOWN").fillna("UNKNOWN")
    df["exchange_rate_hpd_po"] = df.get("exchange_rate_hpd_po", 1.0).replace({0: np.nan}).fillna(1.0)
    return df

# -----------------------------------------------------------------------------
# 2a) Rule‑based features (rules 1,2,3,5)
# -----------------------------------------------------------------------------

def add_rule_metrics(df: pd.DataFrame,
                     split_days: int = 60,
                     price_var_days: int = 365) -> pd.DataFrame:
    """Add features mirroring Baldota P2P rules.

    * vm_count/value = aggregation for **same vendor+material** within *split_days*
    * vm_price_var_pct = price deviation (%) vs mean of past *price_var_days*
    * mat_vendor_cnt & mat_price_var_pct analogues for material across vendors
    """
    df = df.copy()
    df.sort_values("purch_doc_date_hpd_po", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Initialise
    df["vm_count_%dd" % split_days] = 0
    df["vm_value_%dd" % split_days] = 0.0
    df["vm_price_var_pct_%dd" % price_var_days] = 0.0
    df["mat_vendor_cnt_%dd" % split_days] = 0
    df["mat_price_var_pct_%dd" % price_var_days] = 0.0

    # Pre‑extract convenient arrays for speed
    po_dates = df["purch_doc_date_hpd_po"].values
    prices = df.get("net_price_doc_curr_src_po").astype(float).values
    vals = df.get("gross_val_po_curr_src_po").astype(float).values

    # --- Same vendor + material group logic ----------------------------------
    for (v, m), idx in df.groupby(["vendor_or_creditor_acct_no_hpd_po", "material_no_src_po"]).groups.items():
        d = po_dates[idx]
        p = prices[idx]
        vls = vals[idx]
        for loc, ridx in enumerate(idx):
            cur = d[loc]
            # split window
            win_mask = (d >= cur - np.timedelta64(split_days, "D")) & (d <= cur)
            df.iat[ridx, df.columns.get_loc("vm_count_%dd" % split_days)] = int(win_mask.sum())
            df.iat[ridx, df.columns.get_loc("vm_value_%dd" % split_days)] = float(vls[win_mask].sum())
            # price variance window
            var_mask = (d >= cur - np.timedelta64(price_var_days, "D")) & (d <= cur)
            if var_mask.sum() > 1:
                mean_price = p[var_mask].mean()
                if mean_price:
                    pct = abs(p[loc] - mean_price) / mean_price * 100
                    df.iat[ridx, df.columns.get_loc("vm_price_var_pct_%dd" % price_var_days)] = pct

    # --- Material‑only group logic -------------------------------------------
    for m, idx in df.groupby("material_no_src_po").groups.items():
        d = po_dates[idx]
        p = prices[idx]
        vendors = df.loc[idx, "vendor_or_creditor_acct_no_hpd_po"].values
        for loc, ridx in enumerate(idx):
            cur = d[loc]
            win_mask = (d >= cur - np.timedelta64(split_days, "D")) & (d <= cur)
            df.iat[ridx, df.columns.get_loc("mat_vendor_cnt_%dd" % split_days)] = int(len(set(vendors[win_mask])))
            var_mask = (d >= cur - np.timedelta64(price_var_days, "D")) & (d <= cur)
            if var_mask.sum() > 1:
                mean_price = p[var_mask].mean()
                if mean_price:
                    pct = abs(p[loc] - mean_price) / mean_price * 100
                    df.iat[ridx, df.columns.get_loc("mat_price_var_pct_%dd" % price_var_days)] = pct

    return df

# -----------------------------------------------------------------------------
# 2b) Value & process metrics
# -----------------------------------------------------------------------------

def add_value_and_timing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["conv_factor_p2o"] = (
        df.get("p2o_unit_conv_num_src_po") / df.get("p2o_unit_conv_denom_src_po")
    ).replace({np.inf: np.nan, -np.inf: np.nan})
    df["conv_factor_o2b"] = (
        df.get("o2b_unit_conv_num_src_po") / df.get("o2b_unit_conv_denom_src_po")
    ).replace({np.inf: np.nan, -np.inf: np.nan})

    df["po_change_lag_days"] = (df.get("doc_change_date_src_po") - df["purch_doc_date_hpd_po"]).dt.days
    df["base_value"] = df.get("gross_val_po_curr_src_po") * df.get("exchange_rate_hpd_po")
    p95 = df["gross_val_po_curr_src_po"].quantile(0.95)
    df["high_value_flag"] = (df["gross_val_po_curr_src_po"] >= p95).astype(int)
    return df

# -----------------------------------------------------------------------------
# 2c) Behavioural rolling windows
# -----------------------------------------------------------------------------

def _rolling_stats(df: pd.DataFrame, group_col: str, days: int,
                   count_col: str, sum_col: str) -> pd.DataFrame:
    df = df.copy()
    df[count_col] = 0
    df[sum_col] = 0.0

    po_dates = df["purch_doc_date_hpd_po"].values
    gross_vals = df["gross_val_po_curr_src_po"].values

    for key, idx in df.groupby(group_col).groups.items():
        dates = po_dates[idx]
        vals = gross_vals[idx]
        for loc, ridx in enumerate(idx):
            cur = dates[loc]
            mask = (dates >= cur - np.timedelta64(days, "D")) & (dates <= cur)
            df.iat[ridx, df.columns.get_loc(count_col)] = int(mask.sum())
            df.iat[ridx, df.columns.get_loc(sum_col)] = float(vals[mask].sum())
    return df


def add_behavioural_stats(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    df = _rolling_stats(df, "requester_name_src_po", window_days,
                        "req_po_count_%dd" % window_days,
                        "req_val_sum_%dd" % window_days)
    df = _rolling_stats(df, "vendor_or_creditor_acct_no_hpd_po", window_days,
                        "vendor_po_count_%dd" % window_days,
                        "vendor_val_sum_%dd" % window_days)
    return df

# -----------------------------------------------------------------------------
# 3) Public orchestrator
# -----------------------------------------------------------------------------

def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """End‑to‑end feature generator (no target leakage)."""
    df = _prep(raw_df)
    df = add_rule_metrics(df)
    df = add_value_and_timing(df)
    df = add_behavioural_stats(df)
    return df
def flag_split_po(df):
    df = df.copy()
    df['split_po_flag'] = 0  # Default 0

    # Apply exclusion filters only for computation
    exclude_doc_types = ["AN", "AR", "MN", "QC", "QI", "QS", "RS", "SC", "SG", "SR", "SS", "ST", "TP", "TR", "UB", "WK"]

    filtered = df[
        (~df['purch_doc_type_hpd_po'].isin(exclude_doc_types)) &
        (df['purch_doc_type_hpd_po'].notna()) &
        (~(df['po_item_del_flag_src_po'] == 'L')) &
        (~df['plant_src_po'].fillna("").astype(str).str.startswith("4")) &
        (df['release_indicator_hpd_po'] == 'R') &
        (df['gross_val_po_curr_src_po'] >= 10) &
        (df['purch_doc_date_hpd_po'].notna()) &
        (df['vendor_or_creditor_acct_no_hpd_po'].notna()) &
        (df['company_code_src_po'].notna())
    ].copy()

    filtered['purch_doc_date_hpd_po'] = pd.to_datetime(filtered['purch_doc_date_hpd_po'])

    def get_group_key(row):
        if pd.notna(row['material_no_src_po']):
            return f"{row['vendor_or_creditor_acct_no_hpd_po']}__{row['material_no_src_po']}"
        elif pd.notna(row['short_text_src_po']):
            return f"{row['vendor_or_creditor_acct_no_hpd_po']}__{row['short_text_src_po']}"
        else:
            return np.nan

    filtered['split_key'] = filtered.apply(get_group_key, axis=1)
    filtered = filtered[filtered['split_key'].notna()].copy()
    filtered.sort_values(['split_key', 'purch_doc_date_hpd_po'], inplace=True)

    flagged_po_set = set()

    for key, group in filtered.groupby('split_key'):
        dates = group['purch_doc_date_hpd_po'].reset_index(drop=True)
        po_nos = group['purch_doc_no_src_po'].reset_index(drop=True)

        for i in range(len(dates)):
            date_i = dates[i]
            po_i = po_nos[i]
            mask = (
                (dates >= date_i - pd.Timedelta(days=14)) &
                (dates <= date_i + pd.Timedelta(days=14)) &
                (po_nos != po_i)
            )
            if mask.sum() > 0:
                flagged_po_set.add(po_i)

    # Assign flag only to matching rows in original df
    df['split_po_flag'] = df['purch_doc_no_src_po'].isin(flagged_po_set).astype(int)
    return df
def flag_intra_po_split(df, gross_threshold=10):
    df = df.copy()
    df['intra_po_split_flag'] = 0  # Default

    df_valid = df[
        df['gross_val_po_curr_src_po'].notna() &
        df['vendor_or_creditor_acct_no_hpd_po'].notna() &
        df['material_no_src_po'].notna() &
        df['purch_doc_no_src_po'].notna()
    ].copy()

    group_cols = ['purch_doc_no_src_po', 'vendor_or_creditor_acct_no_hpd_po', 'material_no_src_po']
    grouped = df_valid.groupby(group_cols)

    flagged_indexes = []

    for _, group in grouped:
        total_gross = group['gross_val_po_curr_src_po'].sum()
        num_items = len(group)
        all_below_threshold = group['gross_val_po_curr_src_po'].all()

        if total_gross >= gross_threshold and num_items > 1 and all_below_threshold:
            flagged_indexes.extend(group.index.tolist())

    df.loc[flagged_indexes, 'intra_po_split_flag'] = 1
    return df
def flag_multiple_pos_per_pr_item(df):
    df = df.copy()
    df['multi_po_per_pr_flag'] = 0  # Default

    # Only consider approved POs with valid PR and PR item
    df_valid = df[
        (df['release_indicator_hpd_po'] == 'R') &
        df['pr_no_src_po'].notna() &
        df['pr_item_no_src_po'].notna() &
        df['purch_doc_no_src_po'].notna()
    ][['pr_no_src_po', 'pr_item_no_src_po', 'purch_doc_no_src_po']].drop_duplicates()

    # Count number of unique POs per PR+Item
    po_counts = df_valid.groupby(['pr_no_src_po', 'pr_item_no_src_po'])['purch_doc_no_src_po'].nunique()

    # Identify PR+Items linked to more than one PO
    multi_po_keys = po_counts[po_counts > 1].index.tolist()

    # Create a set for fast lookup
    multi_po_set = set(multi_po_keys)

    # Flag in the main DataFrame
    df['multi_po_per_pr_flag'] = df.apply(
        lambda row: 1 if (row['pr_no_src_po'], row['pr_item_no_src_po']) in multi_po_set else 0,
        axis=1
    )

    return df
def flag_same_vendor_price_increase(df, price_increase_threshold=0.05, months_range=6, flag_column='same_vendor_price_increase_flag'):
    df = df.copy()
    df[flag_column] = 0

    df_valid = df[
        (df['release_indicator_hpd_po'] == 'R') &
        (df['material_no_src_po'].notna()) &
        (df['vendor_or_creditor_acct_no_hpd_po'].notna()) &
        (df['net_price_doc_curr_src_po'].notna()) &
        (df['purch_doc_date_hpd_po'].notna())
    ].copy()

    df_valid['purch_doc_date_hpd_po'] = pd.to_datetime(df_valid['purch_doc_date_hpd_po'])
    df_valid.sort_values(['vendor_or_creditor_acct_no_hpd_po', 'material_no_src_po', 'purch_doc_date_hpd_po'], inplace=True)

    group_cols = ['vendor_or_creditor_acct_no_hpd_po', 'material_no_src_po']
    flagged_indices = []

    for _, group in df_valid.groupby(group_cols):
        group = group.sort_values('purch_doc_date_hpd_po').reset_index()

        for i in range(1, len(group)):
            current_row = group.loc[i]
            current_date = current_row['purch_doc_date_hpd_po']
            current_price = current_row['net_price_doc_curr_src_po']

            mask = group.loc[:i-1, 'purch_doc_date_hpd_po'] >= current_date - pd.DateOffset(months=months_range)
            past_group = group.loc[:i-1][mask]

            if not past_group.empty:
                last_price = past_group['net_price_doc_curr_src_po'].iloc[-1]
                if last_price > 0 and ((current_price - last_price) / last_price) >= price_increase_threshold:
                    flagged_indices.append(current_row['index'])

    df.loc[flagged_indices, flag_column] = 1
    return df
def flag_diff_vendor_price_variance(df, price_variance_threshold=0.05, months_range=6, flag_column='diff_vendor_price_variance_flag'):
    df = df.copy()
    df[flag_column] = 0

    df_valid = df[
        (df['release_indicator_hpd_po'] == 'R') &
        (df['material_no_src_po'].notna()) &
        (df['vendor_or_creditor_acct_no_hpd_po'].notna()) &
        (df['net_price_doc_curr_src_po'].notna()) &
        (df['purch_doc_date_hpd_po'].notna())
    ].copy()

    df_valid['purch_doc_date_hpd_po'] = pd.to_datetime(df_valid['purch_doc_date_hpd_po'])
    df_valid = df_valid.sort_values(['material_no_src_po', 'purch_doc_date_hpd_po'])

    for material, mat_group in df_valid.groupby('material_no_src_po'):
        mat_group = mat_group.sort_values('purch_doc_date_hpd_po').reset_index()

        for i in range(len(mat_group)):
            current_row = mat_group.loc[i]
            current_date = current_row['purch_doc_date_hpd_po']
            current_price = current_row['net_price_doc_curr_src_po']

            past_window = mat_group[
                (mat_group['purch_doc_date_hpd_po'] < current_date) &
                (mat_group['purch_doc_date_hpd_po'] >= current_date - pd.DateOffset(months=months_range))
            ]

            vendor_prices = past_window.groupby('vendor_or_creditor_acct_no_hpd_po')['net_price_doc_curr_src_po'].mean()
            if not vendor_prices.empty:
                max_price = vendor_prices.max()
                min_price = vendor_prices.min()
                if max_price > 0 and (max_price - min_price) / max_price >= price_variance_threshold:
                    df.loc[current_row['index'], flag_column] = 1

    return df

def final_parsing(df):
        # === Parse Date Columns ===
    date_cols = [
        "doc_change_date_src_po",
        "doc_change_date_hpd_po",
        "purch_doc_date_hpd_po"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # === Feature Engineering ===
    
    # A. Date Features
    df["po_doc_age_days"] = (df["doc_change_date_hpd_po"] - df["purch_doc_date_hpd_po"]).dt.days
    df["lead_time_po_vs_pr"] = (df["purch_doc_date_hpd_po"] - df["doc_change_date_src_po"]).dt.days
    df["po_day_of_week"] = df["purch_doc_date_hpd_po"].dt.dayofweek
    df["po_day_of_month"] = df["purch_doc_date_hpd_po"].dt.day
    df["po_month"] = df["purch_doc_date_hpd_po"].dt.month
    
    # B. Price & Value Features
    df["price_per_unit"] = df["net_val_po_curr_src_po"] / df["quantity_src_po"].replace(0, np.nan)
    df["net_vs_gross_delta"] = df["gross_val_po_curr_src_po"] - df["net_val_po_curr_src_po"]
    df["price_variance_percent"] = (df["net_val_po_curr_src_po"] - df["gross_val_po_curr_src_po"]) / df["gross_val_po_curr_src_po"].replace(0, np.nan)
    df["outline_agrmt_coverage"] = df["outline_agrmt_tgt_val_doc_curr_src_po"] / df["net_val_po_curr_src_po"].replace(0, np.nan)
    
    # C. PO-PR Linkage Flags
    df["has_pr_link"] = df["pr_no_src_po"].notna().astype(int)
    df["has_pr_item_link"] = df["pr_item_no_src_po"].notna().astype(int)
    
    # D. Flags & Indicators
    binary_cols = [
        "gr_indicator_src_po", "gr_invoice_verif_flag_src_po",
        "inv_receipt_indicator_src_po", "release_indicator_hpd_po",
        "release_status_hpd_po", "doc_release_incompl_flag_hpd_po"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col + "_flag"] = df[col].notna().astype(int)
    
    # E. Missing Signal
    critical_cols = [
        "material_type_src_po", "material_no_src_po",
        "vendor_or_creditor_acct_no_hpd_po", "gross_val_po_curr_src_po",
        "net_price_doc_curr_src_po"
    ]
    df["missing_critical_fields"] = df[critical_cols].isnull().sum(axis=1)
    
    # F. Currency & Exchange Rate
    df["has_exchange_rate"] = df["exchange_rate_hpd_po"].notna().astype(int)
    df["log_exchange_rate"] = np.log1p(df["exchange_rate_hpd_po"].fillna(0))
    
    # G. Unit Conversion Features
    df["p2o_conversion_ratio"] = df["p2o_unit_conv_num_src_po"] / df["p2o_unit_conv_denom_src_po"].replace(0, np.nan)
    df["o2b_conversion_ratio"] = df["o2b_unit_conv_num_src_po"] / df["o2b_unit_conv_denom_src_po"].replace(0, np.nan)
    
    # H. Behavioral Flags
    df["is_same_vendor_pr_po"] = (
        df["vendor_or_creditor_acct_no_hpd_po"].notna() & df["base_id_src_po"].notna()
    ).astype(int)
    df["has_rfq_status"] = df["rfq_status_hpd_po"].notna().astype(int)
    df["purch_group_org_same"] = (df["purch_group_hpd_po"] == df["purch_org_hpd_po"]).astype(int)
    
    # I. Rare Category Flagging
    rare_cat_cols = ["purch_doc_type_hpd_po", "purch_group_hpd_po", "vendor_or_creditor_acct_no_hpd_po"]
    for col in rare_cat_cols:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[f"{col}_is_rare"] = df[col].map(freq_map) < 0.01

    return df

##############################################################################
#  Cell 3 — Create / load features
##############################################################################
if FEAT_PO.exists():
    print("⚡ Loading cached features")
    df = pd.read_pickle(FEAT_PO)
else:
    print("🚧 Generating features …")
    raw_df  = pd.read_pickle(RAW_PO)
    columns_to_drop = [
    'base_id_po','rule_ids_po','P2P02067','P2P02068','P2P02070','P2P02072']
    raw_df.drop(columns=[col for col in columns_to_drop if col in raw_df.columns], inplace=True)
    feat_df = build_features(raw_df)
    df=flag_split_po(feat_df)
    df=flag_intra_po_split(df)
    df = flag_multiple_pos_per_pr_item(df)
    # Same vendor price jump
    df = flag_same_vendor_price_increase(df, months_range=6, flag_column='same_vendor_price_increase_6m_flag')
    df = flag_same_vendor_price_increase(df, months_range=12, flag_column='same_vendor_price_increase_12m_flag')
    
    # Diff vendor price variance
    df = flag_diff_vendor_price_variance(df, months_range=6, flag_column='diff_vendor_price_variance_6m_flag')
    df = flag_diff_vendor_price_variance(df, months_range=12, flag_column='diff_vendor_price_variance_12m_flag')
    df=final_parsing(df)
    df.to_pickle(FEAT_PO)
    print("Feature shape:", df.shape)

##############################################################################
#  Cell 4 — Fast train (no hyper-parameter search, no RFECV)
##############################################################################
# =======================
# Reproducibility
# =======================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =======================
# Step 1: Prepare Data
# =======================
target = "rft_by_engine_po"
df[target] = df[target].fillna(0).astype(int)
y = df[target]
X = df.drop(columns=[target])

num_cols = [col for col in X.columns if X[col].dtype.kind in 'if']
cat_cols = [col for col in X.columns if X[col].dtype.kind not in 'if']

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10))
    ]), cat_cols)
])

X_scaled = preprocessor.fit_transform(X)

# Split for training and testing
X_train_unsupervised = X_scaled[y == 0]
X_test = X_scaled
y_test = y.values

# =======================
# Step 2: Define Autoencoder
# =======================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_1, hidden_2, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# =======================
# Step 3: Train Final Model
# =======================

# 🔧 Best Hyperparameters from Optuna
best_params = {
    'hidden_1': 200,
    'hidden_2': 92,
    'bottleneck': 23,
    'lr': 0.000518,
    'batch_size': 256
}

input_dim = X_train_unsupervised.shape[1]
model = Autoencoder(input_dim, best_params['hidden_1'], best_params['hidden_2'], best_params['bottleneck'])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

X_train_tensor = torch.tensor(X_train_unsupervised, dtype=torch.float32)
batch_size = best_params['batch_size']
n_epochs = 100
patience = 5
best_loss = float('inf')
patience_counter = 0

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for i in range(0, X_train_tensor.size(0), batch_size):
        batch = X_train_tensor[i:i+batch_size]
        output = model(batch)
        loss = criterion(output, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (X_train_tensor.size(0) // batch_size)
    print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# =======================
# Step 4: Evaluate
# =======================
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
model.eval()
with torch.no_grad():
    reconstructed = model(X_test_tensor)
    recon_error = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).numpy()

# Dynamic threshold based on normal (non-fraud) population
threshold = np.percentile(recon_error[y_test == 0], 95)
print(f"\n🚨 Reconstruction Error Threshold: {threshold:.4f}")

predicted_labels = (recon_error > threshold).astype(int)

# =======================
# Step 5: Report & Confusion Matrix
# =======================
print("\n📊 Classification Report (Autoencoder):")
print(classification_report(y_test, predicted_labels))

#cm = confusion_matrix(y_test, predicted_labels)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Risk", "Fraud"])
#disp.plot(cmap='Blues')
#plt.title("Confusion Matrix - Final Autoencoder")
#plt.show()

# =======================
# Step 6: Add Back to DataFrame
# =======================
df["ae_fraud_score"] = recon_error
df["ae_predicted_flag"] = predicted_labels

torch.save(model.state_dict(), "Po_Invoice_Data/autoencoder_model.pth")
joblib.dump(preprocessor, "Po_Invoice_Data/preprocessor.pkl")
joblib.dump(best_params, "Po_Invoice_Data/best_params.pkl")
joblib.dump(threshold, "Po_Invoice_Data/autoencoder_threshold.pkl")
print("✅ Baseline model saved")