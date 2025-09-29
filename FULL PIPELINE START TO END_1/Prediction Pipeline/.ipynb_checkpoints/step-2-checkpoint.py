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
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import sys, os

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# 2) Feature engineering functions (pure, chainable)
# -----------------------------------------------------------------------------

def preprocessing_feature_engineering_prediction(df_final):
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
    
    def auto_encoder_pipeline(df_new):
        # ========== Step 1: Load Files Auto Encoder ==========
        model_weights_path = "Po_Invoice_Data/autoencoder_model.pth"
        preprocessor_path = "Po_Invoice_Data/preprocessor.pkl"
        best_params_path = "Po_Invoice_Data/best_params.pkl"
        # Load precomputed threshold
        threshold_path = "Po_Invoice_Data/autoencoder_threshold.pkl"
    
        preprocessor = joblib.load(preprocessor_path)
        best_params = joblib.load(best_params_path)
        
        # ========== Step 3: Preprocess the New Data ==========
        X_new_scaled = preprocessor.transform(df_new)
        
        # ========== Step 5: Initialize & Load Model ==========
        input_dim = X_new_scaled.shape[1]
        model = Autoencoder(input_dim, best_params['hidden_1'], best_params['hidden_2'], best_params['bottleneck'])
        model.load_state_dict(torch.load(model_weights_path))
        model.eval()
        
        # ========== Step 6: Get Reconstruction Error ==========
        X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            reconstructed = model(X_tensor)
            recon_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        # ========== Step 7: Apply Threshold to Get Prediction ==========
        # Load the same threshold used during training or recompute if needed
        # For now, assume same threshold as earlier
        threshold =joblib.load(threshold_path) #np.percentile(recon_error, 95)  # or hardcode: threshold = 0.0383
        print(f"🚨 Threshold used: {threshold:.4f}")
        
        predicted_labels = (recon_error > threshold).astype(int)
        
        # ========== Step 8: Add to DataFrame and Save ==========
        df_new["ae_fraud_score"] = recon_error
        df_new["ae_predicted_flag"] = predicted_labels
        
    
        return df_new
    
    def gbt_pipeline(df):
        # === CONFIG ===
        MODEL_PATH = Path("Po_Invoice_Data/po_gbdt_model.pkl")
        print("📦 Loading model from:", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
    
        print("🔍 Predicting risk probabilities…")
        X = df.copy()
        fraud_flag = model.predict(X)
        fraud_score = model.predict_proba(X)[:, 1]
        
        print("📝 Adding prediction columns…")
        df["gbt_fraud_score"] = fraud_score
        df["gbt_model_flag"] = fraud_flag
    
        return df
    
    def predict_isolation_forest(new_df: pd.DataFrame):
        ARTIFACT_DIR = Path("Po_Invoice_Data")
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        PREPROC_PKL   = ARTIFACT_DIR / "iso_preprocessor.pkl"
        MODEL_PKL     = ARTIFACT_DIR / "isoforest_model.pkl"
        THRESHOLD_NPY = ARTIFACT_DIR / "isoforest_threshold.npy"
        """
        Loads saved preprocessor, model, and threshold. Returns scores + flags for new data.
        """
        preproc = joblib.load(PREPROC_PKL)
        iso = joblib.load(MODEL_PKL)
        threshold = float(np.load(THRESHOLD_NPY)[0])
    
        out = new_df.copy()
        X_enc = preproc.transform(new_df)
        scores = iso.score_samples(X_enc)
        flags = (scores < threshold).astype(int)  # 1 = anomaly
    
        #out = new_df.copy()"iso_fraud_score", "iso_predicted_flag"
        out["iso_fraud_score"] = scores
        out["iso_predicted_flag"]  = flags
        return out
    
    # === Drop same columns as training ===
    columns_to_drop = [
        # IDs and references
        "purch_doc_no_src_po", "purch_doc_item_no_src_po", "pr_no_src_po", "pr_item_no_src_po","base_id_src_po", 
        "principal_purch_agrmt_item_no_src_po", "principal_purch_agrmt_no_hpd_po",
    
        # Text
        "short_text_src_po", "requester_name_src_po", "resp_vendor_salesperson_hpd_po",
    
        # Dates (used to create features)
        "doc_change_date_src_po", "doc_change_date_hpd_po", "purch_doc_date_hpd_po",
    
        # Sparse or incomplete
        "po_item_del_flag_src_po", "doc_release_incompl_flag_hpd_po", "control_indicator_hpd_po",
    
        # Replaced with ratios / logs / engineered
        "p2o_unit_conv_denom_src_po", "p2o_unit_conv_num_src_po",
        "o2b_unit_conv_denom_src_po", "o2b_unit_conv_num_src_po",
        "gross_val_po_curr_src_po", "net_val_po_curr_src_po",
        "outline_agrmt_tgt_val_doc_curr_src_po", "quantity_src_po",
        "exchange_rate_hpd_po", "net_price_doc_curr_src_po",]
    def predict_sub_risks(raw_df) -> pd.DataFrame:
        # === Load MLP model and transformers ===
        MODEL_PATH = "Po_Invoice_Data/rf_multilabel_model_sub_risks.pkl"
        MLB_PATH = "Po_Invoice_Data/mlb.pkl"
        ENCODERS_PATH = "Po_Invoice_Data/label_encoders.pkl"
        IMPUTER_PATH = "Po_Invoice_Data/imputer.pkl"
        
        model = joblib.load(MODEL_PATH)
        mlb = joblib.load(MLB_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        
        # Step 2: Drop unused columns
        columns_to_drop_before_preprocessing= [
        'rft_by_engine_po','base_id_po', 'rule_ids_po','P2P02067', 'P2P02068', 'P2P02070', 'P2P02072','gbt_fraud_score', 'gbt_model_flag', 'ae_fraud_score',
           'ae_predicted_flag',"iso_fraud_score", "iso_predicted_flag",'risk_score','risk_level',]#,'model_flag' ]
        raw_df.drop(columns=[col for col in columns_to_drop_before_preprocessing if col in raw_df.columns], inplace=True)
        
        print("🧮 Generating features…")
        # Step 3: Feature engineering
        df = build_features(raw_df)
        df = flag_split_po(df)
        df = flag_intra_po_split(df)
        df = flag_multiple_pos_per_pr_item(df)
        df = flag_same_vendor_price_increase(df, months_range=6, flag_column='same_vendor_price_increase_6m_flag')
        df = flag_same_vendor_price_increase(df, months_range=12, flag_column='same_vendor_price_increase_12m_flag')
        df = flag_diff_vendor_price_variance(df, months_range=6, flag_column='diff_vendor_price_variance_6m_flag')
        df = flag_diff_vendor_price_variance(df, months_range=12, flag_column='diff_vendor_price_variance_12m_flag')
        df = final_parsing(df)
        df_with_base_id_src_po=df.copy()
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
        # Step 4: Encode categorical
        categorical_cols = df.select_dtypes(include='object').columns
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).fillna("Unknown")
                df[col] = le.transform(df[col])
            else:
                df[col] = 0  # fallback if unseen
    
        # Step 5: Impute numeric
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_cols] = imputer.transform(df[numeric_cols])
        
        print("📦 Loading model from:", MODEL_PATH)
        #  Step 6: Predict
        X = df.copy()
        
        print("🔍 Predicting risk probabilities…")
        y_pred = model.predict(X)
        predicted_sub_risks = mlb.inverse_transform(y_pred)
    
        # Step 7: Output DataFrame
        print("📝 Adding prediction columns…")
        df_with_base_id_src_po["predicted_risks"] = predicted_sub_risks
        # merge with data on base_id_src_po
        #new_df=pd.merge(final_df,df_with_base_id_src_po[['base_id_src_po','Predicted Risks']],on='base_id_src_po')
        #new_df.drop_duplicates(inplace=True)
        # Save results
        #new_df.to_pickle(output_pickle_path)
        #print(f"✅ Predictions saved to {output_pickle_path}")
    
        return df_with_base_id_src_po
    
    def compute_risk_level(row):
        gbt = int(row['gbt_model_flag'])
        ae  = int(row['ae_predicted_flag'])
        iso = int(row['iso_predicted_flag'])
    
        # 1) All three flag → Very High Risk
        if gbt == 1 and ae == 1 and iso == 1:
            return "Very High Risk"
        if gbt== 0 and ae == 0 and iso == 0:
            return "No Risk"
        # 2) GBT flags (regardless of others, except the all-three case above) → High Risk
        if gbt == 1:
            return "High Risk"
    
        # 3) GBT=0 and at least one of AE/ISO flags → Needs Validation
        if gbt == 0 and (ae == 1 or iso == 1):
            return "Needs Validation"
    
    
    def compute_model_flag(row):
        gbt = int(row['gbt_model_flag'])
        ae  = int(row['ae_predicted_flag'])
        iso = int(row['iso_predicted_flag'])
        # 0 only if all three are 0
        return 0 if (gbt == 0 and ae == 0 and iso == 0) else 1
    
    def fraud_weighted_score(amount, prob):
        """
        Score in [0,1] = minmax(amount) * prob_0_1
        - amount: array-like or pandas Series (e.g., net_val_po_curr_src_po)
        - prob: array-like or pandas Series (e.g., gbt_fraud_score; % or 0–1)
        """
        a = pd.to_numeric(pd.Series(amount), errors="coerce").astype(float)
        p = pd.to_numeric(pd.Series(prob),   errors="coerce").astype(float)
    
        # normalize prob to [0,1] if it looks like percentage
        maxp = np.nanmax(p.values) if len(p) else np.nan
        if np.isfinite(maxp) and 1.0 < maxp <= 100.0:
            p = p / 100.0
        p = p.clip(0.0, 1.0)
    
        # min-max scale amount
        amin, amax = a.min(skipna=True), a.max(skipna=True)
        if pd.isna(amin) or pd.isna(amax) or amax == amin:
            a_mm = pd.Series(0.0, index=a.index)
        else:
            a_mm = (a - amin) / (amax - amin)
    
        # final score
        return (0.3*a_mm + 0.7*p).fillna(0.0)
    
    
    def sub_risk_screen(output_df):
            # 🔹 Step 2: Ensure all values in 'Predicted Risks' are tuples
        output_df["predicted_risks"] = output_df["predicted_risks"].apply(lambda x: tuple(x) if not isinstance(x, tuple) else x)
        
        # 🔹 Step 3: Fill empty risks based on model_flag
        output_df["predicted_risks"] = output_df.apply(
            lambda row: ("Price Variance Risk",) if row["model_flag"] == 1 and row["predicted_risks"] in [(), None, np.nan]
            else ("No Risk",) if row["model_flag"] == 0 and row["predicted_risks"] in [(), None, np.nan]
            else row["predicted_risks"],
            axis=1
        )
        
        # 🔹 Step 4: Main Risk Scenario
        output_df["main_risk_scenario"] = output_df["model_flag"].apply(
            lambda x: "Procurement Risk" if x == 1 else "No Risk"
        )
        
        # 🔹 Step 5: Drop existing Sub Risk columns if any (optional cleanup)
        output_df.drop(columns=[col for col in output_df.columns if col.startswith("sub_risk_")], errors='ignore', inplace=True)
        
        # 🔹 Step 6: Expand risk tuple to individual columns — safe even if only one risk
        risk_df = pd.DataFrame(output_df["predicted_risks"].tolist())
        
        # 🔹 Fix mismatched index
        risk_df.reset_index(drop=True, inplace=True)
        output_df.reset_index(drop=True, inplace=True)
        
        # 🔹 Rename dynamically based on max number of sub-risks
        risk_df.columns = [f"sub_risk_{i+1}" for i in range(risk_df.shape[1])]
        
        # 🔹 Step 7: Safe concat
        output_df = pd.concat([output_df, risk_df], axis=1)
        return output_df
    
    def get_impact_ordered(row, col1="sub_risk_1", col2="sub_risk_2"):
        s1 = str(row.get(col1, "") or "")
        s2 = str(row.get(col2, "") or "")
        impacts = []
    
        def add(x):
            if x not in impacts:
                impacts.append(x)
    
        # Map contributions from sub_risk_1 (first)
        if "Price Variance Risk" in s1:
            add("Profitability")
            add("Cash Flow")
        if "Split PO" in s1:
            add("Efficiency")
    
        # Then contributions from sub_risk_2 (second)
        if "Price Variance Risk" in s2:
            add("Profitability")
            add("Cash Flow")
        if "Split PO" in s2:
            add("Efficiency")
    
        else:
            add('None')
    
        return impacts
    
    def split_impact_column(df, impact_col="impact", max_impacts=3):
        """
        Splits a list-based 'Impact' column into separate columns: Impact 1, Impact 2, ..., Impact n.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            impact_col (str): Name of the column containing the list of impacts.
            max_impacts (int): Maximum number of impact columns to create.
        
        Returns:
            pd.DataFrame: DataFrame with new 'Impact 1'...'Impact n' columns added.
        """
        if impact_col not in df.columns:
            raise ValueError(f"Column '{impact_col}' not found in DataFrame.")
        
        # Ensure all rows are lists, then pad with None to match max_impacts
        impact_expanded = df[impact_col].apply(
            lambda x: (x if isinstance(x, list) else []) + [None] * (max_impacts - len(x))
        )
        
        # Create new columns
        new_cols = [f"impact_{i+1}" for i in range(max_impacts)]
        df[new_cols] = pd.DataFrame(impact_expanded.tolist(), index=df.index)
        
        return df
    
    def preprocess_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = ['rft_by_engine_po','base_id_po', 'rule_ids_po', 'P2P02067', 'P2P02068', 'P2P02070', 'P2P02072']
        raw_df.drop(columns=[col for col in columns_to_drop if col in raw_df.columns], errors="ignore", inplace=True)
        df = build_features(raw_df)
        df = flag_split_po(df)
        df = flag_intra_po_split(df)
        df = flag_multiple_pos_per_pr_item(df)
        df = flag_same_vendor_price_increase(df, months_range=6, flag_column='same_vendor_price_increase_6m_flag')
        df = flag_same_vendor_price_increase(df, months_range=12, flag_column='same_vendor_price_increase_12m_flag')
        df = flag_diff_vendor_price_variance(df, months_range=6, flag_column='diff_vendor_price_variance_6m_flag')
        df = flag_diff_vendor_price_variance(df, months_range=12, flag_column='diff_vendor_price_variance_12m_flag')
        df = final_parsing(df)
    
        
        return df
    
    def predict_rft_flags(po_data_clean) -> pd.DataFrame:#predict_rft_flags( raw_path: Union[str, Path]) -> pd.DataFrame:
        print("📥 Loading raw PO data…")
        #raw_df = pd.read_pickle(raw_path) if str(raw_path).endswith(".pkl") else pd.read_csv(raw_path)
        raw_df=po_data_clean
        
        # Split into non-NaN and NaN base_id_src_po
        df_with_id = raw_df[raw_df['base_id_src_po'].notna()].copy()
        df_without_id = raw_df[raw_df['base_id_src_po'].isna()].copy()
    
        ### ==== PROCESS NON-NaN DATA ====
        print("🧮 Processing data with base_id_src_po...")
        data_for_gbt = preprocess_pipeline(df_with_id)
        data_for_auto = data_for_gbt.copy()
        data_for_iso=data_for_gbt.copy()
        df_gbt= gbt_pipeline(data_for_gbt)
        df_auto= auto_encoder_pipeline(data_for_auto)
        df_iso=predict_isolation_forest(data_for_iso)
        # 🔹 Fix mismatched index
        new_df=(df_gbt[["base_id_src_po", "gbt_fraud_score", "gbt_model_flag"]]
        .merge(
            df_auto[["base_id_src_po", "ae_fraud_score", "ae_predicted_flag"]],
            on="base_id_src_po",
            how="outer"
        )
        .merge(
            df_iso[["base_id_src_po", "iso_fraud_score", "iso_predicted_flag"]],
            on="base_id_src_po",
            how="outer"
        )
    )
    
    
        gbt_auto_df = pd.merge(df_with_id, new_df, on='base_id_src_po', how='inner').drop_duplicates().reset_index(drop=True)
    
        # Risk levels
        gbt_auto_df["risk_score"] = fraud_weighted_score(gbt_auto_df["net_val_po_curr_src_po"], gbt_auto_df["gbt_fraud_score"])
        gbt_auto_df['risk_level'] = gbt_auto_df.apply(compute_risk_level, axis=1)
        gbt_auto_df['model_flag'] = gbt_auto_df.apply(compute_model_flag, axis=1)
    
        # Sub-risk prediction
        sub_risk_data = predict_sub_risks(gbt_auto_df.copy())
        ml_with_all_data_df = pd.merge(
            gbt_auto_df, 
            sub_risk_data[['base_id_src_po', 'predicted_risks']],
            on='base_id_src_po',
            how='left'
        ).drop_duplicates().reset_index(drop=True)
    
        result_df_with_id = sub_risk_screen(ml_with_all_data_df)
    
        ### ==== PROCESS NaN DATA SEPARATELY ====
        if not df_without_id.empty:
            print("🧮 Processing data WITHOUT base_id_src_po...")
            df_without_id.reset_index(drop=True, inplace=True)
            data_for_gbt_nan = preprocess_pipeline(df_without_id)
            data_for_auto_nan = data_for_gbt_nan.copy()
            data_for_iso_nan=data_for_gbt_nan.copy()
    
            df_gbt_nan = gbt_pipeline(data_for_gbt_nan)
            df_auto_nan = auto_encoder_pipeline(data_for_auto_nan)
            df_iso_nan=predict_isolation_forest(data_for_iso_nan)
    
            df_gbt_nan.reset_index(drop=True, inplace=True)
            df_auto_nan.reset_index(drop=True, inplace=True)
            df_iso_nan.reset_index(drop=True, inplace=True)
    
            # Merge manually on index
            nan_df = pd.concat([df_without_id, df_gbt_nan[['gbt_fraud_score', 'gbt_model_flag']], df_auto_nan[['ae_fraud_score', 'ae_predicted_flag']],
                               df_iso_nan[['iso_fraud_score', 'iso_predicted_flag']]], axis=1)
            nan_df["risk_score"] = fraud_weighted_score(nan_df["net_val_po_curr_src_po"], nan_df["gbt_fraud_score"])
            nan_df['risk_level'] = nan_df.apply(compute_risk_level, axis=1)
            nan_df['model_flag'] = nan_df.apply(compute_model_flag, axis=1)
            nan_df['predicted_risks'] = [()] * len(nan_df)  # Empty risks for now
    
            result_df_without_id = sub_risk_screen(nan_df)
        else:
            result_df_without_id = pd.DataFrame()
    
        ### ==== CONCAT FINAL OUTPUT ====
        final_result_df = pd.concat([result_df_with_id, result_df_without_id], axis=0).reset_index(drop=True)
        final_result_df["impact"] = final_result_df.apply(get_impact_ordered, axis=1)
        final_result_df = split_impact_column(final_result_df, impact_col="impact", max_impacts=3)
    
        return final_result_df

    # === Example run ===
    #if __name__ == "__main__":
    final_result_df=predict_rft_flags(df_final.copy()) #"po_output.pkl")

    return final_result_df