from __future__ import annotations
# ============================================
# FortifAI — Verification + Word-style 5 Sections (Markdown tables)
# ============================================
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
import sys, os

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"
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

warnings.filterwarnings("ignore", category=FutureWarning)


import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def evidence_part_1(final_result_df_2):
    # =====================================================
    # CONFIG
    # =====================================================
    SPLIT_WINDOW_DAYS    = 60
    MAX_LINES_PER_RULE   = 50
    CURRENCY_MUST_MATCH  = True
    EPS_ZERO_DELTA_PCT   = 0.0  # treat |Δ%| <= EPS as "zero" (exclude)
    
    # NEW: hide empty subsections (that would otherwise print "No Risk")
    HIDE_NO_RISK_SUBSECTIONS = True
    
    # Doc-type exclusions
    DOC_TYPE_EXCLUDE_70 = {
        "NULL","AN","AR","MN","QC","QI","QS","RS","SC","SG","SR","SS","ST","TP","TR","UB","WK"
    }
    DOC_TYPE_EXCLUDE_6768 = {
        "NULL","AN","AR","MN","QC","QI","QS","RS","SC","SG","SR","SS","ST","TP","TR","UB","WK",
        "FO","RS1","RS2","RS3","RS4","PS1","PS2","PS3","PS4","ZSB"
    }
    
    # =====================================================
    # SYNONYMS & STANDARDIZATION
    # =====================================================
    COL_SYNONYMS = {
        "purch_doc_no_src_po":            ["purch_doc_no_src_po", "po_no", "purch_doc_no"],
        "purch_doc_item_no_src_po":       ["purch_doc_item_no_src_po", "po_item", "item_no"],
        "purch_doc_type_hpd_po":          ["purch_doc_type_hpd_po", "purch_doc_type", "doc_type"],
        "release_indicator_hpd_po":       ["release_indicator_hpd_po", "release_indicator", "rel_ind", "rel_ind_hpd_po"],
        "doc_change_date_hpd_po":         ["doc_change_date_hpd_po", "doc_change_date"],
        "purch_doc_date_hpd_po":          ["purch_doc_date_hpd_po", "purch_doc_date", "po_date"],
        "vendor_or_creditor_acct_no_hpd_po": ["vendor_or_creditor_acct_no_hpd_po", "vendor_code", "vendor_id"],
        "vendor_name_1":                  ["vendor_name_1", "vendor_name"],
        "material_no_src_po":             ["material_no_src_po", "material_code", "material"],
        "short_text_src_po":              ["short_text_src_po", "short_text", "item_desc", "description"],
        "order_uom_src_po":               ["order_uom_src_po", "uom"],
        "quantity_src_po":                ["quantity_src_po", "qty", "quantity"],
        "net_val_po_curr_src_po":         ["net_val_po_curr_src_po", "net_value", "net_val"],
        "gross_val_po_curr_src_po":       ["gross_val_po_curr_src_po", "gross_value", "gross_val"],
        "currency_hpd_po":                ["currency_hpd_po", "currency", "curr"],
        "company_code_hpd_po":            ["company_code_hpd_po", "company_code", "bukrs"],
        "plant_src_po":                   ["plant_src_po", "plant", "werk"],
        "purch_org_hpd_po":               ["purch_org_hpd_po", "purch_org", "ekorg"],
        "requester_name_src_po":          ["requester_name_src_po", "requestor_name", "requester_name"],
        "po_item_del_flag_src_po":        ["po_item_del_flag_src_po", "deletion_indicator", "del_flag"],
        "pr_no_src_po":                   ["pr_no_src_po", "pr_no", "purchase_req_no"],
        "pr_item_no_src_po":              ["pr_item_no_src_po", "pr_item", "pr_item_no", "purchase_req_item", "preq_item"],
        "matl_group_src_po":              ["matl_group_src_po", "material_group", "mat_group", "matl_group"],
        "net_price_doc_curr_src_po":      ["net_price_doc_curr_src_po", "net_unit_price", "unit_price", "price", "unit_rate"],
        "base_id_src_po":                 ["base_id_src_po", "base_id", "id_src"],
        "material_type_src_po":           ["material_type_src_po", "material_type", "mat_type", "mtart"],
        "on_release_total_value_hpd_po":  ["on_release_total_value_hpd_po", "on_release_total_value", "po_total_value", "total_po_value"],
        "doc_change_date_src_po":         ["doc_change_date_src_po", "doc_change_date_src"],}
    
    def _first_present(df, names):
        m = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in m:
                return m[n.lower()]
        return None
    
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure canonical columns exist and compute net unit price if missing."""
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        # synonyms -> canonical
        for canon, variants in COL_SYNONYMS.items():
            if canon not in df.columns:
                src = _first_present(df, variants)
                if src is not None:
                    df[canon] = df[src]
        # derive unit price if missing
        if "net_price_doc_curr_src_po" not in df.columns or df["net_price_doc_curr_src_po"].isna().all():
            nv_col = _first_present(df, COL_SYNONYMS["net_val_po_curr_src_po"])
            q_col  = _first_present(df, COL_SYNONYMS["quantity_src_po"])
            if nv_col and q_col:
                nv  = pd.to_numeric(df[nv_col], errors="coerce")
                qty = pd.to_numeric(df[q_col], errors="coerce").replace({0: np.nan})
                df["net_price_doc_curr_src_po"] = nv / qty
            else:
                df["net_price_doc_curr_src_po"] = np.nan
        df["net_price_doc_curr_src_po"]=pd.to_numeric(df["net_price_doc_curr_src_po"], errors="coerce")
        df["net_val_po_curr_src_po"]=pd.to_numeric(df["net_val_po_curr_src_po"], errors="coerce")
        df["gross_val_po_curr_src_po"]=pd.to_numeric(df["gross_val_po_curr_src_po"], errors="coerce")
        df["on_release_total_value_hpd_po"]=pd.to_numeric(df["on_release_total_value_hpd_po"], errors="coerce")
        df["exchange_rate_hpd_po"]=pd.to_numeric(df["exchange_rate_hpd_po"], errors="coerce")
        df['net_price_doc_curr_src_po_with_exchange_rate']=df["net_price_doc_curr_src_po"]*df["exchange_rate_hpd_po"]
        df['net_val_po_curr_src_po_with_exchange_rate']=df["net_val_po_curr_src_po"]*df["exchange_rate_hpd_po"]
        df['gross_val_po_curr_src_po_with_exchange_rate']=df["gross_val_po_curr_src_po"]*df["exchange_rate_hpd_po"]
        df['on_release_total_value_hpd_po_with_exchange_rate']=df["on_release_total_value_hpd_po"]*df["exchange_rate_hpd_po"]
        # ensure required columns exist
        required = [
            "purch_doc_no_src_po","purch_doc_item_no_src_po","purch_doc_type_hpd_po",
            "release_indicator_hpd_po","doc_change_date_hpd_po","purch_doc_date_hpd_po",
            "vendor_or_creditor_acct_no_hpd_po","vendor_name_1",
            "material_no_src_po","short_text_src_po","matl_group_src_po","material_type_src_po",
            "order_uom_src_po","quantity_src_po","net_price_doc_curr_src_po",
            "net_val_po_curr_src_po","gross_val_po_curr_src_po",
            "currency_hpd_po","company_code_hpd_po","po_item_del_flag_src_po",
            "plant_src_po","purch_org_hpd_po",
            "pr_no_src_po","pr_item_no_src_po","base_id_src_po","doc_change_date_src_po",
            "on_release_total_value_hpd_po",'net_price_doc_curr_src_po_with_exchange_rate','net_val_po_curr_src_po_with_exchange_rate',
            'gross_val_po_curr_src_po_with_exchange_rate','on_release_total_value_hpd_po_with_exchange_rate',
            # optional risk/impact fields (created empty if absent)
            "risk_score","risk_level","main_risk_scenario","sub_risk_1","sub_risk_2",
            "impact_1","impact_2","impact_3",
        ]
        for c in required:
            if c not in df.columns:
                df[c] = np.nan
        return df
    
    # =====================================================
    # LOW-LEVEL HELPERS
    # =====================================================
    def _clean(x):
        if x is None or pd.isna(x): return ""
        return str(x).strip()
    
    def _num(x):
        try:
            if x is None or (isinstance(x,str) and not x.strip()): return np.nan
            return float(x)
        except Exception:
            return np.nan
    
    def _to_date(s):
        if pd.isna(s): return pd.NaT
        return pd.to_datetime(s, errors="coerce")
    
    def _best_date(row):
        dt = _to_date(row.get("purch_doc_date_hpd_po"))
        if pd.notna(dt):
            return dt.normalize()
        return pd.NaT
    
    def _date_str(dt):
        return dt.date().isoformat() if pd.notna(dt) else "-"
    
    def _starts_with_4(s):
        s = _clean(s)
        return s.startswith("4")
    
    def _is_deleted(flag):
        f = _clean(flag).upper()
        return f in {"L","X"}
    
    def _is_subject_skip(flag):
        return _clean(flag).upper() == "L"
    
    def _uom_price(row):
        return _num(row.get("net_price_doc_curr_src_po_with_exchange_rate"))
    
    def _within_days(a, b, days):
        if pd.isna(a) or pd.isna(b): return False
        return abs((a - b).days) <= int(days)
    
    def _delta_pct(ref_price, cmp_price):
        """
        Your sign convention:
          positive  => subject unit price > compared unit price
          negative  => subject unit price < compared unit price
        """
        if pd.isna(ref_price) or ref_price == 0 or pd.isna(cmp_price):
            return np.nan
        return 100.0 * (ref_price - cmp_price) / abs(ref_price)
    
    def _risk_score_2dp(x):
        v = _num(x)
        return "-" if pd.isna(v) else f"{v:.2f}"
    
    def _po_total_value(df, subject_row):
        """Compute total PO value (sum of net values for that PO). Fallback to on_release_total_value_hpd_po if present."""
        po  = _clean(subject_row.get("purch_doc_no_src_po"))
        #cur = _clean(subject_row.get("currency_hpd_po")) or ""
        cur = "INR"
        if not po:
            tv = _num(subject_row.get("on_release_total_value_hpd_po_with_exchange_rate"))
            return cur, (None if pd.isna(tv) else tv)
        df_po = df[df["purch_doc_no_src_po"].astype(str).str.strip() == po]
        total_net = pd.to_numeric(df_po["net_val_po_curr_src_po_with_exchange_rate"], errors="coerce").sum()
        if total_net and not pd.isna(total_net) and total_net > 0:
            return cur, total_net
        tv = _num(subject_row.get("on_release_total_value_hpd_po_with_exchange_rate"))
        return cur, (None if pd.isna(tv) else tv)
    
    def _opt_join(*vals):
        got = [str(v).strip() for v in vals if v is not None and str(v).strip() and str(v).strip().lower() not in {"none","nan"}]
        return ", ".join(got)
    
    # =====================================================
    # SUBJECT SELECTOR
    # =====================================================
    def select_subject_row(df, po_no=None, po_item=None, base_id=None, idx=None):
        d = standardize(df)
        if idx is not None:
            return d.iloc[int(idx)]
        if base_id:
            m = d[d["base_id_src_po"].astype(str).str.strip() == _clean(base_id)]
            if len(m): return m.iloc[0]
        if po_no and po_item:
            m = d[(d["purch_doc_no_src_po"].astype(str).str.strip()==_clean(po_no)) &
                  (d["purch_doc_item_no_src_po"].astype(str).str.strip()==_clean(po_item))]
            if len(m): return m.iloc[0]
        raise ValueError("Subject not found. Provide (po_no & po_item) or base_id or idx.")
    
    # =====================================================
    # BASE FILTERS FOR RULES (ONLY release_indicator_hpd_po == 'R')
    # =====================================================
    def _released_only(df):
        return df[df["release_indicator_hpd_po"].astype(str).str.upper() == "R"].copy()
    
    def _base_filter_common(df):
        out = _released_only(df)
        out = out[out["material_no_src_po"].astype(str).str.strip() != ""]
        out = out[~out["po_item_del_flag_src_po"].apply(_is_deleted)]
        out = out[~out["plant_src_po"].apply(_starts_with_4)]
        return out
    
    def _filter_for_67(df):
        out = _base_filter_common(df)
        out = out[~out["purch_doc_type_hpd_po"].isin(DOC_TYPE_EXCLUDE_6768)]
        return out
    
    def _filter_for_68(df):
        out = _base_filter_common(df)
        out = out[~out["purch_doc_type_hpd_po"].isin(DOC_TYPE_EXCLUDE_6768)]
        return out
    
    def _filter_for_70(df):
        out = _released_only(df)
        out = out[~out["purch_doc_type_hpd_po"].isin(DOC_TYPE_EXCLUDE_70)]
        out = out[~out["po_item_del_flag_src_po"].apply(_is_deleted)]
        out = out[~out["plant_src_po"].apply(_starts_with_4)]
        gv = pd.to_numeric(out["gross_val_po_curr_src_po_with_exchange_rate"], errors="coerce")
        nv = pd.to_numeric(out["net_val_po_curr_src_po_with_exchange_rate"], errors="coerce")
        out = out[gv.fillna(nv).fillna(0) >= 0.0]
        return out
    
    def _norm_keys(df):
        b = df.copy()
        for c in [
            "vendor_or_creditor_acct_no_hpd_po","company_code_hpd_po","currency_hpd_po",
            "material_no_src_po","short_text_src_po","purch_doc_no_src_po","purch_doc_item_no_src_po",
            "pr_no_src_po","pr_item_no_src_po"
        ]:
            if c in b.columns:
                b[c] = b[c].astype(str).str.strip()
        return b
    
    
    
    def comps_67(subject, df) -> pd.DataFrame:
        base = _filter_for_67(df)
        b = _norm_keys(base)
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        mat    = _clean(subject.get("material_no_src_po"))
        cur    = _clean(subject.get("currency_hpd_po"))
        if not vendor or not mat:
            return pd.DataFrame()
        cond = (b["vendor_or_creditor_acct_no_hpd_po"] == vendor) & (b["material_no_src_po"] == mat)
        #if CURRENCY_MUST_MATCH and cur:
            #cond = cond & (b["currency_hpd_po"] == cur)
        cand = b[cond].copy()
        # drop same line
        cand = cand[~(
            (cand["purch_doc_no_src_po"] == _clean(subject["purch_doc_no_src_po"])) &
            (cand["purch_doc_item_no_src_po"] == _clean(subject["purch_doc_item_no_src_po"]))
        )]
        cand["__date"]  = cand.apply(_best_date, axis=1)
        cand["__price"] = pd.to_numeric(cand['net_price_doc_curr_src_po_with_exchange_rate'], errors="coerce")
        cand["__delta_pct"] = cand["__price"].apply(lambda up: _delta_pct(_uom_price(subject), up))
        cand = cand.dropna(subset=["__price"])
        cand = cand[cand["__delta_pct"].abs() != EPS_ZERO_DELTA_PCT]
        cand=cand.sort_values(by=["__delta_pct"],ascending=False)
        return cand
    
    def comps_68(subject, df) -> pd.DataFrame:
        base = _filter_for_68(df)
        b = _norm_keys(base)
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        mat    = _clean(subject.get("material_no_src_po"))
        cur    = _clean(subject.get("currency_hpd_po"))
        if not mat:
            return pd.DataFrame()
        cond = (b["material_no_src_po"] == mat)
        #if CURRENCY_MUST_MATCH and cur:
            #cond = cond & (b["currency_hpd_po"] == cur)
        if vendor:
            cond = cond & (b["vendor_or_creditor_acct_no_hpd_po"] != vendor)
        cand = b[cond].copy()
        cand = cand[~(
            (cand["purch_doc_no_src_po"] == _clean(subject["purch_doc_no_src_po"])) &
            (cand["purch_doc_item_no_src_po"] == _clean(subject["purch_doc_item_no_src_po"]))
        )]
        cand["__date"]  = cand.apply(_best_date, axis=1)
        cand["__price"] = pd.to_numeric(cand['net_price_doc_curr_src_po_with_exchange_rate'], errors="coerce")
        cand["__delta_pct"] = cand["__price"].apply(lambda up: _delta_pct(_uom_price(subject), up))
        cand = cand.dropna(subset=["__price"])
        cand = cand[cand["__delta_pct"].abs() != EPS_ZERO_DELTA_PCT]
        cand=cand.sort_values(by=["__delta_pct"],ascending=False)
        return cand
    
    def comps_70(subject, df) -> pd.DataFrame:
        base = _filter_for_70(df)
        b = _norm_keys(base)
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        cc     = _clean(subject.get("company_code_hpd_po"))
        cur    = _clean(subject.get("currency_hpd_po"))
        mat    = _clean(subject.get("material_no_src_po"))
        st     = _clean(subject.get("short_text_src_po")).lower()
        d0     = _best_date(subject)
        if not (vendor and cc and cur and pd.notna(d0)):
            return pd.DataFrame()
        cond = (
            (b["vendor_or_creditor_acct_no_hpd_po"] == vendor) &
            (b["company_code_hpd_po"] == cc)# &
            #(b["currency_hpd_po"] == cur)
        )
        if mat:
            cond = cond & (b["material_no_src_po"] == mat)
        else:
            cond = cond & (b["short_text_src_po"].str.lower() == st)
        cand = b[cond].copy()
        # different PO line
        cand = cand[~(
            (cand["purch_doc_no_src_po"] == _clean(subject["purch_doc_no_src_po"])) &
            (cand["purch_doc_item_no_src_po"] == _clean(subject["purch_doc_item_no_src_po"])))]
        # date window
        cand["__date"] = pd.to_datetime(cand["purch_doc_date_hpd_po"], errors="coerce")
        m = cand["__date"].isna()
        cand.loc[m, "__date"] = pd.to_datetime(cand.loc[m, "doc_change_date_hpd_po"], errors="coerce")
        m = cand["__date"].isna()
        cand.loc[m, "__date"] = pd.to_datetime(cand.loc[m, "doc_change_date_src_po"], errors="coerce")
        cand = cand.dropna(subset=["__date"])
        lo, hi = d0 - pd.Timedelta(days=SPLIT_WINDOW_DAYS), d0 + pd.Timedelta(days=SPLIT_WINDOW_DAYS)
        cand = cand[(cand["__date"] >= lo) & (cand["__date"] <= hi)]
        # price/delta
        cand["__price"] = pd.to_numeric(cand['net_price_doc_curr_src_po_with_exchange_rate'], errors="coerce")
        cand["__delta_pct"] = cand["__price"].apply(lambda up: _delta_pct(_uom_price(subject), up))
        cand = cand.dropna(subset=["__price"])
        cand=cand.sort_values(by=["__date"],ascending=True)
        return cand
    
    def comps_72(subject, df, strict_pr_item: bool = True) -> pd.DataFrame:
        d = _released_only(standardize(df))
        b = _norm_keys(d)
        pr   = _clean(subject.get("pr_no_src_po"))
        prit = _clean(subject.get("pr_item_no_src_po"))
        if not pr:
            return pd.DataFrame()
        if strict_pr_item:
            subset = b[(b["pr_no_src_po"] == pr) & (b["pr_item_no_src_po"] == prit)]
        else:
            subset = b[(b["pr_no_src_po"] == pr)]
        subset = subset[~(
            (subset["purch_doc_no_src_po"] == _clean(subject["purch_doc_no_src_po"])) &
            (subset["purch_doc_item_no_src_po"] == _clean(subject["purch_doc_item_no_src_po"])))]
        subset = subset.copy()
        subset["__date"]  = subset.apply(_best_date, axis=1)
        subset["__price"] = pd.to_numeric(subset['net_price_doc_curr_src_po_with_exchange_rate'], errors="coerce")
        subset["__delta_pct"] = subset["__price"].apply(lambda up: _delta_pct(_uom_price(subject), up))
        subset = subset.dropna(subset=["__price"])
        subset=subset.sort_values(by=["__date"],ascending=True)
        return subset
    
    # =====================================================
    # BUSINESS IMPACT HELPERS
    # =====================================================
    def _format_money(cur, val):
        if val is None or pd.isna(val): return "-"
        try:
            return f"{cur} {val:,.2f}".strip()
        except Exception:
            return f"{cur} {val}"
    
    def _header_share(row, df):
        po = _clean(row.get("purch_doc_no_src_po"))
        #cur = _clean(row.get("currency_hpd_po")) or ""
        cur='INR'
        line_val = _num(row.get("net_val_po_curr_src_po_with_exchange_rate"))
        if pd.isna(line_val):
            line_val = _num(row.get("gross_val_po_curr_src_po_with_exchange_rate"))
        if not po or pd.isna(line_val):
            return cur, line_val, None
        header_sum = pd.to_numeric(df[df["purch_doc_no_src_po"].astype(str) == po]["net_val_po_curr_src_po_with_exchange_rate"], errors="coerce").sum()
        if not header_sum or header_sum == 0:
            return cur, line_val, None
        return cur, line_val, round(100.0 * line_val / header_sum, 2)
    
    # =====================================================
    # RENDER: Word-like 5 sections as Markdown tables
    # =====================================================
    def _fmt_inr(val, cur):
        if pd.isna(val): return ""
        try:
            return f"{cur} {float(val):,.2f}".strip()
        except Exception:
            return f"{cur} {val}"
    
    def _fmt_pct(p):
        if pd.isna(p): return ""
        return f"{p:+.1f}%"
    
    def _qty_uom_price(row):
        """
        Returns a single string exactly like:
          "<qty> <uom> @ <CUR> <unit_price>"
        Example: "490 L @ INR 85.03"
        Falls back gracefully if some pieces are missing.
        """
        qty = pd.to_numeric(row.get("quantity_src_po"), errors="coerce")
        uom = _clean(row.get("order_uom_src_po"))
        #cur = (_clean(row.get("currency_hpd_po")) or "").upper()
        cur = "INR"
        up  = _uom_price(row)
    
        def _fmt_qty(q):
            if pd.isna(q):
                return ""
            s = f"{float(q):.3f}".rstrip("0").rstrip(".")
            return s
    
        qty_s = _fmt_qty(qty)
        uom_s = uom
        price_s = "" if pd.isna(up) else f"{cur} {float(up):,.2f}".strip()
    
        left = " ".join([p for p in (qty_s, uom_s) if p])
        right = price_s
    
        if left and right:
            return f"{left} @ {right}"
        elif left:
            return left
        elif right:
            return right
        else:
            return ""
    
    def _subject_key_table(s, df_full):
        #cur = _clean(s.get("currency_hpd_po")) or ""
        cur = "INR"
        po   = _clean(s.get("purch_doc_no_src_po"))
        it   = _clean(s.get("purch_doc_item_no_src_po"))
        prn  = _clean(s.get("pr_no_src_po"))
        pri  = _clean(s.get("pr_item_no_src_po"))
        rel  =_clean(s.get("release_indicator_hpd_po")) #change #_clean(s.get("on_release_total_value_hpd_po_with_exchange_rate"))
        ven  = _clean(s.get("vendor_or_creditor_acct_no_hpd_po"))
        venm = _clean(s.get("vendor_name_1"))
        mat  = _clean(s.get("material_no_src_po"))
        mtp  = _clean(s.get("material_type_src_po"))
        txt  = _clean(s.get("short_text_src_po"))
        plant= _clean(s.get("plant_src_po"))
        org  = _clean(s.get("purch_org_hpd_po"))
        req  = _clean(s.get("requester_name_src_po"))
        pdt  = _date_str(_best_date(s))
        delF = _clean(s.get("po_item_del_flag_src_po"))
    
        qty_price = _qty_uom_price(s)
        netv = pd.to_numeric(s.get("net_val_po_curr_src_po_with_exchange_rate"), errors="coerce")
        grossv = pd.to_numeric(s.get("gross_val_po_curr_src_po_with_exchange_rate"), errors="coerce")
        cur_tv, total_po_val = _po_total_value(df_full, s)
    
        rows = [
            ("PO / Item & PR Ref",      f"{po} / {it} & PR {prn} / {pri}".strip()),
            ("Release Status",          rel),
            ("Vendor no - Name",        f"{ven} – {venm}".strip(" –")),
            ("Material – Text, Type",   f"{mat} — {txt}, Type {mtp}".strip(" ,")),
            ("Plant / Org",             f"{plant} / {org}".strip(" /")),
            ("Requester",               req),
            ("Purchase Date",           pdt),
            ("Quantity / Unit & Unit Price", qty_price),  # ← exact format "490 L @ INR 85.03"
            ("Net / Gross Value",       _fmt_inr(netv if not pd.isna(netv) else grossv, cur)),
            ("Total PO Value",          "-" if total_po_val is None else _fmt_inr(total_po_val, cur_tv)),
            ("Deletion Indicator",      "(blank)" if not delF else delF),
        ]
        out = ["| Field | Value |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in rows]
        return "\n".join(out)
    
    def _risk_drivers_block(res, subject_row):
        drivers = []
        if len(res["rule_67"]["table"]) > 0:
            drivers.append("Price variance within same vendor.")
        if len(res["rule_68"]["table"]) > 0:
            drivers.append("Cross-vendor price variance.")
        if len(res["rule_70"]["table"]) > 0:
            drivers.append("Split POs (±60 days, same vendor/material).")
        if len(res["rule_72"]["table"]) > 0:
            drivers.append("Split POs PR-based (same PR line).")
        return drivers if drivers else ["No Risk"]
    
    def _variance_cols(comps_df, subject_row):
        cur_up = _uom_price(subject_row)
        qty = pd.to_numeric(subject_row.get("quantity_src_po"), errors="coerce")
        df = comps_df.copy()
        df["Δ vs Current"] = df["__delta_pct"].apply(_fmt_pct)                   # uses your sign convention
        df["Variance Value/Unit"] = cur_up - df["__price"]                       # subject - compared
        df["Variance Value"] = np.where(pd.notna(qty), df["Variance Value/Unit"] * qty, np.nan)
        return df
    
    # UPDATED: return None (hide) when there are no rows and HIDE_NO_RISK_SUBSECTIONS=True
    def _mk_rule_table(header_title, comps_df, subject_row, hide_no_risk=HIDE_NO_RISK_SUBSECTIONS):
        if comps_df is None or len(comps_df) == 0:
            if hide_no_risk:
                return None
            return f"**{header_title}**\n\nNo Risk\n"
        #cur = _clean(subject_row.get("currency_hpd_po")) or ""
        cur = "INR"
        df = _variance_cols(comps_df, subject_row)
        disp = pd.DataFrame({
            "PO → Item": df["purch_doc_no_src_po"].astype(str).str.strip() + "/" + df["purch_doc_item_no_src_po"].astype(str).str.strip(),
            "Vendor": df["vendor_or_creditor_acct_no_hpd_po"].astype(str).str.strip(),
            "Price": df["__price"].apply(lambda v: _fmt_inr(v, cur)).astype(str),
            "Qty": df.get("quantity_src_po", pd.Series(index=df.index)).astype(str).replace("nan",""),
            "Δ vs Current": df["Δ vs Current"].fillna(""),
            "Variance Value/Unit": df["Variance Value/Unit"].apply(lambda v: "" if pd.isna(v) else f"{v:,.2f}").astype(str),
            "Variance Value": df["Variance Value"].apply(lambda v: "" if pd.isna(v) else f"{v:,.2f}").astype(str),
            "Date": df["__date"].apply(_date_str),
            "Material": df["material_no_src_po"].astype(str).str.strip(),
            "Text": df["short_text_src_po"].astype(str).str.strip(),
            "PR": df["pr_no_src_po"].astype(str).str.strip() + "/" + df["pr_item_no_src_po"].astype(str).str.strip(),
            "Deletion Indicator": df["po_item_del_flag_src_po"].astype(str).str.strip().replace({"nan": ""}),
        })
        md = [f"**{header_title}**", ""]
        md.append("| " + " | ".join(disp.columns) + " |")
        md.append("|" + "|".join(["---"] * len(disp.columns)) + "|")
        for _, r in disp.iterrows():
            md.append("| " + " | ".join(r.values) + " |")
        md.append("")
        return "\n".join(md)
    
    # =====================================================
    # VERIFY & RENDER
    # =====================================================
    def verify_transaction_all(df_std: pd.DataFrame, po_no=None, po_item=None, base_id=None, idx=None,
                               strict_pr_item_72: bool = True):
        """
        Returns a dict with:
          - 'Key Description' : Series
          - 'skip_reason' : str | None (if subject has del flag L)
          - 'rule_67'/'68'/'70'/'72' : {'lines': [...], 'table': DataFrame (comps)}
          - 'df_full_std' : DataFrame (for totals)
        """
        subject = select_subject_row(df_std, po_no=po_no, po_item=po_item, base_id=base_id, idx=idx)
        skip_reason = "Subject has deletion indicator 'L'." if _is_subject_skip(subject.get("po_item_del_flag_src_po")) else None
        if skip_reason:
            empty = pd.DataFrame()
            return {
                "Key Description": subject,
                "skip_reason": skip_reason,
                "rule_67": {"lines": [], "table": empty},
                "rule_68": {"lines": [], "table": empty},
                "rule_70": {"lines": [], "table": empty},
                "rule_72": {"lines": [], "table": empty},
                "df_full_std": df_std,
            }
    
        c67 = comps_67(subject, df_std)
        c68 = comps_68(subject, df_std)
        c70 = comps_70(subject, df_std)
        c72 = comps_72(subject, df_std, strict_pr_item=strict_pr_item_72)
    
        out = {
            "Key Description": subject,
            "skip_reason": None,
            "rule_67": {"lines": [], "table": c67},
            "rule_68": {"lines": [], "table": c68},
            "rule_70": {"lines": [], "table": c70},
            "rule_72": {"lines": [], "table": c72},
            "df_full_std": df_std,
        }
        return out
    
    def render_all_docstyle(result_dict, df_full=None):
        """
        Returns a single markdown string with:
          1. Context & Trigger
          2. Key Description (two-col table)
          3. Business Impact (bullets)
          4. Risk Drivers (bullets / 'No Risk')
          5. Compared Transactions (A/B/C/D subtables or 'No Risk')
        """
        s = result_dict["Key Description"]
        if df_full is None:
            df_full = result_dict.get("df_full_std", pd.DataFrame([s]))
    
        # Section 1
        score = _risk_score_2dp(s.get("risk_score"))
        risk_level = _clean(s.get("risk_level")) or "No Risk"
        main_scenario = _clean(s.get("main_risk_scenario")) or "No Risk"
        sub1 = _clean(s.get("sub_risk_1")); sub2 = _clean(s.get("sub_risk_2"))
        subs = _opt_join(sub1, sub2)
        sec1 = [
            "**1. Context & Trigger**",
            f"Sara Risk Score: {score} → {risk_level}",
            f"Risk Scenario: {main_scenario}" + (f" → **Sub-risk**: {subs}" if subs else ""),
            ""
        ]
    
        # Section 2
        sec2 = ["**2. Key Description (PO Details)**", _subject_key_table(s, df_full), ""]
    
        # Section 3 (skip vs normal)
        if result_dict.get("skip_reason"):
            sec3 = ["**3. Business Impact**", "Deletion indicator = L. This transaction is excluded from analysis.", ""]
            sec4 = ["**4. Risk Drivers**", "No Risk", ""]
            sec5 = ["**5. Compared Transactions**", "No Risk", ""]
            return "\n".join(sec1 + sec2 + sec3 + sec4 + sec5)
    
        cur_s, line_val, _share = _header_share(s, df_full)
        cur_tv, total_po_val = _po_total_value(df_full, s)
        flagged = _fmt_inr(line_val, cur_s)
        total   = "-" if total_po_val is None else _fmt_inr(total_po_val, cur_tv)
        impacts = _opt_join(_clean(s.get("impact_1")), _clean(s.get("impact_2")), _clean(s.get("impact_3")))
        sec3 = ["**3. Business Impact**",
                f"Flagged Value: {flagged} out of PO total {total}.",
                f"Impact Areas: {impacts}." if impacts else "",
                ""]
    
        # Section 4
        drivers = _risk_drivers_block(result_dict, s)
        sec4 = ["**4. Risk Drivers**"] + [d for d in drivers if d] + [""]
    
        # Section 5 (A/B/C/D)
        c67 = result_dict["rule_67"]["table"]
        c68 = result_dict["rule_68"]["table"]
        c70 = result_dict["rule_70"]["table"]
        c72 = result_dict["rule_72"]["table"]
    
        blocks = [
            _mk_rule_table("A. Price Variance — Same Vendor & Material (vs current unit price)", c67, s),
            _mk_rule_table("B. Price Variance — Cross Vendor (Same Material)", c68, s),
            _mk_rule_table(f"C. Split PO Activity (±{SPLIT_WINDOW_DAYS} Days, Same Vendor & Material)", c70, s),
            _mk_rule_table("D. Split PO PR-based (Same PR Line)", c72, s),
        ]
        # Keep only those sub-sections that have evidence (non-None strings)
        blocks = [b for b in blocks if b]
    
        if not blocks:
            sec5 = ["**5. Compared Transactions**", "No Risk", ""]
        else:
            sec5 = ["**5. Compared Transactions**", *blocks]
    
        return "\n".join(sec1 + sec2 + sec3 + sec4 + sec5)
    
    # =====================================================
    # APPLY TO EACH ROW (keep inside DataFrame)
    # =====================================================
    def build_word_style_explanations(df_std: pd.DataFrame, dest_col: str = "llm_explanation"):
        out = []
        n = len(df_std)
        #for i, (_, row) in enumerate(df_std.iterrows(), 1):
        for _, row in df_std.iterrows():
            try:
                res = verify_transaction_all(
                    df_std,
                    po_no=str(row["purch_doc_no_src_po"]).strip(),
                    po_item=str(row["purch_doc_item_no_src_po"]).strip(),
                    strict_pr_item_72=True
                )
                out.append(render_all_docstyle(res, df_full=df_std))
                #if i % 200 == 0:
                #    print(f"Processed {i}/{n} rows...")
            except Exception as e:
                out.append(f"ERROR for {row.get('purch_doc_no_src_po')}/{row.get('purch_doc_item_no_src_po')}: {e}")
        df_std[dest_col] = out
        return df_std
    
    
    
    
    
    # =====================================================
    # MAIN (example) — integrate with your DF
    # =====================================================
    #if __name__ == "__main__":
        # Replace 'a.copy()' with your actual DataFrame variable
    df_full = final_result_df_2.copy()  # e.g., pd.read_pickle(...)
    
        # 1) Standardize once
    df_std = standardize(df_full)
    
        # 2) Build Word-style 5-section explanations (Markdown tables) per row
    df_std = build_word_style_explanations(df_std, dest_col="llm_explanation")
    
        # 3) (Optional) Save:
        # df_std.to_pickle("output_with_word_style_explanations.pkl")
        # df_std.to_csv("output_with_word_style_explanations.csv", index=False)
        # df_std.to_excel("output_with_word_style_explanations.xlsx", index=False)

    return df_std