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


def evidence_part_1(final_result_df_2, invoice_df=None):
    # =====================================================
    # IMPORTS (local to keep function self-contained)
    # =====================================================
    import time, logging
    from typing import Optional
    import numpy as np
    import pandas as pd

    # =====================================================
    # CONFIG
    # =====================================================
    SPLIT_WINDOW_DAYS    = 60
    MAX_LINES_PER_RULE   = 50
    CURRENCY_MUST_MATCH  = True
    EPS_ZERO_DELTA_PCT   = 0.0
    HIDE_NO_RISK_SUBSECTIONS = True

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
        "purch_doc_no_src_po":            ["purch_doc_no_src_po", "po_no", "purch_doc_no", "purch_doc_no_po"],
        "purch_doc_item_no_src_po":       ["purch_doc_item_no_src_po", "po_item", "item_no"],
        "purch_doc_type_hpd_po":          ["purch_doc_type_hpd_po", "purch_doc_type", "doc_type"],
        "release_indicator_hpd_po":       ["release_indicator_hpd_po", "release_indicator", "rel_ind", "rel_ind_hpd_po", "release_status_po"],
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
        "doc_change_date_src_po":         ["doc_change_date_src_po", "doc_change_date_src"],
        "exchange_rate_hpd_po":           ["exchange_rate_hpd_po", "exchange_rate"],
        "record_creation_dt":             ["record_creation_dt", "vendor_record_creation_dt"],
        "tax_no_3":                       ["tax_no_3", "tax_id_3"],
    }

    INV_SYNONYMS = {
        "purch_doc_mapping_invoice":  ["purch_doc_mapping_invoice", "base_id", "base_id_src_po", "po_item_key"],
        "amt_doc_curr_src_invoice":   ["amt_doc_curr_src_invoice", "invoice_amount", "inv_amount", "amt_doc_curr"],
        "quantity_src_invoice":       ["quantity_src_invoice", "invoice_quantity", "inv_quantity", "quantity"],
        "reversal_doc_no_hpd_invoice":["reversal_doc_no_hpd_invoice", "reversal_doc_no", "reversal_doc"],
        "base_id_src_invoice":        ["base_id_src_invoice", "invoice_no", "invoice_id",
                                       "accounting_doc_no_src_invoice", "belnr", "acc_doc_no"],
    }

    def _first_present(df, names):
        m = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in m:
                return m[n.lower()]
        return None

    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        for canon, variants in COL_SYNONYMS.items():
            if canon not in df.columns:
                src = _first_present(df, variants)
                if src is not None:
                    df[canon] = df[src]

        for c in ["net_price_doc_curr_src_po","net_val_po_curr_src_po",
                  "gross_val_po_curr_src_po","on_release_total_value_hpd_po",
                  "exchange_rate_hpd_po","quantity_src_po"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df['net_price_doc_curr_src_po_with_exchange_rate']       = df["net_price_doc_curr_src_po"] * df["exchange_rate_hpd_po"]
        df['net_val_po_curr_src_po_with_exchange_rate']          = df["net_val_po_curr_src_po"] * df["exchange_rate_hpd_po"]
        df['gross_val_po_curr_src_po_with_exchange_rate']        = df["gross_val_po_curr_src_po"] * df["exchange_rate_hpd_po"]
        df['on_release_total_value_hpd_po_with_exchange_rate']   = df["on_release_total_value_hpd_po"] * df["exchange_rate_hpd_po"]

        required = [
            "purch_doc_no_src_po","purch_doc_item_no_src_po","purch_doc_type_hpd_po",
            "release_indicator_hpd_po","doc_change_date_hpd_po","purch_doc_date_hpd_po",
            "vendor_or_creditor_acct_no_hpd_po","vendor_name_1",
            "material_no_src_po","short_text_src_po","matl_group_src_po","material_type_src_po",
            "order_uom_src_po","quantity_src_po","net_price_doc_curr_src_po",
            "net_val_po_curr_src_po","gross_val_po_curr_src_po",
            "currency_hpd_po","company_code_hpd_po","po_item_del_flag_src_po",
            "plant_src_po","purch_org_hpd_po","pr_no_src_po","pr_item_no_src_po",
            "base_id_src_po","doc_change_date_src_po",
            "on_release_total_value_hpd_po",
            "net_price_doc_curr_src_po_with_exchange_rate",
            "net_val_po_curr_src_po_with_exchange_rate",
            "gross_val_po_curr_src_po_with_exchange_rate",
            "on_release_total_value_hpd_po_with_exchange_rate",
            "risk_score","risk_level","main_risk_scenario","sub_risk_1","sub_risk_2","sub_risk_3",
            "impact_1","impact_2","impact_3","impact_4","impact_5",
            "exchange_rate_hpd_po","record_creation_dt","tax_no_3"
        ]
        for c in required:
            if c not in df.columns:
                df[c] = np.nan

        for c in ["purch_doc_date_hpd_po","doc_change_date_hpd_po","doc_change_date_src_po","record_creation_dt"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    def standardize_invoice(inv: pd.DataFrame) -> pd.DataFrame:
        inv = inv.copy()
        inv.columns = [c.strip() for c in inv.columns]
        for canon, variants in INV_SYNONYMS.items():
            if canon not in inv.columns:
                src = _first_present(inv, variants)
                if src is not None:
                    inv[canon] = inv[src]
        for c in ["amt_doc_curr_src_invoice","quantity_src_invoice"]:
            if c in inv.columns:
                inv[c] = pd.to_numeric(inv[c], errors="coerce")
        for c in ["purch_doc_mapping_invoice","amt_doc_curr_src_invoice","quantity_src_invoice",
                  "reversal_doc_no_hpd_invoice","base_id_src_invoice"]:
            if c not in inv.columns:
                inv[c] = np.nan
        return inv

    # =====================================================
    # HELPERS
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

    #def _best_date(row):
        #dt = _to_date(row.get("purch_doc_date_hpd_po"))
        #if pd.notna(dt):
            #return dt.normalize()
        #return pd.NaT

    

    #def _date_str(dt):
        #return dt.date().isoformat() if pd.notna(dt) else "-"
    def _best_date(row):
        # safely extract and convert to datetime
        val = row.get("purch_doc_date_hpd_po", None)
        if pd.isna(val) or val in ["", None]:
            return pd.NaT
    
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        return dt.normalize()  # ensure it's a proper pandas.Timestamp


    def _date_str(dt):
        # handles both pd.Timestamp and string gracefully
        if pd.isna(dt) or dt is None or dt == "":
            return "-"
        try:
            # if string, convert to datetime
            if isinstance(dt, str):
                dt = pd.to_datetime(dt, errors="coerce")
            if pd.isna(dt):
                return "-"
            return dt.strftime("%d-%m-%Y")
        except Exception as e:
            print("⚠️ Error while formatting date:", e)
            return "-"

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

    def _delta_pct(ref_price, cmp_price):
        if pd.isna(ref_price) or ref_price == 0 or pd.isna(cmp_price):
            return np.nan
        return 100.0 * (ref_price - cmp_price) / abs(ref_price)

    def _po_total_value(df, subject_row):
        po  = _clean(subject_row.get("purch_doc_no_src_po"))
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
    # FILTERS + COMPARATORS (existing 67/68/70/72)
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
            "pr_no_src_po","pr_item_no_src_po","base_id_src_po"
        ]:
            if c in b.columns:
                b[c] = b[c].astype(str).str.strip()
        return b

    def comps_67(subject, df) -> pd.DataFrame:
        base = _filter_for_67(df); b = _norm_keys(base)
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        mat    = _clean(subject.get("material_no_src_po"))
        if not vendor or not mat: return pd.DataFrame()
        cand = b[(b["vendor_or_creditor_acct_no_hpd_po"] == vendor) & (b["material_no_src_po"] == mat)].copy()
        cand = cand[~((cand["purch_doc_no_src_po"] == _clean(subject["purch_doc_no_src_po"])) &
                      (cand["purch_doc_item_no_src_po"] == _clean(subject["purch_doc_item_no_src_po"])))]
        cand["__date"]  = cand.apply(_best_date, axis=1)
        cand["__price"] = pd.to_numeric(cand['net_price_doc_curr_src_po_with_exchange_rate'], errors="coerce")
        cand["__delta_pct"] = cand["__price"].apply(lambda up: _delta_pct(_uom_price(subject), up))
        cand = cand.dropna(subset=["__price"])
        cand = cand[cand["__delta_pct"] >EPS_ZERO_DELTA_PCT]
        cand = cand.sort_values(by=["__delta_pct"], ascending=False)
        return cand

    def comps_68(subject, df) -> pd.DataFrame:
        base = _filter_for_68(df); b = _norm_keys(base)
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        mat    = _clean(subject.get("material_no_src_po"))
        if not mat: return pd.DataFrame()
        cond = (b["material_no_src_po"] == mat)
        if vendor: cond = cond & (b["vendor_or_creditor_acct_no_hpd_po"] != vendor)
        cand = b[cond].copy()
        cand = cand[~((cand["purch_doc_no_src_po"] == _clean(subject["purch_doc_no_src_po"])) &
                      (cand["purch_doc_item_no_src_po"] == _clean(subject["purch_doc_item_no_src_po"])))]
        cand["__date"]  = cand.apply(_best_date, axis=1)
        cand["__price"] = pd.to_numeric(cand['net_price_doc_curr_src_po_with_exchange_rate'], errors="coerce")
        cand["__delta_pct"] = cand["__price"].apply(lambda up: _delta_pct(_uom_price(subject), up))
        cand = cand.dropna(subset=["__price"])
        cand = cand[cand["__delta_pct"] >EPS_ZERO_DELTA_PCT]
        cand = cand.sort_values(by=["__delta_pct"], ascending=False)
        return cand

    def comps_70(subject, df) -> pd.DataFrame:
        base = _filter_for_70(df); b = _norm_keys(base)
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        cc     = _clean(subject.get("company_code_hpd_po"))
        mat    = _clean(subject.get("material_no_src_po"))
        st     = _clean(subject.get("short_text_src_po")).lower()
        d0     = _best_date(subject)
        if not (vendor and cc and pd.notna(d0)): return pd.DataFrame()
        cond = ((b["vendor_or_creditor_acct_no_hpd_po"] == vendor) & (b["company_code_hpd_po"] == cc))
        if mat:
            cond = cond & (b["material_no_src_po"] == mat)
        else:
            cond = cond & (b["short_text_src_po"].str.lower() == st)
        cand = b[cond].copy()
        cand = cand[~((cand["purch_doc_no_src_po"] == _clean(subject["purch_doc_no_src_po"])) &
                      (cand["purch_doc_item_no_src_po"] == _clean(subject["purch_doc_item_no_src_po"])))]
        cand["__date"] = pd.to_datetime(cand["purch_doc_date_hpd_po"], errors="coerce")
        m = cand["__date"].isna()
        cand.loc[m, "__date"] = pd.to_datetime(cand.loc[m, "doc_change_date_hpd_po"], errors="coerce")
        m = cand["__date"].isna()
        cand.loc[m, "__date"] = pd.to_datetime(cand.loc[m, "doc_change_date_src_po"], errors="coerce")
        cand = cand.dropna(subset=["__date"])
        lo, hi = d0 - pd.Timedelta(days=SPLIT_WINDOW_DAYS), d0 + pd.Timedelta(days=SPLIT_WINDOW_DAYS)
        cand = cand[(cand["__date"] >= lo) & (cand["__date"] <= hi)]
        cand["__price"] = pd.to_numeric(cand['net_price_doc_curr_src_po_with_exchange_rate'], errors="coerce")
        cand["__delta_pct"] = cand["__price"].apply(lambda up: _delta_pct(_uom_price(subject), up))
        cand = cand.dropna(subset=["__price"])
        cand = cand.sort_values(by=["__date"], ascending=True)
        return cand

    def comps_72(subject, df, strict_pr_item: bool = True) -> pd.DataFrame:
        d = _released_only(standardize(df)); b = _norm_keys(d)
        pr   = _clean(subject.get("pr_no_src_po"))
        prit = _clean(subject.get("pr_item_no_src_po"))
        if not pr: return pd.DataFrame()
        subset = b[(b["pr_no_src_po"] == pr) & (b["pr_item_no_src_po"] == prit)] if strict_pr_item else b[(b["pr_no_src_po"] == pr)]
        subset = subset[~((subset["purch_doc_no_src_po"] == _clean(subject["purch_doc_no_src_po"])) &
                          (subset["purch_doc_item_no_src_po"] == _clean(subject["purch_doc_item_no_src_po"])))]
        subset = subset.copy()
        subset["__date"]  = subset.apply(_best_date, axis=1)
        subset["__price"] = pd.to_numeric(subset['net_price_doc_curr_src_po_with_exchange_rate'], errors="coerce")
        subset["__delta_pct"] = subset["__price"].apply(lambda up: _delta_pct(_uom_price(subject), up))
        subset = subset.dropna(subset=["__price"])
        subset = subset.sort_values(by=["__date"], ascending=True)
        return subset

    # =====================================================
    # INVOICE>PO PRECOMPUTE (Amount & Quantity split)
    # =====================================================
    inv_std = None
    PRE_MAP = {}

    def _precompute_invoice_vs_po_flags(df_std: pd.DataFrame, inv_std_: pd.DataFrame) -> pd.DataFrame:
        """
        Gate:
          - PO rows: release_indicator_hpd_po == 'R' AND po_item_del_flag_src_po is NaN
          - INV rows: reversal_doc_no_hpd_invoice is NaN
        Aggregation:
          - group by base_id_src_po / purch_doc_mapping_invoice -> 'base_id'
          - inv totals: sum(amt_doc_curr_src_invoice), sum(quantity_src_invoice)
          - po refs: max(net_val_po_curr_src_po), max(quantity_src_po)
        Tolerances:
          - Amount: round(2); inv_total_amt > po_val_ref + 0.01
          - Quantity: round(6); inv_total_qty > po_qty_ref + 1e-9
        """
        if inv_std_ is None or df_std is None or df_std.empty:
            return pd.DataFrame(columns=[
                "base_id","invoice_amt_gt_po_flag","invoice_qty_gt_po_flag",
                "inv_total_amt","inv_total_qty","po_val_ref","po_qty_ref","currency",
                "amt_excess","qty_excess"
            ])

        # ---- PO gate (R & not deleted) ----
        d = df_std.copy()
        d["release_indicator_hpd_po"] = d["release_indicator_hpd_po"].astype(str).str.upper()
        po_gate = d[(d["release_indicator_hpd_po"] == "R") & (d["po_item_del_flag_src_po"].isna())].copy()
        po_gate["net_val_po_curr_src_po"] = pd.to_numeric(po_gate["net_val_po_curr_src_po"], errors="coerce")
        po_gate["quantity_src_po"]        = pd.to_numeric(po_gate["quantity_src_po"], errors="coerce")

        po_refs = (po_gate.groupby("base_id_src_po", dropna=False)
                   .agg(po_val_ref=("net_val_po_curr_src_po","max"),
                        po_qty_ref=("quantity_src_po","max"),
                        currency=("currency_hpd_po", lambda s: next((str(x) for x in s if pd.notna(x)), "INR")))
                   .reset_index().rename(columns={"base_id_src_po":"base_id"}))

        # ---- INV gate (not reversed) ----
        inv_gate = inv_std_[inv_std_["reversal_doc_no_hpd_invoice"].isna()].copy()
        inv_gate["amt_doc_curr_src_invoice"] = pd.to_numeric(inv_gate["amt_doc_curr_src_invoice"], errors="coerce")
        inv_gate["quantity_src_invoice"]     = pd.to_numeric(inv_gate["quantity_src_invoice"], errors="coerce")

        inv_grp = (inv_gate.groupby("purch_doc_mapping_invoice", dropna=False)
                   .agg(inv_total_amt=("amt_doc_curr_src_invoice","sum"),
                        inv_total_qty=("quantity_src_invoice","sum"))
                   .reset_index().rename(columns={"purch_doc_mapping_invoice":"base_id"}))

        # ---- Merge & compare with tolerances ----
        m = po_refs.merge(inv_grp, on="base_id", how="left")

        m["inv_total_amt"] = m["inv_total_amt"].round(2)
        m["po_val_ref"]    = m["po_val_ref"].round(2)
        m["inv_total_qty"] = m["inv_total_qty"].round(6)
        m["po_qty_ref"]    = m["po_qty_ref"].round(6)

        AMT_ATOL = 0.01
        QTY_ATOL = 1e-9

        m["invoice_amt_gt_po_flag"] = (
            m["inv_total_amt"].notna() & m["po_val_ref"].notna() &
            (m["inv_total_amt"] > (m["po_val_ref"] + AMT_ATOL))
        ).astype(int)

        m["invoice_qty_gt_po_flag"] = (
            m["inv_total_qty"].notna() & m["po_qty_ref"].notna() &
            (m["inv_total_qty"] > (m["po_qty_ref"] + QTY_ATOL))
        ).astype(int)

        # explicit differences
        m["amt_excess"] = (m["inv_total_amt"] - m["po_val_ref"]).round(2)
        m["qty_excess"] = (m["inv_total_qty"] - m["po_qty_ref"]).round(6)

        return m

    # =====================================================
    # RENDER HELPERS
    # =====================================================
    def _fmt_inr(val, cur):
        if val is None or pd.isna(val): return ""
        try: return f"{cur} {float(val):,.2f}".strip()
        except Exception: return f"{cur} {val}"

    def _fmt_pct(p):
        if pd.isna(p): return ""
        return f"{p:+.1f}%"

    def _qty_uom_price(row):
        qty = pd.to_numeric(row.get("quantity_src_po"), errors="coerce")
        uom = _clean(row.get("order_uom_src_po"))
        cur = "INR"
        up  = _uom_price(row)
        def _fmt_qty(q):
            if pd.isna(q): return ""
            s = f"{float(q):.3f}".rstrip("0").rstrip(".")
            return s
        left  = " ".join([p for p in (_fmt_qty(qty), uom) if p])
        right = "" if pd.isna(up) else f"{cur} {float(up):,.2f}".strip()
        if left and right: return f"{left} @ {right}"
        if left:           return left
        if right:          return right
        return ""

    def _mk_markdown_table_from_df(df: pd.DataFrame, title: str):
        if df is None or len(df) == 0:
            if HIDE_NO_RISK_SUBSECTIONS: return None
            return f"**{title}**\n\nNo Risk\n"
        md = [f"**{title}**", ""]
        md.append("| " + " | ".join(df.columns) + " |")
        md.append("|" + "|".join(["---"] * len(df.columns)) + "|")
        for _, r in df.iterrows():
            md.append("| " + " | ".join("" if pd.isna(x) else str(x) for x in r.values) + " |")
        md.append("")
        return "\n".join(md)

    def _subject_key_table(s, df_full):
        cur = "INR"
        po   = _clean(s.get("purch_doc_no_src_po"))
        it   = _clean(s.get("purch_doc_item_no_src_po"))
        prn  = _clean(s.get("pr_no_src_po"))
        pri  = _clean(s.get("pr_item_no_src_po"))
        rel  = _clean(s.get("release_indicator_hpd_po"))
        ven  = _clean(s.get("vendor_or_creditor_acct_no_hpd_po"))
        venm = _clean(s.get("vendor_name_1"))
        mat  = _clean(s.get("material_no_src_po"))
        mtp  = _clean(s.get("material_type_src_po"))
        txt  = _clean(s.get("short_text_src_po"))
        plant= _clean(s.get("plant_src_po"))
        org  = _clean(s.get("purch_org_hpd_po"))
        req  = _clean(s.get("requester_name_src_po"))
        pdt = _date_str(_best_date(s))
        delF = _clean(s.get("po_item_del_flag_src_po"))

        qty_price = _qty_uom_price(s)
        netv   = pd.to_numeric(s.get("net_val_po_curr_src_po_with_exchange_rate"), errors="coerce")
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
            ("Quantity / Unit & Unit Price", qty_price),
            ("Net / Gross Value",       _fmt_inr(netv if not pd.isna(netv) else grossv, cur)),
            ("Total PO Value",          "-" if total_po_val is None else _fmt_inr(total_po_val, cur_tv)),
            ("Deletion Indicator",      "(blank)" if not delF else delF),
        ]
        out = ["| Field | Value |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in rows]
        return "\n".join(out)

    def _variance_cols(comps_df, subject_row):
        cur_up = _uom_price(subject_row)
        qty = pd.to_numeric(subject_row.get("quantity_src_po"), errors="coerce")
        df = comps_df.copy()
        df["Δ vs Current"] = df["__delta_pct"].apply(_fmt_pct)
        df["Variance Value/Unit"] = cur_up - df["__price"]
        df["Variance Value"] = np.where(pd.notna(qty), df["Variance Value/Unit"] * qty, np.nan)
        return df

    def _mk_rule_table(header_title, comps_df, subject_row):
        if comps_df is None or len(comps_df) == 0:
            if HIDE_NO_RISK_SUBSECTIONS: return None
            return f"**{header_title}**\n\nNo Risk\n"
        cur = "INR"
        df = _variance_cols(comps_df, subject_row)
        disp = pd.DataFrame({
            "PO → Item": df["purch_doc_no_src_po"].astype(str).str.strip() + "/" + df["purch_doc_item_no_src_po"].astype(str).str.strip(),
            "Vendor": df["vendor_or_creditor_acct_no_hpd_po"].astype(str).str.strip(),
            "Price/Unit": df["__price"].apply(lambda v: _fmt_inr(v, cur)).astype(str),
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

    # ---------- E: Invoice > PO (split into amount vs quantity) ----------
    def ev_invoice_more_than_po(subject, df_full) -> str | None:
        """
        E: Invoice > PO sub-rules:
           - E.Amt: Invoice Amount > PO Value (tol ₹0.01)
           - E.Qty: Invoice Qty   > PO Qty   (tol 1e-9)
        Gate: PO released & not deleted; invoices not reversed.
        Shows: contributing invoice IDs per subtype and differences.
        """
        base_id = _clean(subject.get("base_id_src_po"))
        if not base_id or base_id not in PRE_MAP:
            return None

        row = PRE_MAP[base_id]
        cur = row.get("currency") or "INR"
        inv_total_amt = row.get("inv_total_amt")
        inv_total_qty = row.get("inv_total_qty")
        po_val_ref    = row.get("po_val_ref")
        po_qty_ref    = row.get("po_qty_ref")
        flag_amt      = int(row.get("invoice_amt_gt_po_flag", 0))
        flag_qty      = int(row.get("invoice_qty_gt_po_flag", 0))
        flag_any      = int(bool(flag_amt or flag_qty))
        amt_excess    = row.get("amt_excess")
        qty_excess    = row.get("qty_excess")

        # NEW: if neither amount nor quantity breaches, hide Rule E entirely
        if flag_any == 0:
            return None

        # Contributing invoices (gated: not reversed) – raw first (for ID lists)
        cond = (inv_std["purch_doc_mapping_invoice"].astype(str).str.strip() == base_id) & (inv_std["reversal_doc_no_hpd_invoice"].isna())
        lines_raw = (inv_std[cond]
                     .groupby("base_id_src_invoice", dropna=True)
                     .agg(Amount=("amt_doc_curr_src_invoice","sum"),
                          Quantity=("quantity_src_invoice","sum"))
                     .reset_index())

        # ID lists per metric (mention IDs when that subtype breaches)
        inv_ids_amt = lines_raw.loc[lines_raw["Amount"]   > 0, "base_id_src_invoice"].astype(str).tolist()
        inv_ids_qty = lines_raw.loc[lines_raw["Quantity"] > 0, "base_id_src_invoice"].astype(str).tolist()

        # Pretty display table
        lines = lines_raw.copy()
        inv_lines_disp = None
        if len(lines):
            lines["Amount"]   = lines["Amount"].apply(lambda v: _fmt_inr(v, cur) if pd.notna(v) else "")
            lines["Quantity"] = lines["Quantity"].apply(lambda v: "" if pd.isna(v) else f"{float(v):,.6f}".rstrip("0").rstrip("."))
            lines.rename(columns={"base_id_src_invoice":"Invoice ID"}, inplace=True)
            inv_lines_disp = _mk_markdown_table_from_df(lines[["Invoice ID","Amount","Quantity"]],
                                                        "E.1 Contributing Invoices (Gated)")

        # Group totals (with rounded differences already computed)
        disp = pd.DataFrame([
            ["Amount (Invoice Total)", _fmt_inr(inv_total_amt, cur),
             _fmt_inr(po_val_ref, cur),
             ("" if (pd.isna(amt_excess)) else _fmt_inr(amt_excess, cur)),
             ("YES" if flag_amt else "NO")],
            ["Quantity (Invoice Total)",
             ("" if pd.isna(inv_total_qty) else f"{inv_total_qty:,.6f}".rstrip("0").rstrip(".")),
             ("" if pd.isna(po_qty_ref) else f"{po_qty_ref:,.6f}".rstrip("0").rstrip(".")),
             ("" if pd.isna(qty_excess) else f"{qty_excess:,.6f}".rstrip("0").rstrip(".")),
             ("YES" if flag_qty else "NO")],
        ], columns=["Metric","Invoice","PO Ref","Excess","Breach?"])

        group_tbl = _mk_markdown_table_from_df(
            disp, f"E.2 Invoice vs PO — Group Totals (E.Amt={flag_amt}, E.Qty={flag_qty}, E.Any={flag_any})"
        )

        # Notes listing invoice IDs per subtype that breached
        notes = []
        if flag_amt:
            notes.append("**E.Amt – Invoices (IDs):** " + (", ".join(inv_ids_amt) if inv_ids_amt else "(none)"))
        if flag_qty:
            notes.append("**E.Qty – Invoices (IDs):** " + (", ".join(inv_ids_qty) if inv_ids_qty else "(none)"))
        #notes_block = "\n\n".join(notes) if notes else None

        return "\n\n".join([b for b in [inv_lines_disp, group_tbl, ] if b])#notes_block

    # ---------- F–I ----------
    def ev_blocked_vendor(subject) -> str | None:
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        vendor_name = _clean(subject.get("vendor_name_1"))
        flags = {
            "Central Deletion Flag": _clean(subject.get("central_deletion_flag")),
            "Central Purchase Block Flag": _clean(subject.get("central_purch_blk_flag")),
            "Central Posting Block Flag": _clean(subject.get("central_posting_blk_flag")),
        }
        any_x = any(v.upper()=="X" for v in flags.values())
        if not any_x:
            return None
        disp = pd.DataFrame(
            [{"Vendor": vendor, "Vendor Name": vendor_name, "Flag": k, "Value": ("X" if v.upper()=="X" else (v or ""))}
             for k, v in flags.items()]
        )
        return _mk_markdown_table_from_df(disp, "F. PO to Blocked Vendor — Block Indicators")

    def ev_new_vendor_gt_tolerance(subject) -> str | None:
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        vendor_name = _clean(subject.get("vendor_name_1"))
        po_dt  = pd.to_datetime(subject.get("purch_doc_date_hpd_po"), errors="coerce")
        rec_dt = pd.to_datetime(subject.get("record_creation_dt"), errors="coerce")
        po_val_inr = _num(subject.get("on_release_total_value_hpd_po_with_exchange_rate"))
        within_30  = (pd.notna(po_dt) and pd.notna(rec_dt) and abs((po_dt - rec_dt).days) <= 30)
        ge_1cr     = (pd.notna(po_val_inr) and po_val_inr >= 1e7)
        if not (within_30 and ge_1cr):
            return None
        disp = pd.DataFrame([{
            "Vendor": vendor,
            "Vendor Name": vendor_name,
            "Vendor Creation Date": _date_str(rec_dt),
            "PO Date": _date_str(po_dt),
            "Δ Days": (abs((po_dt - rec_dt).days) if (pd.notna(po_dt) and pd.notna(rec_dt)) else ""),
            "PO Value": _fmt_inr(po_val_inr, "INR"),
            "Checks": "≤ 30 days & ≥ 1 Cr"
        }])
        return _mk_markdown_table_from_df(disp, "G. PO to New Vendor > Tolerance (≤30 days & ≥ ₹1 Cr)")

    def _extract_pan_from_taxno3(val):
        s = _clean(val).upper()
        if not s: return None
        cand = s[2:-3] if len(s) > 5 else ""
        if len(cand) != 10 or not cand.isalnum():
            return None
        return cand

    def ev_pan4P_ge_1cr(subject) -> str | None:
        vendor = _clean(subject.get("vendor_or_creditor_acct_no_hpd_po"))
        vendor_name = _clean(subject.get("vendor_name_1"))
        pan = subject.get("pan_extracted")
        if pd.isna(pan) or not _clean(pan):
            pan = _extract_pan_from_taxno3(subject.get("tax_no_3"))
        po_val_inr = _num(subject.get("on_release_total_value_hpd_po_with_exchange_rate"))
        cond = (pan is not None) and (len(pan)==10) and (pan[3]=="P") and (pd.notna(po_val_inr) and po_val_inr>=1e7)
        if not cond:
            return None
        disp = pd.DataFrame([{
            "Vendor": vendor,
            "Vendor Name": vendor_name,
            "Original Tax No. 3": _clean(subject.get("tax_no_3")),
            "Extracted PAN": pan,
            "4th Char": pan[3],
            "PO Value": _fmt_inr(po_val_inr, "INR"),
            "Checks": "4th = 'P' & ≥ 1 Cr"
        }])
        return _mk_markdown_table_from_df(disp, "H. PAN 4th Char 'P' & Amount ≥ ₹1 Cr")

    def ev_missing_tax_id(subject) -> str | None:
        if pd.isna(subject.get("tax_no_3")) or _clean(subject.get("tax_no_3"))=="":
            disp = pd.DataFrame([{
                "Vendor": _clean(subject.get("vendor_or_creditor_acct_no_hpd_po")),
                "Vendor Name": _clean(subject.get("vendor_name_1")),
                "Tax No. 3": "(missing)"
            }])
            return _mk_markdown_table_from_df(disp, "I. Missing Tax ID (KYC)")
        return None

    # =====================================================
    # VERIFY & RENDER
    # =====================================================
    def verify_transaction_all(df_std: pd.DataFrame, po_no=None, po_item=None, base_id=None, idx=None,
                               strict_pr_item_72: bool = True):
        subject = select_subject_row(df_std, po_no=po_no, po_item=po_item, base_id=base_id, idx=idx)

        skip_reason = "Subject has deletion indicator 'L'." if _is_subject_skip(subject.get("po_item_del_flag_src_po")) else None
        if skip_reason:
            empty = pd.DataFrame()
            return {
                "Key Description": subject,
                "skip_reason": skip_reason,
                "rule_67": {"table": empty},
                "rule_68": {"table": empty},
                "rule_70": {"table": empty},
                "rule_72": {"table": empty},
                "rule_E": None, "rule_F": None, "rule_G": None, "rule_H": None, "rule_I": None,
                "df_full_std": df_std,
            }

        c67 = comps_67(subject, df_std)
        c68 = comps_68(subject, df_std)
        c70 = comps_70(subject, df_std)
        c72 = comps_72(subject, df_std, strict_pr_item=strict_pr_item_72)

        eE = ev_invoice_more_than_po(subject, df_std)
        eF = ev_blocked_vendor(subject)
        eG = ev_new_vendor_gt_tolerance(subject)
        eH = ev_pan4P_ge_1cr(subject)
        eI = ev_missing_tax_id(subject)

        return {
            "Key Description": subject,
            "skip_reason": None,
            "rule_67": {"table": c67},
            "rule_68": {"table": c68},
            "rule_70": {"table": c70},
            "rule_72": {"table": c72},
            "rule_E": eE, "rule_F": eF, "rule_G": eG, "rule_H": eH, "rule_I": eI,
            "df_full_std": df_std,
        }

    def _header_share(row, df):
        cur='INR'
        line_val = _num(row.get("net_val_po_curr_src_po_with_exchange_rate"))
        if pd.isna(line_val):
            line_val = _num(row.get("gross_val_po_curr_src_po_with_exchange_rate"))
        po = _clean(row.get("purch_doc_no_src_po"))
        if not po or pd.isna(line_val):
            return cur, line_val, None
        header_sum = pd.to_numeric(df[df["purch_doc_no_src_po"].astype(str) == po]["net_val_po_curr_src_po_with_exchange_rate"], errors="coerce").sum()
        if not header_sum or header_sum == 0:
            return cur, line_val, None
        return cur, line_val, round(100.0 * line_val / header_sum, 2)

    def _risk_drivers_block(res, subject_row):
        drivers = []
        if len(res["rule_67"]["table"]) > 0: drivers.append("Price variance within same vendor.")
        if len(res["rule_68"]["table"]) > 0: drivers.append("Cross-vendor price variance.")
        if len(res["rule_70"]["table"]) > 0: drivers.append("Split POs (±60 days, same vendor/material).")
        if len(res["rule_72"]["table"]) > 0: drivers.append("Split POs PR-based (same PR line).")
        if res.get("rule_E"): drivers.append("Invoice amount or quantity exceeds PO.")
        if res.get("rule_F"): drivers.append("PO placed to blocked vendor.")
        if res.get("rule_G"): drivers.append("New vendor + high-value PO (≤30 days & ≥ ₹1 Cr).")
        if res.get("rule_H"): drivers.append("PAN 4th char = 'P' with amount ≥ ₹1 Cr.")
        if res.get("rule_I"): drivers.append("Vendor missing KYC (Tax ID).")
        return drivers if drivers else ["No Risk"]

    def render_all_docstyle(result_dict, df_full=None):
        s = result_dict["Key Description"]
        if df_full is None:
            df_full = result_dict.get("df_full_std", pd.DataFrame([s]))

        # 1. Context & Trigger
        score = _num(s.get("risk_score"))
        score_s = "-" if pd.isna(score) else f"{score:.2f}"
        risk_level = _clean(s.get("risk_level")) or "No Risk"
        main_scenario = _clean(s.get("main_risk_scenario")) or "No Risk"
        sub1 = _clean(s.get("sub_risk_1")); sub2 = _clean(s.get("sub_risk_2")); sub3 = _clean(s.get("sub_risk_3"))
        subs = _opt_join(sub1, sub2, sub3)
        sec1 = [
            "**1. Context & Trigger**",
            f"Sara Risk Score: {score_s} → {risk_level}",
            f"Risk Scenario: {main_scenario}" + (f" → **Sub-risk**: {subs}" if subs else ""),
            ""
        ]

        # 2. Key Description
        sec2 = ["**2. Key Description (PO Details)**", _subject_key_table(s, df_full), ""]

        # 3. Business Impact
        cur_s, line_val, _share = _header_share(s, df_full)
        cur_tv, total_po_val = _po_total_value(df_full, s)
        flagged = _fmt_inr(line_val, cur_s)
        total   = "-" if total_po_val is None else _fmt_inr(total_po_val, cur_tv)
        impacts = _opt_join(_clean(s.get("impact_1")), _clean(s.get("impact_2")), _clean(s.get("impact_3")), _clean(s.get("impact_4")), _clean(s.get("impact_5")))
        sec3 = ["**3. Business Impact**",
                f"Flagged Value: {flagged} out of PO total {total}.",
                f"Impact Areas: {impacts}." if impacts else "",
                ""]

        # 4. Risk Drivers
        drivers = _risk_drivers_block(result_dict, s)
        sec4 = ["**4. Risk Drivers**"] + [d for d in drivers if d] + [""]

        # 5. Evidence Blocks (A–D existing, E split, F–I existing)
        c67 = result_dict["rule_67"]["table"]
        c68 = result_dict["rule_68"]["table"]
        c70 = result_dict["rule_70"]["table"]
        c72 = result_dict["rule_72"]["table"]
        blocks = [
            _mk_rule_table("A. Price Variance — Same Vendor & Material (vs current unit price)", c67, s),
            _mk_rule_table("B. Price Variance — Cross Vendor (Same Material)", c68, s),
            _mk_rule_table(f"C. Split PO Activity (±{SPLIT_WINDOW_DAYS} Days, Same Vendor & Material)", c70, s),
            _mk_rule_table("D. Split PO PR-based (Same PR Line)", c72, s),
            result_dict.get("rule_E"),
            result_dict.get("rule_F"),
            result_dict.get("rule_G"),
            result_dict.get("rule_H"),
            result_dict.get("rule_I"),
        ]
        blocks = [b for b in blocks if b]

        sec5 = ["**5. Compared Transactions**", "No Risk", ""]
        if blocks:
            sec5 = ["**5. Compared Transactions**", *blocks]

        return "\n".join(sec1 + sec2 + sec3 + sec4 + sec5)

    # =====================================================
    # APPLY TO EACH ROW
    # =====================================================
    def select_subject_row(df, po_no=None, po_item=None, base_id=None, idx=None):
        d = standardize(df)
        if idx is not None: return d.iloc[int(idx)]
        if base_id:
            m = d[d["base_id_src_po"].astype(str).str.strip() == _clean(base_id)]
            if len(m): return m.iloc[0]
        if po_no and po_item:
            m = d[(d["purch_doc_no_src_po"].astype(str).str.strip()==_clean(po_no)) &
                  (d["purch_doc_item_no_src_po"].astype(str).str.strip()==_clean(po_item))]
            if len(m): return m.iloc[0]
        raise ValueError("Subject not found. Provide (po_no & po_item) or base_id or idx.")

    def build_word_style_explanations(df_std: pd.DataFrame,dest_col: str = "llm_explanation",progress_every: int = 200,use_tqdm: bool = True,logger: Optional[logging.Logger] = None,):
        n = len(df_std)
        out = []

        pbar = None
        if use_tqdm:
            try:
                from tqdm import tqdm as _tqdm
                pbar = _tqdm(total=n, desc="Building evidence", unit="rows")
            except Exception:
                pbar = None

        start = time.perf_counter()
        last_tick = start
        last_count = 0

        for i, (_, row) in enumerate(df_std.iterrows(), 1):
            try:
                res = verify_transaction_all(
                    df_std,
                    po_no=str(row.get("purch_doc_no_src_po", "")).strip(),
                    po_item=str(row.get("purch_doc_item_no_src_po", "")).strip(),
                    strict_pr_item_72=True
                )
                out.append(render_all_docstyle(res, df_full=df_std))
            except Exception as e:
                out.append(f"ERROR for {row.get('purch_doc_no_src_po')}/{row.get('purch_doc_item_no_src_po')}: {e}")

            if pbar:
                pbar.update(1)

            if progress_every and (i % progress_every == 0 or i == n):
                now = time.perf_counter()
                span = now - last_tick
                total_span = now - start
                delta_rows = i - last_count
                rps = (delta_rows / span) if span > 0 else float("inf")
                total_rps = (i / total_span) if total_span > 0 else float("inf")

                msg = (f"[evidence] processed {i:,}/{n:,} rows | "
                       f"chunk {delta_rows} rows in {span:.2f}s ({rps:.1f} rows/s) | "
                       f"total {total_span:.2f}s ({total_rps:.1f} rows/s)")
                if logger:
                    logger.info(msg)
                else:
                    print(msg)

                last_tick = now
                last_count = i

        if pbar:
            pbar.close()

        total_sec = time.perf_counter() - start
        summary = f"[evidence] done: {n:,} rows in {total_sec:.2f}s ({(n/total_sec if total_sec>0 else 0):.1f} rows/s)"
        if logger:
            logger.info(summary)
        else:
            print(summary)

        df_std[dest_col] = out
        return df_std

    # =====================================================
    # MAIN
    # =====================================================
    df_full = final_result_df_2.copy()
    df_std  = standardize(df_full)

    # Standardize & precompute invoice>PO flags (amount & qty) once
    if invoice_df is not None and len(invoice_df) > 0:
        inv_std = standardize_invoice(invoice_df)  # used by E block

        pre = _precompute_invoice_vs_po_flags(df_std, inv_std)
        if not pre.empty:
            # For evidence lookup
            PRE_MAP = pre.set_index("base_id").to_dict(orient="index")

            # Attach flags + differences onto output rows
            df_std = df_std.merge(
                pre[["base_id","invoice_amt_gt_po_flag","invoice_qty_gt_po_flag","amt_excess","qty_excess",
                     "inv_total_amt","inv_total_qty","po_val_ref","po_qty_ref"]],
                left_on="base_id_src_po", right_on="base_id", how="left"
            )

            # Rename merged cols to clearer names for downstream use if needed
            df_std.rename(columns={
                "amt_excess": "invoice_amt_excess",
                "qty_excess": "invoice_qty_excess"
            }, inplace=True)

            # Clean types
            for c in ["invoice_amt_gt_po_flag","invoice_qty_gt_po_flag"]:
                if c in df_std.columns:
                    df_std[c] = df_std[c].fillna(0).astype(int)
            df_std["invoice_more_than_po_flag"] = df_std[["invoice_amt_gt_po_flag","invoice_qty_gt_po_flag"]].max(axis=1).astype(int)

    # build explanations
    df_std = build_word_style_explanations(df_std, progress_every=100, use_tqdm=True)
    return df_std
