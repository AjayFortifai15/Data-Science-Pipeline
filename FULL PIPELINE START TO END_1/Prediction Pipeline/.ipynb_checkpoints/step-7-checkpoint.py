from __future__ import annotations
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
import sys, os

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"
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

warnings.filterwarnings("ignore", category=FutureWarning)


import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import math

def price_variance_data_cleaning(df_updated):
    # ============================================
    # Price Variance Metrics from markdown "updated_evidence"
    # Handles: both sections present, one present, or none (No Risk)
    # ============================================
    
    # ---------------- CONFIG ----------------
    EVIDENCE_COL = "updated_evidence_text"
    SLIGHT_THRESHOLD_PCT = 0.0   # |Δ%| < 3% → "slightly above/below"
    ABOUT_SAME_PCT = 0.0         # |Δ%| < 0.5% → "about the same"
    EPS = 0.0
    
    # ---------------- HELPERS ----------------
    _keep_num = re.compile(r"[^\d\.\-]")  # keep digits, dot, minus
    
    def to_float(s):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return np.nan
        t = str(s).strip()
        if not t:
            return np.nan
        t = _keep_num.sub("", t)  # strip currency, commas, spaces
        if t in ("", "-", ".", "-.", ".-"):
            return np.nan
        try:
            return float(t)
        except:
            return np.nan
    
    def _strip_bold(cell):
        c = str(cell).strip()
        if c.startswith("**") and c.endswith("**"):
            return c[2:-2].strip()
        return c
    
    def _norm(s):
        return re.sub(r"\s+", " ", str(s)).strip().lower()
    
    def _split_md_row(line):
        # split a markdown row like: "| a | b | c |"
        parts = [p.strip() for p in line.strip().split("|")]
        # first and last are empty due to leading/trailing pipes
        parts = [p for p in parts if p != ""]
        return [_strip_bold(p) for p in parts]
    
    def _find_section_lines(lines, startswith_any):
        """
        Find the line index where a section header occurs (matches any string in startswith_any, case-insensitive).
        Returns index or -1.
        """
        candidates = [s.lower() for s in startswith_any]
        for i, ln in enumerate(lines):
            raw = _strip_bold(ln)
            if any(_norm(raw).startswith(_norm(c)) for c in candidates):
                return i
        return -1
    
    def _collect_table_lines(lines, start_idx):
        """
        From the line after start_idx, collect consecutive table lines that start with '|'.
        Stops when a non-table line is encountered.
        """
        tbl = []
        i = start_idx + 1
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            tbl.append(lines[i])
            i += 1
        return tbl
    
    def _parse_md_table(table_lines):
        """
        Parse a markdown table (as lines starting with '|') into a DataFrame.
        Returns empty DataFrame if cannot parse.
        """
        if not table_lines:
            return pd.DataFrame()
    
        # Expect header, separator, then rows
        # Find the first header line (has at least two columns)
        header_idx = -1
        for k, ln in enumerate(table_lines[:3]):  # header usually within first 3 lines
            if "|" in ln:
                cols = _split_md_row(ln)
                if len(cols) >= 2:
                    header_idx = k
                    break
        if header_idx == -1 or header_idx + 1 >= len(table_lines):
            return pd.DataFrame()
    
        header = _split_md_row(table_lines[header_idx])
        # Skip separator row if present (---)
        data_start = header_idx + 1
        if re.search(r"-{3,}", table_lines[data_start]):
            data_start += 1
    
        rows = []
        for ln in table_lines[data_start:]:
            if not ln.lstrip().startswith("|"):
                break
            row = _split_md_row(ln)
            if len(row) < len(header):
                # pad if row is short
                row = row + [""] * (len(header) - len(row))
            rows.append(row[:len(header)])
    
        if not rows:
            return pd.DataFrame(columns=header)
    
        df = pd.DataFrame(rows, columns=header)
    
        # Normalize common column names
        rename_map = {}
        for c in df.columns:
            lc = _norm(c)
            if lc in ("po → item", "po -> item", "po → item ", "po item"):
                rename_map[c] = "po_item"
            elif lc == "vendor":
                rename_map[c] = "vendor"
            elif lc in ("price", "unit price"):
                rename_map[c] = "price"
            elif lc in ("qty", "quantity"):
                rename_map[c] = "qty"
            elif "Δ vs current" in lc or "delta vs current" in lc or "change vs current" in lc:
                rename_map[c] = "delta_vs_current_pct"
            elif "variance value/unit" in lc:
                rename_map[c] = "variance_value_per_unit"
            elif lc == "variance value":
                rename_map[c] = "variance_value"
            elif lc == "date":
                rename_map[c] = "date"
            elif lc == "material":
                rename_map[c] = "material"
            elif lc == "text":
                rename_map[c] = "text"
            elif lc == "pr":
                rename_map[c] = "pr"
            elif "deletion indicator" in lc or lc == "deletion":
                rename_map[c] = "deletion_indicator"
        if rename_map:
            df = df.rename(columns=rename_map)
    
        # Coerce numerics
        if "price" in df.columns:
            df["price_num"] = df["price"].map(to_float)
        if "qty" in df.columns:
            df["qty_num"] = df["qty"].map(to_float)
        if "variance_value_per_unit" in df.columns:
            df["variance_per_unit_num"] = df["variance_value_per_unit"].map(to_float)
        if "variance_value" in df.columns:
            df["variance_value_num"] = df["variance_value"].map(to_float)
    
        # Standardize deletion_indicator
        if "deletion_indicator" in df.columns:
            df["del_ind_norm"] = df["deletion_indicator"].fillna("").astype(str).str.strip().str.upper()
        else:
            df["del_ind_norm"] = ""
    
        return df
    
    def _parse_section2_kv(md_lines):
        """
        Parse Section 2 Key Description table (Field | Value) → dict
        """
        # locate section 2 header
        idx2 = _find_section_lines(
            md_lines,
            startswith_any=[
                "**2. Key Description",     # exact from sample
                "2. Key Description",       # fallback
            ],
        )
        if idx2 == -1:
            return {}
    
        tbl_lines = _collect_table_lines(md_lines, idx2)
        df2 = _parse_md_table(tbl_lines)
        if df2.empty or not set(df2.columns) >= {"Field", "Value"}:
            return {}
    
        kv = {}
        for _, r in df2.iterrows():
            k = _strip_bold(r["Field"])
            v = r["Value"]
            kv[k] = v
        return kv
    
    def _parse_section_A(md_lines):
        idxA = _find_section_lines(
            md_lines,
            startswith_any=[
                "**A. Price Variance", "A. Price Variance",
                "**A.", "A."
            ],
        )
        if idxA == -1:
            return pd.DataFrame()
        tblA = _collect_table_lines(md_lines, idxA)
        return _parse_md_table(tblA)
    
    def _parse_section_B(md_lines):
        idxB = _find_section_lines(
            md_lines,
            startswith_any=[
                "**B. Price Variance", "B. Price Variance",
                "**B.", "B."
            ],
        )
        if idxB == -1:
            return pd.DataFrame()
        tblB = _collect_table_lines(md_lines, idxB)
        return _parse_md_table(tblB)
    
    def _verdict(delta_pct):
        if pd.isna(delta_pct):
            return "insufficient data"
        ap = abs(delta_pct)
        if ap < ABOUT_SAME_PCT:
            return "about the same"
        if ap < SLIGHT_THRESHOLD_PCT:
            return "slightly higher" if delta_pct > 0 else "slightly lower"
        return "higher" if delta_pct > 0 else "lower"
    
    def _currency_prefix(s):
        # best-effort: detect e.g. "INR" from a string like "INR 85.03"
        if not isinstance(s, str):
            return ""
        m = re.match(r"\s*([A-Za-z]{3,})\b", s.strip())
        return (m.group(1) + " ") if m else ""
    
    def compute_price_variance_from_evidence(df, evidence_col=EVIDENCE_COL):
        out_cols = [
            "current_qty",
            "current_unit_price",
            "currency_hint",
            "avg_unit_price_same_vendor",
            "avg_unit_price_cross_vendor",
            "avg_unit_price_simple_all",
            "n_rows_same_vendor",
            "n_rows_cross_vendor",
            "delta_per_unit",
            "delta_pct",
            "variance_value_total",
            "verdict",
            "status",
            "used_sections",
        ]
        for c in out_cols:
            df[c] = np.nan
    
        df["verdict"] = None
        df["status"] = None
        df["used_sections"] = None
        df["currency_hint"] = ""
    
        for idx, text in df[evidence_col].fillna("").astype(str).items():
            txt = text.strip()
            if not txt:
                df.at[idx, "status"] = "missing evidence"
                continue
    
            # Fast path: No Risk message
            if "no risk for this line item" in txt.lower():
                df.at[idx, "status"] = "no risk (as per evidence)"
                continue
    
            lines = [ln.rstrip("\n") for ln in txt.splitlines() if ln.strip() != "" or ln.startswith("|")]
    
            # ---------- Section 2 (current qty & unit price; deletion indicator; currency) ----------
            kv = _parse_section2_kv(lines)
            if not kv:
                df.at[idx, "status"] = "missing section 2"
                # keep going—we might still get averages, but delta will be NaN without current price/qty
    
            # Deletion check for current line
            cur_del = str(kv.get("Deletion Indicator", "")).strip().upper() if kv else ""
            if cur_del == "L":
                df.at[idx, "status"] = "current item deleted (L) — skipped"
                continue
    
            # Quantity / Unit & Unit Price e.g. "490 L @ INR 85.03"
            cur_qty = np.nan
            cur_unit_price = np.nan
            cur_currency_hint = ""
            if kv:
                qpu = kv.get("Quantity / Unit & Unit Price", "") or kv.get("Qty / Unit & Unit Price", "")
                if qpu:
                    # try to capture quantity and unit price
                    # patterns like "490 L @ INR 85.03" or "490 @ 85.03"
                    m = re.search(r"(?P<qty>[\d,\.]+)\s*[A-Za-z]*\s*@\s*(?P<cur>.+)$", qpu)
                    if m:
                        cur_qty = to_float(m.group("qty"))
                        cur_unit_price = to_float(m.group("cur"))
                        cur_currency_hint = _currency_prefix(m.group("cur"))
                # If not found, try to look into "Unit Price" field directly (if your format ever splits it)
                if pd.isna(cur_unit_price):
                    up = kv.get("Unit Price", "")
                    if up:
                        cur_unit_price = to_float(up)
                        cur_currency_hint = _currency_prefix(up)
                # Quantity fallback
                if pd.isna(cur_qty):
                    q = kv.get("Quantity", "")
                    if q:
                        cur_qty = to_float(q)
    
            df.at[idx, "current_qty"] = cur_qty
            df.at[idx, "current_unit_price"] = cur_unit_price
            df.at[idx, "currency_hint"] = cur_currency_hint
    
            # ---------- Section A & B (compared transactions) ----------
            dfA = _parse_section_A(lines)
            dfB = _parse_section_B(lines)
    
            # Filter out deletion flag L
            def usable(dfX):
                if dfX.empty:
                    return dfX
                good = dfX.copy()
                good = good[(~good["price_num"].isna()) & (~good["qty_num"].isna()) & (good["qty_num"] > 0)]
                good = good[good["del_ind_norm"].fillna("") != "L"]
                return good
    
            A_use = usable(dfA)
            B_use = usable(dfB)
    
            # counts
            nA, nB = len(A_use), len(B_use)
            df.at[idx, "n_rows_same_vendor"] = nA
            df.at[idx, "n_rows_cross_vendor"] = nB
    
            used_secs = []
            if nA > 0: used_secs.append("A")
            if nB > 0: used_secs.append("B")
            df.at[idx, "used_sections"] = ",".join(used_secs) if used_secs else "none"
    
            # Averages (per-section)
            avgA = 0.0
            if nA > 0:
                avgA = (A_use["price_num"] * A_use["qty_num"]).sum() / A_use["qty_num"].sum()
            avgB = 0.0
            if nB > 0:
                avgB = (B_use["price_num"] * B_use["qty_num"]).sum() / B_use["qty_num"].sum()
            df.at[idx, "avg_unit_price_same_vendor"] = avgA
            df.at[idx, "avg_unit_price_cross_vendor"] = avgB
            
            total_avg=0.0
            ## Combined averages (weighted + simple) across whichever sections exist
            #all_use = pd.concat([A_use, B_use], ignore_index=True) if (nA + nB) > 0 else pd.DataFrame()
            #if not all_use.empty:
                #w_avg = (all_use["price_num"] * all_use["qty_num"]).sum() / max(all_use["qty_num"].sum(), EPS)
                #s_avg = all_use["price_num"].mean()
                #df.at[idx, "avg_unit_price_weighted_all"] = w_avg
                #df.at[idx, "avg_unit_price_simple_all"] = s_avg
                #df.at[idx, "n_rows_used"] = len(all_use)
            if avgA > 0.0 or avgB > 0.0:
                total_avg = (avgA*nA + avgB*nB)/(nA+nB)
                df.at[idx, "avg_unit_price_simple_all"] = total_avg
            else:
                df.at[idx, "status"] = (df.at[idx, "status"] or "") + " | no comparison rows"
    
            # ---------- Deltas & totals ----------
            cur_p = df.at[idx, "current_unit_price"]
            base_avg = df.at[idx, "avg_unit_price_simple_all"]  # prefer weighted average
            if not pd.isna(cur_p) and not pd.isna(base_avg) and abs(base_avg) > EPS:
                delta_per_unit = cur_p - base_avg
                delta_pct = (delta_per_unit / base_avg) * 100.0
                verdict = _verdict(delta_pct)
                df.at[idx, "delta_per_unit"] = delta_per_unit
                df.at[idx, "delta_pct"] = delta_pct
                q = df.at[idx, "current_qty"]
                if not pd.isna(q):
                    df.at[idx, "variance_value_total"] = delta_per_unit * q
                df.at[idx, "verdict"] = verdict
                if not df.at[idx, "status"]:
                    df.at[idx, "status"] = "ok"
            else:
                # If we can’t compute delta, mark status
                if not df.at[idx, "status"]:
                    df.at[idx, "status"] = "insufficient data for delta"
    
        # Optional: pretty summary column
        def _fmt_money(v, cur):
            if pd.isna(v): return "NaN"
            # Simple formatting, keeps sign
            return f"{cur}{abs(v):,.2f}" if cur else f"{v:,.2f}"
    
        def mk_summary(r):
            cur = r.get("currency_hint", "") or ""
            cp = r.get("current_unit_price", np.nan)
            ap = r.get("avg_unit_price_simple_all", np.nan)
            dp = r.get("delta_per_unit", np.nan)
            dpp = r.get("delta_pct", np.nan)
            tv = r.get("variance_value_total", np.nan)
            verdict = r.get("verdict", None) or "insufficient data"
            used = r.get("used_sections", "none")
    
            if pd.isna(cp) and pd.isna(ap):
                return "No price info found."
            side = "higher" if (not pd.isna(dpp) and dpp > 0) else ("lower" if not pd.isna(dpp) else "unknown")
    
            return (
                f"Current Price/Unit: {_fmt_money(cp, cur)} | "
                f"Avg Price/Unit ({used}): {_fmt_money(ap, cur)} | "
                f"Δ/Unit: {_fmt_money(dp, cur)} ({'' if pd.isna(dpp) else f'{dpp:+.2f}%'}); "
                f"→ Current PO price is {verdict}. "
                f"Total Variance Value: {_fmt_money(tv, cur)}"
            )
    
        df["variance_summary"] = df[out_cols].apply(mk_summary, axis=1)
        return df
    
    # ---------------- USAGE ----------------
    df_updated_2 = compute_price_variance_from_evidence(df_updated.copy(), evidence_col="updated_evidence_text")
    # Now you can inspect:
    # df[[
    #     "current_qty","current_unit_price","avg_unit_price_same_vendor","avg_unit_price_cross_vendor",
    #     "avg_unit_price_weighted_all","delta_per_unit","delta_pct","variance_value_total",
    #     "verdict","status","used_sections","variance_summary"
    # ]]

    drop_some=df_updated_2.copy()
    df_updated_final=drop_some.drop(columns=[
    'current_qty','current_unit_price','currency_hint','avg_unit_price_same_vendor',
    'avg_unit_price_cross_vendor','avg_unit_price_weighted_all','avg_unit_price_simple_all',
    'n_rows_same_vendor', 'n_rows_cross_vendor', 'n_rows_used',
           'delta_per_unit', 'delta_pct', 'variance_value_total', 'verdict',
           'status', 'used_sections'],errors="ignore")
    return df_updated_final