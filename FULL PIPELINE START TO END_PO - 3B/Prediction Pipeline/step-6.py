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
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
from typing import Optional

from sklearn.preprocessing import MultiLabelBinarizer
import sys, os

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"
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
import re
import pandas as pd
import numpy as np

def no_risk_over_ride(
    df: pd.DataFrame,
    text_col: str = "evidence_text",
    out_text_col: str = "updated_evidence_text",
) -> pd.DataFrame:
    df = df.copy()
    if out_text_col not in df.columns:
        df[out_text_col] = pd.NA

    base_col_map = {
        "risk_level":          "updated_risk_level",
        "main_risk_scenario":  "updated_main_risk_scenario",
        "sub_risk_1":          "updated_sub_risk_1",
        "sub_risk_2":          "updated_sub_risk_2",
        "sub_risk_3":          "updated_sub_risk_3",
        "impact_1":            "updated_impact_1",
        "impact_2":            "updated_impact_2",
        "impact_3":            "updated_impact_3",
        "impact_4":            "updated_impact_4",
        "impact_5":            "updated_impact_5",
    }
    for old, newc in base_col_map.items():
        if newc not in df.columns:
            df[newc] = df[old] if old in df.columns else pd.NA

    CANON = {
        "PRICE_VARIANCE": "Price Variance Risk",
        "SPLIT_PO": "Split PO",
        "INV_EXCEEDS_PO": "Invoice Exceeds PO",
        "VENDOR_MISSING_KYC": "PO to Vendor with missing KYC",
        "NEW_VENDOR_TOL": "PO to new vendor > tolerance level",
        "PAN_P_NON_COMPANY": "PO to Non Company Vendors",
        "BLOCKED_VENDOR": "PO to block vendor",
    }

    DRIVER_TO_CANON = [
        (re.compile(r"\bprice\s*variance\b", re.I), CANON["PRICE_VARIANCE"]),
        (re.compile(r"\bprice\s*varaince\b", re.I), CANON["PRICE_VARIANCE"]),
        (re.compile(r"(?:same|cross)[-\s]*vendor.*price\s*variance", re.I), CANON["PRICE_VARIANCE"]),
        (re.compile(r"\b(unit\s*price|rate)\s*variance\b", re.I), CANON["PRICE_VARIANCE"]),
        (re.compile(r"\bsplit\s*po(s)?\b", re.I), CANON["SPLIT_PO"]),
        (re.compile(r"(?:same\s*pr\s*line|pr[-\s]*based|±?\s*60\s*days)", re.I), CANON["SPLIT_PO"]),
        (re.compile(r"\binvoice\s*amount|qty|quantity?\s*exceeds?\s*po\b", re.I), CANON["INV_EXCEEDS_PO"]),
        #(re.compile(r"\binvoice\s*exceeds?\s*po\b", re.I), CANON["INV_EXCEEDS_PO"]),
        (re.compile(r"\bvendor.*missing\s*(?:kyc|tax\s*id)\b|\bmissing\s*(?:kyc|tax\s*id)\b", re.I), CANON["VENDOR_MISSING_KYC"]),
        (re.compile(r"\bnew\s*vendor\b.*(≤|<=)\s*30\s*days.*(≥|>=|>\s*)\s*(₹?\s*)?1\s*cr", re.I), CANON["NEW_VENDOR_TOL"]),
        (re.compile(r"\bnew\s*vendor\b.*(high[-\s]*value|toler)", re.I), CANON["NEW_VENDOR_TOL"]),
        (re.compile(r"\bpan\b.*4(?:th)?\s*char.*\bp\b", re.I), CANON["PAN_P_NON_COMPANY"]),
        (re.compile(r"\bblocked\s*vendor\b|\bpo\s*placed\s*to\s*blocked\s*vendor\b", re.I), CANON["BLOCKED_VENDOR"]),
    ]

    CANON_IMPACTS = {
        "Price Variance Risk": ["Profitability", "Cash Flow"],
        "Invoice Exceeds PO":  ["Profitability", "Cash Flow"],
        "Split PO":            ["Efficiency"],
        "PO to block vendor":  ["Cash Flow", "Efficiency"],
        "PO to new vendor > tolerance level": ["Cash Flow", "Efficiency", "Regulatory"],
        "PO to Non Company Vendors": ["Cash Flow", "Regulatory"],
        "PO to Vendor with missing KYC": ["Efficiency", "Regulatory"],
    }

    def says_no_risk(text: str) -> bool:
        return isinstance(text, str) and bool(re.search(r"\bno\s*risk\b", text, re.I))

    PAT_DRIVERS_BLOCK = re.compile(
        r"^\*\*4\.\s*Risk\s*Drivers\*\*(?P<body>.*?)(?=^\*\*\d+\.\s|\Z)",
        re.I | re.S | re.M
    )
    def extract_driver_lines(txt: str) -> list[str]:
        if not isinstance(txt, str) or not txt.strip():
            return []
        m = PAT_DRIVERS_BLOCK.search(txt)
        if not m:
            return []
        body = m.group("body")
        lines = []
        for raw in body.splitlines():
            s = raw.strip()
            if not s:
                continue
            if s.startswith("|"):  # table rows
                continue
            if re.fullmatch(r"[_\-\s]{6,}", s):
                continue
            s = re.sub(r'^[\s•·▪●◦\-\–\—]+', '', s).strip()
            if not s or s.startswith("**"):
                continue
            s = s.rstrip(".").strip()
            if s:
                lines.append(s)
        seen, out = set(), []
        for s in lines:
            if s not in seen:
                seen.add(s); out.append(s)
        return out

    def to_canonical(line: str) -> str | None:
        if not isinstance(line, str):
            return None
        for pat, canon in DRIVER_TO_CANON:
            if pat.search(line):
                return canon
        return None

    PAT_RS_LINE = re.compile(
        r"(?mi)^(?P<bullet>\s*(?:[-•]\s*)?)"
        r"(?:\*\*)?\s*Risk\s*Scenario:\s*(?:\*\*)?"
        r"(?P<after>[^\n]*)$"
    )
    def _extract_scenario(after: str) -> str:
        core = (after or "").split("→", 1)[0].replace("**", "").strip(" :")
        return core if core else "Procurement Risk"

    def rewrite_subrisk(text: str, subrisks_display: list[str]) -> str:
        m = PAT_RS_LINE.search(text or "")
        if not m:
            return text
        bullet = m.group("bullet") or ""
        scenario = _extract_scenario(m.group("after"))
        joined = ", ".join(subrisks_display) if subrisks_display else ""
        new_line = f"{bullet}**Risk Scenario:** {scenario}"
        if joined:
            new_line += f" → **Sub-risk**: {joined}"
        return text[:m.start()] + new_line + text[m.end():]

    PAT_BI_SECTION = re.compile(
        r"(?P<head>^\*\*3\.\s*Business\s*Impact\*\*)"
        r"(?P<body>.*?)(?=^\*\*\d+\.\s|\Z)",
        re.I | re.S | re.M
    )
    PAT_FLAGGED = re.compile(
        r"(?mi)^\s*(?:[-•·]\s*)?(?:\*\*)?\s*Flagged\s*Value\s*:?\s*(?:\*\*)?\s*(?P<val>.+?)\s*\.?\s*$"
    )
    PAT_BI_LINES_TO_DROP = re.compile(
        r"(?mi)^\s*(?:[-•·]\s*)?(?:\*\*)?\s*(?:Flagged\s*Value|Impact\s*Areas)\s*:.*?$"
    )
    def rewrite_impacts(text: str, impacts: list[str]) -> str:
        m = PAT_BI_SECTION.search(text or "")
        if not m:
            return text
        body = m.group("body")
        fv_match = PAT_FLAGGED.search(body)
        flagged_val = fv_match.group("val").strip().rstrip(".") if fv_match else None
        body_clean = PAT_BI_LINES_TO_DROP.sub("", body)
        body_clean = re.sub(r"^\s*\n", "", body_clean, flags=re.M)

        bullets = [""]
        if flagged_val:
            bullets.append(f"• **Flagged Value:** {flagged_val}.")
        bullets.append(f"• **Impact Areas:** {', '.join(impacts)}.")
        joiner = "" if (body_clean.startswith("\n") or body_clean == "") else "\n"
        new_body = "\n".join(bullets) + joiner + body_clean
        return text[:m.start("body")] + new_body + text[m.end("body"):]

    def compute_impacts(canon_list: list[str]) -> list[str]:
        out, seen = [], set()
        CANON_IMPACTS = {
            "Price Variance Risk": ["Profitability", "Cash Flow"],
            "Invoice Exceeds PO":  ["Profitability", "Cash Flow"],
            "Split PO":            ["Efficiency"],
            "PO to block vendor":  ["Cash Flow", "Efficiency"],
            "PO to new vendor > tolerance level": ["Cash Flow", "Efficiency", "Regulatory"],
            "PO to Non Company Vendors": ["Cash Flow", "Regulatory"],
            "PO to Vendor with missing KYC": ["Efficiency", "Regulatory"],
        }
        for c in canon_list:
            if c in CANON_IMPACTS:
                for imp in CANON_IMPACTS[c]:
                    if imp not in seen:
                        seen.add(imp); out.append(imp)
        return out

    per_row_subrisks = {}
    per_row_impacts = {}
    updated_text = df[out_text_col].copy()

    max_sr = 0
    max_imp = 0

    for i in df.index:
        txt = str(df.at[i, text_col]) if pd.notna(df.at[i, text_col]) else ""

        # Global "No Risk" override — also update level here
        if says_no_risk(txt):
            per_row_subrisks[i] = ["No Risk"]
            per_row_impacts[i] = ["No Risk"]
            updated_text.at[i] = "There is No Risk for this line item."
            df.at[i, "updated_risk_level"] = "No Risk"   # <<< THIS LINE DOES THE TRICK
            max_sr = max(max_sr, 1)
            max_imp = max(max_imp, 1)
            continue

        raw_lines = extract_driver_lines(txt)

        seen_display = set()
        display_list = []
        seen_canon = set()
        canon_for_impacts = []

        for ln in raw_lines:
            canon = to_canonical(ln)
            if canon:
                if canon not in seen_display:
                    seen_display.add(canon)
                    display_list.append(canon)
                if canon not in seen_canon:
                    seen_canon.add(canon)
                    canon_for_impacts.append(canon)
            else:
                if ln not in seen_display:
                    seen_display.add(ln)
                    display_list.append(ln)

        if not display_list:
            if pd.isna(updated_text.at[i]):
                updated_text.at[i] = txt
            continue

        impacts = compute_impacts(canon_for_impacts)
        new_txt = rewrite_subrisk(txt, display_list)
        new_txt = rewrite_impacts(new_txt, impacts)
        updated_text.at[i] = new_txt

        per_row_subrisks[i] = display_list
        per_row_impacts[i]  = impacts
        max_sr  = max(max_sr, len(display_list))
        max_imp = max(max_imp, len(impacts))

    for k in range(1, max(3, max_sr) + 1):
        col = f"updated_sub_risk_{k}"
        if col not in df.columns:
            df[col] = pd.NA
    for k in range(1, max(5, max_imp) + 1):
        col = f"updated_impact_{k}"
        if col not in df.columns:
            df[col] = pd.NA

    for i in df.index:
        subs = per_row_subrisks.get(i)
        imps = per_row_impacts.get(i)

        if subs is not None:
            for k in range(1, max(3, max_sr) + 1):
                df.at[i, f"updated_sub_risk_{k}"] = subs[k-1] if k-1 < len(subs) else ("No Risk" if subs == ["No Risk"] else "None")

        if imps is not None:
            for k in range(1, max(5, max_imp) + 1):
                df.at[i, f"updated_impact_{k}"] = imps[k-1] if k-1 < len(imps) else ("No Risk" if imps == ["No Risk"] else "None")

    df[out_text_col] = updated_text.where(updated_text.notna(), df[text_col])
    
    df_updated=df.copy()
    return df_updated