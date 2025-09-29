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
import sys, os

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"
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

def evidence_part_2(df_std):
    SEPARATOR = "_" * 152          # exactly 40 underscores
    BULLET = "•\t"                # bullet + a single tab
    
    # ---------- helpers ----------
    def _block_range(lines, start_pat, next_pat):
        n = len(lines); si = None
        for i in range(n):
            if re.match(start_pat, lines[i]):
                si = i; break
        if si is None: return None, None
        if not next_pat: return si, n - 1
        for j in range(si + 1, n):
            if re.match(next_pat, lines[j]): return si, j - 1
        return si, n - 1
    
    def _first_after(prefix_pat, text, default=""):
        m = re.search(prefix_pat + r"([^\n]*)", text)
        return (m.group(1).strip() if m else default)
    
    def _grab_table_after(lines, head_idx):
        """Capture contiguous markdown table lines immediately after a heading index."""
        i = head_idx + 1
        while i < len(lines) and lines[i].strip() == "": i += 1
        if i >= len(lines) or not lines[i].lstrip().startswith("|"): return None, head_idx
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"): j += 1
        return "\n".join(lines[i:j]), j - 1
    
    def _parse_md_table_full(table_text):
        """Return (header_cells, data_rows), all strings; width-normalized."""
        if not table_text: return [], []
        rows = [r for r in table_text.splitlines() if r.strip()]
        header = None; data = []
        for r in rows:
            if not r.startswith("|"): continue
            cells = [c.strip() for c in r.strip("|").split("|")]
            if header is None:
                header = cells
                continue
            # divider?
            if all(re.match(r"^:?[-]+:?$", c) for c in cells):
                continue
            data.append(cells)
        if header is None: return [], []
        # normalize widths
        max_len = max(len(header), *(len(r) for r in data)) if data else len(header)
        header = header + [""] * (max_len - len(header))
        data = [r + [""] * (max_len - len(r)) for r in data]
        return header, data
    
    def _render_md_table(header_cells, data_rows):
        """Render a markdown table from header + rows, escaping | in cell text."""
        if not header_cells:
            return []
        def esc(s): return str(s).replace("|", r"\|").strip()
        hdr = "| " + " | ".join(esc(h) for h in header_cells) + " |"
        divider = "|" + "|".join("---" for _ in header_cells) + "|"
        out = [hdr, divider]
        for row in data_rows:
            out.append("| " + " | ".join(esc(c) for c in row) + " |")
        return out
    
    def _boldify_header(header_cells):
        return [c if (c.startswith("**") and c.endswith("**")) else f"**{c}**" for c in header_cells]
    
    def _boldify_first_column(data_rows):
        out = []
        for r in data_rows:
            if not r:
                out.append(r); continue
            c0 = r[0]
            c0_b = c0 if (str(c0).startswith("**") and str(c0).endswith("**")) else f"**{c0}**"
            out.append([c0_b] + r[1:])
        return out
    
    # ---------- main reformatter ----------
    def reformat_to_spec_with_bold_tables(md_text: str) -> str:
        """
        Word-style text with bold markers:
          - **Evidence**, **1–4 headings**
          - Section 1 bullets: **labels** bold
          - Section 2: markdown table; **header bold** and **first column bold**
          - Section 3 bullets: **labels** bold
          - Section 4 bullets: plain (heading bold)
          - Compared transactions A/B/C/D tables: **header bold** (data unchanged)
          - All other content preserved as-is (no rows added/removed).
        """
        if not md_text:
            return md_text
    
        lines = md_text.splitlines()
    
        # 1) Context & Trigger -> bullets with bold labels
        c_si, c_ei = _block_range(lines,
            r"^\s*\*{0,2}1\.\s*Context\s*&\s*Trigger\s*\*{0,2}\s*$",
            r"^\s*\*{0,2}2\."
        )
        part1, covered = [], set()
        if c_si is not None:
            bt = "\n".join(lines[c_si:c_ei+1])
            sara = _first_after(r"Sara Risk Score:\s*", bt, "")
            scen = _first_after(r"Risk Scenario:\s*", bt, "")
            part1 = [
                SEPARATOR,
                "**Evidence**",SEPARATOR,
                "**1. Context & Trigger**",
                f"{BULLET}**Sara Risk Score:** {sara}",
                f"{BULLET}**Risk Scenario:** {scen}",
                SEPARATOR,
            ]
            for idx in range(c_si, c_ei+1): covered.add(idx)
    
        # 2) Key Description -> markdown table; header bold + first column bold
        kd_si, kd_ei = _block_range(lines,
            r"^\s*\*{0,2}2\.\s*Key Description\s*\(PO Details\)\s*\*{0,2}\s*$",
            r"^\s*\*{0,2}3\."
        )
        part2 = []
        if kd_si is not None:
            part2.append("**2. Key Description (PO Details)**")
            table_text, tbl_end = _grab_table_after(lines, kd_si)
    
            # If table exists, rewrite with bolded header & field column; otherwise try to reconstruct from pairs
            if table_text:
                hdr, rows = _parse_md_table_full(table_text)
                if hdr:
                    hdr = _boldify_header(hdr)  # **Field**, **Value**
                if rows:
                    rows = _boldify_first_column(rows)  # **PO / Item & PR Ref**, …
                part2 += _render_md_table(hdr, rows)
                for idx in range(kd_si, kd_ei+1): covered.add(idx)
                for idx in range(kd_si+1, tbl_end+1): covered.add(idx)
            else:
                # try to rebuild from "**Field:** Value" or "Field<TAB>Value"
                pairs = []
                i = kd_si + 1
                while i <= kd_ei and lines[i].strip() == "": i += 1
                while i <= kd_ei and lines[i].strip():
                    ln = lines[i]
                    if "\t" in ln:
                        k, v = ln.split("\t", 1); pairs.append((k.strip(), v.strip())); i += 1; continue
                    m = re.match(r"^\s*\*\*(.+?)\:\*\*\s*(.*)$", ln)  # **Field:** Value
                    if m:
                        pairs.append((m.group(1).strip(), m.group(2).strip())); i += 1; continue
                    break
                if pairs:
                    # Render header bold + first column bold
                    hdr = _boldify_header(["Field", "Value"])
                    rows = _boldify_first_column([[k, v] for k, v in pairs])
                    part2 += _render_md_table(hdr, rows)
                for idx in range(kd_si, kd_ei+1): covered.add(idx)
            part2.append(SEPARATOR)
    
        # 3) Business Impact -> bullets with bold labels
        bi_si, bi_ei = _block_range(lines,
            r"^\s*\*{0,2}3\.\s*Business Impact\s*\*{0,2}\s*$",
            r"^\s*\*{0,2}4\."
        )
        part3 = []
        if bi_si is not None:
            bt = "\n".join(lines[bi_si:bi_ei+1])
            flagged = _first_after(r"Flagged Value:\s*", bt, "")
            areas   = _first_after(r"Impact Areas:\s*", bt, "")
            part3 = ["**3. Business Impact**"]
            if flagged: part3.append(f"{BULLET}**Flagged Value:** {flagged}")
            if areas:   part3.append(f"{BULLET}**Impact Areas:** {areas}")
            part3.append(SEPARATOR)
            for idx in range(bi_si, bi_ei+1): covered.add(idx)
    
        # 4) Risk Drivers -> bullets (plain body text), heading bold
        rd_si, rd_ei = _block_range(lines,
            r"^\s*\*{0,2}4\.\s*Risk Drivers\s*\*{0,2}\s*$",
            r"^\s*\*{0,2}5\."
        )
        part4 = []
        if rd_si is not None:
            raw = [ln for ln in lines[rd_si+1:rd_ei+1] if ln.strip()]
            # strip leading '-', '*' or '•'
            drivers = [re.sub(r"^[\-\*\u2022]+\s*", "", ln.strip()) for ln in raw]
            part4 = ["**4. Risk Drivers**"] + [f"{BULLET}{d}" for d in drivers] + [SEPARATOR]
            for idx in range(rd_si, rd_ei+1): covered.add(idx)
    
        # Remainder (e.g., section 5+): keep text, but boldify table headers under A/B/C/D
        remainder = [lines[i] for i in range(len(lines)) if i not in covered]
    
        def _boldify_compared_headers_in_remainder(rem_lines):
            out = []; i = 0; n = len(rem_lines)
            while i < n:
                ln = rem_lines[i]
                # Match subsection headings like '**B. ...**' or 'B. ...'
                if re.match(r"^\s*\*{0,2}[ABCD]\.\s", ln):
                    out.append(ln)  # keep heading as-is
                    # If a table immediately follows, rewrite with bold header (data unchanged)
                    tbl, end_idx = _grab_table_after(rem_lines, i)
                    if tbl:
                        hdr, rows = _parse_md_table_full(tbl)
                        if hdr:
                            hdr = _boldify_header(hdr)  # e.g., **PO → Item**, **Vendor**, …
                        out += _render_md_table(hdr, rows)
                        i = end_idx + 1
                        continue
                    i += 1
                    continue
                out.append(ln)
                i += 1
            return out
    
        remainder = _boldify_compared_headers_in_remainder(remainder)
    
        # stitch together with exact spacing (no extra blank lines)
        out = []
        if part1: out += part1
        if part2: out += part2
        if part3: out += part3
        if part4: out += part4
        out += remainder
    
        return "\n".join(out)
    # df_updated has columns: base_id_src_po, llm_explanation
    df_std["evidence_text"] = df_std["llm_explanation"].apply(reformat_to_spec_with_bold_tables)
    return df_std

   
