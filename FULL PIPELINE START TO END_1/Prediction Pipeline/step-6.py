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
def no_risk_over_ride(df_std_2):
    # =====================================================
    # APPLY LLM "No Risk" OVERRIDE INTO NEW COLUMNS
    # =====================================================
    def apply_llm_no_risk_overrides(df, llm_col="evidence_text",out_llm_col="updated_evidence_text"):
        """
        Creates updated_* columns based on llm_explanation.
        Rule: If the text contains 'No Risk' (any case), set ALL updated columns to 'No Risk';
              else keep the existing/original values.
    
        New columns created:
          - updated_risk_level
          - updated_main_risk_scenario
          - updated_sub_risk_1
          - updated_sub_risk_2
          - updated_impact_1
          - updated_impact_2
          - updated_impact_3
        """
        if llm_col not in df.columns:
            raise KeyError(f"Column '{llm_col}' not found in DataFrame.")
    
        # 1) Build a mask: does LLM text say "No Risk" anywhere? (case-insensitive)
        #    \s* allows "No   Risk" or line breaks between the words.
        mask_no_risk = df[llm_col].astype(str).str.contains(r"\bno\s*risk\b", case=False, na=False)
    
        # 2) Map of original -> new updated columns
        col_map = {
            "risk_level":          "updated_risk_level",
            "main_risk_scenario":  "updated_main_risk_scenario",
            "sub_risk_1":          "updated_sub_risk_1",
            "sub_risk_2":          "updated_sub_risk_2",
            "impact_1":            "updated_impact_1",
            "impact_2":            "updated_impact_2",
            "impact_3":            "updated_impact_3",
            
        }
    
        # 3) Initialize updated_* columns with existing/original values
        for old_col, new_col in col_map.items():
            if old_col not in df.columns:
                df[old_col] = pd.NA
            df[new_col] = df[old_col]
    
        # 4) Override to "No Risk" wherever the LLM text contains "No Risk"
        for new_col in col_map.values():
            df.loc[mask_no_risk, new_col] = "No Risk"
    
         # Build updated_llm_explanation
        df[out_llm_col] = df[llm_col]
        df.loc[mask_no_risk, out_llm_col] = "There is No Risk for this line item."
    
        return df
    
    # ---------- Example usage ----------
    # After you've built df_std["llm_explanation"]:
    # df_std = build_word_style_explanations(df_std, dest_col="llm_explanation")
    df_updated = apply_llm_no_risk_overrides(df_std_2.copy(), llm_col="evidence_text")
    
    return df_updated
