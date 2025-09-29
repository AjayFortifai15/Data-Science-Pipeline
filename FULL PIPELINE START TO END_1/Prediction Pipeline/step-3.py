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

def vendor_information_and_extra_data_cleaning(final_result_df):
    ## cleaning .0 from vendor
    s = final_result_df['vendor_or_creditor_acct_no_hpd_po'].astype(str).str.strip()           # ensure string & tidy spaces
    mask = s.str.upper().eq('UNKNOWN')                 # rows to leave as-is
    final_result_df['vendor_or_creditor_acct_no_hpd_po'] = s.where(mask, s.str.replace(r'\.0$', '', regex=True))
    
    ## getting vendor name from vendor master DB
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
    			user='fortifai_ng_user_ro',
    			password='user@123!',
    			port='5432',
                sslmode="require"
    		)
    cur = conn.cursor()
    cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
            """)
    tables = [row[0] for row in cur.fetchall()]
    vendor_data= pd.read_sql_query(f"SELECT * FROM ingest_db.{'vendor_master_general_section'}", conn)
    a=pd.merge(final_result_df,vendor_data[['vendor_or_creditor_acct_no','vendor_name_1']],left_on='vendor_or_creditor_acct_no_hpd_po',right_on='vendor_or_creditor_acct_no',how='left')
    # single or multiple
    a = a.drop(columns=["vendor_or_creditor_acct_no"], errors="ignore")  # errors='ignore' skips missing cols
    a['vendor_name_1'] = a['vendor_name_1'].fillna('UNKNOWN')
    cols = [
            'p2o_unit_conv_denom_src_po','o2b_unit_conv_denom_src_po',
            'p2o_unit_conv_num_src_po','o2b_unit_conv_num_src_po',
            'material_no_src_po','matl_group_src_po','exchange_rate_hpd_po',
    
            'principal_purch_agrmt_item_no_src_po','principal_purch_agrmt_no_hpd_po'
        ]
    a[cols] = (a[cols].astype("string")
                         .apply(lambda s: s.str.replace(r'(?<=\d)\.0+$', '', regex=True))
                         .fillna(""))
    #a.shape
    final_result_df_2=a.copy()
    return final_result_df_2