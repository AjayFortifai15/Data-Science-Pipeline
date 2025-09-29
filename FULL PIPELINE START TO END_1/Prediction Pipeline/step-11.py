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
from datetime import date
warnings.filterwarnings("ignore", category=FutureWarning)


import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import math


def db_update(doc_4):
    ## staging_db.po_header_lineitem_merged_with_risks truncate
    
    #conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    #			database='baldota-dev-db',
    #			user='fortifai_ng_user_ro',
    #			password='user@123!',
    #			port='5432',
    #            sslmode="require"
    #		)
    
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
    			user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			port='5432',
                sslmode="require"
    		)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM staging_db.po_header_lineitem_merged_with_risks;")
        conn.commit()
    
    print("✅ Cleared all rows from staging_db.po_header_lineitem_merged_with_risks")
    
    
    
    ## staging DB data upload
    #conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    #			database='baldota-dev-db',
    #			user='fortifai_ng_user_ro',
    #			password='user@123!',
    #			port='5432',
    #            sslmode="require"
    #		)
    
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
    			user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			port='5432',
                sslmode="require"
    		)
    cur = conn.cursor()
    cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
            """)
    tables = [row[0] for row in cur.fetchall()]
    
    ## table == name of table from transform_db or semantic db
    #data= pd.read_sql_query(f"SELECT * FROM transform_db.{table}", conn)
    #data= pd.read_sql_query(f"SELECT * FROM semantic_db.{table}", conn)
    
    #cur.close()
    #conn.close()
    #df['risk_summary_object']=
    new_df_1=doc_4[doc_4['purch_doc_no_src_po'].notna()].copy()
    #df = df.rename(columns=new_names, errors="ignore")
    new_df_1 = new_df_1.rename(columns={
        "llm_refined_explanation": "risk_summary_object",
    })
    import io
    import numpy as np
    
    schema = "staging_db"
    table  = "po_header_lineitem_merged_with_risks"
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ordinal_position
        """, (schema, table))
        db_cols = [r[0] for r in cur.fetchall()]
    
    df_to_insert = new_df_1[db_cols].replace({np.nan: None})  # align cols & fix NaN
    
    buf = io.StringIO()
    df_to_insert.to_csv(buf, index=False, header=False)
    buf.seek(0)
    
    copy_sql = f"COPY {schema}.{table} ({', '.join(db_cols)}) FROM STDIN WITH CSV"
    with conn.cursor() as cur:
        cur.copy_expert(copy_sql, buf)
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Inserted {len(df_to_insert)} rows into {schema}.{table}")
    
    
    
    
    
    ## transform_db.transaction_details truncate
    #conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    #			database='baldota-dev-db',
    #			user='fortifai_ng_user_ro',
    #			password='user@123!',
    #			port='5432',
    #            sslmode="require"
    #		)
    
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
    			user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			port='5432',
                sslmode="require"
    		)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM transform_db.transaction_details;")
        conn.commit()
    
    print("✅ Cleared all rows from transform_db.transaction_details")
    
    
    
    
    ## data fetch from merged db after transaction_date_update
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
    			user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			port='5432',
                sslmode="require"
    		)
    cur = conn.cursor()
    cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
            """)
    
    tables = [row[0] for row in cur.fetchall()]
    po_data= pd.read_sql_query(f"SELECT * FROM staging_db.{'po_header_lineitem_merged_with_risks'}", conn)
    cur.close()
    conn.close()
    #p2p_header_po_data.head()
    #print(po_data.shape)
    #po_data.head()
    
    
    
    
    
    #conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    #			database='baldota-dev-db',
    #			user='fortifai_ng_user_ro',
    #			password='user@123!',
    #			port='5432',
    #            sslmode="require"
    #		)
    
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
    			user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			port='5432',
                sslmode="require"
    		)
    cur = conn.cursor()
    cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
            """)
    tables = [row[0] for row in cur.fetchall()]
    
    ## table == name of table from transform_db or semantic db
    #data= pd.read_sql_query(f"SELECT * FROM transform_db.{table}", conn)
    #data= pd.read_sql_query(f"SELECT * FROM semantic_db.{table}", conn)
    
    #p2p_header_po_data= pd.read_sql_query(f"SELECT * FROM transform_db.{'transaction_details'}", conn)
    #df['stage']='PO'
    #df['region']='India'
    po_data['stage']='PO'
    po_data['region']='India'
    new_df=po_data[[ 'purch_doc_no_src_po', 'purch_doc_item_no_src_po','short_text_src_po','vendor_or_creditor_acct_no_hpd_po','net_val_po_curr_src_po',
    'currency_hpd_po','stage','region','purch_doc_date_hpd_po','vendor_name_1']]
    rename_map = {
        'purch_doc_no_src_po': 'doc_number',
        'purch_doc_item_no_src_po': 'line_item_number',
        'short_text_src_po': 'item_description',
        'vendor_or_creditor_acct_no_hpd_po': 'vendor_code',
        'net_val_po_curr_src_po': 'amount',
        'currency_hpd_po': 'currency',
        'stage': 'document_stage',
        'region': 'region',
        'purch_doc_date_hpd_po': 'transaction_date','vendor_name_1':'vendor_name',
    }
    
    new_df_1 = new_df.rename(columns=rename_map)
    new_df_1=new_df_1[new_df_1['doc_number'].notna()].copy()
    new_df_1['module_name']=None
    new_df_1['document_type']=None	
    
    
    
    schema = "transform_db"
    table  = "transaction_details"
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ordinal_position
        """, (schema, table))
        db_cols = [r[0] for r in cur.fetchall()]
    
    df_to_insert = new_df_1[db_cols].replace({np.nan: None})  # align cols & fix NaN
    
    buf = io.StringIO()
    df_to_insert.to_csv(buf, index=False, header=False)
    buf.seek(0)
    
    copy_sql = f"COPY {schema}.{table} ({', '.join(db_cols)}) FROM STDIN WITH CSV"
    with conn.cursor() as cur:
        cur.copy_expert(copy_sql, buf)
    
    p2p_header_po_data= pd.read_sql_query(f"SELECT * FROM transform_db.{'transaction_details'}", conn)
    p2p_header_po_data.head()
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Inserted {len(df_to_insert)} rows into {schema}.{table}")
    
    #p2p_header_po_data= pd.read_sql_query(f"SELECT * FROM transform_db.{'transaction_details'}", conn)
    #p2p_header_po_data
    
    
    ##transform_db.transaction_risk_analysis truncate
    #conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    #			database='baldota-dev-db',
    #			user='fortifai_ng_user_ro',
    #			password='user@123!',
    #			port='5432',
    #            sslmode="require"
    #		)
    
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
    			user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			port='5432',
                sslmode="require"
    		)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM transform_db.transaction_risk_analysis;")
        conn.commit()
    
    print("✅ Cleared all rows from transform_db.transaction_risk_analysis")
    
    
    
    
    
    
    #conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    #			database='baldota-dev-db',
    #			user='fortifai_ng_user_ro',
    #			password='user@123!',
    #			port='5432',
    #            sslmode="require"
    #		)
    
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
    			user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			port='5432',
                sslmode="require"
    		)
    cur = conn.cursor()
    cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
            """)
    tables = [row[0] for row in cur.fetchall()]
    
    ## table == name of table from transform_db or semantic db
    #data= pd.read_sql_query(f"SELECT * FROM transform_db.{table}", conn)
    #data= pd.read_sql_query(f"SELECT * FROM semantic_db.{table}", conn)
    new_df=po_data[['purch_doc_date_hpd_po', 'purch_doc_no_src_po', 'purch_doc_item_no_src_po','sub_risk_1','main_risk_scenario',
               'risk_score','impact_1','risk_level','risk_summary_object']]
    
    rename_map = {
        'purch_doc_date_hpd_po':'transaction_date', 
        'purch_doc_no_src_po':'doc_number',
           'purch_doc_item_no_src_po':'line_item_number',
        'sub_risk_1':'risk_definition',
        'main_risk_scenario':'risk_category',
           'risk_score':'risk_score', 
        'impact_1':'risk_impact_type', 
        'risk_level':'risk_severity_level',
        'risk_summary_object':'risk_description'
    }
    
    new_df_1 = new_df.rename(columns=rename_map)
    new_df_1=new_df_1[new_df_1['doc_number'].notna()].copy()
    new_df_1['identified_by']='AI Engine'
    #new_df_1['risk_description']='Coming Soon'
    new_df_1['risk_date']=date.today().strftime("%Y-%m-%d")
    new_df_1['mitigation_status']=None
    new_df_1['comments']=None
    new_df_1['risk_module']='Procure-to-Pay (P2P)'
    
    
    schema = "transform_db"
    table  = "transaction_risk_analysis"
    
    # 1️⃣ Get DB column names
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ordinal_position
        """, (schema, table))
        db_cols = [r[0] for r in cur.fetchall()]
    
    # 2️⃣ Remove 'risk_id' from insertion columns
    insert_cols = [c for c in db_cols if c.lower() != "risk_id"]
    
    # 3️⃣ Align DataFrame columns and replace NaNs
    df_to_insert = new_df_1[insert_cols].replace({np.nan: None})
    
    # 4️⃣ Create CSV buffer
    buf = io.StringIO()
    df_to_insert.to_csv(buf, index=False, header=False)
    buf.seek(0)
    
    # 5️⃣ COPY into Postgres (excluding risk_id)
    copy_sql = f"COPY {schema}.{table} ({', '.join(insert_cols)}) FROM STDIN WITH CSV"
    with conn.cursor() as cur:
        cur.copy_expert(copy_sql, buf)
    p2p_header_po_data= pd.read_sql_query(f"SELECT * FROM transform_db.{'transaction_risk_analysis'}", conn)
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Inserted {len(df_to_insert)} rows into {schema}.{table} (risk_id auto-generated)")

