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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error
)
import torch
import torch.nn as nn
from typing import Optional
# Set options to show full DataFrame output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
			database='baldota-dev-db',
			user='fortifai_ng_user_ro',
			password='user@123!',
			port='5432',
            sslmode="require"
		)
    
#conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
#    			database='baldota-dev-db',
#    			user='fortifai_ng_ai_user_rw',
#    			password='AIPwd@123!',
#    			port='5432',
#                sslmode="require"
#    		)
cur = conn.cursor()
cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
        """)
tables = [row[0] for row in cur.fetchall()]

## table == name of table from transform_db or semantic db
#data= pd.read_sql_query(f"SELECT * FROM transform_db.{table}", conn)
#data= pd.read_sql_query(f"SELECT * FROM semantic_db.{table}", conn)

 # --- Step 4: Commit and close ---
#conn.commit()
#cur.close()
#conn.close()
line_item_keys = ["Purch.Doc.","Item", "Purch.Req.","Item.1",
    "Buyer Name", "Changed On", "CoCd", "Eq. To", "Denom.", "Conv.", "Conv..1", "Ct", "Customer", "D",
    "GR", "GR Date", "GR-IV", "Gross value", "IR","MTyp", "Material", "Matl Group",
    "Net Price", "Net Value", "PO Quantity.1", "PO Quantity", "Plnt",
    "Reference Document for PO Trac", "S", "Short Text", "Targ. Qty", "Target Value", "TrackingNo",
    "EKPO-CHG_FPLNR", "Acknowledgment", "Agmt.", "Item.2", "Status of purchasing doc. item"
]
line_item_values = ["purch_doc_no", "purch_doc_item_no","pr_no", "pr_item_no",
    "requester_name", "doc_change_date", "company_code", "p2o_unit_conv_denom", "o2b_unit_conv_denom",
    "p2o_unit_conv_num", "o2b_unit_conv_num", "acct_assgnmt_category", "customer_no", "po_item_del_flag",
    "gr_indicator", "latest_gr_dt", "gr_invoice_verif_flag", "gross_val_po_curr", "inv_receipt_indicator",
    "material_type", "material_no", "matl_group", "net_price_doc_curr",
    "net_val_po_curr", "order_uom", "quantity", "plant", "tpop_crm_ref_ordr_no",
    "rfq_status", "short_text", "target_qty", "outline_agrmt_tgt_val_doc_curr", "reqmt_tracking_no",
    "no_invoice_flag", "order_ack_no", "principal_purch_agrmt_no", "principal_purch_agrmt_item_no",
    "purch_doc_item_status"
]


# Step 2: Create the dictionary
my_dict_line_item = dict(zip(line_item_keys, line_item_values))

# Step 3: Print the result
print("Line items key with baldota: mapped values SAP fixed",my_dict_line_item)

header_keys=["Purch.Doc.","Supplier",
    "C", "CoCd", "Crcy", "Created By", "Created On", "Ctl", "D", "Doc. Date",
    "Doc.Cond.", "Exch. Rate", "PGr", "POrg", "PayT", "Proc.state",
    "R", "Rel", "Release", "S", "Salespers.", "Tot. value", "Type",
    "VP Start", "VPer.End"]
header_values=["purch_doc_no","vendor_or_creditor_acct_no",
    "purch_doc_category", "company_code", "currency", "object_created_by", "doc_change_date",
    "control_indicator", "po_item_del_flag", "purch_doc_date", "principal_purch_agrmt_no",
    "exchange_rate", "purch_group", "purch_org", "pymnt_terms", "processing_status", "doc_release_incompl_flag", "release_indicator", "release_status",
    "rfq_status", "resp_vendor_salesperson",
    "on_release_total_value", "purch_doc_type", "validity_start_dt", "validity_end_dt"
]
# Step 2: Create the dictionary
my_header_dict = dict(zip(header_keys, header_values))

# Step 3: Print the result
print("Header key with baldota: mapped values SAP fixed",my_header_dict)   



#tables = [row[0] for row in cur.fetchall()]
#p2p_line_item_po_data
for table in tables:
    if table == 'purchasing_document_item':
        p2p_line_item_po_data= pd.read_sql_query(f"SELECT * FROM ingest_db.{table}", conn)
        
df_line_item_po_data=p2p_line_item_po_data.copy()
## dropping '4500009180^00030' at index 0 as its order_uom is 0.0 must be added for testing
df_line_item_po_data.drop(index=0, inplace=True)

# convert to str to maintain
df_line_item_po_data['purch_doc_no'] = df_line_item_po_data['purch_doc_no'].astype(float).astype('int64').astype(str)
#df_line_item_po_data['pr_no'] = df_line_item_po_data['pr_no'].astype(float).astype('int64').astype(str)

### bring values for item number to orginal form, 10.0 to 00010 etc
df_line_item_po_data['purch_doc_item_no'] = df_line_item_po_data['purch_doc_item_no'].fillna(0).astype(float).astype(int).astype(str).str.zfill(5)
df_line_item_po_data['pr_item_no'] = df_line_item_po_data['pr_item_no'].fillna(0).astype(float).astype(int).astype(str).str.zfill(5)
#df_line_item_po_data['principal_purch_agrmt_item_no'] = df_line_item_po_data['principal_purch_agrmt_item_no'].fillna(0).astype(float).astype(int).astype(str).str.zfill(5)

## converting all column values dtype to string except for ingestion_timestamp :: will be updating dtype for values in code when those are required
#df_line_item_po_data.loc[:, df_line_item_po_data.columns != 'ingestion_timestamp'] = df_line_item_po_data.loc[:, df_line_item_po_data.columns != 'ingestion_timestamp'].astype(str)

##main data using the mapped columns from baldota
df_line_item=df_line_item_po_data[line_item_values]
## to merge with label data later on
df_line_item['base_id']=df_line_item["purch_doc_no"] + "^" + df_line_item["purch_doc_item_no"]
# adding src to line item data columns to distinguish with header column data
df_line_item_renamed = df_line_item.rename(columns={col: f"{col}_src" for col in df_line_item.columns})


#p2p_header_po_data
for table in tables:
    if table == 'purchasing_document_header':
        p2p_header_po_data= pd.read_sql_query(f"SELECT * FROM ingest_db.{table}", conn)
df_header_po_data=p2p_header_po_data.copy()

df_header_po_data['purch_doc_no'] = df_header_po_data['purch_doc_no'].astype(float).astype('int64').astype(str)
## converting all column values dtype to string except for ingestion_timestamp :: will be updating dtype for values in code when those are required
#df_header_po_data.loc[:, df_header_po_data.columns != 'ingestion_timestamp'] = df_header_po_data.loc[:, df_header_po_data.columns != 'ingestion_timestamp'].astype(str)
##main data using the mapped columns from baldota
df_header=df_header_po_data[header_values]
# adding hpd to header data columns to distinguish with line_item column data
df_header_renamed = df_header.rename(columns={col: f"{col}_hpd" for col in df_header.columns})

merged_df_before_label=pd.merge(df_line_item_renamed,df_header_renamed,left_on='purch_doc_no_src',right_on='purch_doc_no_hpd',how='outer')
merged_df_before_label.shape




## rule label output
#line_item=pd.read_excel("baldota rule label data/Line Item Transaction Summary.xlsx")
line_item=pd.read_csv("Line Item Transaction Summary 24_07.csv")
## taking output po label at line item
po_line_item_label=line_item[line_item['stage']=='PO']

## merging main data df with label data df
merged_df_after_label=pd.merge(merged_df_before_label,po_line_item_label[['base_id','rule_ids','rft_by_engine']],left_on='base_id_src',right_on='base_id',how="outer")

### adding false to rft_by_engine where rft by engine in null nan
df=merged_df_after_label.copy()
null_count = df['rft_by_engine'].isna().sum()
none_str_count = (df['rft_by_engine'] == 'None').sum()

total_missing = null_count + none_str_count
print(f"Total missing values in 'rft_by_engine': {total_missing}")

# Step 2: Replace NaN and 'None' with False
df['rft_by_engine'] = df['rft_by_engine'].replace('None', pd.NA)
df['rft_by_engine'] = df['rft_by_engine'].fillna(False)

#df = df.drop(columns=['id', 'run_id',
       #'config_id', 'transaction_id', 'base_id', 'header_id', 'header_base_id',
       #'is_data_quality_approved', 'transaction_date', 'transaction_value',
       #'stage', 'location', 'department', 'vendor_code', 'employee_code',
       #'rule_ids', 'risk_score', 'price', 'people', 'process','flag_by_audit'])

# Convert dates
df["doc_change_date_src"] = pd.to_datetime(df["doc_change_date_src"], errors='coerce')
df["doc_change_date_hpd"] = pd.to_datetime(df["doc_change_date_hpd"], errors='coerce')
df["purch_doc_date_hpd"] = pd.to_datetime(df["purch_doc_date_hpd"], errors='coerce')

# drop all rows where all values are Nan
df = df.dropna(axis=1, how='all')
#df.info()
#df['purch_doc_mapping']=df["purch_doc_no_src"] + "^" + df["purch_doc_item_no_src"]
df_final = df.rename(columns={col: f"{col}_po" for col in df.columns})

# Step 1: Get unique rule IDs from all rows (comma-separated strings)
rule_sets = df_final['rule_ids_po'].dropna().apply(lambda x: [r.strip() for r in x.split(',')])
unique_rules = sorted(set(r for sublist in rule_sets for r in sublist))

# Step 2: Create columns for each rule ID with 1 if present, else 0
for rule in unique_rules:
    df_final[rule] = df_final['rule_ids_po'].apply(lambda x: int(rule in x.split(',')) if pd.notna(x) else 0)

df_final.to_pickle("po_output.pkl")
#df_final_copy.to_pickle("po_output_tushar.pkl")
cur.close()
conn.close()