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


line_item_keys = ["DocumentNo","InvItem","Purch.Doc.","Item", "Amount", "BlR", "Central Contract", "Central Contract Item", "CoCd", "D/C", "FIn",
    "GR/IR Clrg", "Indicator for Differential Invoicing", "Material", "Plnt",
     "OUn", "Quantity", "OPUn", "Qty in OPUn", "Reference", "SAA", "Supplier",
    "Tax Jur.", "Year", "Year.1"
]
line_item_values = ["accounting_doc_no","doc_line_item_no","purch_doc_no", "purch_doc_item_no","amt_doc_curr", "block_reason_field", "central_contract",
                    "central_contract_item_no","company_code", "debit_credit_flag", "final_inv_flag",
    "ext_gr_ir_clrg_flag", "diff_invoicing_flag", 
    "material_no", "plant", "order_uom", "quantity", "po_uom",
    "po_qty_order_uom", "ref_doc_no", "acct_assgnmt_seq_no", "vendor_or_creditor_acct_no",
    "tax_jurisdiction_code", "fiscal_year", "ref_doc_fiscal_year"
]


# Step 2: Create the dictionary
my_dict_line_item = dict(zip(line_item_keys, line_item_values))

# Step 3: Print the result
print("Line items key with baldota: mapped values SAP fixed",my_dict_line_item)

header_keys= ["Doc. No.",
    "Bline Date", "CoCd", "Crcy", "Del.Costs", "Doc. Date",  "Doc.Header Text",
    "Entry Dte", "Exch.rate", "G/L", "Gross Amnt", "I", "IV cat", "InR.Ref.no", "Inv. Pty",
    "PBk", "PM", "PayT", "Payer", "Paymt Ref.", "Reference", "Rel.", "Rvrsd by",
    "St", "TCode", "Time", "Type", "User Name"
]
header_values=["accounting_doc_no","baseline_date", "company_code", "currency", "unplanned_dlvry_costs", "doc_date",
     "doc_header_text", "doc_entry_date", "exchange_rate", "gl_account",
    "gross_inv_amt_doc_curr", "post_inv_flag", "logistics_inv_verif_orig_type", "txn_invoice_no",
    "vendor_or_creditor_acct_no", "house_bank_short_key", "pymnt_method", "pymnt_terms",
    "payee_or_payer_name", "assignment_no", "ref_doc_no", "sap_release", "reversal_doc_no",
    "invoice_doc_status", "txn_code", "entry_time", "doc_type", "username"
]
# Step 2: Create the dictionary
my_header_dict = dict(zip(header_keys, header_values))

# Step 3: Print the result
print("Header key with baldota: mapped values SAP fixed",my_header_dict)


#tables = [row[0] for row in cur.fetchall()]
#p2p_line_item_invoice_data
for table in tables:
    if table == 'invoice_receipt_items':
        p2p_line_item_invoice_data= pd.read_sql_query(f"SELECT * FROM ingest_db.{table}", conn)
        
df_line_item_invoice_data=p2p_line_item_invoice_data.copy()
# there are na values in accounting_doc_no--> dropping
df_line_item_invoice_data= df_line_item_invoice_data.dropna(subset=['accounting_doc_no'])

# convert to str to maintain
df_line_item_invoice_data['accounting_doc_no'] = df_line_item_invoice_data['accounting_doc_no'].astype(float).astype('int64').astype(str)
df_line_item_invoice_data['purch_doc_no'] = df_line_item_invoice_data['purch_doc_no'].astype(float).astype('int64').astype(str)

### bring values for item number to orginal form, 10.0 to 00010 etc
df_line_item_invoice_data['purch_doc_item_no'] = df_line_item_invoice_data['purch_doc_item_no'].fillna(0).astype(float).astype(int).astype(str).str.zfill(5)
# convert doc_line_item_no to str
df_line_item_invoice_data['doc_line_item_no'] = df_line_item_invoice_data['doc_line_item_no'].fillna(0).astype(float).astype(int).astype(str)



##main data using the mapped columns from baldota
df_line_item=df_line_item_invoice_data[line_item_values]
## to merge with label data later on
df_line_item['base_id']=df_line_item["accounting_doc_no"] + "^" + df_line_item["doc_line_item_no"]
# adding src to line item data columns to distinguish with header column data
df_line_item_renamed = df_line_item.rename(columns={col: f"{col}_src" for col in df_line_item.columns})


#p2p_header_invoice_data
for table in tables:
    if table == 'invoice_receipt_header':
        p2p_header_invoice_data= pd.read_sql_query(f"SELECT * FROM ingest_db.{table}", conn)
df_header_invoice_data=p2p_header_invoice_data.copy()

# there are na values in accounting_doc_no--> dropping
df_header_invoice_data= df_header_invoice_data.dropna(subset=['accounting_doc_no'])

# str maintain
df_header_invoice_data['accounting_doc_no'] = df_header_invoice_data['accounting_doc_no'].astype(float).astype('int64').astype(str)
## converting all column values dtype to string except for ingestion_timestamp :: will be updating dtype for values in code when those are required
#df_header_po_data.loc[:, df_header_po_data.columns != 'ingestion_timestamp'] = df_header_po_data.loc[:, df_header_po_data.columns != 'ingestion_timestamp'].astype(str)
##main data using the mapped columns from baldota
df_header=df_header_invoice_data[header_values]
# adding hpd to header data columns to distinguish with line_item column data
df_header_renamed = df_header.rename(columns={col: f"{col}_hpd" for col in df_header.columns})

merged_df_before_label=pd.merge(df_line_item_renamed,df_header_renamed,left_on='accounting_doc_no_src',right_on='accounting_doc_no_hpd',how='outer')
merged_df_before_label.shape




## rule label output
#line_item=pd.read_excel("baldota rule label data/Line Item Transaction Summary.xlsx")
line_item=pd.read_csv("Line Item Transaction Summary 24_07.csv")
## taking output po label at line item
invoice_line_item_label=line_item[line_item['stage']=='Invoice']

## merging main data df with label data df
merged_df_after_label=pd.merge(merged_df_before_label,invoice_line_item_label[['base_id','rule_ids','rft_by_engine']],left_on='base_id_src',right_on='base_id',how="outer")

### adding false to rft_by_engine where rft by engine in null nan
df=merged_df_after_label.copy()
null_count = df['rft_by_engine'].isna().sum()
none_str_count = (df['rft_by_engine'] == 'None').sum()

total_missing = null_count + none_str_count
print(f"Total missing values in 'rft_by_engine': {total_missing}")

# Step 2: Replace NaN and 'None' with False
df['rft_by_engine'] = df['rft_by_engine'].replace('None', pd.NA)
df['rft_by_engine'] = df['rft_by_engine'].fillna(False)


# Convert dates
df["baseline_date_hpd"] = pd.to_datetime(df["baseline_date_hpd"], errors='coerce')
df["doc_date_hpd"] = pd.to_datetime(df["doc_date_hpd"], errors='coerce')
df["doc_entry_date_hpd"] = pd.to_datetime(df["doc_entry_date_hpd"], errors='coerce')

# drop all rows where all values are Nan
df = df.dropna(axis=1, how='all')
#df.info()
df['purch_doc_mapping']=df["purch_doc_no_src"] + "^" + df["purch_doc_item_no_src"]
df_final = df.rename(columns={col: f"{col}_invoice" for col in df.columns})
#df_final.info()

## there are 175 invoice that have data in header but not in line item

# Step 1: Get unique rule IDs from all rows (comma-separated strings)
rule_sets = df_final['rule_ids_invoice'].dropna().apply(lambda x: [r.strip() for r in x.split(',')])
unique_rules = sorted(set(r for sublist in rule_sets for r in sublist))

# Step 2: Create columns for each rule ID with 1 if present, else 0
for rule in unique_rules:
    df_final[rule] = df_final['rule_ids_invoice'].apply(lambda x: int(rule in x.split(',')) if pd.notna(x) else 0)

df_final.to_pickle("invoice_output.pkl")

cur.close()
conn.close()