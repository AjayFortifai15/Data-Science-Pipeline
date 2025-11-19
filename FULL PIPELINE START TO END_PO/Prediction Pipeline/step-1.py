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
import sys, os

import re
# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"

def data_load_and_cleaning_po():
        ### setting connection
    #conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			#database='baldota-dev-db',
    			#user='fortifai_ng_ai_user_rw',
    			#password='AIPwd@123!',
    			#port='5432',
                #sslmode="require"
    		#)
    
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
                            user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			#user='fortifai_ng_user_ro',
    			#password='user@123!',
    			port='5432',
                sslmode="require"
    		)
    cur = conn.cursor()
    cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
            """)
    tables = [row[0] for row in cur.fetchall()]
    
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
    
    
    df=merged_df_before_label.copy()
    # Convert dates
    df["doc_change_date_src"] = pd.to_datetime(df["doc_change_date_src"], errors='coerce')
    df["doc_change_date_hpd"] = pd.to_datetime(df["doc_change_date_hpd"], errors='coerce')
    df["purch_doc_date_hpd"] = pd.to_datetime(df["purch_doc_date_hpd"], errors='coerce')
    
    # drop all rows where all values are Nan
    df = df.dropna(axis=1, how='all')
    #df.info()
    #df['purch_doc_mapping']=df["purch_doc_no_src"] + "^" + df["purch_doc_item_no_src"]
    df_final_po= df.rename(columns={col: f"{col}_po" for col in df.columns})
    df_final_po.info()
    cur.close()
    conn.close()
    # there are 3 po that have data in header but not in line item
    
    
    
    
    ### Invoice Data ####
    
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
                            user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			#user='fortifai_ng_user_ro',
    			#password='user@123!',
    			port='5432',
                sslmode="require"
    		)
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
    
    df=merged_df_before_label.copy()
    # Convert dates
    df["baseline_date_hpd"] = pd.to_datetime(df["baseline_date_hpd"], errors='coerce')
    df["doc_date_hpd"] = pd.to_datetime(df["doc_date_hpd"], errors='coerce')
    df["doc_entry_date_hpd"] = pd.to_datetime(df["doc_entry_date_hpd"], errors='coerce')
    
    # drop all rows where all values are Nan
    df = df.dropna(axis=1, how='all')
    #df.info()
    df['purch_doc_mapping']=df["purch_doc_no_src"] + "^" + df["purch_doc_item_no_src"]
    df_final_invoice = df.rename(columns={col: f"{col}_invoice" for col in df.columns})
    df_final_invoice.info()
    
    ## there are 175 invoice that have data in header but not in line item
    cur.close()
    conn.close()
    
    
    po_invoice=pd.merge(df_final_po,df_final_invoice, left_on='base_id_src_po',right_on='purch_doc_mapping_invoice',how='outer')
    
    ## cleaning .0 from vendor
    s = po_invoice['vendor_or_creditor_acct_no_hpd_po'].astype(str).str.strip()           # ensure string & tidy spaces
    mask = s.str.upper().eq('UNKNOWN')                 # rows to leave as-is
    po_invoice['vendor_or_creditor_acct_no_hpd_po'] = s.where(mask, s.str.replace(r'\.0$', '', regex=True))
    
    ## getting vendor name from vendor master DB
    conn = psycopg2.connect(host='fortifai-ng-dev-db.postgres.database.azure.com',
    			database='baldota-dev-db',
                            user='fortifai_ng_ai_user_rw',
    			password='AIPwd@123!',
    			#user='fortifai_ng_user_ro',
    			#password='user@123!',
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
    lfa_1=pd.read_excel('LFA1 - Vendor Master.xlsx')
    lfa_1["Supplier"] = lfa_1["Supplier"].astype(str)
    lfa_1_1=lfa_1[['Supplier','DelF','DeBl','B','B.1','Status']]
    lfa_1_2 = lfa_1_1.rename(columns={
        "Supplier": "vendor_or_creditor_acct_no",
        'DelF':"central_deletion_flag",
        "DeBl": "central_del_block_flg",
        "B": "central_posting_blk_flag",
        "B.1": "central_purch_blk_flag",
        "Status": "data_transfer_status",
    })
    
    KEY = "vendor_or_creditor_acct_no"
    cols_to_update = ["central_deletion_flag",
        "central_del_block_flg",
        "central_posting_blk_flag",
        "central_purch_blk_flag",
        "data_transfer_status",
    ]
    
    
    vendor= vendor_data.drop(columns=cols_to_update)
    
    
    # Merge and overwrite columns
    merged = vendor.merge(
        lfa_1_2,
        on=KEY,
        how="outer")
    vendor_data_updated = merged
    # drop all rows where all values are Nan
    vendor_data_1 = vendor_data_updated.dropna(axis=1, how='all')
    po_invoice_vendor=pd.merge(po_invoice,vendor_data_1,left_on='vendor_or_creditor_acct_no_hpd_po',right_on='vendor_or_creditor_acct_no',how='outer')
    
    # ---------- 0) Invoice reversed flag ----------
    po_invoice_vendor["invoice_reversed"] = np.where(
        po_invoice_vendor["reversal_doc_no_hpd_invoice"].isnull(), 0, 1
    )
    
    # ---------- 1) Make sure inputs are numeric ----------
    for col in ["amt_doc_curr_src_invoice", "quantity_src_invoice",
                "net_val_po_curr_src_po", "quantity_src_po"]:
        po_invoice_vendor[col] = pd.to_numeric(po_invoice_vendor[col], errors="coerce")
    
    # ---------- 2) Build the gate ----------
    if "reversal_doc_no_hpd_invoice" in po_invoice_vendor.columns:
        inv_rev_ok = po_invoice_vendor["reversal_doc_no_hpd_invoice"].isna()
    elif "invoice_reversed" in po_invoice_vendor.columns:
        inv_rev_ok = po_invoice_vendor["invoice_reversed"].fillna(0).eq(0)
    else:
        inv_rev_ok = True  # broadcasts
    
    gate = (
        (po_invoice_vendor["release_indicator_hpd_po"] == "R") &
        (po_invoice_vendor["po_item_del_flag_src_po"].isna()) &
        inv_rev_ok
    )
    
    KEY = "purch_doc_mapping_invoice"
    
    # ---------- 3) Work only on gated rows ----------
    cols_needed = [KEY, "amt_doc_curr_src_invoice", "quantity_src_invoice",
                   "net_val_po_curr_src_po", "quantity_src_po"]
    gated = po_invoice_vendor.loc[gate, cols_needed].copy()
    
    # Group totals ONLY from gated rows
    gated["invoice_total_amount"]   = gated.groupby(KEY)["amt_doc_curr_src_invoice"].transform("sum")
    gated["invoice_total_quantity"] = gated.groupby(KEY)["quantity_src_invoice"].transform("sum")
    
    # Pick PO references within gated rows (use max to be conservative)
    gated["po_value_ref"] = gated.groupby(KEY)["net_val_po_curr_src_po"].transform("max")
    gated["po_qty_ref"]   = gated.groupby(KEY)["quantity_src_po"].transform("max")
    
    # ---------- 4) Kill float noise and compare with tolerance ----------
    AMT_ATOL = 0.01   # ₹0.01 tolerance
    QTY_ATOL = 1e-9   # effectively exact for integers
    
    # Round values before comparing (money: 2dp; qty: 6dp)
    gated["invoice_total_amount"]   = gated["invoice_total_amount"].round(2)
    gated["po_value_ref"]           = gated["po_value_ref"].round(2)
    gated["invoice_total_quantity"] = gated["invoice_total_quantity"].round(6)
    gated["po_qty_ref"]             = gated["po_qty_ref"].round(6)
    
    amt_breach = gated["invoice_total_amount"]   > (gated["po_value_ref"] + AMT_ATOL)
    qty_breach = gated["invoice_total_quantity"] > (gated["po_qty_ref"]   + QTY_ATOL)
    
    gated["invoice_more_than_po_flag"] = (amt_breach | qty_breach).astype(int)
    
    # If any row in a gated group breaches → mark ALL gated rows in that group
    gated["invoice_more_than_po_flag"] = gated.groupby(KEY)["invoice_more_than_po_flag"].transform("max")
    
    # ---------- 5) Map results back; non-gated rows remain 0 ----------
    po_invoice_vendor["invoice_total_amount"]          = 0.0
    po_invoice_vendor["invoice_total_quantity"]        = 0.0
    po_invoice_vendor["invoice_more_than_po_flag"]     = 0
    
    po_invoice_vendor.loc[gated.index, "invoice_total_amount"]   = gated["invoice_total_amount"].values
    po_invoice_vendor.loc[gated.index, "invoice_total_quantity"] = gated["invoice_total_quantity"].values
    po_invoice_vendor.loc[gated.index, "invoice_more_than_po_flag"] = gated["invoice_more_than_po_flag"].values
    
    # (Optional) ensure the flag is int, not float
    #po_invoice_vendor["invoice_more_than_po_flag"] = po_invoice_vendor["invoice_more_than_po_flag"].astype(int)
        
    
    
    
    label_data=pd.read_excel("Rulewise summary - 6 rules.xlsx")
    one=pd.merge(po_invoice_vendor,label_data,left_on='base_id_src_po',right_on='base_id',how='left')
    
    
    
    one_1=one[['purch_doc_no_src_po', 'purch_doc_item_no_src_po', 'pr_no_src_po',
           'pr_item_no_src_po', 'requester_name_src_po', 'doc_change_date_src_po',
           'company_code_src_po', 'p2o_unit_conv_denom_src_po',
           'o2b_unit_conv_denom_src_po', 'p2o_unit_conv_num_src_po',
           'o2b_unit_conv_num_src_po', 'po_item_del_flag_src_po',
           'gr_indicator_src_po', 'gr_invoice_verif_flag_src_po',
           'gross_val_po_curr_src_po', 'inv_receipt_indicator_src_po',
           'material_type_src_po', 'material_no_src_po', 'matl_group_src_po',
           'net_price_doc_curr_src_po', 'net_val_po_curr_src_po',
           'order_uom_src_po', 'quantity_src_po', 'plant_src_po',
           'short_text_src_po', 'target_qty_src_po',
           'outline_agrmt_tgt_val_doc_curr_src_po', 'reqmt_tracking_no_src_po',
           'principal_purch_agrmt_item_no_src_po', 'base_id_src_po',
           'purch_doc_no_hpd_po', 'vendor_or_creditor_acct_no_hpd_po',
           'purch_doc_category_hpd_po', 'company_code_hpd_po', 'currency_hpd_po',
           'object_created_by_hpd_po', 'doc_change_date_hpd_po',
           'control_indicator_hpd_po', 'purch_doc_date_hpd_po',
           'principal_purch_agrmt_no_hpd_po', 'exchange_rate_hpd_po',
           'purch_group_hpd_po', 'purch_org_hpd_po', 'pymnt_terms_hpd_po',
           'processing_status_hpd_po', 'doc_release_incompl_flag_hpd_po',
           'release_indicator_hpd_po', 'release_status_hpd_po',
           'rfq_status_hpd_po', 'resp_vendor_salesperson_hpd_po',
           'on_release_total_value_hpd_po', 'purch_doc_type_hpd_po','vendor_or_creditor_acct_no', 'country_code', 'vendor_name_1',
           'vendor_name_2', 'vendor_name_3', 'vendor_name_4', 'city',
           'postal_code', 'street_address', 'item_manual_addr_no',
           'matchcode_search_term_1', 'record_creation_dt', 'object_created_by',
           'vendor_acct_group',
           'tax_no_1', 'vendor_telephone_no', 'second_telephone_no', 'tax_no_3',
           'tax_no_5','central_deletion_flag', 'central_purch_blk_flag','central_posting_blk_flag','data_transfer_status',
               "invoice_more_than_po_flag"]]
    
    #one_1.drop_duplicates(inplace=True)
    
    #rule_2
    # ensure the flag is numeric 0/1 (or numeric)
    one_1["invoice_more_than_po_flag"] = (
        pd.to_numeric(one_1["invoice_more_than_po_flag"], errors="coerce")
          .fillna(0).astype(int)
    )
    
    # keep rows where the flag equals the group's max; drop the rest
    grp_max = one_1.groupby("base_id_src_po")["invoice_more_than_po_flag"].transform("max")
    one_1 = one_1.loc[
        one_1["invoice_more_than_po_flag"].eq(grp_max)
    ].copy()
    one_1 = (one_1
             .sort_values(["base_id_src_po", "invoice_more_than_po_flag"], ascending=[True, False])
             .drop_duplicates("base_id_src_po", keep="first"))
    
    
    # rule label output
    #line_item=pd.read_excel("baldota rule label data/Line Item Transaction Summary.xlsx")
    rft=pd.read_csv("Line Item Transaction Summary 24_07.csv")
    ## taking output po label at line item
    rft_po=rft[rft['stage']=='PO']
    one_2=one_1[one_1['base_id_src_po'].notna()].copy()
    ## merging main data df with label data df
    one_3=pd.merge(one_2,rft_po[['base_id','rule_ids','rft_by_engine']],left_on='base_id_src_po',right_on='base_id',how="left")
    #one_3.head()
    ## rule_3
    block_cols = ["central_deletion_flag", "central_purch_blk_flag", "central_posting_blk_flag"]
    
    one_3["po_to_blocked_vendor"] = np.where(
        (one_3[block_cols].eq("X").any(axis=1)) &
        (one_3["release_indicator_hpd_po"] == "R") &
        (one_3["po_item_del_flag_src_po"].isna()),
        1,
        0
    )
    
    
    ## rule_3
    # Ensure both columns are datetime
    one_3["purch_doc_date_hpd_po"] = pd.to_datetime(one_3["purch_doc_date_hpd_po"], errors="coerce")
    one_3["record_creation_dt"] = pd.to_datetime(one_3["record_creation_dt"], errors="coerce")
    
    # Create flag: 1 if purch_doc_date_hpd_po is within 30 days of record_creation_dt
    one_3["purch_doc_within_30days"] = (
        (one_3["purch_doc_date_hpd_po"] - one_3["record_creation_dt"]).abs().dt.days <= 30
    ).astype(int)
    
    ## rule_4
    # Ensure numeric types
    one_3["on_release_total_value_hpd_po"] = pd.to_numeric(one_3["on_release_total_value_hpd_po"], errors="coerce")
    one_3["exchange_rate_hpd_po"] = pd.to_numeric(one_3["exchange_rate_hpd_po"], errors="coerce")
    
    # Calculate PO value in INR (or base currency)
    one_3["po_value_inr"] = one_3["on_release_total_value_hpd_po"] * one_3["exchange_rate_hpd_po"]
    
    # Create flag: 1 if > 1 crore (1 Cr = 10,000,000)
    one_3["po_gt_1cr_flag"] = np.where(one_3["po_value_inr"] > 1e7, 1, 0)
    
    one_3["po_to_new_vendor_gt_tolerance"] = np.where(
        (one_3["purch_doc_within_30days"] == 1) & (one_3["po_gt_1cr_flag"] == 1) & (one_3["release_indicator_hpd_po"] == 'R')
        & (one_3["po_item_del_flag_src_po"].isna()),
        1,
        0
    )
    ## rule_5
    df = one_3.copy() # or your dataframe name
    
    # 1) Extract PAN: remove first 2 chars and last 3 chars, clean up casing/whitespace
    df["pan_extracted"] = (
        df["tax_no_3"]
          .astype(str)
          .str.strip()
          .str.upper()
          .str[2:-3]                 # remove first 2 and last 3
    )
    
    # (Optional) Keep only plausible PANs (10 chars, alphanumeric)
    pan_valid = df["pan_extracted"].str.len().eq(10) & df["pan_extracted"].str.isalnum()
    df.loc[~pan_valid, "pan_extracted"] = np.nan
    
    # 2) Amount in INR (or base) and >= 1 Cr flag
    df["on_release_total_value_hpd_po"] = pd.to_numeric(df["on_release_total_value_hpd_po"], errors="coerce")
    df["exchange_rate_hpd_po"] = pd.to_numeric(df["exchange_rate_hpd_po"], errors="coerce")
    df["po_value_inr"] = df["on_release_total_value_hpd_po"] * df["exchange_rate_hpd_po"]
    
    amount_ge_1cr = df["po_value_inr"] >= 1e7  # 1 crore = 10,000,000
    
    # 3) 4th PAN character == 'P'
    fourth_char_is_P = df["pan_extracted"].str[3].eq("P")
    
    # 4) Final flag: 4th PAN char 'P' AND amount ≥ 1 Cr  -> 1 else 0
    df["pan4P_amt_ge_1cr_flag"] = np.where(fourth_char_is_P & amount_ge_1cr &  (df["release_indicator_hpd_po"] == 'R')
        & (df["po_item_del_flag_src_po"].isna()), 1, 0)
    
    ##rule_6
    df["missing_tax_id_flag"] = np.where(df["tax_no_3"].isna() &  (df["release_indicator_hpd_po"] == 'R')
        & (df["po_item_del_flag_src_po"].isna()), 1, 0)
    
    
    one_3=df.copy()
    
    
    
    # 1) Flag: True if either column is true
    # List your 5 columns
    cols_to_check = ["invoice_more_than_po_flag", "po_to_blocked_vendor", "po_to_new_vendor_gt_tolerance", "pan4P_amt_ge_1cr_flag","missing_tax_id_flag"]
    
    # output_flag will be True if any column == 1, else False
    one_3["new_model_output"] = df[cols_to_check].eq(1).any(axis=1)
    
    
    
    # List the columns to check
    cols_to_check = ["invoice_more_than_po_flag", "po_to_blocked_vendor", "po_to_new_vendor_gt_tolerance", "pan4P_amt_ge_1cr_flag","missing_tax_id_flag"]
    
    # Build rule_summary by joining column names where value == 1
    one_3["new_rule_summary"] = (
        df[cols_to_check]
        .apply(lambda row: ",".join(row.index[row.eq(1)]), axis=1)
    )
    
    # If you prefer NaN instead of empty string when no rules matched:
    # df["rule_summary"] = df["rule_summary"].replace("", np.nan)
    #one_3.head()
    
    
    
    try:
        to_bool
    except NameError:
        TRUEY = {"true","t","yes","y","1","ok","on"}
        def to_bool(x):
            if pd.isna(x):
                return False
            if isinstance(x, (bool, np.bool_)):
                return bool(x)
            if isinstance(x, (int, np.integer, float, np.floating)):
                # treat 1 or True-like as True
                return x == 1 or x is True
            return str(x).strip().lower() in TRUEY
    
    try:
        split_rules
    except NameError:
        def split_rules(val):
            if pd.isna(val):
                return []
            if isinstance(val, (list, tuple, set)):
                parts = list(val)
            else:
                # accept commas/semicolons
                parts = [p.strip() for chunk in str(val).split(';') for p in chunk.split(',')]
            parts = [p for p in parts if p and p.lower() != "nan"]
            # de-dupe preserve order
            seen, out = set(), []
            for p in parts:
                if p not in seen:
                    seen.add(p); out.append(p)
            return out
    
    
    
    # 1) Flag: True if either column is true
    one_3["rft_by_engine_7"] = (
        one_3["rft_by_engine"].map(to_bool) | one_3["new_model_output"].map(to_bool)
    )
    
    # (Optional) whitelist to avoid free-text becoming a "rule"
    ALLOWED_RULES = None  # e.g., {"invoice_before_po_flag", "invoice_more_than_po_flag"}
    
    def merge_rules_guarded(row):
        # Build candidate rules from both columns
        parts = split_rules(row.get("new_rule_summary")) + split_rules(row.get("rule_ids"))
        parts = list(dict.fromkeys(parts))  # de-dupe, keep order
        if ALLOWED_RULES is not None:
            parts = [p for p in parts if p in ALLOWED_RULES]
    
        # ENFORCE: if flag is False -> no rules
        if not to_bool(row.get("rft_by_engine_7", False)):
            return []
    
        return parts
    
    # Build final columns with gating
    one_3["rule_ids_7_list"] = one_3.apply(merge_rules_guarded, axis=1)
    one_3["rule_ids_7"] = one_3["rule_ids_7_list"].apply(lambda lst: ",".join(lst) if lst else np.nan)
    
    
    # Step 1: Get unique rule IDs from all rows (comma-separated strings)
    rule_sets = one_3['rule_ids_7'].dropna().apply(lambda x: [r.strip() for r in x.split(',')])
    unique_rules = sorted(set(r for sublist in rule_sets for r in sublist))
    
    # Step 2: Create columns for each rule ID with 1 if present, else 0
    for rule in unique_rules:
        one_3[rule] = one_3['rule_ids_7'].apply(lambda x: int(rule in x.split(',')) if pd.notna(x) else 0)
    
    #Invoice before PO date
    #Invoice Exceeds PO
    # Final rule to sub-risk mapping
    def get_sub_risks(row):
        rule_to_subrisk = {"P2P02067": "Price Variance Risk",
            "P2P02068": "Price Variance Risk",
            "P2P02070": "Split PO",
            "P2P02072": "Split PO",
            "invoice_more_than_po_flag": "Invoice Exceeds PO",
            "po_to_blocked_vendor":"PO to block vendor",
            "po_to_new_vendor_gt_tolerance":"PO to new vendor > tolerance level",
            "pan4P_amt_ge_1cr_flag":"PO to Non Company Vendors",
            "missing_tax_id_flag":"PO to Vendor with missing KYC",
                           
                           
            #rule_1": "Invoice before PO date",
        }
        risks = {rule_to_subrisk[rule] for rule in rule_to_subrisk if row.get(rule, 0) == 1}
        return list(risks) if risks else ["No Risk"]
    def sub_risk(df):    
        
        # Step 1: Assign Main Risk Scenario
        df["main_risk_scenario"] = df["rft_by_engine_7"].apply(
            lambda x: "Procurement Risk" if x else "No Risk"
        )
        
        # Step 2: Generate clean list of sub risks
        #def get_sub_risks(row):
            #risks = {rule_to_subrisk[rule] for rule in rule_to_subrisk if row.get(rule, 0) == 1}
            #return list(risks) if risks else ["No Risk"]
        
        df["sub_risks"] = df.apply(get_sub_risks, axis=1)
        
        return df
    
    invoice_data_2=sub_risk(one_3)
    #invoice_data_2.head()
    
    
    
    po_data_2=invoice_data_2[['purch_doc_no_src_po', 'purch_doc_item_no_src_po', 'pr_no_src_po',
           'pr_item_no_src_po', 'requester_name_src_po', 'doc_change_date_src_po',
           'company_code_src_po', 'p2o_unit_conv_denom_src_po',
           'o2b_unit_conv_denom_src_po', 'p2o_unit_conv_num_src_po',
           'o2b_unit_conv_num_src_po', 'po_item_del_flag_src_po',
           'gr_indicator_src_po', 'gr_invoice_verif_flag_src_po',
           'gross_val_po_curr_src_po', 'inv_receipt_indicator_src_po',
           'material_type_src_po', 'material_no_src_po', 'matl_group_src_po',
           'net_price_doc_curr_src_po', 'net_val_po_curr_src_po',
           'order_uom_src_po', 'quantity_src_po', 'plant_src_po',
           'short_text_src_po', 'target_qty_src_po',
           'outline_agrmt_tgt_val_doc_curr_src_po', 'reqmt_tracking_no_src_po',
           'principal_purch_agrmt_item_no_src_po', 'base_id_src_po',
           'purch_doc_no_hpd_po', 'vendor_or_creditor_acct_no_hpd_po',
           'purch_doc_category_hpd_po', 'company_code_hpd_po', 'currency_hpd_po',
           'object_created_by_hpd_po', 'doc_change_date_hpd_po',
           'control_indicator_hpd_po', 'purch_doc_date_hpd_po',
           'principal_purch_agrmt_no_hpd_po', 'exchange_rate_hpd_po',
           'purch_group_hpd_po', 'purch_org_hpd_po', 'pymnt_terms_hpd_po',
           'processing_status_hpd_po', 'doc_release_incompl_flag_hpd_po',
           'release_indicator_hpd_po', 'release_status_hpd_po',
           'rfq_status_hpd_po', 'resp_vendor_salesperson_hpd_po',
                                  'on_release_total_value_hpd_po', 'purch_doc_type_hpd_po',
           'vendor_or_creditor_acct_no', 'country_code', 'vendor_name_1',
           'vendor_name_2', 'vendor_name_3', 'vendor_name_4', 'city',
           'postal_code', 'street_address', 'item_manual_addr_no',
           'matchcode_search_term_1', 'record_creation_dt', 'object_created_by',
           'vendor_acct_group', 'tax_no_1', 'vendor_telephone_no',
           'second_telephone_no', 'tax_no_3', 'tax_no_5', 'central_deletion_flag',
           'central_purch_blk_flag', 'central_posting_blk_flag',
           'data_transfer_status', 'invoice_more_than_po_flag', 
           'po_to_blocked_vendor', 'purch_doc_within_30days', 'po_value_inr',
           'po_gt_1cr_flag', 'po_to_new_vendor_gt_tolerance', 'pan_extracted',
           'pan4P_amt_ge_1cr_flag', 'missing_tax_id_flag','rft_by_engine_7','main_risk_scenario', 'sub_risks']]
    
    #po_data_2.head()po_invoice
    #po_data_2.to_pickle('po_data_02_10.pkl')
    df_final=po_data_2.copy()
    #return po_data_2

    return df_final,df_final_invoice