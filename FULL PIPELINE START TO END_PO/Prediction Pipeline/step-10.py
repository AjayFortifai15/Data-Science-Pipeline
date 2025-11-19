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

warnings.filterwarnings("ignore", category=FutureWarning)


import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import math

def update_after_llm_explanation(df_updated_rft,updated_llm):
    doc_1=df_updated_rft.merge(updated_llm[['base_id_src_po','llm_refined_explanation']],on='base_id_src_po',how='outer')
    doc_2=doc_1[doc_1['purch_doc_no_src_po'].notna()].copy()
    
    doc_3=doc_2[['purch_doc_no_src_po', 'purch_doc_item_no_src_po', 'pr_no_src_po',
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
           'data_transfer_status',
           'gbt_fraud_score', 'gbt_model_flag', 'ae_fraud_score',
           'ae_predicted_flag', 'iso_fraud_score', 'iso_predicted_flag',
           'risk_score','updated_risk_level', 'model_flag','updated_main_risk_scenario', 'predicted_risks', 
                 'updated_sub_risk_1', 'updated_sub_risk_2','updated_sub_risk_3','impact', 'updated_impact_1','updated_impact_2', 'updated_impact_3',
          'updated_impact_4','updated_impact_5', 'llm_refined_explanation',]]
    
    doc_3["llm_refined_explanation"] = doc_3["llm_refined_explanation"].fillna(
        "There is No Risk for this line item."
    )
    
    new_names = {
        "updated_risk_level": "risk_level",
        "updated_main_risk_scenario":"main_risk_scenario","updated_sub_risk_1":'sub_risk_1',"updated_sub_risk_2":'sub_risk_2',"updated_sub_risk_3":'sub_risk_3',
        'updated_impact_1':'impact_1','updated_impact_2':'impact_2','updated_impact_3':'impact_3','updated_impact_4':'impact_4','updated_impact_5':'impact_5',
    }
    doc_4 =doc_3.rename(columns=new_names, errors="ignore")
    
    
    ##updating predicted_risks
    mask = (
        doc_4["main_risk_scenario"].astype(str).str.strip().str.casefold() == "no risk"
    )
    doc_4["predicted_risks"] = np.where(mask, "(No Risk)", doc_4["predicted_risks"])
    
    #updating impact
    
    mask = (
        doc_4["main_risk_scenario"].astype(str).str.strip().str.casefold() == "no risk"
    )
    doc_4["impact"] = np.where(mask, "[None]", doc_4["impact"])
    doc_4.to_excel('data_before_db_update.xlsx',index=False)

    #return doc_4
