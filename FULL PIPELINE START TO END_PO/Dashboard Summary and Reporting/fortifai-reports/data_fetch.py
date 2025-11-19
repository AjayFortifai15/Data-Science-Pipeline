# data_fetch.py
import os
import pandas as pd
import psycopg2

PG_DSN_SOURCE = os.getenv("PG_DSN_SOURCE")  # DB that holds your PO data

def fetch_po_data(date_from: str, date_to: str) -> pd.DataFrame:
    """
    Pulls the data window requested by the UI.
    Ensure it includes a usable date column (e.g., 'purch_doc_date_hpd_po').
    """
    if not PG_DSN_SOURCE:
        raise RuntimeError("PG_DSN_SOURCE not set")
    sql = """
    SELECT *
    FROM po_invoice_vendor  -- TODO: your real table or materialized view
    WHERE purch_doc_date_hpd_po >= %s::date
      AND purch_doc_date_hpd_po <= %s::date
    """
    with psycopg2.connect(PG_DSN_SOURCE) as conn, conn.cursor() as cur:
        cur.execute(sql, (date_from, date_to))
        colnames = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame.from_records(rows, columns=colnames)
