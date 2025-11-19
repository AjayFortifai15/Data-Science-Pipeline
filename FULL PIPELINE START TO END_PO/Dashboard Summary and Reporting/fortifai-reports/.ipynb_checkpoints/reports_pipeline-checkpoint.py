# reports_pipeline.py
from __future__ import annotations
import os, re, json, uuid
from typing import Iterable, Literal, Optional, Dict, Any, List
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import psycopg2

# ------------------------------
# ENV / CONFIG
# ------------------------------
BASE_OUT            = os.getenv("BASE_OUT", "generate_reports")
TZ_NAME             = os.getenv("TZ_NAME", "Asia/Kolkata")
RUN_DATE            = pd.Timestamp.now(tz=TZ_NAME).date()
DATE_COL            = None  # force a date column name if you need
BLOB_CONTAINER      = os.getenv("BLOB_CONTAINER", "reports")
AZURE_STORAGE_CONN  = os.getenv("AZURE_STORAGE_CONN")  # required
PG_DSN              = os.getenv("PG_DSN")              # required for DB logging

# ------------------------------
# Excel helpers (auto select engine)
# ------------------------------
def _pick_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

ENGINE = _pick_engine()

def _writer_with_formats(path):
    w = pd.ExcelWriter(path, engine=ENGINE)
    if ENGINE == "xlsxwriter":
        wb = w.book
        fmt = {"money": wb.add_format({"num_format": "#,##0.00"}),
               "text":  wb.add_format({"num_format": "@"})}
        return w, fmt
    return w, None

def _autofit(writer, sheet_name, df, fmt):
    if ENGINE == "xlsxwriter":
        ws = writer.sheets[sheet_name]
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(df), len(df.columns)-1)
        for i, c in enumerate(df.columns):
            if c in {"document_id", "line_item", "cost_center"}:
                ws.set_column(i, i, 20, fmt["text"])
            elif c.lower() in {"value","impact value","impact_value","total_value","total_impact"}:
                ws.set_column(i, i, 16, fmt["money"])
            elif c.lower() in {"vendor","vendor_name"}:
                ws.set_column(i, i, 34)
            else:
                ws.set_column(i, i, 18)
    else:
        from openpyxl.utils import get_column_letter
        ws = writer.sheets[sheet_name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for i, _ in enumerate(df.columns, start=1):
            ws.column_dimensions[get_column_letter(i)].width = 18

# ------------------------------
# Date-column logic & periods
# ------------------------------
DATE_CANDIDATES = ['purch_doc_date_hpd_po']

def _choose_date_col(df):
    if DATE_COL and DATE_COL in df.columns:
        return DATE_COL
    for c in DATE_CANDIDATES:
        if c in df.columns:
            as_dt = pd.to_datetime(df[c], errors="coerce")
            if as_dt.notna().any():
                return c
    raise ValueError("No usable date column found; set DATE_COL or update DATE_CANDIDATES.")

def _clamp_to_six_months(df, dcol):
    six_months_ago = RUN_DATE - relativedelta(months=6)
    data_min = pd.to_datetime(df[dcol], errors="coerce").min().date()
    start = max(six_months_ago, data_min)
    return start, RUN_DATE

def _iter_daily(start: date, end: date):
    d = start
    while d <= end:
        yield d, d
        d += timedelta(days=1)

def _iter_weekly(start: date, end: date):
    w_start = start - timedelta(days=start.weekday())  # Monday
    while w_start <= end:
        w_end = min(w_start + timedelta(days=6), end)
        s = max(w_start, start)
        yield s, w_end
        w_start += timedelta(days=7)

def _iter_monthly(start: date, end: date):
    m_start = date(start.year, start.month, 1)
    while m_start <= end:
        next_m = (pd.Timestamp(m_start) + relativedelta(months=1)).date()
        m_end = min(next_m - timedelta(days=1), end)
        s = max(m_start, start)
        yield s, m_end
        m_start = next_m

def _weekly_folder_name(s: date, e: date):
    iso_year, iso_week, _ = s.isocalendar()
    return f"{iso_year}-W{iso_week:02d}_{s:%Y-%m-%d}_to_{e:%Y-%m-%d}"

def _monthly_folder_name(s: date, e: date):
    return f"{s:%Y-%m}"

# ------------------------------
# Report implementations (YOUR logic)
# ------------------------------
def report_top10_vendors_and_txns(df_in, out_path):
    COLUMNS = {
        "vendor_id":   "vendor_or_creditor_acct_no_hpd_po",
        "vendor_name": "vendor_name_1",
        "document_id": "base_id_src_po",
        "risk_level":  "risk_level",
        "cost_center": "plant_src_po",
        "total_value": "on_release_total_value_hpd_po",
        "impact_value":"net_val_po_curr_src_po",
        "main_scenario": "main_risk_scenario",
    }
    df = df_in.copy()
    present = {k:v for k,v in COLUMNS.items() if v in df.columns}
    df = df.rename(columns={present[k]:k for k in present})
    required = ["vendor_id","vendor_name","document_id","risk_level","total_value"]
    if any(c not in df.columns for c in required): return False
    for opt in ["vendor_category","cost_center","impact_value","process","main_scenario","del_flag"]:
        if opt not in df.columns: df[opt] = pd.NA
    df = df[~df["del_flag"].fillna("").astype(str).str.upper().eq("L")]
    if "process" in df.columns and df["process"].notna().any():
        df = df[df["process"].fillna("").str.upper().eq("P2P")]
    elif "main_scenario" in df.columns and df["main_scenario"].notna().any():
        df = df[df["main_scenario"].fillna("").str.lower().eq("procurement risk")]
    rs = df["risk_level"].fillna("").astype(str).str.strip()
    df_risky = df[(~rs.str.lower().eq("no risk")) & (rs != "")]
    if df_risky.empty: return False
    if "impact_value" not in df_risky.columns: df_risky["impact_value"] = df_risky["total_value"]
    df_risky["impact_value"] = pd.to_numeric(df_risky["impact_value"], errors="coerce").fillna(0.0)
    df_risky["total_value"]  = pd.to_numeric(df_risky["total_value"],  errors="coerce").fillna(0.0)
    agg = (df_risky.groupby(["vendor_id","vendor_name","vendor_category"], dropna=False)
                  .agg(risky_txn_rows=("document_id","count"),
                       distinct_docs=("document_id","nunique"),
                       total_risky_value=("total_value","sum"),
                       total_impact=("impact_value","sum")).reset_index())
    top10 = agg.sort_values(["risky_txn_rows","total_impact"], ascending=[False,False]).head(10).reset_index(drop=True)
    if top10.empty: return False
    top10_vendor_ids = set(top10["vendor_id"].dropna().astype(str))
    detail_cols = [c for c in ["vendor_id","vendor_name","vendor_category","document_id","risk_level",
                               "cost_center","total_value","impact_value"] if c in df_risky.columns]
    detail = (df_risky[df_risky["vendor_id"].astype(str).isin(top10_vendor_ids)][detail_cols]
                    .sort_values(["vendor_id","impact_value"], ascending=[True,False]).reset_index(drop=True))
    writer, fmt = _writer_with_formats(out_path)
    top10.to_excel(writer, "Top10_Vendors", index=False)
    detail.to_excel(writer, "Risky_Txns_Top10", index=False)
    _autofit(writer, "Top10_Vendors", top10, fmt)
    _autofit(writer, "Risky_Txns_Top10", detail, fmt)
    writer.close()
    return True

def report_highrisk_txns(df_in, out_path):
    COLUMNS = {
        "document_id": 'purch_doc_no_src_po',
        "risk_level": "risk_level",
        "cost_center": "plant_src_po",
        "vendor_id": "vendor_or_creditor_acct_no_hpd_po",
        "vendor_name":"vendor_name_1",
        "line_item": 'purch_doc_item_no_src_po',
        "total_value":"on_release_total_value_hpd_po",
        "impact_value":"net_val_po_curr_src_po",
        "main_scenario": "main_risk_scenario",
    }
    RISKY = {"High Risk","Very High Risk","Needs Validation"}
    df = df_in.copy()
    present = {k:v for k,v in COLUMNS.items() if v in df.columns}
    df = df.rename(columns={present[k]:k for k in present})
    for opt in ["process","main_scenario","del_flag","impact_value","vendor_id","vendor_name","line_item","value"]:
        if opt not in df.columns: df[opt] = pd.NA
    df = df[~df["del_flag"].fillna("").astype(str).str.upper().eq("L")]
    if df["process"].notna().any():
        df = df[df["process"].fillna("").str.upper().eq("P2P")]
    elif df["main_scenario"].notna().any():
        df = df[df["main_scenario"].fillna("").str.lower().eq("procurement risk")]
    rs = df["risk_level"].fillna("").astype(str).str.strip()
    df_hr = df[rs.str.lower().isin({x.lower() for x in RISKY})].copy()
    if df_hr.empty: return False
    for c in ["impact_value","value"]:
        if c in df_hr.columns: df_hr[c] = pd.to_numeric(df_hr[c], errors="coerce")
    df_hr["Value"] = df_hr["impact_value"].where(df_hr["impact_value"].notna(), df_hr["value"]).fillna(0.0)
    df_hr["Vendor"] = df_hr.get("vendor_id").astype("string").fillna("")
    vn = df_hr.get("vendor_name")
    if vn is not None:
        df_hr["Vendor"] = df_hr["Vendor"].mask(df_hr["Vendor"].eq(""), vn.fillna("").astype(str))
        both = df_hr["Vendor"].ne("") & vn.notna() & (vn.astype(str) != "")
        df_hr.loc[both, "Vendor"] = df_hr.loc[both, "Vendor"] + " – " + vn.loc[both].astype(str)
    for tcol in ["document_id","line_item","cost_center"]:
        if tcol in df_hr.columns: df_hr[tcol] = df_hr[tcol].astype("string")
    out_cols = [c for c in ["document_id","risk_level","cost_center","Vendor","line_item","Value"] if c in df_hr.columns]
    top10 = df_hr.sort_values("Value", ascending=False).loc[:, out_cols].head(10).reset_index(drop=True)
    all_tx = df_hr.loc[:, out_cols].sort_values(["risk_level","Value"], ascending=[True,False]).reset_index(drop=True)
    if top10.empty and all_tx.empty: return False
    writer, fmt = _writer_with_formats(out_path)
    top10.to_excel(writer, "Top10_HighRisk_Txns", index=False)
    all_tx.to_excel(writer, "All_HighRisk_Txns", index=False)
    _autofit(writer, "Top10_HighRisk_Txns", top10, fmt)
    _autofit(writer, "All_HighRisk_Txns", all_tx, fmt)
    writer.close()
    return True

def report_split_po(df_in, out_path):
    import re
    COLUMNS = {
        "document_id":"purch_doc_no_src_po","risk_level":"risk_level","cost_center":"plant_src_po",
        "vendor_id":"vendor_or_creditor_acct_no_hpd_po","vendor_name":"vendor_name_1",
        "line_item":"purch_doc_item_no_src_po","value":"on_release_total_value_hpd_po",
        "impact_value":"net_val_po_curr_src_po","main_scenario":"main_risk_scenario",
        "sub_risk_1":"sub_risk_1","sub_risk_2":"sub_risk_2",
    }
    pat = re.compile(r"\bsplit[\s_-]*po\b", re.I)
    df = df_in.copy()
    present = {k:v for k,v in COLUMNS.items() if v in df.columns}
    df = df.rename(columns={present[k]:k for k in present})
    for opt in ["process","main_scenario","del_flag","vendor_id","vendor_name","line_item","value","impact_value","sub_risk_1","sub_risk_2"]:
        if opt not in df.columns: df[opt] = pd.NA
    df = df[~df["del_flag"].fillna("").astype(str).str.upper().eq("L")]
    if df["process"].notna().any():
        df = df[df["process"].fillna("").str.upper().eq("P2P")]
    elif df["main_scenario"].notna().any():
        df = df[df["main_scenario"].fillna("").str.lower().eq("procurement risk")]
    def _has_split(x):
        if pd.isna(x): return False
        if isinstance(x,(list,tuple,set)): return any(bool(pat.search(str(v))) for v in x)
        return bool(pat.search(str(x)))
    cols = [c for c in ["sub_risk_1","sub_risk_2"] if c in df.columns]
    if not cols: return False
    mask = False
    for c in cols: mask = mask | df[c].apply(_has_split)
    df = df[mask].copy()
    if df.empty: return False
    df["impact_value"] = pd.to_numeric(df.get("impact_value"), errors="coerce").fillna(0.0)
    for t in ["document_id","line_item","cost_center"]:
        if t in df.columns: df[t] = df[t].astype("string")
    out_cols = [c for c in ["document_id","risk_level","cost_center","vendor_id","vendor_name","line_item","impact_value"] if c in df.columns]
    df["Vendor"] = df.get("vendor_id").astype("string").fillna("")
    vn = df.get("vendor_name")
    if vn is not None:
        df["Vendor"] = df["Vendor"].mask(df["Vendor"].eq(""), vn.fillna("").astype(str))
        both = df["Vendor"].ne("") & vn.notna() & (vn.astype(str) != "")
        df.loc[both, "Vendor"] = df.loc[both, "Vendor"] + " – " + vn.loc[both].astype(str)
    out_cols = [c for c in ["document_id","risk_level","cost_center","Vendor","line_item","impact_value"] if c in df.columns]
    sheet = df.loc[:, out_cols].sort_values("impact_value", ascending=False)
    writer, fmt = _writer_with_formats(out_path)
    sheet.to_excel(writer, "SplitPO_Txns", index=False)
    _autofit(writer, "SplitPO_Txns", sheet, fmt)
    writer.close()
    return True

def report_price_variance(df_in, out_path):
    import re
    COLUMNS = {
        "document_id":"purch_doc_no_src_po","risk_level":"risk_level","cost_center":"plant_src_po",
        "vendor_id":"vendor_or_creditor_acct_no_hpd_po","vendor_name":"vendor_name_1",
        "line_item":"purch_doc_item_no_src_po","value":"on_release_total_value_hpd_po",
        "impact_value":"net_val_po_curr_src_po","main_scenario":"main_risk_scenario",
        "sub_risk_1":"sub_risk_1","sub_risk_2":"sub_risk_2",
    }
    pat = re.compile(r"\bprice[\s_-]*variance\b", re.I)
    df = df_in.copy()
    present = {k:v for k,v in COLUMNS.items() if v in df.columns}
    df = df.rename(columns={present[k]:k for k in present})
    for opt in ["process","main_scenario","del_flag","vendor_id","vendor_name","line_item","value","impact_value","sub_risk_1","sub_risk_2"]:
        if opt not in df.columns: df[opt] = pd.NA
    df = df[~df["del_flag"].fillna("").astype(str).str.upper().eq("L")]
    if df["process"].notna().any():
        df = df[df["process"].fillna("").str.upper().eq("P2P")]
    elif df["main_scenario"].notna().any():
        df = df[df["main_scenario"].fillna("").str.lower().eq("procurement risk")]
    def _has_pv(x):
        if pd.isna(x): return False
        if isinstance(x,(list,tuple,set)): return any(bool(pat.search(str(v))) for v in x)
        return bool(pat.search(str(x)))
    cols = [c for c in ["sub_risk_1","sub_risk_2"] if c in df.columns]
    if not cols: return False
    mask = False
    for c in cols: mask = mask | df[c].apply(_has_pv)
    df = df[mask].copy()
    if df.empty: return False
    df["impact_value"] = pd.to_numeric(df.get("impact_value"), errors="coerce").fillna(0.0)
    for t in ["document_id","line_item","cost_center"]:
        if t in df.columns: df[t] = df[t].astype("string")
    df["Vendor"] = df.get("vendor_id").astype("string").fillna("")
    vn = df.get("vendor_name")
    if vn is not None:
        df["Vendor"] = df["Vendor"].mask(df["Vendor"].eq(""), vn.fillna("").astype(str))
        both = df["Vendor"].ne("") & vn.notna() & (vn.astype(str) != "")
        df.loc[both, "Vendor"] = df.loc[both, "Vendor"] + " – " + vn.loc[both].astype(str)
    out_cols = [c for c in ["document_id","risk_level","cost_center","Vendor","line_item","impact_value"] if c in df.columns]
    sheet = df.loc[:, out_cols].sort_values("impact_value", ascending=False)
    writer, fmt = _writer_with_formats(out_path)
    sheet.to_excel(writer, "PriceVariance_Txns", index=False)
    _autofit(writer, "PriceVariance_Txns", sheet, fmt)
    writer.close()
    return True

# ------------------------------
# Blob uploader
# ------------------------------
class BlobUploader:
    def __init__(self, conn_str: str, container: str):
        if not conn_str:
            raise RuntimeError("AZURE_STORAGE_CONN not set")
        from azure.storage.blob import BlobServiceClient
        self.service = BlobServiceClient.from_connection_string(conn_str)
        self.cc = self.service.get_container_client(container)
        try: self.cc.create_container()
        except Exception: pass
        self.container = container

    def upload_file(self, local_path: str, blob_path: str, content_type: Optional[str] = None) -> Dict[str, Any]:
        from azure.storage.blob import ContentSettings
        with open(local_path, "rb") as f: data = f.read()
        cs = ContentSettings(content_type=content_type) if content_type else None
        self.cc.upload_blob(name=blob_path, data=data, overwrite=True, content_settings=cs)
        return {"blob_path": blob_path, "bytes": len(data)}

# ------------------------------
# Registry
# ------------------------------
ReportType  = Literal["top10_vendors","highrisk_txns","split_po","price_variance"]
Granularity = Literal["Daily","Weekly","Monthly"]

REPORTS = {
    "top10_vendors":  ("Top 10 high-risk vendors by risky transactions.xlsx", report_top10_vendors_and_txns, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "highrisk_txns":  ("Top 10 high-risk transactions.xlsx",                 report_highrisk_txns,            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "split_po":       ("Split PO Report.xlsx",                               report_split_po,                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "price_variance": ("Price Variance Risk Report.xlsx",                    report_price_variance,           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
}

def _ensure_dirs():
    for sub in ["Daily","Weekly","Monthly"]:
        os.makedirs(os.path.join(BASE_OUT, sub), exist_ok=True)

def _blob_prefix(tenant_id: str, job_id: str) -> str:
    now = pd.Timestamp.utcnow()
    return f"{tenant_id}/{now:%Y/%m/%d}/{job_id}"

def _subfolder(g: Granularity, s: date, e: date) -> str:
    if g == "Daily": return f"{s:%Y-%m-%d}"
    if g == "Weekly": return _weekly_folder_name(s, e)
    if g == "Monthly": return _monthly_folder_name(s, e)
    return f"{s:%Y-%m-%d}_to_{e:%Y-%m-%d}"

def _iter_periods(g: Granularity, s: date, e: date):
    if g == "Daily":   yield from _iter_daily(s, e)
    if g == "Weekly":  yield from _iter_weekly(s, e)
    if g == "Monthly": yield from _iter_monthly(s, e)

# ------------------------------
# DB persistence
# ------------------------------
def _db():
    if not PG_DSN: raise RuntimeError("PG_DSN not set")
    return psycopg2.connect(PG_DSN)

def _persist(manifest: Dict[str, Any]) -> None:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
        insert into report_jobs (
          job_id, tenant_id, report_types, granularities, date_from, date_to, blob_prefix,
          status, created_at, run_at_utc, total_artifacts, notes
        ) values (
          %(job_id)s,%(tenant_id)s,%(report_types)s,%(granularities)s,%(date_from)s,%(date_to)s,
          %(blob_prefix)s,%(status)s, now(), %(run_at_utc)s, %(total_artifacts)s, %(notes)s
        )
        on conflict (job_id) do update set
          report_types=excluded.report_types,
          granularities=excluded.granularities,
          date_from=excluded.date_from,
          date_to=excluded.date_to,
          blob_prefix=excluded.blob_prefix,
          status=excluded.status,
          run_at_utc=excluded.run_at_utc,
          total_artifacts=excluded.total_artifacts,
          notes=excluded.notes
        """, {
            "job_id": manifest["job_id"],
            "tenant_id": manifest["tenant_id"],
            "report_types": json.dumps(manifest["filters"]["report_types"]),
            "granularities": json.dumps(manifest["filters"]["granularities"]),
            "date_from": manifest["filters"]["date_from"],
            "date_to": manifest["filters"]["date_to"],
            "blob_prefix": manifest["blob_prefix"],
            "status": manifest["status"],
            "run_at_utc": manifest["run_at_utc"],
            "total_artifacts": len(manifest["artifacts"]),
            "notes": None
        })
        for a in manifest["artifacts"]:
            cur.execute("""
            insert into report_artifacts (
              job_id, report_type, granularity, period_start, period_end,
              blob_path, content_type, rows_generated, bytes_generated
            ) values (
              %(job_id)s, %(report_type)s, %(granularity)s, %(period_start)s, %(period_end)s,
              %(blob_path)s, %(content_type)s, %(rows_generated)s, %(bytes_generated)s
            )
            on conflict (job_id, report_type, granularity, period_start, period_end, blob_path)
            do update set rows_generated=excluded.rows_generated, bytes_generated=excluded.bytes_generated
            """, {
                "job_id": manifest["job_id"],
                "report_type": a["report_type"],
                "granularity": a["granularity"],
                "period_start": a["period_start"],
                "period_end": a["period_end"],
                "blob_path": a["blob_path"],
                "content_type": a["content_type"],
                "rows_generated": a.get("rows"),
                "bytes_generated": a.get("bytes"),
            })

# ------------------------------
# Core runner exposed to API
# ------------------------------
def run_pipeline(
    po_data: pd.DataFrame,
    tenant_id: str,
    report_types: Iterable[ReportType],
    granularities: Iterable[Granularity] = ("Daily","Weekly","Monthly"),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    job_id: Optional[str] = None,
    upload_to_blob: bool = True,
    persist_to_db: bool = True,
) -> Dict[str, Any]:

    if po_data is None or po_data.empty:
        raise ValueError("po_data is empty")
    dcol = _choose_date_col(po_data)
    df = po_data.copy()
    df["__report_date"] = pd.to_datetime(df[dcol], errors="coerce")
    df = df[df["__report_date"].notna()]
    if df.empty: raise ValueError(f"No valid dates in '{dcol}'")

    if date_from and date_to:
        start, end = pd.to_datetime(date_from).date(), pd.to_datetime(date_to).date()
    else:
        start, end = _clamp_to_six_months(df, "__report_date")

    os.makedirs(BASE_OUT, exist_ok=True); _ensure_dirs()
    job_id = job_id or str(uuid.uuid4())
    prefix_rel = _blob_prefix(tenant_id, job_id)
    blob_prefix = f"{BLOB_CONTAINER}/{prefix_rel}"

    # 1) Generate locally
    local_artifacts: List[Dict[str, Any]] = []
    for g in granularities:
        for s, e in _iter_periods(g, start, end):
            # slice
            mask = (df["__report_date"].dt.date >= s) & (df["__report_date"].dt.date <= e)
            df_slice = df.loc[mask].copy()
            if df_slice.empty: continue
            subfolder = (
                f"Daily/{s:%Y-%m-%d}" if g=="Daily"
                else f"Monthly/{_monthly_folder_name(s,e)}" if g=="Monthly"
                else f"Weekly/{_weekly_folder_name(s,e)}"
            )
            folder = os.path.join(BASE_OUT, subfolder)
            os.makedirs(folder, exist_ok=True)

            for rt in report_types:
                if rt not in REPORTS: continue
                filename, fn, _ct = REPORTS[rt]
                out_path = os.path.join(folder, filename)
                ok = fn(df_slice, out_path)
                if ok:
                    local_artifacts.append({
                        "report_type": rt,
                        "granularity": g,
                        "period_start": f"{s:%Y-%m-%d}",
                        "period_end": f"{e:%Y-%m-%d}",
                        "local_path": out_path,
                        "content_type": _ct,
                        "rows": None,
                    })

    # 2) Upload to blob
    artifacts: List[Dict[str, Any]] = []
    uploader = BlobUploader(AZURE_STORAGE_CONN, BLOB_CONTAINER) if upload_to_blob else None
    for a in local_artifacts:
        lp, g, rt = a["local_path"], a["granularity"], a["report_type"]
        if g == "Daily":
            rel = f"{prefix_rel}/Daily/{a['period_start']}/{os.path.basename(lp)}"
        elif g == "Monthly":
            rel = f"{prefix_rel}/Monthly/{a['period_start'][:7]}/{os.path.basename(lp)}"
        else:
            rel = f"{prefix_rel}/Weekly/{a['period_start']}_to_{a['period_end']}/{os.path.basename(lp)}"

        if uploader:
            meta = uploader.upload_file(lp, rel, content_type=a["content_type"])
            artifacts.append({
                "report_type": rt, "granularity": g,
                "period_start": a["period_start"], "period_end": a["period_end"],
                "blob_path": f"{BLOB_CONTAINER}/{rel}",
                "content_type": a["content_type"],
                "rows": a.get("rows"), "bytes": meta["bytes"],
            })
        else:
            artifacts.append({
                "report_type": rt, "granularity": g,
                "period_start": a["period_start"], "period_end": a["period_end"],
                "local_path": lp, "content_type": a["content_type"], "rows": a.get("rows")
            })

    # 3) Manifest
    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "tenant_id": tenant_id,
        "run_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "filters": {
            "date_from": f"{start:%Y-%m-%d}",
            "date_to": f"{end:%Y-%m-%d}",
            "granularities": list(granularities),
            "report_types": list(report_types),
        },
        "blob_prefix": blob_prefix,
        "artifacts": artifacts,
        "status": "SUCCEEDED",
    }

    # 4) Upload manifest.json
    if uploader:
        from azure.storage.blob import ContentSettings
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        self_rel = f"{prefix_rel}/manifest.json"
        uploader.cc.upload_blob(name=self_rel, data=manifest_bytes, overwrite=True,
                                content_settings=ContentSettings(content_type="application/json"))
        manifest["manifest_blob_path"] = f"{BLOB_CONTAINER}/{self_rel}"

    # 5) DB logging
    if persist_to_db: _persist(manifest)
    return manifest
