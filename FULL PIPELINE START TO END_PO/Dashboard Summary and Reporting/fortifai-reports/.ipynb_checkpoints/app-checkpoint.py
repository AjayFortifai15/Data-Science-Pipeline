# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import pandas as pd

from reports_pipeline import run_pipeline, ReportType, Granularity
from data_fetch import fetch_po_data

# -------- FastAPI setup --------
app = FastAPI(title="FortifAI Reports API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),  # set to your UI origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant/org code")
    report_types: List[ReportType] = Field(..., min_items=1)
    granularities: List[Granularity] = Field(default=["Daily","Weekly","Monthly"])
    date_from: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: Optional[str]   = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    job_id: Optional[str] = None
    # If you already have data in UI/backend, you can POST it (optional)
    po_data_records: Optional[list] = None  # list of dict rows

@app.post("/run")
def run_reports(req: RunRequest):
    try:
        # 1) Get data: either posted (po_data_records) or fetch from DB
        if req.po_data_records is not None:
            po_data = pd.DataFrame(req.po_data_records)
            # If no date window given and data is posted, pipeline will clamp to last 6 months available
        else:
            # Require dates if we have to fetch
            if not (req.date_from and req.date_to):
                raise HTTPException(400, "date_from & date_to are required when po_data is not provided.")
            po_data = fetch_po_data(req.date_from, req.date_to)

        # 2) Run pipeline
        manifest = run_pipeline(
            po_data=po_data,
            tenant_id=req.tenant_id,
            report_types=req.report_types,
            granularities=req.granularities,
            date_from=req.date_from,
            date_to=req.date_to,
            job_id=req.job_id,
            upload_to_blob=True,
            persist_to_db=True,
        )
        return manifest
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
