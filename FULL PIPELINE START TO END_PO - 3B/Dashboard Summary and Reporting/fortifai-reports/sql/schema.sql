-- Jobs summary
create table if not exists report_jobs (
  job_id uuid primary key,
  tenant_id text not null,
  report_types jsonb not null,
  granularities jsonb not null,
  date_from date not null,
  date_to date not null,
  blob_prefix text not null,
  status text not null default 'SUCCEEDED',
  created_at timestamptz not null default now(),
  run_at_utc timestamptz not null,
  total_artifacts int not null,
  notes text
);

-- Per-file artifacts
create table if not exists report_artifacts (
  job_id uuid not null references report_jobs(job_id) on delete cascade,
  report_type text not null,
  granularity text not null,
  period_start date not null,
  period_end date not null,
  blob_path text not null,
  content_type text not null,
  rows_generated bigint,
  bytes_generated bigint,
  created_at timestamptz not null default now(),
  primary key (job_id, report_type, granularity, period_start, period_end, blob_path)
);
