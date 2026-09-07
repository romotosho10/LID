"""LID API — gateway layer.

Auth-checks the request, then enqueues mining/checking as a background
job instead of doing the work inline. The actual mine/check/re-derive
logic lives in worker.py, run as a separate process:

    uvicorn api:app --reload          # this process
    rq worker lid-jobs --url $REDIS_URL   # separate process, does the work

Poll GET /jobs/{job_id} for the result once a job completes.
"""
from __future__ import annotations
import os

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from redis import Redis
from rq import Queue
from rq.job import Job

from lid.storage import get_engine, get_session, init_db, get_or_create_tenant, get_tenant_by_key
import worker

app = FastAPI(title="LID API", version="0.2.0")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue("lid-jobs", connection=redis_conn)

engine = get_engine()
init_db(engine)


def _authenticate(x_api_key: str | None) -> str:
    """Every route requires an X-API-Key header. Keys are tenant-scoped --
    this is the multi-tenancy boundary. Swap for OAuth/JWT in production;
    this is intentionally the simplest thing that enforces isolation."""
    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key header")
    session = get_session(engine)
    tenant = get_tenant_by_key(session, x_api_key)
    if tenant is None:
        raise HTTPException(403, "Unknown API key")
    return x_api_key


@app.post("/tenants")
def create_tenant(name: str):
    """Dev-only convenience endpoint to mint a tenant + API key.
    In production this is an admin action or a signup flow, not an open route."""
    import secrets
    session = get_session(engine)
    api_key = secrets.token_hex(16)
    tenant = get_or_create_tenant(session, name, api_key)
    return {"tenant": tenant.name, "api_key": tenant.api_key}


@app.post("/fingerprints/{table_name}")
async def create_fingerprint(table_name: str, file: UploadFile = File(...), x_api_key: str = Header(None)):
    api_key = _authenticate(x_api_key)
    content = await file.read()
    job = queue.enqueue(worker.mine_job, api_key, table_name, content)
    return {"job_id": job.id, "status": "queued"}


@app.post("/check/{table_name}")
async def check_batch(table_name: str, file: UploadFile = File(...), batch_label: str = "unlabeled", x_api_key: str = Header(None)):
    api_key = _authenticate(x_api_key)
    content = await file.read()
    job = queue.enqueue(worker.check_job, api_key, table_name, content, batch_label)
    return {"job_id": job.id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str = Header(None)):
    _authenticate(x_api_key)
    job = Job.fetch(job_id, connection=redis_conn)
    if job.is_finished:
        return {"status": "finished", "result": job.result}
    if job.is_failed:
        return {"status": "failed", "error": str(job.exc_info)}
    return {"status": job.get_status()}


@app.get("/health")
def health():
    return {"status": "ok"}
