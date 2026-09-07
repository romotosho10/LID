"""Minimal FastAPI wrapper around LID — a starting point for the SaaS API.

Run locally:
    uvicorn api:app --reload

Endpoints:
    POST /fingerprints/{table_name}   — upload historical CSV, mine + store fingerprint
    POST /check/{table_name}          — upload new-batch CSV, check against stored fingerprint
    GET  /fingerprints/{table_name}   — inspect a stored fingerprint

This in-memory/on-disk version is for local dev only — see README's
"Making it a real SaaS API" section for what changes in production
(auth, per-tenant storage, a real DB, async checking, etc).
"""
from __future__ import annotations
import io
import json
import os

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException

from lid import mine_invariants, check_violations, re_derive_formula, explain_drift

app = FastAPI(title="LID API", version="0.1.0")

FINGERPRINT_DIR = "fingerprints"
os.makedirs(FINGERPRINT_DIR, exist_ok=True)


def _fp_path(table_name: str) -> str:
    safe = "".join(c for c in table_name if c.isalnum() or c in "-_")
    return os.path.join(FINGERPRINT_DIR, f"{safe}.json")


@app.post("/fingerprints/{table_name}")
async def create_fingerprint(table_name: str, file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    fingerprint = mine_invariants(df)
    with open(_fp_path(table_name), "w") as f:
        json.dump(fingerprint, f)
    return {"table": table_name, "invariant_count": len(fingerprint), "invariants": fingerprint}


@app.get("/fingerprints/{table_name}")
def get_fingerprint(table_name: str):
    path = _fp_path(table_name)
    if not os.path.exists(path):
        raise HTTPException(404, f"No fingerprint stored for '{table_name}'")
    with open(path) as f:
        return json.load(f)


@app.post("/check/{table_name}")
async def check_batch(table_name: str, file: UploadFile = File(...)):
    path = _fp_path(table_name)
    if not os.path.exists(path):
        raise HTTPException(404, f"No fingerprint stored for '{table_name}' — POST to /fingerprints/{table_name} first")
    with open(path) as f:
        fingerprint = json.load(f)

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    violations = check_violations(df, fingerprint)

    numeric_cols = list(df.select_dtypes(include="number").columns)
    explained = []
    for v in violations:
        if v["type"] != "linear_arithmetic":
            explained.append({"invariant": v, "explanation": f"{v['type']} invariant broke on {v}."})
            continue
        result = re_derive_formula(df, v, numeric_cols)
        explanation = explain_drift(v["formula"], result["formula"], v["old_support"], v["new_support"])
        explained.append({"invariant": v, "re_derived": result, "explanation": explanation})

    return {"table": table_name, "violation_count": len(violations), "violations": explained}


@app.get("/health")
def health():
    return {"status": "ok"}
