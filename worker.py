"""Background worker — runs mining and checking jobs off the request path.

Run it with:
    rq worker lid-jobs --url $REDIS_URL

api.py enqueues jobs here instead of doing the work inline. This is what
lets a slow re-derivation search (bounded but not free -- see
lid/miner.py's complexity notes) happen without holding an HTTP
connection open, and gives you a natural place to add retries,
dead-letter handling, and horizontal worker scaling later.
"""
from __future__ import annotations
import io

import pandas as pd

from lid import mine_invariants, check_violations, re_derive_formula, explain_drift
from lid.storage import (
    get_engine, get_session, get_tenant_by_key, save_fingerprint_version,
    get_latest_fingerprint, record_violations,
)


def mine_job(api_key: str, table_name: str, csv_bytes: bytes) -> dict:
    """Job: mine a fresh fingerprint from an uploaded historical CSV."""
    engine = get_engine()
    session = get_session(engine)
    tenant = get_tenant_by_key(session, api_key)
    if tenant is None:
        return {"error": "unknown api key"}

    df = pd.read_csv(io.BytesIO(csv_bytes))
    invariants = mine_invariants(df)
    fp = save_fingerprint_version(session, tenant, table_name, invariants)
    return {"table": table_name, "version": fp.version, "invariant_count": len(invariants)}


def check_job(api_key: str, table_name: str, csv_bytes: bytes, batch_label: str = "unlabeled") -> dict:
    """Job: check a new batch against the latest stored fingerprint,
    re-deriving and explaining any violations found."""
    engine = get_engine()
    session = get_session(engine)
    tenant = get_tenant_by_key(session, api_key)
    if tenant is None:
        return {"error": "unknown api key"}

    fp = get_latest_fingerprint(session, tenant, table_name)
    if fp is None:
        return {"error": f"no fingerprint stored yet for '{table_name}' -- mine one first"}

    df = pd.read_csv(io.BytesIO(csv_bytes))
    violations = check_violations(df, fp.invariants)

    numeric_cols = list(df.select_dtypes(include="number").columns)
    explained = []
    for v in violations:
        if v["type"] != "linear_arithmetic":
            explained.append({"invariant": v, "explanation": f"{v['type']} invariant broke on {v}."})
            continue
        result = re_derive_formula(df, v, numeric_cols)
        explanation = explain_drift(v["formula"], result["formula"], v["old_support"], v["new_support"])
        explained.append({"invariant": v, "re_derived": result, "explanation": explanation})

    record_violations(session, fp, batch_label, explained)

    return {
        "table": table_name,
        "fingerprint_version": fp.version,
        "violation_count": len(violations),
        "violations": explained,
    }
