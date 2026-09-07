# LID
Mines implicit business-logic rules from data (e.g. net = gross - refunds - discounts) and catches when they silently break — no ML required. Personal project.




# LID — Learned Invariant Drift Detection

Mines implicit business-logic rules (e.g. `net = gross - refunds - discounts`)
directly from historical data, then cheaply checks whether those rules still
hold on every new batch — catching silent *logic* drift that statistical
monitors miss.

```
Historical Data ──▶ [1. Invariant Miner] ──▶ Invariant Fingerprint (JSON)
                                                       │
New Data Batch ───────────────────────────────────────┤
                                                       ▼
                                          [2. Violation Checker] ──▶ pass / fail
                                                       │ (only on failure)
                                                       ▼
                                          [3. Symbolic Re-Deriver] ──▶ new formula
                                                       │
                                                       ▼
                                          [4. Diff Explainer] ──▶ human-readable alert
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic demo data (embeds a realistic "chargebacks added" drift)
python -m examples.generate_data

# 2. Run the full mine -> check -> re-derive -> explain pipeline
python -m examples.demo

# 3. Run tests
python -m pytest tests/ -q
```

Expected demo output: the miner discovers `net_revenue = gross_revenue -
discounts - refunds` from historical data with 100% support, the checker
flags it as broken on the drifted batch, the re-deriver finds
`net_revenue = gross_revenue - discounts - refunds - chargebacks`, and the
explainer prints exactly what changed.

## Project layout

```
lid/                  core library
  miner.py            Component 1 — mines invariants from a DataFrame
  checker.py           Component 2 — cheap pure-arithmetic violation check
  rederiver.py         Components 3 & 4 — symbolic re-derivation + explanation
  utils.py             shared helpers (linear fit, rational snapping, etc.)
examples/
  generate_data.py     synthetic dataset with embedded ground-truth invariants
  demo.py               CLI walkthrough of the full pipeline
tests/
  test_lid.py           mine / check / re-derive round-trip tests
api.py                 minimal FastAPI wrapper (see "Making it a SaaS API" below)
```

## Design notes

- **Mining is offline/periodic** (O(k^m), k=candidate columns, m=max formula
  size ≤3–4) — run nightly or on schema change.
- **Checking is online/per-batch** (O(k), pure vectorized arithmetic) — the
  cheap always-on pre-filter that scales to thousands of tables per batch.
- **Re-derivation only runs on violation**, and only searches around the
  specific broken relationship, not the whole table.
- Invariant classes covered: linear arithmetic, ratio bounds, functional
  dependency, monotonicity, sum-to-total.

## Making it a real SaaS API

`api.py` is a working local starting point (`uvicorn api:app --reload`) with
`POST /fingerprints/{table}`, `POST /check/{table}`, `GET /fingerprints/{table}`.
It's intentionally minimal — see the project discussion for what's needed to
take it from local dev to a deployable multi-tenant service (auth, real
storage, async batch checking, webhooks/alerting, etc).
