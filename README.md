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

The API is now split into a gateway (`api.py`) and a background worker
(`worker.py`), talking through a Postgres-backed fingerprint store
(`lid/storage.py`) and a Redis-backed job queue. This is the part of the
project aimed at the "how would you actually deploy this" question:

```
Client --> API gateway (auth) --> Redis queue --> Worker pool --> Postgres
                                                         |
                                                         v
                                                    Alerting (on violation)
```

**Why the queue.** The original single-process version ran mining/checking
inline in the HTTP handler. A checking pass on a wide table runs a bounded
but non-trivial combinatorial search (see "Scaling" below) — doing that
synchronously means a slow request blocks the connection and a burst of
checks queues up behind the event loop instead of behind an actual queue
with retry/backoff semantics. Splitting it into gateway + queue + worker
means the worker pool scales independently of the API, and a failed job
retries without the client having to re-upload.

**Why Postgres instead of local JSON files.** Fingerprints are now
versioned rows, not overwritten files — `get_fingerprint_history()` lets
you ask "when did this invariant relationship first start drifting,"
not just "is it broken right now." `violations` are recorded per
fingerprint version too, so drift history survives a re-mine.

**Multi-tenancy.** Every route requires an `X-API-Key` header, scoped to
a `Tenant` row. `POST /tenants` is a dev-only convenience for minting one
locally — in production that becomes an admin action or a signup flow,
not an open route.

### Running the async stack locally

```bash
docker compose up -d          # postgres + redis
cp .env.example .env

uvicorn api:app --reload      # terminal 1 — the gateway
rq worker lid-jobs --url redis://localhost:6379/0   # terminal 2 — the worker

curl -X POST "http://localhost:8000/tenants?name=demo"   # returns an api_key
# then POST /fingerprints/{table} and /check/{table} with X-API-Key set,
# and GET /jobs/{job_id} to poll for the result
```

### Scaling notes (miner)

- `mine_invariants` now takes `max_rows` (default 20k) — mining runs on a
  random sample for larger tables rather than the full history, since
  checking (not mining) is what needs to see every row.
- `correlation_prune_k` (default 8) caps the candidate columns considered
  per target before the combinatorial formula search runs, cutting the
  O(k^m) cost by roughly two orders of magnitude on wide tables. See
  `lid/utils.py::prune_by_correlation` for the reasoning.
