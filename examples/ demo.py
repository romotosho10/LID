"""End-to-end CLI demo: mine -> check -> re-derive -> explain.

Run from the repo root:
    python -m examples.generate_data   # writes the three CSVs
    python -m examples.demo
"""
import pandas as pd
from lid import mine_invariants, save_fingerprint, load_fingerprint, check_violations, re_derive_formula, explain_drift


def main():
    print("=" * 70)
    print("STEP 1: Mining invariants from historical data")
    print("=" * 70)
    historical = pd.read_csv("historical_orders.csv", index_col="timestamp", parse_dates=True)
    fingerprint = mine_invariants(historical)
    save_fingerprint(fingerprint, "fingerprint.json")
    print(f"Discovered {len(fingerprint)} invariants:")
    for inv in fingerprint:
        label = inv.get("formula") or f"{inv['type']} on {inv.get('columns') or inv.get('column') or inv.get('from')}"
        print(f"  [{inv['type']}] {label}  (support={inv.get('support', 0):.2%})")

    print()
    print("=" * 70)
    print("STEP 2a: Checking a CLEAN new batch (should show no violations)")
    print("=" * 70)
    clean = pd.read_csv("new_batch_clean.csv", index_col="timestamp", parse_dates=True)
    violations = check_violations(clean, fingerprint)
    print(f"Violations found: {len(violations)}")

    print()
    print("=" * 70)
    print("STEP 2b: Checking a DRIFTED new batch (chargebacks silently added)")
    print("=" * 70)
    drifted = pd.read_csv("new_batch_drifted.csv", index_col="timestamp", parse_dates=True)
    violations = check_violations(drifted, fingerprint)
    print(f"Violations found: {len(violations)}")
    for v in violations:
        print(f"  BROKEN: {v.get('formula', v)}")

    print()
    print("=" * 70)
    print("STEP 3: Symbolic re-derivation of the broken invariant")
    print("=" * 70)
    numeric_cols = list(drifted.select_dtypes(include="number").columns)
    for v in violations:
        if v["type"] != "linear_arithmetic":
            continue
        result = re_derive_formula(drifted, v, numeric_cols)
        print(f"  New best-fit formula: {result['formula']}  (support={result['support']:.2%})")

        print()
        print("=" * 70)
        print("STEP 4: Human-readable diff explanation")
        print("=" * 70)
        explanation = explain_drift(v["formula"], result["formula"], v["old_support"], v["new_support"])
        print(" ", explanation)


if __name__ == "__main__":
    main()
