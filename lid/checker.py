"""Component 2 — Violation Checker.

Cheap, deterministic, pure-arithmetic check of a stored fingerprint
against a new batch of data. No ML inference here by design.
"""
from __future__ import annotations
import pandas as pd

from .utils import (
    apply_formula, fraction_within_tolerance, safe_divide, fraction_outside,
    is_functionally_dependent, is_monotonic, sum_matches_total,
)


def check_violations(new_batch: pd.DataFrame, fingerprint, tolerance=0.01, alert_threshold=0.10):
    violations = []
    for inv in fingerprint:
        t = inv["type"]

        if t == "linear_arithmetic":
            X = new_batch[inv["subset"]]
            predicted = apply_formula(X, inv["formula_coeffs"])
            support_now = fraction_within_tolerance(new_batch[inv["target"]], predicted, tolerance)
            if (inv["support"] - support_now) > alert_threshold:
                violations.append({**inv, "old_support": inv["support"], "new_support": support_now})

        elif t == "ratio_bound":
            a, b = inv["columns"]
            ratio = safe_divide(new_batch[a], new_batch[b])
            out_of_range_frac = fraction_outside(ratio, inv["range"])
            if out_of_range_frac > alert_threshold:
                violations.append({**inv, "out_of_range_frac": out_of_range_frac})

        elif t == "functional_dependency":
            if not is_functionally_dependent(new_batch, inv["from"], inv["to"], 0.95):
                violations.append(inv)

        elif t == "monotonic":
            if not is_monotonic(new_batch[inv["column"]], 0.95):
                violations.append(inv)

        elif t == "sum_to_total":
            if not sum_matches_total(new_batch, inv["total"], inv["group_by"], inv["parts"], tolerance, 0.95):
                violations.append(inv)

    return violations  # empty list == all invariants held
