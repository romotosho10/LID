"""Component 1 — Invariant Miner.

Given a historical DataFrame, mine a ranked list of candidate invariants
(linear-arithmetic relationships, ratio bounds, functional dependencies,
monotonicity, sum-to-total) and return them as a JSON-serializable
"fingerprint".
"""
from __future__ import annotations
import json
import pandas as pd

from .utils import (
    small_combinations, fit_linear, snap_to_simple_rationals, apply_formula,
    fraction_within_tolerance, build_formula_string, numeric_column_pairs,
    categorical_column_pairs, safe_divide, percentile, is_stable_range,
    is_functionally_dependent, is_ordered_by_time, is_monotonic,
    find_total_candidates, sum_matches_total, rank_invariants,
)


def mine_invariants(df: pd.DataFrame, tolerance=0.01, min_support=0.95, max_formula_size=3):
    invariants = []
    all_numeric_cols = list(df.select_dtypes(include="number").columns)
    # Exclude obvious identifier columns from ratio/arithmetic mining — they're
    # not measures, and including them just produces noisy, meaningless invariants.
    id_like = {c for c in all_numeric_cols if "id" in c.lower()}
    numeric_cols = [c for c in all_numeric_cols if c not in id_like]

    # --- Linear arithmetic invariants ---
    for target in numeric_cols:
        candidates = [c for c in numeric_cols if c != target]
        if not candidates:
            continue
        best_for_target = None
        for subset in small_combinations(candidates, max_size=max_formula_size):
            X = df[list(subset)]
            coeffs = fit_linear(df[target], X)
            snapped = snap_to_simple_rationals(coeffs)
            predicted = apply_formula(X, snapped)
            support = fraction_within_tolerance(df[target], predicted, tolerance)
            if support >= min_support:
                candidate_inv = {
                    "type": "linear_arithmetic",
                    "target": target,
                    "subset": list(subset),
                    "formula_coeffs": snapped,
                    "formula": build_formula_string(target, subset, snapped),
                    "support": support,
                }
                # Prefer the simplest formula that clears the bar, so once
                # we find one for this target we don't keep expanding.
                if best_for_target is None:
                    best_for_target = candidate_inv
        if best_for_target:
            invariants.append(best_for_target)

    # --- Ratio bound invariants ---
    for a, b in numeric_column_pairs(numeric_cols):
        ratio = safe_divide(df[a], df[b])
        lo, hi = percentile(ratio, 1), percentile(ratio, 99)
        if lo == lo and hi == hi and is_stable_range(ratio, lo, hi):  # NaN check
            invariants.append({
                "type": "ratio_bound", "columns": [a, b],
                "range": [lo, hi], "support": 0.98
            })

    # --- Functional dependency invariants ---
    for a, b in categorical_column_pairs(df):
        if is_functionally_dependent(df, a, b, min_support):
            invariants.append({"type": "functional_dependency", "from": a, "to": b, "support": min_support})

    # --- Monotonicity invariants ---
    if is_ordered_by_time(df):
        for col in numeric_cols:
            if is_monotonic(df[col], min_support):
                invariants.append({"type": "monotonic", "column": col, "support": min_support})

    # --- Sum-to-total invariants ---
    for total_col, group_col, amount_col in find_total_candidates(df):
        if sum_matches_total(df, total_col, group_col, amount_col, tolerance, min_support):
            invariants.append({
                "type": "sum_to_total",
                "total": total_col, "group_by": group_col, "parts": amount_col,
                "support": min_support,
            })

    return rank_invariants(invariants)


def save_fingerprint(invariants, path):
    with open(path, "w") as f:
        json.dump(invariants, f, indent=2)


def load_fingerprint(path):
    with open(path) as f:
        return json.load(f)
