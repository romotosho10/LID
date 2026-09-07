"""Component 3 — Symbolic Re-Deriver (bounded {+,-,*,/} search)
   Component 4 — Diff Explainer (human-readable, math-verified alert text)
"""
from __future__ import annotations
from .utils import (
    small_combinations, fit_linear, snap_to_simple_rationals, apply_formula,
    fraction_within_tolerance, build_formula_string, find_newly_correlated_columns,
    parse_terms,
)


def re_derive_formula(new_batch, broken_invariant, all_numeric_columns, max_formula_size=4):
    target = broken_invariant["target"]
    candidate_columns = broken_invariant["subset"]
    expanded_candidates = list(candidate_columns) + find_newly_correlated_columns(
        new_batch, target, candidate_columns
    )
    expanded_candidates = list(dict.fromkeys(expanded_candidates))  # dedupe, keep order

    best_formula, best_coeffs, best_subset, best_support = None, None, None, 0.0
    for subset in small_combinations(expanded_candidates, max_size=max_formula_size):
        X = new_batch[list(subset)]
        coeffs = fit_linear(new_batch[target], X)
        snapped = snap_to_simple_rationals(coeffs)
        predicted = apply_formula(X, snapped)
        support = fraction_within_tolerance(new_batch[target], predicted, 0.01)
        if support > best_support:
            best_formula = build_formula_string(target, subset, snapped)
            best_coeffs, best_subset, best_support = snapped, list(subset), support

    return {
        "target": target,
        "formula": best_formula,
        "formula_coeffs": best_coeffs,
        "subset": best_subset,
        "support": best_support,
    }


def explain_drift(old_formula, new_formula, old_support, new_support):
    old_terms = parse_terms(old_formula)
    new_terms = parse_terms(new_formula) if new_formula else set()
    added = new_terms - old_terms
    removed = old_terms - new_terms

    explanation = f"'{old_formula}' held {old_support:.0%} of the time, now only {new_support:.0%}. "
    if added:
        explanation += f"New term(s) detected: {', '.join(sorted(added))}. "
    if removed:
        explanation += f"Term(s) no longer present: {', '.join(sorted(removed))}. "
    if new_formula:
        explanation += f"Best-fit new formula: '{new_formula}'."
    else:
        explanation += "No stable replacement formula was found — this may be genuine data corruption rather than a logic change."
    return explanation
