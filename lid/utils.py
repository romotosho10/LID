"""Shared helpers for Learned Invariant Drift (LID) detection."""
from __future__ import annotations
from itertools import combinations
from fractions import Fraction
import numpy as np
import pandas as pd


def small_combinations(items, max_size=3):
    """Yield all combinations of `items` of size 1..max_size."""
    out = []
    for size in range(1, max_size + 1):
        out.extend(combinations(items, size))
    return out


def fit_linear(y: pd.Series, X: pd.DataFrame):
    """Least-squares fit y ~ X (no intercept search beyond a constant term).
    Returns a dict {col: coeff, "_intercept": b}.
    """
    X_mat = np.column_stack([X[c].values for c in X.columns] + [np.ones(len(X))])
    y_vec = y.values
    coeffs, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    result = {c: coeffs[i] for i, c in enumerate(X.columns)}
    result["_intercept"] = coeffs[-1]
    return result


def snap_to_simple_rationals(coeffs: dict, max_denominator=4, snap_tol=0.03):
    """Snap near-integer / near-simple-fraction coefficients to exact values.
    Leaves a coefficient alone if it isn't close to a simple rational.
    """
    snapped = {}
    for k, v in coeffs.items():
        frac = Fraction(v).limit_denominator(max_denominator)
        if abs(float(frac) - v) <= snap_tol:
            snapped[k] = float(frac)
        else:
            snapped[k] = v
    # Drop the intercept if it's ~0
    if "_intercept" in snapped and abs(snapped["_intercept"]) < snap_tol:
        snapped["_intercept"] = 0.0
    return snapped


def apply_formula(X: pd.DataFrame, formula: dict):
    """Apply a {col: coeff, "_intercept": b} formula to a DataFrame."""
    total = pd.Series(np.zeros(len(X)), index=X.index)
    for k, v in formula.items():
        if k == "_intercept":
            total = total + v
        else:
            total = total + X[k] * v
    return total


def fraction_within_tolerance(actual: pd.Series, predicted: pd.Series, tolerance=0.01):
    """Fraction of rows where |actual - predicted| is within `tolerance`
    (relative tolerance, falls back to absolute for near-zero actuals).
    """
    denom = actual.abs().clip(lower=1e-9)
    rel_err = (actual - predicted).abs() / denom
    abs_err = (actual - predicted).abs()
    ok = (rel_err <= tolerance) | (abs_err <= tolerance)
    return float(ok.mean())


def build_formula_string(target, subset, coeffs):
    terms = []
    for c in subset:
        v = coeffs.get(c, 0.0)
        if v == 0:
            continue
        sign = "+" if v >= 0 else "-"
        mag = abs(v)
        mag_str = "" if abs(mag - 1.0) < 1e-6 else f"{mag:.4g}*"
        terms.append(f"{sign} {mag_str}{c}")
    intercept = coeffs.get("_intercept", 0.0)
    if intercept:
        sign = "+" if intercept >= 0 else "-"
        terms.append(f"{sign} {abs(intercept):.4g}")
    body = " ".join(terms).lstrip("+ ").strip()
    if body.startswith("- "):
        body = "-" + body[2:]
    return f"{target} = {body}" if body else f"{target} = 0"


def numeric_column_pairs(cols):
    return list(combinations(cols, 2))


def categorical_column_pairs(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return list(combinations(cat_cols, 2))


def safe_divide(a: pd.Series, b: pd.Series):
    b_safe = b.replace(0, np.nan)
    return a / b_safe


def percentile(series: pd.Series, p):
    return float(np.nanpercentile(series.dropna().values, p)) if len(series.dropna()) else np.nan


def is_stable_range(ratio: pd.Series, lo, hi, min_coverage=0.95):
    ratio = ratio.dropna()
    if len(ratio) == 0:
        return False
    inside = ((ratio >= lo) & (ratio <= hi)).mean()
    return inside >= min_coverage


def fraction_outside(ratio: pd.Series, rng):
    lo, hi = rng
    ratio = ratio.dropna()
    if len(ratio) == 0:
        return 0.0
    return float(((ratio < lo) | (ratio > hi)).mean())


def is_functionally_dependent(df: pd.DataFrame, a, b, min_support=0.95):
    """True if, for `min_support` fraction of rows, each value of `a` maps
    to a single dominant value of `b`."""
    grouped = df.groupby(a)[b].agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else 0)
    if len(grouped) == 0:
        return False
    return float(grouped.mean()) >= min_support


def is_ordered_by_time(df: pd.DataFrame):
    """Heuristic: True if the DataFrame index or a datetime-like column is sorted."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.is_monotonic_increasing
    dt_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    return len(dt_cols) > 0


def is_monotonic(series: pd.Series, min_support=0.95):
    series = series.dropna()
    if len(series) < 2:
        return True
    diffs = series.diff().dropna()
    return float((diffs >= 0).mean()) >= min_support


def find_total_candidates(df: pd.DataFrame):
    """Heuristic search for (total_col, group_col, amount_col) triples:
    any numeric column whose name suggests a total, paired with any
    categorical column and any other numeric column."""
    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    total_like = [c for c in numeric_cols if "total" in c.lower()]
    triples = []
    for total_col in total_like:
        for group_col in cat_cols:
            for amount_col in numeric_cols:
                if amount_col == total_col:
                    continue
                triples.append((total_col, group_col, amount_col))
    return triples


def sum_matches_total(df, total_col, group_col, amount_col, tolerance=0.01, min_support=0.95):
    grouped = df.groupby(group_col)[amount_col].sum()
    totals = df.groupby(group_col)[total_col].first()
    common = grouped.index.intersection(totals.index)
    if len(common) == 0:
        return False
    ok = fraction_within_tolerance(totals.loc[common], grouped.loc[common], tolerance)
    return ok >= min_support


def find_newly_correlated_columns(df, target, existing_cols, top_n=3):
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                     if c != target and c not in existing_cols]
    if not numeric_cols:
        return []
    corr = df[numeric_cols + [target]].corr()[target].drop(target).abs()
    return list(corr.sort_values(ascending=False).head(top_n).index)


def rank_invariants(invariants):
    def score(inv):
        support = inv.get("support", 0.9)
        simplicity = 1.0
        if inv["type"] == "linear_arithmetic":
            terms = inv["formula"].split("=")[1].count("*") + inv["formula"].split("=")[1].count("+") + 1
            simplicity = 1.0 / terms
        return support * (0.5 + 0.5 * simplicity)
    return sorted(invariants, key=score, reverse=True)


def parse_terms(formula: str):
    rhs = formula.split("=", 1)[1]
    rhs = rhs.replace("-", "+-")
    raw_terms = [t.strip() for t in rhs.split("+") if t.strip()]
    cleaned = set()
    for t in raw_terms:
        t = t.lstrip("-").strip()
        # strip numeric coefficient like "0.5*col" -> "col"
        if "*" in t:
            t = t.split("*", 1)[1]
        cleaned.add(t)
    return cleaned
