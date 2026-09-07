import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lid import mine_invariants, check_violations, re_derive_formula, explain_drift


def make_df(n=500, drifted=False, seed=1):
    rng = np.random.default_rng(seed)
    gross = np.round(rng.uniform(50, 500, n), 2)
    discounts = np.round(gross * rng.uniform(0, 0.2, n), 2)
    refunds = np.round(gross * rng.uniform(0, 0.1, n), 2)
    chargebacks = np.round(gross * rng.uniform(0, 0.05, n), 2)
    net = gross - discounts - refunds - (chargebacks if drifted else 0)
    return pd.DataFrame({
        "gross_revenue": gross, "discounts": discounts,
        "refunds": refunds, "chargebacks": chargebacks, "net_revenue": net,
    })


def test_mine_finds_known_invariant():
    df = make_df(drifted=False)
    fp = mine_invariants(df)
    targets = {inv["target"] for inv in fp if inv["type"] == "linear_arithmetic"}
    assert "net_revenue" in targets


def test_clean_batch_has_no_violations():
    historical = make_df(drifted=False, seed=1)
    clean_new = make_df(drifted=False, seed=2)
    fp = mine_invariants(historical)
    violations = check_violations(clean_new, fp)
    assert violations == []


def test_drifted_batch_is_caught_and_rederived():
    historical = make_df(drifted=False, seed=1)
    drifted_new = make_df(drifted=True, seed=3)
    fp = mine_invariants(historical)
    violations = check_violations(drifted_new, fp)
    assert any(v["target"] == "net_revenue" for v in violations if v["type"] == "linear_arithmetic")

    broken = next(v for v in violations if v["target"] == "net_revenue")
    numeric_cols = list(drifted_new.select_dtypes(include="number").columns)
    result = re_derive_formula(drifted_new, broken, numeric_cols)
    assert "chargebacks" in result["formula"]
    assert result["support"] > 0.95

    explanation = explain_drift(broken["formula"], result["formula"], broken["old_support"], broken["new_support"])
    assert "chargebacks" in explanation
