"""Generates synthetic order-level data with embedded ground-truth
invariants, plus a "drifted" batch where net_revenue silently starts
subtracting chargebacks too (the demo scenario from the spec).
"""
import numpy as np
import pandas as pd

np.random.seed(7)


def make_batch(n, drifted=False, start_date="2026-01-01"):
    dates = pd.date_range(start_date, periods=n, freq="h")
    gross_revenue = np.round(np.random.uniform(50, 500, n), 2)
    discounts = np.round(gross_revenue * np.random.uniform(0.0, 0.2, n), 2)
    refunds = np.round(gross_revenue * np.random.uniform(0.0, 0.1, n), 2)
    chargebacks = np.round(gross_revenue * np.random.uniform(0.0, 0.05, n), 2)

    if drifted:
        net_revenue = gross_revenue - discounts - refunds - chargebacks
    else:
        net_revenue = gross_revenue - discounts - refunds

    regions = np.random.choice(["NA", "EMEA", "APAC"], n)
    customer_ids = np.random.randint(1000, 1050, n)
    # customer_id -> region functional dependency: fix each customer to one region
    fixed_region = {cid: np.random.choice(["NA", "EMEA", "APAC"]) for cid in range(1000, 1050)}
    regions = [fixed_region[c] for c in customer_ids]

    discount_rate = discounts / gross_revenue

    cumulative_total = np.cumsum(np.round(np.random.uniform(10, 100, n), 2))

    df = pd.DataFrame({
        "timestamp": dates,
        "customer_id": customer_ids,
        "region": regions,
        "gross_revenue": gross_revenue,
        "discounts": discounts,
        "refunds": refunds,
        "chargebacks": chargebacks,
        "net_revenue": net_revenue,
        "discount_rate": discount_rate,
        "cumulative_total": cumulative_total,
    })
    df = df.set_index("timestamp")
    return df


def make_category_totals(n_groups=20, drifted=False):
    """Sum-to-total demo table: order_total should equal sum of line items."""
    rows = []
    for order_id in range(n_groups):
        n_items = np.random.randint(2, 5)
        amounts = np.round(np.random.uniform(5, 80, n_items), 2)
        total = round(float(amounts.sum()), 2)
        if drifted and order_id % 5 == 0:
            total += 15.0  # e.g. shipping silently added to total but not itemized
        for amt in amounts:
            rows.append({"order_id": order_id, "line_item_amount": amt, "order_total": total})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    historical = make_batch(2000, drifted=False)
    historical.to_csv("historical_orders.csv")

    new_clean = make_batch(500, drifted=False, start_date="2026-04-01")
    new_clean.to_csv("new_batch_clean.csv")

    new_drifted = make_batch(500, drifted=True, start_date="2026-04-01")
    new_drifted.to_csv("new_batch_drifted.csv")

    print("Wrote historical_orders.csv, new_batch_clean.csv, new_batch_drifted.csv")
