"""Small pipeline with switchable injected failures for the Auto-Healer demo."""

from __future__ import annotations

import pandas as pd


def run_pipeline(failure: str = "none") -> pd.DataFrame:
    """Run a tiny transform. Select a failure to demonstrate log diagnosis."""
    raw = pd.DataFrame({"customer_id": [101, 102, 103], "order_total": [25.5, 18.0, 42.25]})
    if failure == "schema_missing_column":
        raw = raw.drop(columns=["customer_id"])
    elif failure == "schema_type_drift":
        raw["order_total"] = ["25.50", "unknown", "42.25"]
    elif failure == "empty_input":
        raw = raw.iloc[0:0]

    required = {"customer_id", "order_total"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("No objects to concatenate: input batch is empty")
    raw["order_total"] = pd.to_numeric(raw["order_total"], errors="raise")
    raw["tax"] = (raw["order_total"] * 0.08).round(2)
    return raw


if __name__ == "__main__":
    print(run_pipeline().to_string(index=False))
