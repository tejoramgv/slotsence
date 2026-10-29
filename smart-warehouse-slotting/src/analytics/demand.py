"""Phase-3 demand analysis: per-SKU demand frequency, units sold, and velocity.

Aggregates over a chosen primary demand scenario (default: A). All figures are used by the
ABC classifier and the optimizer's demand term.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_order_items(processed: Path, scenario: str = "A") -> pd.DataFrame:
    """Canonical order lines for one demand scenario.

    ``order_items.csv`` is shared across scenarios; ``orders.csv`` carries ``scenario_id``,
    so the scenario filter joins through the order header.
    """
    items = pd.read_csv(processed / "order_items.csv")
    orders = pd.read_csv(processed / "orders.csv", usecols=["order_id", "scenario_id"])
    order_ids = orders.loc[orders["scenario_id"] == scenario, "order_id"]
    return items[items["order_id"].isin(order_ids)].reset_index(drop=True)


def demand_summary(order_items: pd.DataFrame, all_skus: list[str] | pd.Series, assign_zero: bool = True) -> pd.DataFrame:
    """Per-SKU demand metrics.

    - demand_frequency = number of distinct orders containing the SKU
    - total_units_sold = sum of quantities
    - velocity = demand_frequency (proxy; recency not modeled in this synthetic data)
    Missing (zero-order) SKUs are included with 0 when assign_zero is True so the table is complete.
    """
    if order_items.empty or "sku_id" not in order_items.columns:
        agg = pd.DataFrame(columns=["sku_id", "demand_frequency", "total_units_sold", "velocity"])
    else:
        grp = order_items.groupby("sku_id")
        agg = grp.agg(
            demand_frequency=("order_id", "nunique"),
            total_units_sold=("quantity", "sum"),
        ).reset_index()
        agg["velocity"] = agg["demand_frequency"]

    if assign_zero:
        sku_ids = pd.Series(list(all_skus), name="sku_id")
        agg = sku_ids.to_frame().merge(agg, on="sku_id", how="left")
        agg["demand_frequency"] = agg["demand_frequency"].fillna(0).astype(int)
        agg["total_units_sold"] = agg["total_units_sold"].fillna(0).astype(int)
        agg["velocity"] = agg["demand_frequency"].fillna(0).astype(int)

    agg = agg.sort_values(["velocity", "sku_id"], ascending=[False, True]).reset_index(drop=True)
    return agg


def demand_frequency(order_items: pd.DataFrame) -> pd.Series:
    """Return per-SKU distinct-order count (indexed by sku_id)."""
    return order_items.groupby("sku_id")["order_id"].nunique()


def write_demand_summary(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_demand_summary(processed: Path, all_skus: pd.Series, scenario: str = "A", out: Path | None = None) -> pd.DataFrame:
    """One-call helper: build and (optionally) persist the demand_summary.csv."""
    items = load_order_items(processed, scenario)
    df = demand_summary(items, all_skus, assign_zero=True)
    if out is not None:
        write_demand_summary(df, out)
    return df