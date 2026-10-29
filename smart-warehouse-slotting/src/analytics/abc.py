"""Phase-3 ABC classification: documented Pareto-based rule.

Thresholds (config): A = first 80% of cumulative demand, B = next 15% (to 95%),
C = remainder. Zero-demand SKUs are classified C. The classification table covers every SKU.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .demand import demand_summary


def abc_classify_summary(
    summary: pd.DataFrame,
    a_share: float = 0.80,
    b_share: float = 0.95,
) -> pd.DataFrame:
    """Classify a demand_summary DataFrame (with velocity) into A/B/C by cumulative share.

    Returns a table of sku_id, demand, rank, cumulative_share, abc_class for every SKU.
    """
    df = summary[["sku_id", "velocity"]].rename(columns={"velocity": "demand"}).copy()
    df = df.sort_values("demand", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    total = df["demand"].sum()
    df["cumulative_share"] = df["demand"].cumsum() / total if total > 0 else 0.0
    df["abc_class"] = "C"
    df.loc[df["cumulative_share"] <= a_share, "abc_class"] = "A"
    df.loc[
        (df["cumulative_share"] > a_share) & (df["cumulative_share"] <= b_share),
        "abc_class",
    ] = "B"
    return df


def write_abc(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_abc(processed: Path, all_skus: pd.Series, a_share: float = 0.80, b_share: float = 0.95, scenario: str = "A", out: Path | None = None):
    """One-call helper: demand_summary -> ABC table, optionally persisted."""
    summary = demand_summary_from_processed(processed, all_skus, scenario)
    abc = abc_classify_summary(summary, a_share, b_share)
    if out is not None:
        write_abc(abc, out)
    return abc, summary


def demand_summary_from_processed(processed: Path, all_skus: pd.Series, scenario: str = "A") -> pd.DataFrame:
    from .demand import load_order_items

    items = load_order_items(processed, scenario)
    return demand_summary(items, all_skus, assign_zero=True)