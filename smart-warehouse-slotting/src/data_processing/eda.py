"""Phase-2 EDA: demand-velocity (Pareto/ABC), affinity & co-occurrence, and volumetric
density / space profiling. Produces charts (matplotlib PNG) and a plain-text report.

The ABC classification computed here is the primary classification used downstream (Phase 3),
using documented configurable Pareto thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.demand import load_order_items


def abc_classify(
    demand_freq: pd.Series,
    a_share: float = 0.80,
    b_share: float = 0.95,
) -> pd.DataFrame:
    """Classify SKUs into A/B/C by cumulative demand share (documented Pareto rule).

    demand_freq: indexed by sku_id, values = number of distinct orders (or units).
    Returns a DataFrame: sku_id, demand (rank by volume), rank, cumulative_share, abc_class.
    """
    df = demand_freq.rename("demand").to_frame()
    df = df.sort_values("demand", ascending=False).reset_index(names="sku_id")
    df["rank"] = range(1, len(df) + 1)
    total = df["demand"].sum()
    df["cumulative_share"] = df["demand"].cumsum() / total
    df["abc_class"] = "C"
    df.loc[df["cumulative_share"] <= a_share, "abc_class"] = "A"
    df.loc[
        (df["cumulative_share"] > a_share) & (df["cumulative_share"] <= b_share),
        "abc_class",
    ] = "B"
    return df


def demand_frequency(order_items: pd.DataFrame) -> pd.Series:
    """demand_frequency = number of distinct orders containing each SKU."""
    return order_items.groupby("sku_id")["order_id"].nunique()


def order_affinity(order_items: pd.DataFrame, top_pairs: int = 200) -> pd.DataFrame:
    """Support, confidence, lift for co-occurring SKU pairs from order contents."""
    n_orders = order_items["order_id"].nunique()
    order_skus = order_items.groupby("order_id")["sku_id"].apply(set)
    records = []
    for pair_skus, count in _co_occurrence(order_items).most_common(top_pairs):
        a, b = pair_skus
        freq_a = order_items.loc[order_items["sku_id"] == a, "order_id"].nunique()
        freq_b = order_items.loc[order_items["sku_id"] == b, "order_id"].nunique()
        support = count / n_orders
        conf_ab = count / freq_a
        conf_ba = count / freq_b
        lift_ab = conf_ab / (freq_b / n_orders)
        records.append(
            {
                "sku_a": a,
                "sku_b": b,
                "co_count": count,
                "support": support,
                "confidence_ab": conf_ab,
                "confidence_ba": conf_ba,
                "lift": lift_ab,
            }
        )
    return pd.DataFrame(records)


def _co_occurrence(order_items: pd.DataFrame):
    from collections import Counter

    order_skus = order_items.groupby("order_id")["sku_id"].apply(set)
    counter: Counter = Counter()
    for sku_set in order_skus:
        lst = sorted(sku_set)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                counter[(lst[i], lst[j])] += 1
    return counter


def volume_density_profile(products: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU unit volume/weight vs. the representative bin capacity class.

    Reports volume as a share of a representative bin so space mismatches surface.
    """
    med_bin_vol = float(locations["usable_volume_cm3"].median())
    out = products[["sku_id", "category_id", "unit_volume_cm3", "unit_weight_kg"]].copy()
    out["median_bin_volume_cm3"] = med_bin_vol
    out["vol_share_of_bin"] = out["unit_volume_cm3"] / med_bin_vol
    out["fits_median_bin"] = out["vol_share_of_bin"] <= 1.0
    return out


def run_eda(processed: Path, out_dir: Path, a_share: float = 0.80, b_share: float = 0.95) -> dict:
    """Run full EDA and write charts + a Markdown/JSON report. Returns a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(processed / "products.csv")
    locations = pd.read_csv(processed / "locations.csv")
    oi_a = load_order_items(processed, "A")
    oi_c = load_order_items(processed, "C")

    # --- 1. Demand velocity (Pareto/ABC) on scenario A ---
    freq = demand_frequency(oi_a).reindex(products["sku_id"], fill_value=0)
    abc = abc_classify(freq, a_share, b_share)
    counts = abc["abc_class"].value_counts().to_dict()
    share = abc.groupby("abc_class")["cumulative_share"].max().to_dict()

    # ABC Pareto curve
    fig, ax = plt.subplots(figsize=(8, 5))
    abc_sorted = abc.sort_values("demand", ascending=False)
    ax.plot(range(1, len(abc_sorted) + 1), abc_sorted["cumulative_share"])
    ax.axhline(a_share, color="gray", ls="--", lw=1)
    ax.axhline(b_share, color="gray", ls="--", lw=1)
    ax.set_xlabel("SKU rank (by demand)")
    ax.set_ylabel("Cumulative demand share")
    ax.set_title("Pareto (ABC) curve — Scenario A")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_abc_pareto.png", dpi=100)
    plt.close(fig)

    # --- 2. Affinity & co-occurrence on scenario C ---
    aff = order_affinity(oi_c, top_pairs=80)
    if len(aff):
        top_lift = aff.nlargest(10, "lift")[["sku_a", "sku_b", "co_count", "support", "lift"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            range(len(top_lift)),
            list(top_lift["lift"]),
            tick_label=[f"{a}↔{b}" for a, b in zip(top_lift["sku_a"], top_lift["sku_b"])],
        )
        ax.set_xlabel("Lift")
        ax.set_title("Top affinity pairs by lift — Scenario C")
        fig.tight_layout()
        fig.savefig(out_dir / "eda_affinity.png", dpi=100)
        plt.close(fig)

    # --- 3. Volume density / space profiling ---
    vd = volume_density_profile(products, locations)
    fits_pct = (vd["fits_median_bin"].mean() * 100)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vd["vol_share_of_bin"], bins=30)
    ax.axvline(1.0, color="red", ls="--", lw=1)
    ax.set_xlabel("SKU unit volume / median bin volume")
    ax.set_ylabel("Count")
    ax.set_title("Volumetric density vs. median bin capacity")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_volume_density.png", dpi=100)
    plt.close(fig)

    report = {
        "abc_counts": counts,
        "abc_cumulative_share_by_class": {"A": share.get("A", 0), "B": share.get("B", 0), "C": share.get("C", 0)},
        "affinity_top10_lift": [] if aff.empty else top_lift.to_dict("records"),
        "vol_share_of_bin_pct_fits_median": round(float(fits_pct), 2),
        "median_bin_volume_cm3": float(vd["median_bin_volume_cm3"].iloc[0]),
        "charts": [p.name for p in out_dir.iterdir() if p.suffix == ".png"],
    }
    (out_dir / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # EDA summary markdown for docs
    lines = [
        "# EDA Summary",
        "",
        "## Demand velocity (Scenario A)",
        f"- ABC counts: {report['abc_counts']}",
        f"- Cumulative share at A boundary: {share.get('A', 0):.3f} (target 0.80)",
        f"- Cumulative share at B boundary: {share.get('B', 0):.3f} (target 0.95)",
        "",
        "## Affinity (Scenario C)",
        "- Top pairs by lift listed in `eda_affinity.png`.",
        "",
        "## Volumetric density / space",
        f"- Median bin volume (cm3): {report['median_bin_volume_cm3']:.0f}",
        f"- % SKUs that fit the median bin: {report['vol_share_of_bin_pct_fits_median']:.1f}%",
        "",
    ]
    (out_dir / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    processed = root.parent / "dataset" / "processed"
    out_dir = root / "outputs" / "eda"
    report = run_eda(processed, out_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())