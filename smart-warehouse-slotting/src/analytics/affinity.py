"""Phase-3 order affinity: co-occurrence, support, confidence, lift, and affinity clusters.

For scenarios with many SKUs the full pair set is large, so the matrix is bounded to the
top pairs by co-occurrence count (configurable). Clusters are derived by connected components
on a lift-thresholded graph.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .demand import load_order_items


def pair_statistics(order_items: pd.DataFrame, top_pairs: int = 20000) -> pd.DataFrame:
    """Support / confidence / lift for co-occurring SKU pairs from order contents."""
    n_orders = order_items["order_id"].nunique()
    order_skus = order_items.groupby("order_id")["sku_id"].apply(set)
    sku_order_count: Counter = Counter()
    pair_count: Counter = Counter()

    for sku_set in order_skus:
        lst = sorted(sku_set)
        for sku in lst:
            sku_order_count[sku] += 1
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                pair_count[(lst[i], lst[j])] += 1

    records = []
    for (a, b), co in pair_count.most_common(top_pairs):
        freq_a = sku_order_count[a]
        freq_b = sku_order_count[b]
        support = co / n_orders
        conf_ab = co / freq_a
        conf_ba = co / freq_b
        lift = conf_ab / (freq_b / n_orders) if freq_b > 0 else float("inf")
        records.append(
            {
                "sku_a": a,
                "sku_b": b,
                "co_count": co,
                "support": support,
                "confidence_ab": conf_ab,
                "confidence_ba": conf_ba,
                "lift": lift,
            }
        )
    return pd.DataFrame(records)


def affinity_clusters(
    pairs: pd.DataFrame,
    lift_threshold: float = 1.0,
    min_co_count: int = 2,
) -> list[list[str]]:
    """Connected components on a lift- and co-occurrence-thresholded SKU graph."""
    adj: defaultdict[str, set] = defaultdict(set)
    for _, r in pairs.iterrows():
        if r["lift"] >= lift_threshold and r["co_count"] >= min_co_count:
            adj[r["sku_a"]].add(r["sku_b"])
            adj[r["sku_b"]].add(r["sku_a"])

    seen: set[str] = set()
    clusters: list[list[str]] = []
    for node in adj:
        if node in seen:
            continue
        seen.add(node)
        stack = [node]
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        clusters.append(sorted(comp))
    # Sort clusters by size descending.
    clusters.sort(key=len, reverse=True)
    return clusters


def build_affinity(
    processed: Path,
    scenario: str = "C",
    top_pairs: int = 20000,
    lift_threshold: float = 5.0,
    min_co_count: int = 8,
    matrix_out: Path | None = None,
    cluster_out: Path | None = None,
) -> tuple[pd.DataFrame, list[list[str]]]:
    """One-call helper: pair statistics + clusters, optionally persisted."""
    items = load_order_items(processed, scenario)
    pairs = pair_statistics(items, top_pairs=top_pairs)
    clusters = affinity_clusters(pairs, lift_threshold, min_co_count)
    if matrix_out is not None:
        matrix_out.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(matrix_out, index=False)
    if cluster_out is not None:
        cluster_out.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"affinity_cluster_id": i, "sku_id": sku} for i, cl in enumerate(clusters) for sku in cl]
        pd.DataFrame(rows).to_csv(cluster_out, index=False)
    return pairs, clusters