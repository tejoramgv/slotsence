"""Derived demand tables: ABC classification and SKU-pair affinity.

These tables are convenience artifacts of the dataset build. Later phases can recompute
them with the analytics modules, but the dataset ships a consistent baseline.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any


ABC_A_THRESHOLD = 0.80
ABC_B_THRESHOLD = 0.95


def build_demand_summary(
    products: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    order_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scenario_by_order = {row["order_id"]: row["scenario_id"] for row in orders}
    scenarios = sorted({row["scenario_id"] for row in orders})
    sku_ids = [product["sku_id"] for product in products]

    frequency: dict[str, dict[str, int]] = {
        scenario: {sku_id: 0 for sku_id in sku_ids} for scenario in scenarios
    }
    units: dict[str, dict[str, int]] = {
        scenario: {sku_id: 0 for sku_id in sku_ids} for scenario in scenarios
    }
    for item in order_items:
        scenario = scenario_by_order[item["order_id"]]
        frequency[scenario][item["sku_id"]] += 1
        units[scenario][item["sku_id"]] += int(item["quantity"])

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        ranked = sorted(sku_ids, key=lambda sku_id: (-frequency[scenario][sku_id], sku_id))
        total = sum(frequency[scenario].values())
        cumulative = 0.0
        for rank, sku_id in enumerate(ranked, start=1):
            previous = cumulative
            cumulative += frequency[scenario][sku_id] / total if total else 0.0
            if previous < ABC_A_THRESHOLD:
                abc_class = "A"
            elif previous < ABC_B_THRESHOLD:
                abc_class = "B"
            else:
                abc_class = "C"
            rows.append(
                {
                    "scenario_id": scenario,
                    "sku_id": sku_id,
                    "demand_frequency": frequency[scenario][sku_id],
                    "total_units": units[scenario][sku_id],
                    "rank": rank,
                    "cumulative_share": round(cumulative, 6),
                    "abc_class": abc_class,
                    "data_source": "derived",
                }
            )
    rows.sort(key=lambda row: (row["scenario_id"], row["rank"]))
    return rows


def build_affinity(
    products: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    order_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scenario_by_order = {row["order_id"]: row["scenario_id"] for row in orders}
    skus_by_order: dict[str, set[str]] = defaultdict(set)
    for item in order_items:
        skus_by_order[item["order_id"]].add(item["sku_id"])

    orders_by_scenario: dict[str, int] = defaultdict(int)
    sku_order_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pair_count: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))

    for order_id, skus in skus_by_order.items():
        scenario = scenario_by_order[order_id]
        orders_by_scenario[scenario] += 1
        for sku_id in skus:
            sku_order_count[scenario][sku_id] += 1
        for sku_a, sku_b in combinations(sorted(skus), 2):
            pair_count[scenario][(sku_a, sku_b)] += 1

    rows: list[dict[str, Any]] = []
    for scenario in sorted(orders_by_scenario):
        n_orders = orders_by_scenario[scenario]
        for (sku_a, sku_b), co_count in sorted(pair_count[scenario].items()):
            if co_count < 2:
                continue
            count_a = sku_order_count[scenario][sku_a]
            count_b = sku_order_count[scenario][sku_b]
            support = co_count / n_orders
            confidence_a_to_b = co_count / count_a
            confidence_b_to_a = co_count / count_b
            lift = support / ((count_a / n_orders) * (count_b / n_orders))
            rows.append(
                {
                    "scenario_id": scenario,
                    "sku_a": sku_a,
                    "sku_b": sku_b,
                    "co_count": co_count,
                    "support": round(support, 6),
                    "confidence_a_to_b": round(confidence_a_to_b, 6),
                    "confidence_b_to_a": round(confidence_b_to_a, 6),
                    "lift": round(lift, 6),
                    "data_source": "derived",
                }
            )
    rows.sort(key=lambda row: (row["scenario_id"], row["sku_a"], row["sku_b"]))
    return rows
