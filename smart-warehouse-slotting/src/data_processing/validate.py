"""Dataset validator.

Loads every table, checks schema, keys, references, values, physical feasibility,
demand consistency, graph geometry, and manifest integrity. Violations are written to
``reports/validation_errors.csv`` instead of deleting or silently ignoring rows.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

from .schema import (
    ABC_CLASSES,
    ERGONOMIC_ZONES,
    HANDLING_TYPES,
    HAZARD_CLASSES,
    NODE_TYPES,
    QUALITY_VALUES,
    SCENARIO_IDS,
    TABLES,
)


@dataclass
class ValidationResult:
    errors: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def hard_errors(self) -> int:
        return sum(1 for error in self.errors if error["severity"] == "hard")

    @property
    def warnings(self) -> int:
        return sum(1 for error in self.errors if error["severity"] == "warning")

    @property
    def ok(self) -> bool:
        return self.hard_errors == 0


def run_validation(dataset_dir: Path) -> ValidationResult:
    result = ValidationResult()
    tables: dict[str, list[dict[str, str]]] = {}

    for table, fields in TABLES.items():
        path = dataset_dir / "processed" / f"{table}.csv"
        if not path.exists():
            _add(result, table, "", "schema", "hard", f"missing file: processed/{table}.csv")
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            expected = [field.name for field in fields]
            if header != expected:
                missing = sorted(set(expected) - set(header))
                extra = sorted(set(header) - set(expected))
                _add(
                    result, table, "", "schema", "hard",
                    f"header mismatch; missing={missing}, extra={extra}",
                )
            tables[table] = list(reader)

    _check_manifest_and_metadata(dataset_dir, result, tables)
    if not tables:
        return result

    products = tables.get("products", [])
    categories = tables.get("categories", [])
    locations = tables.get("locations", [])
    warehouses = tables.get("warehouses", [])
    inventory = tables.get("inventory", [])
    orders = tables.get("orders", [])
    order_items = tables.get("order_items", [])
    assignments = tables.get("storage_assignments_baseline", [])
    rules = tables.get("compatibility_rules", [])
    graph = tables.get("location_graph", [])
    demand = tables.get("demand_summary", [])
    affinity = tables.get("sku_affinity", [])

    category_ids = _unique_values(result, categories, "category_id", "categories")
    product_ids = _unique_values(result, products, "sku_id", "products")
    location_ids = _unique_values(result, locations, "location_id", "locations")
    warehouse_ids = _unique_values(result, warehouses, "warehouse_id", "warehouses")
    order_ids = _unique_values(result, orders, "order_id", "orders")

    _check_unique(result, inventory, ("sku_id", "location_id"), "inventory")
    _check_unique(result, order_items, ("order_id", "sku_id"), "order_items")
    _check_unique(result, assignments, ("sku_id", "location_id"), "storage_assignments_baseline")
    _check_unique(result, rules, ("category_a", "category_b"), "compatibility_rules")
    _check_unique(result, graph, ("from_node_id", "to_node_id"), "location_graph")
    _check_unique(result, demand, ("scenario_id", "sku_id"), "demand_summary")
    _check_unique(result, affinity, ("scenario_id", "sku_a", "sku_b"), "sku_affinity")

    _check_references(
        result, products, "products", "category_id", category_ids,
        "categories", "foreign_key",
    )
    _check_references(
        result, locations, "locations", "warehouse_id", warehouse_ids,
        "warehouses", "foreign_key",
    )
    _check_references(
        result, order_items, "order_items", "order_id", order_ids,
        "orders", "foreign_key",
    )
    _check_references(
        result, order_items, "order_items", "sku_id", product_ids,
        "products", "foreign_key",
    )
    _check_references(
        result, inventory, "inventory", "sku_id", product_ids,
        "products", "foreign_key",
    )
    _check_references(
        result, inventory, "inventory", "location_id", location_ids,
        "locations", "foreign_key",
    )
    _check_references(
        result, assignments, "storage_assignments_baseline", "sku_id", product_ids,
        "products", "foreign_key",
    )
    _check_references(
        result, assignments, "storage_assignments_baseline", "location_id", location_ids,
        "locations", "foreign_key",
    )
    _check_references(
        result, rules, "compatibility_rules", "category_a", category_ids,
        "categories", "foreign_key",
    )
    _check_references(
        result, rules, "compatibility_rules", "category_b", category_ids,
        "categories", "foreign_key",
    )
    _check_references(
        result, demand, "demand_summary", "sku_id", product_ids,
        "products", "foreign_key",
    )
    _check_references(
        result, affinity, "sku_affinity", "sku_a", product_ids,
        "products", "foreign_key",
    )
    _check_references(
        result, affinity, "sku_affinity", "sku_b", product_ids,
        "products", "foreign_key",
    )

    product_by_sku = {row["sku_id"]: row for row in products}
    location_by_id = {row["location_id"]: row for row in locations}
    category_by_id = {row["category_id"]: row for row in categories}

    _check_product_values(result, products)
    _check_location_values(result, locations)
    _check_order_values(result, orders)
    _check_item_values(result, order_items)
    _check_assignment_values(result, assignments)
    _check_rule_values(result, rules, category_by_id)
    _check_graph_values(result, graph, location_ids)
    _check_demand_values(result, demand, affinity)

    _check_timestamp_window(dataset_dir, result, orders)
    _check_order_composition(dataset_dir, result, orders, order_items)
    _check_demand_math(result, products, orders, order_items, demand)
    _check_affinity_math(result, products, orders, order_items, affinity)
    _check_physical_feasibility(
        result, products, locations, inventory, assignments, product_by_sku, location_by_id
    )
    _check_graph_geometry(dataset_dir, result, locations, graph)

    result.stats["row_counts"] = {table: len(rows) for table, rows in tables.items()}
    result.stats["capacity"] = _capacity_stats(products, locations, assignments, product_by_sku)
    result.stats["demand"] = _demand_stats(demand)
    result.stats["scenario_counts"] = _scenario_counts(orders, order_items)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add(
    result: ValidationResult,
    table: str,
    row_id: str,
    check: str,
    severity: str,
    message: str,
) -> None:
    result.errors.append(
        {
            "table": table,
            "row_id": str(row_id),
            "check": check,
            "severity": severity,
            "message": message,
        }
    )


def _unique_values(
    result: ValidationResult, rows: list[dict[str, str]], key: str, table: str
) -> set[str]:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        value = row.get(key, "")
        if not value:
            _add(result, table, str(index), "unique_key", "hard", f"missing {key}")
            continue
        if value in seen:
            _add(result, table, value, "unique_key", "hard", f"duplicate {key}")
        seen.add(value)
    return seen


def _check_unique(
    result: ValidationResult,
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
    table: str,
) -> None:
    seen: set[tuple[str, ...]] = set()
    for index, row in enumerate(rows, start=2):
        value = tuple(row.get(key, "") for key in keys)
        if any(not part for part in value):
            _add(result, table, str(index), "unique_key", "hard", f"missing key fields {keys}")
            continue
        if value in seen:
            _add(result, table, "/".join(value), "unique_key", "hard", f"duplicate key {keys}")
        seen.add(value)


def _check_references(
    result: ValidationResult,
    rows: list[dict[str, str]],
    table: str,
    field_name: str,
    valid_values: set[str],
    target_table: str,
    check: str,
) -> None:
    for index, row in enumerate(rows, start=2):
        value = row.get(field_name, "")
        if value not in valid_values:
            _add(
                result, table, str(index), check, "hard",
                f"{field_name}={value!r} does not exist in {target_table}",
            )


def _to_float(result: ValidationResult, table: str, row_id: str, value: str, field_name: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        _add(result, table, row_id, "value_range", "hard", f"{field_name}={value!r} is not numeric")
        return None


def _to_int(result: ValidationResult, table: str, row_id: str, value: str, field_name: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        _add(result, table, row_id, "value_range", "hard", f"{field_name}={value!r} is not an integer")
        return None


def _check_product_values(result: ValidationResult, products: list[dict[str, str]]) -> None:
    for index, row in enumerate(products, start=2):
        row_id = row.get("sku_id", str(index))
        dims = []
        for field_name in ("width_cm", "height_cm", "depth_cm"):
            value = _to_float(result, "products", row_id, row.get(field_name, ""), field_name)
            if value is not None and value <= 0:
                _add(result, "products", row_id, "value_range", "hard", f"{field_name} must be > 0")
            dims.append(value)
        weight = _to_float(result, "products", row_id, row.get("unit_weight_kg", ""), "unit_weight_kg")
        volume = _to_float(result, "products", row_id, row.get("unit_volume_cm3", ""), "unit_volume_cm3")
        if weight is not None and weight <= 0:
            _add(result, "products", row_id, "value_range", "hard", "unit_weight_kg must be > 0")
        if volume is not None and volume <= 0:
            _add(result, "products", row_id, "value_range", "hard", "unit_volume_cm3 must be > 0")
        if all(value is not None for value in dims) and volume is not None:
            expected = dims[0] * dims[1] * dims[2]
            if abs(expected - volume) > max(0.01, expected * 1e-4):
                _add(result, "products", row_id, "volume_consistency", "hard", "unit_volume_cm3 != w*h*d")
        if row.get("handling_type") not in HANDLING_TYPES:
            _add(result, "products", row_id, "value_range", "hard", "invalid handling_type")
        if row.get("hazard_class") not in HAZARD_CLASSES:
            _add(result, "products", row_id, "value_range", "hard", "invalid hazard_class")
        if row.get("data_quality_status") not in QUALITY_VALUES:
            _add(result, "products", row_id, "value_range", "warning", "unknown data_quality_status")


def _check_location_values(result: ValidationResult, locations: list[dict[str, str]]) -> None:
    for index, row in enumerate(locations, start=2):
        row_id = row.get("location_id", str(index))
        for field_name in ("width_cm", "height_cm", "depth_cm", "max_weight_kg", "usable_volume_cm3"):
            value = _to_float(result, "locations", row_id, row.get(field_name, ""), field_name)
            if value is not None and value <= 0:
                _add(result, "locations", row_id, "value_range", "hard", f"{field_name} must be > 0")
        volume = _to_float(result, "locations", row_id, row.get("usable_volume_cm3", ""), "usable_volume_cm3")
        width = _to_float(result, "locations", row_id, row.get("width_cm", ""), "width_cm")
        height = _to_float(result, "locations", row_id, row.get("height_cm", ""), "height_cm")
        depth = _to_float(result, "locations", row_id, row.get("depth_cm", ""), "depth_cm")
        if None not in (volume, width, height, depth):
            expected = width * height * depth
            if abs(expected - volume) > max(0.01, expected * 1e-4):
                _add(result, "locations", row_id, "volume_consistency", "hard", "usable_volume_cm3 != w*h*d")
        if row.get("ergonomic_zone") not in ERGONOMIC_ZONES:
            _add(result, "locations", row_id, "value_range", "hard", "invalid ergonomic_zone")
        allowed_handling = row.get("allowed_handling", "").split("|")
        allowed_hazard = row.get("allowed_hazard", "").split("|")
        if not set(allowed_handling) <= set(HANDLING_TYPES):
            _add(result, "locations", row_id, "value_range", "hard", "invalid allowed_handling")
        if not set(allowed_hazard) <= set(HAZARD_CLASSES):
            _add(result, "locations", row_id, "value_range", "hard", "invalid allowed_hazard")
        bay = _to_int(result, "locations", row_id, row.get("bay", ""), "bay")
        shelf = _to_int(result, "locations", row_id, row.get("shelf_level", ""), "shelf_level")
        if bay is not None and bay < 0:
            _add(result, "locations", row_id, "value_range", "hard", "bay must be >= 0")
        if shelf is not None and shelf < 0:
            _add(result, "locations", row_id, "value_range", "hard", "shelf_level must be >= 0")


def _check_order_values(result: ValidationResult, orders: list[dict[str, str]]) -> None:
    for index, row in enumerate(orders, start=2):
        row_id = row.get("order_id", str(index))
        if row.get("scenario_id") not in SCENARIO_IDS:
            _add(result, "orders", row_id, "value_range", "hard", "invalid scenario_id")
        if row.get("order_status") != "completed":
            _add(result, "orders", row_id, "value_range", "warning", f"unexpected status {row.get('order_status')!r}")


def _check_item_values(result: ValidationResult, order_items: list[dict[str, str]]) -> None:
    for index, row in enumerate(order_items, start=2):
        row_id = f"{row.get('order_id', '')}/{row.get('sku_id', '')}"
        quantity = _to_int(result, "order_items", row_id, row.get("quantity", ""), "quantity")
        if quantity is not None and quantity <= 0:
            _add(result, "order_items", row_id, "value_range", "hard", "quantity must be > 0")


def _check_assignment_values(result: ValidationResult, assignments: list[dict[str, str]]) -> None:
    for index, row in enumerate(assignments, start=2):
        row_id = f"{row.get('sku_id', '')}/{row.get('location_id', '')}"
        quantity = _to_int(
            result, "storage_assignments_baseline", row_id, row.get("assigned_quantity", ""), "assigned_quantity"
        )
        if quantity is not None and quantity <= 0:
            _add(result, "storage_assignments_baseline", row_id, "value_range", "hard", "assigned_quantity must be > 0")
        if row.get("assignment_type") != "baseline":
            _add(result, "storage_assignments_baseline", row_id, "value_range", "hard", "assignment_type must be baseline")


def _check_rule_values(
    result: ValidationResult,
    rules: list[dict[str, str]],
    category_by_id: dict[str, dict[str, str]],
) -> None:
    for index, row in enumerate(rules, start=2):
        row_id = row.get("rule_id", str(index))
        if row.get("allowed") not in ("true", "false"):
            _add(result, "compatibility_rules", row_id, "value_range", "hard", "allowed must be true or false")
        if row.get("severity") not in ("hard", "soft", "info"):
            _add(result, "compatibility_rules", row_id, "value_range", "hard", "invalid severity")
        if row.get("category_a") not in category_by_id or row.get("category_b") not in category_by_id:
            continue


def _check_graph_values(
    result: ValidationResult, graph: list[dict[str, str]], location_ids: set[str]
) -> None:
    node_types: dict[str, str] = {}
    for index, row in enumerate(graph, start=2):
        row_id = f"{row.get('from_node_id', '')}->{row.get('to_node_id', '')}"
        for prefix in ("from", "to"):
            node_id = row.get(f"{prefix}_node_id", "")
            node_type = row.get(f"{prefix}_node_type", "")
            if not node_id:
                _add(result, "location_graph", row_id, "foreign_key", "hard", f"missing {prefix}_node_id")
                continue
            if node_type not in NODE_TYPES:
                _add(result, "location_graph", row_id, "value_range", "hard", f"invalid {prefix}_node_type")
            if node_id in node_types and node_types[node_id] != node_type:
                _add(result, "location_graph", row_id, "value_range", "hard", "node id has conflicting types")
            node_types[node_id] = node_type
            if node_type == "location" and node_id not in location_ids:
                _add(result, "location_graph", row_id, "foreign_key", "hard", "location node does not exist")
        distance = _to_float(result, "location_graph", row_id, row.get("distance_m", ""), "distance_m")
        if distance is not None and distance <= 0:
            _add(result, "location_graph", row_id, "value_range", "hard", "distance_m must be > 0")
    if "DEPOT" not in node_types:
        _add(result, "location_graph", "", "graph_connectivity", "hard", "missing DEPOT node")
    for location_id in location_ids:
        if node_types.get(location_id) != "location":
            _add(
                result, "location_graph", location_id, "graph_connectivity", "hard",
                "location is missing from the graph",
            )


def _check_demand_values(
    result: ValidationResult,
    demand: list[dict[str, str]],
    affinity: list[dict[str, str]],
) -> None:
    for index, row in enumerate(demand, start=2):
        row_id = f"{row.get('scenario_id', '')}/{row.get('sku_id', '')}"
        if row.get("scenario_id") not in SCENARIO_IDS:
            _add(result, "demand_summary", row_id, "value_range", "hard", "invalid scenario_id")
        if row.get("abc_class") not in ABC_CLASSES:
            _add(result, "demand_summary", row_id, "value_range", "hard", "invalid abc_class")
        frequency = _to_int(result, "demand_summary", row_id, row.get("demand_frequency", ""), "demand_frequency")
        if frequency is not None and frequency < 0:
            _add(result, "demand_summary", row_id, "value_range", "hard", "demand_frequency must be >= 0")
        share = _to_float(result, "demand_summary", row_id, row.get("cumulative_share", ""), "cumulative_share")
        if share is not None and not 0.0 <= share <= 1.0 + 1e-9:
            _add(result, "demand_summary", row_id, "value_range", "hard", "cumulative_share out of range")

    for index, row in enumerate(affinity, start=2):
        row_id = f"{row.get('scenario_id', '')}/{row.get('sku_a', '')}/{row.get('sku_b', '')}"
        if row.get("scenario_id") not in SCENARIO_IDS:
            _add(result, "sku_affinity", row_id, "value_range", "hard", "invalid scenario_id")
        if row.get("sku_a", "") >= row.get("sku_b", ""):
            _add(result, "sku_affinity", row_id, "value_range", "hard", "sku_a must sort before sku_b")
        for field_name in ("co_count", "support", "confidence_a_to_b", "confidence_b_to_a", "lift"):
            if field_name == "co_count":
                value = _to_int(result, "sku_affinity", row_id, row.get(field_name, ""), field_name)
            else:
                value = _to_float(result, "sku_affinity", row_id, row.get(field_name, ""), field_name)
            if value is not None and value <= 0:
                _add(result, "sku_affinity", row_id, "value_range", "hard", f"{field_name} must be > 0")


def _check_timestamp_window(
    dataset_dir: Path, result: ValidationResult, orders: list[dict[str, str]]
) -> None:
    manifest = _read_manifest(dataset_dir)
    config = manifest.get("config", {}) if manifest else {}
    window = config.get("order_window", {})
    start_text = window.get("start_date")
    days = window.get("days")
    if not start_text or not days:
        _add(result, "orders", "", "timestamp", "warning", "manifest has no order window; range not checked")
        return
    start = datetime.strptime(start_text, "%Y-%m-%d")
    end = start + timedelta(days=int(days))
    for index, row in enumerate(orders, start=2):
        row_id = row.get("order_id", str(index))
        text = row.get("order_timestamp", "")
        try:
            timestamp = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            _add(result, "orders", row_id, "timestamp", "hard", f"invalid UTC timestamp {text!r}")
            continue
        if not start <= timestamp < end:
            _add(result, "orders", row_id, "timestamp", "hard", "timestamp outside the configured window")


def _check_order_composition(
    dataset_dir: Path,
    result: ValidationResult,
    orders: list[dict[str, str]],
    order_items: list[dict[str, str]],
) -> None:
    lines_by_order: dict[str, int] = defaultdict(int)
    for item in order_items:
        lines_by_order[item["order_id"]] += 1
    for order in orders:
        count = lines_by_order.get(order["order_id"], 0)
        if count == 0:
            _add(result, "orders", order["order_id"], "demand_consistency", "hard", "order has no line items")
    manifest = _read_manifest(dataset_dir)
    config = manifest.get("config", {}) if manifest else {}
    target = config.get("n_order_lines_per_scenario")
    if target:
        counts: dict[str, int] = defaultdict(int)
        scenario_by_order = {row["order_id"]: row["scenario_id"] for row in orders}
        for item in order_items:
            counts[scenario_by_order[item["order_id"]]] += 1
        for scenario in SCENARIO_IDS:
            if counts[scenario] != int(target):
                _add(
                    result, "order_items", scenario, "demand_consistency", "hard",
                    f"scenario {scenario} has {counts[scenario]} lines, expected {target}",
                )


def _check_demand_math(
    result: ValidationResult,
    products: list[dict[str, str]],
    orders: list[dict[str, str]],
    order_items: list[dict[str, str]],
    demand: list[dict[str, str]],
) -> None:
    scenario_by_order = {row["order_id"]: row["scenario_id"] for row in orders}
    frequency: dict[tuple[str, str], int] = defaultdict(int)
    units: dict[tuple[str, str], int] = defaultdict(int)
    for item in order_items:
        scenario = scenario_by_order[item["order_id"]]
        frequency[(scenario, item["sku_id"])] += 1
        units[(scenario, item["sku_id"])] += int(item["quantity"])

    table = {(row["scenario_id"], row["sku_id"]): row for row in demand}
    for product in products:
        for scenario in SCENARIO_IDS:
            row = table.get((scenario, product["sku_id"]))
            if row is None:
                _add(
                    result, "demand_summary", f"{scenario}/{product['sku_id']}",
                    "demand_consistency", "hard", "missing SKU row",
                )
                continue
            if int(row["demand_frequency"]) != frequency[(scenario, product["sku_id"])]:
                _add(
                    result, "demand_summary", f"{scenario}/{product['sku_id']}",
                    "demand_consistency", "hard", "demand_frequency mismatch",
                )
            if int(row["total_units"]) != units[(scenario, product["sku_id"])]:
                _add(
                    result, "demand_summary", f"{scenario}/{product['sku_id']}",
                    "demand_consistency", "hard", "total_units mismatch",
                )

    for scenario in SCENARIO_IDS:
        rows = [row for row in demand if row["scenario_id"] == scenario]
        if not rows:
            _add(result, "demand_summary", scenario, "abc_partition", "hard", "scenario missing")
            continue
        rows.sort(key=lambda row: int(row["rank"]))
        class_cumulative = {"A": 0.0, "B": 0.0, "C": 0.0}
        for row in rows:
            class_cumulative[row["abc_class"]] = max(
                class_cumulative[row["abc_class"]], float(row["cumulative_share"])
            )
        if class_cumulative["A"] < 0.80 - 1e-6:
            _add(result, "demand_summary", scenario, "abc_partition", "hard", "A share below 80%")
        if class_cumulative["B"] < 0.95 - 1e-6:
            _add(result, "demand_summary", scenario, "abc_partition", "hard", "B share below 95%")
        total = float(rows[-1]["cumulative_share"])
        if abs(total - 1.0) > 1e-5:
            _add(result, "demand_summary", scenario, "abc_partition", "hard", "cumulative share does not reach 1")


def _check_affinity_math(
    result: ValidationResult,
    products: list[dict[str, str]],
    orders: list[dict[str, str]],
    order_items: list[dict[str, str]],
    affinity: list[dict[str, str]],
) -> None:
    scenario_by_order = {row["order_id"]: row["scenario_id"] for row in orders}
    skus_by_order: dict[str, set[str]] = defaultdict(set)
    for item in order_items:
        skus_by_order[item["order_id"]].add(item["sku_id"])

    orders_by_scenario: dict[str, int] = defaultdict(int)
    sku_counts: dict[tuple[str, str], int] = defaultdict(int)
    pair_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for order_id, skus in skus_by_order.items():
        scenario = scenario_by_order[order_id]
        orders_by_scenario[scenario] += 1
        for sku_id in skus:
            sku_counts[(scenario, sku_id)] += 1
        for sku_a, sku_b in combinations(sorted(skus), 2):
            pair_counts[(scenario, sku_a, sku_b)] += 1

    expected: dict[tuple[str, str, str], tuple[int, float, float, float, float]] = {}
    for key, co_count in pair_counts.items():
        if co_count < 2:
            continue
        scenario, sku_a, sku_b = key
        n_orders = orders_by_scenario[scenario]
        count_a = sku_counts[(scenario, sku_a)]
        count_b = sku_counts[(scenario, sku_b)]
        support = co_count / n_orders
        expected[key] = (
            co_count,
            support,
            co_count / count_a,
            co_count / count_b,
            support / ((count_a / n_orders) * (count_b / n_orders)),
        )

    actual: dict[tuple[str, str, str], dict[str, str]] = {
        (row["scenario_id"], row["sku_a"], row["sku_b"]): row for row in affinity
    }
    for key in sorted(set(expected) - set(actual)):
        _add(result, "sku_affinity", "/".join(key), "affinity_math", "hard", "missing affinity pair")
    for key in sorted(set(actual) - set(expected)):
        _add(result, "sku_affinity", "/".join(key), "affinity_math", "hard", "extra affinity pair")
    for key in sorted(set(expected) & set(actual)):
        co_count, support, conf_ab, conf_ba, lift = expected[key]
        row = actual[key]
        checks = (
            (int(row["co_count"]), co_count, "co_count"),
            (float(row["support"]), support, "support"),
            (float(row["confidence_a_to_b"]), conf_ab, "confidence_a_to_b"),
            (float(row["confidence_b_to_a"]), conf_ba, "confidence_b_to_a"),
            (float(row["lift"]), lift, "lift"),
        )
        for actual_value, expected_value, field_name in checks:
            if abs(actual_value - expected_value) > 1e-5:
                _add(
                    result, "sku_affinity", "/".join(key), "affinity_math", "hard",
                    f"{field_name} mismatch",
                )
    if not any(row["scenario_id"] == "C" for row in affinity):
        _add(result, "sku_affinity", "C", "affinity_math", "warning", "scenario C has no affinity pairs")


def _check_physical_feasibility(
    result: ValidationResult,
    products: list[dict[str, str]],
    locations: list[dict[str, str]],
    inventory: list[dict[str, str]],
    assignments: list[dict[str, str]],
    product_by_sku: dict[str, dict[str, str]],
    location_by_id: dict[str, dict[str, str]],
) -> None:
    assigned_units: dict[str, int] = defaultdict(int)
    assigned_volume: dict[str, float] = defaultdict(float)
    assigned_weight: dict[str, float] = defaultdict(float)
    shelf_weight: dict[tuple[str, str], float] = defaultdict(float)
    rack_weight: dict[str, float] = defaultdict(float)
    shelf_limit: dict[tuple[str, str], float] = {}
    rack_limit: dict[str, float] = {}
    sku_by_location: dict[str, str] = {}

    for row in assignments:
        sku_id = row["sku_id"]
        location_id = row["location_id"]
        quantity = int(row["assigned_quantity"])
        product = product_by_sku.get(sku_id)
        location = location_by_id.get(location_id)
        if product is None or location is None:
            continue
        row_id = f"{sku_id}/{location_id}"
        if location_id in sku_by_location and sku_by_location[location_id] != sku_id:
            _add(result, "storage_assignments_baseline", row_id, "one_sku_per_bin", "hard", "bin holds multiple SKUs")
        sku_by_location[location_id] = sku_id

        dimensions_ok = (
            float(product["width_cm"]) <= float(location["width_cm"])
            and float(product["height_cm"]) <= float(location["height_cm"])
            and float(product["depth_cm"]) <= float(location["depth_cm"])
        )
        if not dimensions_ok:
            _add(result, "storage_assignments_baseline", row_id, "physical_fit", "hard", "SKU does not fit the bin")
        if product["handling_type"] not in location["allowed_handling"].split("|"):
            _add(result, "storage_assignments_baseline", row_id, "handling_compatibility", "hard", "handling not allowed")
        if product["hazard_class"] not in location["allowed_hazard"].split("|"):
            _add(result, "storage_assignments_baseline", row_id, "hazard_compatibility", "hard", "hazard not allowed")
        if product["handling_type"] == "heavy" and int(location["shelf_level"]) > 2:
            _add(result, "storage_assignments_baseline", row_id, "heavy_height", "hard", "heavy SKU above shelf level 2")
        if product["handling_type"] == "fragile" and location["zone_type"] == "heavy":
            _add(result, "storage_assignments_baseline", row_id, "fragile_zone", "hard", "fragile SKU in heavy zone")
        if product["hazard_class"] != "NONE" and location["zone_type"] != "hazard":
            _add(result, "storage_assignments_baseline", row_id, "hazard_compatibility", "hard", "hazard SKU outside hazard zone")

        weight = quantity * float(product["unit_weight_kg"])
        volume = quantity * float(product["unit_volume_cm3"])
        if weight > float(location["max_weight_kg"]) + 1e-6:
            _add(result, "storage_assignments_baseline", row_id, "weight_capacity", "hard", "bin weight exceeded")
        if volume > float(location["usable_volume_cm3"]) + 1e-6:
            _add(result, "storage_assignments_baseline", row_id, "volume_capacity", "hard", "bin volume exceeded")

        shelf_key = (location["rack_id"], location["shelf_level"])
        shelf_weight[shelf_key] += weight
        rack_weight[location["rack_id"]] += weight
        shelf_limit[shelf_key] = float(location["shelf_max_weight_kg"])
        rack_limit[location["rack_id"]] = float(location["rack_max_weight_kg"])
        assigned_units[sku_id] += quantity
        assigned_volume[sku_id] += volume
        assigned_weight[sku_id] += weight

    for shelf_key, weight in shelf_weight.items():
        if weight > shelf_limit[shelf_key] + 1e-6:
            _add(result, "locations", "/".join(shelf_key), "shelf_capacity", "hard", "shelf weight exceeded")
    for rack_id, weight in rack_weight.items():
        if weight > rack_limit[rack_id] + 1e-6:
            _add(result, "locations", rack_id, "rack_capacity", "hard", "rack weight exceeded")

    inventory_by_sku: dict[str, int] = defaultdict(int)
    for row in inventory:
        inventory_by_sku[row["sku_id"]] += int(row["quantity_on_hand"])
    assignment_quantity = {
        (row["sku_id"], row["location_id"]): int(row["assigned_quantity"]) for row in assignments
    }
    for row in inventory:
        key = (row["sku_id"], row["location_id"])
        if assignment_quantity.get(key) != int(row["quantity_on_hand"]):
            _add(
                result, "inventory", f"{row['sku_id']}/{row['location_id']}",
                "inventory_balance", "hard", "inventory does not match the baseline assignment",
            )
    for product in products:
        sku_id = product["sku_id"]
        if assigned_units[sku_id] <= 0:
            _add(result, "products", sku_id, "assignment_complete", "hard", "SKU is not assigned to any bin")
        if inventory_by_sku[sku_id] != assigned_units[sku_id]:
            _add(result, "inventory", sku_id, "inventory_balance", "hard", "inventory total differs from assignments")
    if assigned_units and len(assigned_units) < len(products):
        _add(result, "products", "", "assignment_complete", "hard", "not every SKU has an assignment")


def _check_graph_geometry(
    dataset_dir: Path,
    result: ValidationResult,
    locations: list[dict[str, str]],
    graph: list[dict[str, str]],
) -> None:
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    edge_map: dict[tuple[str, str], float] = {}
    node_types: dict[str, str] = {}
    for row in graph:
        left = row["from_node_id"]
        right = row["to_node_id"]
        distance = float(row["distance_m"])
        adjacency[left].append((right, distance))
        edge_map[(left, right)] = distance
        node_types[left] = row["from_node_type"]
        node_types[right] = row["to_node_type"]
    for (left, right), distance in edge_map.items():
        reverse = edge_map.get((right, left))
        if reverse is None:
            _add(result, "location_graph", f"{left}->{right}", "graph_symmetry", "hard", "missing reverse edge")
        elif abs(reverse - distance) > 1e-5:
            _add(result, "location_graph", f"{left}->{right}", "graph_symmetry", "hard", "asymmetric distance")

    if "DEPOT" not in adjacency:
        return
    distances = {node: float("inf") for node in adjacency}
    distances["DEPOT"] = 0.0
    heap: list[tuple[float, str]] = [(0.0, "DEPOT")]
    while heap:
        current, node = heapq.heappop(heap)
        if current > distances[node]:
            continue
        for neighbour, weight in adjacency[node]:
            candidate = current + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                heapq.heappush(heap, (candidate, neighbour))

    for location in locations:
        location_id = location["location_id"]
        if distances.get(location_id, float("inf")) == float("inf"):
            _add(result, "location_graph", location_id, "graph_connectivity", "hard", "location is unreachable")
            continue
        if "distance_to_dispatch_m" in location and location["distance_to_dispatch_m"] not in ("", None):
            if abs(distances[location_id] - float(location["distance_to_dispatch_m"])) > 0.01:
                _add(result, "location_graph", location_id, "graph_distance", "hard", "distance_to_dispatch_m mismatch")

    manifest = _read_manifest(dataset_dir)
    config = manifest.get("config", {}) if manifest else {}
    warehouse = config.get("warehouse", {})
    lateral = float(warehouse.get("aisle_width_m", 0.0)) / 2.0 + float(warehouse.get("rack_depth_m", 0.0)) / 2.0
    if lateral > 0:
        for row in graph:
            if row["from_node_type"] == "location" and row["to_node_type"] == "aisle_column":
                if abs(float(row["distance_m"]) - lateral) > 1e-6:
                    _add(
                        result, "location_graph", f"{row['from_node_id']}->{row['to_node_id']}",
                        "graph_geometry", "hard", "lateral distance does not match the layout",
                    )


def _check_manifest_and_metadata(
    dataset_dir: Path,
    result: ValidationResult,
    tables: dict[str, list[dict[str, str]]],
) -> None:
    manifest = _read_manifest(dataset_dir)
    if manifest is None:
        _add(result, "metadata", "", "manifest", "hard", "missing metadata/generation_manifest.json")
    else:
        for table in TABLES:
            relative = f"processed/{table}.csv"
            entry = manifest.get("files", {}).get(relative)
            if entry is None:
                _add(result, "metadata", table, "manifest", "hard", f"{relative} missing from the manifest")
                continue
            path = dataset_dir / relative
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != entry.get("sha256"):
                    _add(result, "metadata", table, "manifest_hash", "hard", f"{relative} hash mismatch")
                if "rows" in entry and int(entry["rows"]) != len(tables.get(table, [])):
                    _add(result, "metadata", table, "manifest", "hard", f"{relative} row count mismatch")
        if not manifest.get("seed"):
            _add(result, "metadata", "", "manifest", "warning", "manifest has no seed")

    dictionary_path = dataset_dir / "metadata" / "data_dictionary.csv"
    if not dictionary_path.exists():
        _add(result, "metadata", "", "metadata", "hard", "missing metadata/data_dictionary.csv")
    else:
        with dictionary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        present = {(row.get("table", ""), row.get("field", "")) for row in rows}
        for table, fields in TABLES.items():
            for field in fields:
                if (table, field.name) not in present:
                    _add(
                        result, "metadata", f"{table}.{field.name}", "metadata", "hard",
                        "field missing from data_dictionary.csv",
                    )

    for name in ("source_manifest.csv", "schema_documentation.md"):
        if not (dataset_dir / "metadata" / name).exists():
            _add(result, "metadata", "", "metadata", "hard", f"missing metadata/{name}")


def _read_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    path = dataset_dir / "metadata" / "generation_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _capacity_stats(
    products: list[dict[str, str]],
    locations: list[dict[str, str]],
    assignments: list[dict[str, str]],
    product_by_sku: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total_volume = sum(float(row["usable_volume_cm3"]) for row in locations)
    total_weight = sum(float(row["max_weight_kg"]) for row in locations)
    used_volume = 0.0
    used_weight = 0.0
    used_locations = set()
    for row in assignments:
        product = product_by_sku[row["sku_id"]]
        quantity = int(row["assigned_quantity"])
        used_volume += quantity * float(product["unit_volume_cm3"])
        used_weight += quantity * float(product["unit_weight_kg"])
        used_locations.add(row["location_id"])
    return {
        "locations": len(locations),
        "used_locations": len(used_locations),
        "empty_locations": len(locations) - len(used_locations),
        "total_usable_volume_cm3": round(total_volume, 2),
        "occupied_volume_cm3": round(used_volume, 2),
        "volume_utilization": round(used_volume / total_volume, 6) if total_volume else 0.0,
        "total_bin_weight_capacity_kg": round(total_weight, 2),
        "stored_weight_kg": round(used_weight, 2),
        "weight_utilization": round(used_weight / total_weight, 6) if total_weight else 0.0,
    }


def _demand_stats(demand: list[dict[str, str]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for scenario in SCENARIO_IDS:
        rows = [row for row in demand if row["scenario_id"] == scenario]
        counts = {abc: sum(1 for row in rows if row["abc_class"] == abc) for abc in ABC_CLASSES}
        stats[scenario] = {
            "skus": len(rows),
            "class_counts": counts,
            "class_share": {
                abc: round(
                    sum(float(row["cumulative_share"]) for row in rows if row["abc_class"] == abc)
                    / max(1, len(rows)),
                    6,
                )
                for abc in ABC_CLASSES
            },
        }
    return stats


def _scenario_counts(
    orders: list[dict[str, str]], order_items: list[dict[str, str]]
) -> dict[str, Any]:
    scenario_by_order = {row["order_id"]: row["scenario_id"] for row in orders}
    result: dict[str, Any] = {}
    for scenario in SCENARIO_IDS:
        result[scenario] = {
            "orders": sum(1 for row in orders if row["scenario_id"] == scenario),
            "lines": sum(1 for row in order_items if scenario_by_order.get(row["order_id"]) == scenario),
        }
    return result


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_validation_reports(dataset_dir: Path, result: ValidationResult) -> None:
    reports_dir = dataset_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    errors_path = reports_dir / "validation_errors.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["table", "row_id", "check", "severity", "message"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.errors)

    lines = [
        "# Dataset Validation Report",
        "",
        f"- Status: **{'PASS' if result.ok else 'FAIL'}**",
        f"- Hard errors: **{result.hard_errors}**",
        f"- Warnings: **{result.warnings}**",
        "",
        "## Row counts",
        "",
        "| table | rows |",
        "|---|---|",
    ]
    for table, count in sorted(result.stats.get("row_counts", {}).items()):
        lines.append(f"| {table} | {count} |")

    capacity = result.stats.get("capacity", {})
    if capacity:
        lines.extend(
            [
                "",
                "## Capacity",
                "",
                f"- Locations: {capacity['locations']} ({capacity['used_locations']} used, {capacity['empty_locations']} empty)",
                f"- Volume utilization: {capacity['volume_utilization'] * 100:.2f}%",
                f"- Weight utilization: {capacity['weight_utilization'] * 100:.2f}%",
                f"- Occupied volume: {capacity['occupied_volume_cm3']:,.0f} cm3 of {capacity['total_usable_volume_cm3']:,.0f} cm3",
                f"- Stored weight: {capacity['stored_weight_kg']:,.1f} kg of {capacity['total_bin_weight_capacity_kg']:,.1f} kg",
            ]
        )

    scenario_counts = result.stats.get("scenario_counts", {})
    if scenario_counts:
        lines.extend(["", "## Demand scenarios", "", "| scenario | orders | lines |", "|---|---|---|"])
        for scenario, counts in sorted(scenario_counts.items()):
            lines.append(f"| {scenario} | {counts['orders']} | {counts['lines']} |")

    demand_stats = result.stats.get("demand", {})
    if demand_stats:
        lines.extend(
            [
                "",
                "## ABC classification",
                "",
                "| scenario | SKUs | A | B | C |",
                "|---|---|---|---|---|",
            ]
        )
        for scenario, stats in sorted(demand_stats.items()):
            counts = stats["class_counts"]
            lines.append(
                f"| {scenario} | {stats['skus']} | {counts['A']} | {counts['B']} | {counts['C']} |"
            )

    lines.extend(["", "## Checks", "", "| check | severity | count |", "|---|---|---|"])
    summary: dict[tuple[str, str], int] = defaultdict(int)
    for error in result.errors:
        summary[(error["check"], error["severity"])] += 1
    for (check, severity), count in sorted(summary.items()):
        lines.append(f"| {check} | {severity} | {count} |")

    if result.errors:
        lines.extend(["", "## Errors (first 100)", "", "| table | row | check | severity | message |", "|---|---|---|---|---|"])
        for error in result.errors[:100]:
            lines.append(
                f"| {error['table']} | {error['row_id']} | {error['check']} | "
                f"{error['severity']} | {error['message']} |"
            )
        lines.extend(["", "Full list: `reports/validation_errors.csv`."])

    lines.extend(
        [
            "",
            "## Assumptions",
            "",
            "- All rows are synthetic or derived; no measured warehouse data is present.",
            "- Volume is the full bounding-box volume; packing clearance is not modelled.",
            "- One SKU per bin; a SKU may occupy several bins.",
            "- Demand scenarios are alternative histories, not additive.",
            "",
        ]
    )
    (reports_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
