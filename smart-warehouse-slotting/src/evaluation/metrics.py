"""Evaluation metrics for baseline-vs-optimized comparison (canonical, inventory-fair).

Rules:
- Both layouts hold the SAME real inventory (a layout only relocates stock; it never changes
  totals). Utilization, weight, and conservation are therefore comparable.
- Identical order workload, geometry, and routing applied to both.
- The baseline is validated too; violations are REPORTED, not hidden.
- Utilization delta in percentage points; distance reduction in %.

Documented definitions (also surfaced by the dashboard):

- Cube utilization (%) = occupied inventory volume / total usable storage volume x 100,
  where occupied volume = sum(on-hand assigned quantity x SKU bounding-box volume). This is
  a volume-based capacity approximation, NOT an exact 3-D packing calculation.
- Assignment feasibility (%) = valid assignments / required assignments x 100. A required
  assignment is one SKU that holds on-hand inventory; it is valid when the whole on-hand
  quantity is placed and no validation check for that SKU (or a bin holding it) failed.
- Rule-check pass rate (%) = passed rule checks / rule checks run x 100.
- Golden-zone demand share (%) = sum(demand x units) stored on golden shelf levels /
  sum(demand x units) over all placed units x 100.
"""

from __future__ import annotations

from typing import Callable

GOLDEN_DEFAULT = (1, 2)


def _slotting_units(slotting_map: dict[str, list[tuple[str, int]]]) -> dict[str, int]:
    return {sku: sum(q for _loc, q in spans) for sku, spans in slotting_map.items()}


def _failed_sku_ids(result) -> set[str]:
    """SKUs implicated by a failed check, including bins that hold them."""
    failed = set()
    checks = getattr(result, "checks", None) or []
    if checks:
        for check in checks:
            if not check.passed and check.sku_id:
                failed.add(check.sku_id)
        return failed
    for v in getattr(result, "violations", []) or []:
        if v.sku_id:
            failed.add(v.sku_id)
    return failed


def assignment_feasibility(
    slotting_map: dict[str, list[tuple[str, int]]],
    inventory: dict[str, int] | None,
    result,
) -> dict:
    """Valid assignments / required assignments, with explicit numerator and denominator."""
    required_skus = sorted(inventory) if inventory else sorted(slotting_map)
    placed = _slotting_units(slotting_map)
    failed = _failed_sku_ids(result)

    # A failure recorded against a bin (no SKU id) invalidates the SKU(s) stored there.
    checks = getattr(result, "checks", None) or []
    if checks and inventory:
        bin_to_sku = {loc: sku for sku, spans in slotting_map.items() for loc, _q in spans}
        for check in checks:
            if check.passed or check.sku_id or not check.location_id:
                continue
            sku = bin_to_sku.get(str(check.location_id))
            if sku:
                failed.add(sku)

    valid = 0
    for sku in required_skus:
        required_qty = inventory.get(sku, 0) if inventory else placed.get(sku, 0)
        if placed.get(sku, 0) == required_qty and sku not in failed:
            valid += 1
    required = len(required_skus)
    pct = (100.0 * valid / required) if required else 0.0
    return {
        "valid_assignments": valid,
        "required_assignments": required,
        "assignment_feasibility_pct": round(pct, 2),
        "unplaced_skus": required - valid,
    }


def golden_zone_metrics(
    slotting_map: dict[str, list[tuple[str, int]]],
    locations: dict,
    demand: dict[str, float] | None,
    abc_classes: dict[str, str] | None,
    golden_levels: tuple[int, ...] = GOLDEN_DEFAULT,
) -> dict:
    """Demand-weighted share of units pickable from the ergonomic golden zone.

    Shelf levels in ``golden_levels`` are the configured golden band. If the topology does
    not carry an ergonomic-zone label, the shelf level decides.
    """
    demand = demand or {}
    abc_classes = abc_classes or {}
    golden = set(golden_levels)
    weighted_total = 0.0
    weighted_golden = 0.0
    class_a_skus = 0
    class_a_outside = 0
    for sku, spans in slotting_map.items():
        qty = sum(q for _loc, q in spans)
        if qty <= 0:
            continue
        w = float(demand.get(sku, 0.0)) * qty
        weighted_total += w
        golden_units = 0
        for loc_id, q in spans:
            loc = locations.get(loc_id)
            if loc is None:
                continue
            if getattr(loc, "shelf_level", None) in golden:
                golden_units += q
                weighted_golden += float(demand.get(sku, 0.0)) * q
        if abc_classes.get(sku) == "A":
            class_a_skus += 1
            if golden_units == 0:
                class_a_outside += 1
    pct = (100.0 * weighted_golden / weighted_total) if weighted_total else 0.0
    return {
        "golden_zone_demand_share_pct": round(pct, 2),
        "class_a_skus": class_a_skus,
        "class_a_skus_outside_golden": class_a_outside,
    }


def affinity_separation(
    pair_weights: dict[tuple[str, str], float],
    primary_bins: dict[str, str],
    distance: Callable[[str, str], float],
) -> dict:
    """Co-occurrence-weighted average aisle distance between related SKU pairs.

    ``pair_weights`` maps an unordered SKU pair to its association weight (co-order count).
    Pairs whose SKUs are unassigned or unreachable are excluded and counted, so the reported
    average never silently pretends full coverage.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    evaluated = 0
    skipped = 0
    for (a, b), weight in pair_weights.items():
        la = primary_bins.get(a)
        lb = primary_bins.get(b)
        if la is None or lb is None or la == lb:
            skipped += 1
            continue
        d = distance(la, lb)
        if d is None or d != d or d == float("inf"):
            skipped += 1
            continue
        weighted_sum += float(weight) * float(d)
        weight_total += float(weight)
        evaluated += 1
    avg = (weighted_sum / weight_total) if weight_total else None
    return {
        "affinity_separation_m": round(avg, 3) if avg is not None else None,
        "affinity_pairs_evaluated": evaluated,
        "affinity_pairs_skipped": skipped,
    }


def summarize_layout(
    slotting_map: dict[str, list[tuple[str, int]]],  # sku -> [(location_id, qty), ...]
    skus: dict,
    locations: dict,
    route_summary: dict,
    result,  # ValidationResult
    runtime_sec: float | None,
    inventory: dict[str, int] | None = None,
    demand: dict[str, float] | None = None,
    abc_classes: dict[str, str] | None = None,
    golden_levels: tuple[int, ...] = GOLDEN_DEFAULT,
    extra: dict | None = None,
) -> dict:
    total_cap = sum(l.usable_volume_cm3 for l in locations.values())
    used_vol = 0.0
    for sku_id, spans in slotting_map.items():
        sku = skus.get(sku_id)
        if sku is None:
            continue
        qty = sum(q for _l, q in spans)
        used_vol += sku.unit_volume_cm3 * qty

    utilization = (used_vol / total_cap * 100) if total_cap > 0 else 0.0
    n_violations = len(result.violations) if result is not None else 0
    checks = result.checks_run if result is not None else 0
    feasibility = assignment_feasibility(slotting_map, inventory, result)
    rule_pass = (100.0 * (checks - n_violations) / checks) if checks else 0.0
    ergo = golden_zone_metrics(slotting_map, locations, demand, abc_classes, golden_levels)

    total_distance = route_summary.get("total_distance_m", 0.0)
    total_orders = route_summary.get("total_orders", 0)
    metrics = {
        "total_travel_distance_m": total_distance,
        "avg_travel_distance_m": route_summary.get("avg_distance_m", 0.0),
        "avg_distance_per_order_m": round(total_distance / total_orders, 2) if total_orders else 0.0,
        "total_route_time_sec": route_summary.get(
            "total_route_time_sec", route_summary.get("total_cycle_time_sec", 0.0)
        ),
        "avg_route_time_sec": route_summary.get(
            "avg_cycle_time_sec", route_summary.get("avg_route_time_sec", 0.0)
        ),
        "avg_time_per_order_sec": route_summary.get("avg_time_per_order_sec", 0.0),
        "throughput_orders_per_hr": route_summary.get("throughput_orders_per_hr", 0.0),
        "cube_utilization_pct": round(utilization, 2),
        "cube_occupied_volume_cm3": round(used_vol, 2),
        "cube_total_usable_volume_cm3": round(total_cap, 2),
        "constraint_violations": n_violations,
        "safety_compliant": n_violations == 0,
        "assignment_feasibility_pct": feasibility["assignment_feasibility_pct"],
        "valid_assignments": feasibility["valid_assignments"],
        "required_assignments": feasibility["required_assignments"],
        "rule_check_pass_pct": round(rule_pass, 2),
        "checks_run": checks,
        "golden_zone_demand_share_pct": ergo["golden_zone_demand_share_pct"],
        "class_a_skus": ergo["class_a_skus"],
        "class_a_skus_outside_golden": ergo["class_a_skus_outside_golden"],
        "optimization_runtime_sec": runtime_sec,
        "n_batches": route_summary.get("n_batches", 0),
        "total_orders": route_summary.get("total_orders", 0),
    }
    if extra:
        metrics.update(extra)
    return metrics


# (metric key, unit, direction, change kind, note) — change kind drives how a comparison
# is expressed: "reduction"/"improvement" = relative %, "points" = percentage points,
# "abs" = absolute only, "bool" = status only.
METRIC_SPECS: list[tuple[str, str, str, str, str]] = [
    ("total_travel_distance_m", "m", "lower is better", "reduction", "Total travel distance over identical pick batches"),
    ("avg_travel_distance_m", "m", "lower is better", "reduction", "Average distance per pick batch"),
    ("avg_distance_per_order_m", "m", "lower is better", "reduction", "Average travel distance per order"),
    ("total_route_time_sec", "s", "lower is better", "reduction", "Total pick-cycle time (travel + pick + setup)"),
    ("avg_route_time_sec", "s", "lower is better", "reduction", "Average pick-cycle time per batch"),
    ("avg_time_per_order_sec", "s", "lower is better", "reduction", "Pick-cycle time per order"),
    ("throughput_orders_per_hr", "orders/h", "higher is better", "improvement", "Orders completed / total pick-cycle hours"),
    ("cube_utilization_pct", "%", "reference", "points", "Occupied inventory volume / usable storage volume"),
    ("golden_zone_demand_share_pct", "%", "higher is better", "points", "Demand-weighted units stored on golden shelf levels"),
    ("class_a_skus_outside_golden", "SKUs", "lower is better", "abs", "Class A SKUs with no unit on a golden shelf level"),
    ("affinity_separation_m", "m", "lower is better", "reduction", "Co-occurrence-weighted aisle distance between related SKUs"),
    ("constraint_violations", "violations", "must be 0", "abs", "Hard constraint violations"),
    ("valid_assignments", "SKUs", "must equal required", "abs", "SKUs fully placed with no failed check"),
    ("required_assignments", "SKUs", "reference", "abs", "SKUs with on-hand inventory requiring a location"),
    ("assignment_feasibility_pct", "%", "must be 100%", "pct", "Valid assignments / required assignments"),
    ("checks_run", "checks", "reference", "abs", "Rule checks evaluated by the validator"),
    ("rule_check_pass_pct", "%", "must be 100%", "pct", "Rule checks passed / rule checks run"),
    ("safety_compliant", "bool", "must be true", "bool", "No hard constraint violations"),
]


def compare_baseline_vs_optimized(baseline_metrics: dict, optimized_metrics: dict) -> list[dict]:
    rows = []
    for key, unit, direction, kind, note in METRIC_SPECS:
        if key not in baseline_metrics and key not in optimized_metrics:
            continue
        b = baseline_metrics.get(key)
        o = optimized_metrics.get(key)
        diff = None
        change = None
        if isinstance(b, bool) or isinstance(o, bool):
            pass
        elif isinstance(b, (int, float)) and isinstance(o, (int, float)):
            diff = o - b
            if b != 0 and kind in ("reduction", "improvement"):
                change = (diff / abs(b)) * 100.0
        rows.append({
            "metric": key,
            "note": note,
            "direction": direction,
            "unit": unit,
            "change_kind": kind,
            "baseline": b,
            "optimized": o,
            "absolute_diff": round(diff, 3) if isinstance(diff, (int, float)) else diff,
            "percent_change": round(change, 2) if isinstance(change, (int, float)) else change,
        })
    return rows


def travel_distance_reduction(baseline: float, optimized: float) -> float:
    """Reduction (%) = (baseline - optimized) / baseline x 100; 0 when baseline is 0."""
    return (baseline - optimized) / baseline * 100.0 if baseline else 0.0


def utilization_delta_pp(baseline: float, optimized: float) -> float:
    return optimized - baseline
