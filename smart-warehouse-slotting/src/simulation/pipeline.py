"""Phase-6 orchestration: baseline-vs-optimized route simulation + evaluation (canonical)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import (
    affinity_separation,
    compare_baseline_vs_optimized,
    summarize_layout,
)
from src.models.entities import AssignmentType, SKU, StorageAssignment
from src.routing.graph import build_graph, depot_distance_map, find_depot_node, location_distance_map
from src.simulation.simulator import RouteResult, SimulationConfig, build_batches, simulate_layout, summarize
from src.validation.audit import write_constraint_audit, write_validation_checks
from src.validation.loader import load_compatibility_rules, load_skus, load_warehouse_from_locations
from src.validation.validator import validate_assignment


def _slotting_map(df: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """sku -> list of (location_id, qty) — preserves multi-bin inventory placement."""
    out: dict[str, list[tuple[str, int]]] = {}
    for _, r in df.iterrows():
        out.setdefault(r["sku_id"], []).append((r["location_id"], int(r["assigned_quantity"])))
    return out


def _to_assignments(slotting: dict[str, list[tuple[str, int]]]) -> list[StorageAssignment]:
    return [
        StorageAssignment(sku_id=s, location_id=loc, assigned_quantity=q, assignment_type=AssignmentType.OPTIMIZED)
        for s, spans in slotting.items()
        for loc, q in spans
    ]


def _primary_bins(slotting: dict[str, list[tuple[str, int]]]) -> dict[str, str]:
    return {sku: sorted({loc for loc, _q in spans})[0] for sku, spans in slotting.items() if spans}


def _pair_distance(location_dist, a: str, b: str) -> float:
    key = (a, b) if a <= b else (b, a)
    value = location_dist.get(key)
    return float("inf") if value is None else float(value)


def _write_route_stops(
    order_items: pd.DataFrame,
    slotting: dict[str, list[tuple[str, int]]],
    results: list[RouteResult],
    cfg: SimulationConfig,
    depot: str,
    path: Path,
    layout: str,
) -> None:
    """Write the ordered stop sequence and leg distances used by the simulator."""
    batches = build_batches(order_items, cfg.batch_size)
    sku_locations = {sku: {loc for loc, _q in spans} for sku, spans in slotting.items()}
    rows = []
    for result, batch in zip(results, batches):
        batch_skus = sorted(set(batch["sku_id"]))
        cumulative = 0.0
        for seq, (node, leg) in enumerate(zip(result.route, (0.0, *result.legs))):
            cumulative += leg
            if node == depot:
                skus = ""
            else:
                skus = ";".join(s for s in batch_skus if node in sku_locations.get(s, set()))
            rows.append({
                "batch_id": result.batch_id,
                "layout": layout,
                "sequence": seq,
                "location_id": node,
                "skus": skus,
                "leg_distance_m": round(leg, 3),
                "cumulative_distance_m": round(cumulative, 3),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_phase6(
    processed: Path,
    scenario: str = "A",
    out: Path | None = None,
    optimized_path: Path | None = None,
    affinity_scenario: str = "C",
) -> dict:
    locations_df = pd.read_csv(processed / "locations.csv")
    products_df = pd.read_csv(processed / "products.csv")
    rules_df = pd.read_csv(processed / "compatibility_rules.csv")
    inventory_df = pd.read_csv(processed / "inventory.csv")
    demand_df = pd.read_csv(processed / "demand_summary.csv")
    affinity_df = pd.read_csv(processed / "sku_affinity.csv")

    skus: dict[str, SKU] = load_skus(products_df)
    wh = load_warehouse_from_locations(locations_df)
    comp = load_compatibility_rules(rules_df)
    inventory = inventory_df.groupby("sku_id")["quantity_on_hand"].sum().to_dict()

    # Demand / ABC for the evaluated scenario (the workload scenario), affinity for the
    # association scenario the optimizer actually used.
    demand_rows = demand_df[demand_df["scenario_id"] == scenario]
    demand = dict(zip(demand_rows["sku_id"], demand_rows["demand_frequency"].astype(float)))
    abc_classes = dict(zip(demand_rows["sku_id"], demand_rows["abc_class"]))

    # Golden band comes from the topology's ergonomic labels, not a UI constant.
    golden_levels = tuple(sorted({
        loc.shelf_level for loc in wh.locations.values()
        if getattr(loc.ergonomic_zone, "value", "") == "golden"
    })) or (1, 2)

    aff = affinity_df[affinity_df["scenario_id"] == affinity_scenario]
    pair_weights: dict[tuple[str, str], float] = {}
    for _, r in aff.iterrows():
        if r["co_count"] < 2 or r["lift"] < 1:
            continue
        key = tuple(sorted((str(r["sku_a"]), str(r["sku_b"]))))
        pair_weights[key] = max(pair_weights.get(key, 0.0), float(r["co_count"]))

    # Build aisle network graph + depot distances (deterministic).
    graph = build_graph(processed / "location_graph.csv")
    depot = find_depot_node(graph)
    if depot is None:
        raise ValueError("no depot node found in location_graph.csv")
    depot_dist = depot_distance_map(graph, depot)

    # Slottings
    baseline_df = pd.read_csv(processed / "storage_assignments_baseline.csv")
    project_root = Path(__file__).resolve().parent.parent.parent
    opt_path = (
        Path(optimized_path)
        if optimized_path is not None
        else project_root / "outputs" / "optimized" / "optimized_assignments.csv"
    )
    if not opt_path.exists():
        raise FileNotFoundError(f"optimized assignments not found: {opt_path}")
    optimized_df = pd.read_csv(opt_path)

    baseline_slot = _slotting_map(baseline_df)
    optimized_slot = _slotting_map(optimized_df)

    # Same real inventory totals are assumed for BOTH layouts (conservation is validator-checked).
    order_items = pd.read_csv(processed / "order_items.csv")
    orders = pd.read_csv(processed / "orders.csv")
    sc_orders = set(orders[orders["scenario_id"] == scenario]["order_id"])
    order_items = order_items[order_items["order_id"].isin(sc_orders)].reset_index(drop=True)

    cfg = SimulationConfig(routing_strategy="nearest_neighbor")

    # Precompute pairwise location distances for the involved bins (deterministic, sorted).
    all_bins = sorted({loc for spans in list(baseline_slot.values()) + list(optimized_slot.values()) for loc, _ in spans})
    location_dist = location_distance_map(graph, all_bins)

    base_slot_locs = {sku: [l for l, _q in spans] for sku, spans in baseline_slot.items()}
    opt_slot_locs = {sku: [l for l, _q in spans] for sku, spans in optimized_slot.items()}

    base_results = simulate_layout(order_items, base_slot_locs, location_dist, depot_dist, cfg, depot)
    opt_results = simulate_layout(order_items, opt_slot_locs, location_dist, depot_dist, cfg, depot)
    base_sum = summarize(base_results, cfg)
    opt_sum = summarize(opt_results, cfg)

    base_assign = _to_assignments(baseline_slot)
    opt_assign = _to_assignments(optimized_slot)
    base_val = validate_assignment(base_assign, skus, wh, comp, inventory)
    opt_val = validate_assignment(opt_assign, skus, wh, comp, inventory)

    base_affinity = affinity_separation(pair_weights, _primary_bins(baseline_slot), lambda a, b: _pair_distance(location_dist, a, b))
    opt_affinity = affinity_separation(pair_weights, _primary_bins(optimized_slot), lambda a, b: _pair_distance(location_dist, a, b))

    runtime = 0.0
    sum_path = opt_path.parent / "optimization_summary.json"
    if sum_path.exists():
        runtime = json.loads(sum_path.read_text()).get("runtime_sec", 0.0)

    base_metrics = summarize_layout(
        baseline_slot,
        skus,
        wh.locations,
        base_sum,
        base_val,
        runtime_sec=None,
        inventory=inventory,
        demand=demand,
        abc_classes=abc_classes,
        golden_levels=golden_levels,
        extra=base_affinity,
    )
    opt_metrics = summarize_layout(
        optimized_slot,
        skus,
        wh.locations,
        opt_sum,
        opt_val,
        runtime_sec=runtime,
        inventory=inventory,
        demand=demand,
        abc_classes=abc_classes,
        golden_levels=golden_levels,
        extra=opt_affinity,
    )
    table = compare_baseline_vs_optimized(base_metrics, opt_metrics)

    result = {
        "dataset": str(processed),
        "scenario": scenario,
        "affinity_scenario": affinity_scenario,
        "inventory_units": sum(inventory.values()),
        "skus_with_inventory": len(inventory),
        "baseline": base_metrics,
        "optimized": opt_metrics,
        "comparison": table,
        "assumptions": {
            "routing_strategy": cfg.routing_strategy,
            "batch_size_orders": cfg.batch_size,
            "picker_speed_mps": cfg.picker_speed_mps,
            "pick_time_sec_per_line": cfg.pick_time_sec,
            "setup_time_sec_per_batch": cfg.setup_time_sec,
            "golden_shelf_levels": list(golden_levels),
            "cube_utilization_basis": "on-hand assigned quantity x SKU bounding-box volume / usable bin volume (volume approximation, not 3-D packing)",
            "throughput_definition": "orders completed / total pick-cycle hours (travel + pick + setup)",
            "affinity_weighting": "co-occurrence count for pairs with co_count >= 2 and lift >= 1",
            "depot_node": depot,
        },
        "route_breakdown": {"baseline": base_sum, "optimized": opt_sum},
        "notes": {
            "identical_orders_geometry_routing": True,
            "routing_strategy": cfg.routing_strategy,
            "batch_size": cfg.batch_size,
            "baseline_was_validated": True,
            "baseline_constraint_violations": len(base_val.violations),
            "optimized_constraint_violations": len(opt_val.violations),
            "baseline_rule_checks": base_val.checks_run,
            "optimized_rule_checks": opt_val.checks_run,
            "baseline_inventory_conservation_violations": sum(1 for v in base_val.violations if v.rule_id.startswith("INV")),
            "optimized_inventory_conservation_violations": sum(1 for v in opt_val.violations if v.rule_id.startswith("INV")),
            "same_inventory_units": sum(inventory.values()),
        },
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        route_cols = ["batch_id", "n_orders", "n_locations", "distance_m", "travel_time_sec", "pick_time_sec", "cycle_time_sec"]
        base_df = pd.DataFrame([{k: getattr(r, k) for k in route_cols} for r in base_results])
        opt_df = pd.DataFrame([{k: getattr(r, k) for k in route_cols} for r in opt_results])
        base_df["layout"] = "baseline"
        opt_df["layout"] = "optimized"
        pd.concat([base_df, opt_df], ignore_index=True).to_csv(out / "pick_routes.csv", index=False)
        _write_route_stops(order_items, baseline_slot, base_results, cfg, depot, out / "pick_route_stops_baseline.csv", "baseline")
        _write_route_stops(order_items, optimized_slot, opt_results, cfg, depot, out / "pick_route_stops_optimized.csv", "optimized")
        pd.DataFrame(table).to_csv(out / "optimization_metrics.csv", index=False)
        write_validation_checks(base_val, out / "validation_checks_baseline.csv")
        write_validation_checks(opt_val, out / "validation_checks_optimized.csv")
        write_constraint_audit(base_assign, base_val, out / "constraint_audit_baseline.csv")
        write_constraint_audit(opt_assign, opt_val, out / "constraint_audit_optimized.csv")
        (out / "evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
