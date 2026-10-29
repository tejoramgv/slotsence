"""Pick-order batching and pick-cycle-time simulation.

Given a fixed order workload, a slotting layout (sku -> list of (location, quantity)), and the
aisle route graph, simulates travel distance and completion time. Cycle time = travel time +
pick time + setup time.

Deterministic: batches/orders/SKUs are sorted before iteration so results do not depend on
hash order across processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SimulationConfig:
    batch_size: int = 5
    routing_strategy: str = "nearest_neighbor"
    picker_speed_mps: float = 1.3
    pick_time_sec: float = 8.0
    setup_time_sec: float = 10.0


@dataclass
class RouteResult:
    batch_id: int
    n_orders: int
    n_locations: int
    distance_m: float
    travel_time_sec: float
    pick_time_sec: float
    cycle_time_sec: float
    route: tuple[str, ...] = ()
    legs: tuple[float, ...] = ()

    @property
    def n_pick_lines(self) -> int:
        return int(round(self.pick_time_sec / 8.0)) if self.pick_time_sec else 0


def build_batches(order_items: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Group orders into batches of up to batch_size orders (sorted for determinism)."""
    orders = sorted(order_items["order_id"].unique())
    batches = []
    for i in range(0, len(orders), batch_size):
        ids = orders[i : i + batch_size]
        batches.append(order_items[order_items["order_id"].isin(ids)])
    return batches


def simulate_layout(
    order_items: pd.DataFrame,
    slotting: dict[str, list[str]],  # sku -> locations where its inventory is stored
    location_dist: dict,
    depot_dist: dict[str, float],
    cfg: SimulationConfig,
    depot: str,
) -> list[RouteResult]:
    """Simulate picking over all orders for one slotting layout.

    A batcher visits the distinct bins holding the SKUs of its orders. Slotting maps each
    SKU to the bin(s) where its stock is placed. The ordered stop sequence and each leg's
    distance are kept so routes can be drawn and audited.
    """
    batches = build_batches(order_items, cfg.batch_size)
    results: list[RouteResult] = []
    from ..routing.heuristics import ROUTING_STRATEGIES, route_legs

    route_fn = ROUTING_STRATEGIES[cfg.routing_strategy]
    for bi, batch in enumerate(batches):
        skus = sorted(set(batch["sku_id"]))
        pick_locations = []
        for s in skus:
            for loc in slotting.get(s, []):
                if loc not in pick_locations:
                    pick_locations.append(loc)
        n_pick_lines = len(batch)
        route, distance = route_fn(pick_locations, location_dist, depot_dist, depot)
        legs = route_legs(route, location_dist, depot_dist, depot)
        travel_time = distance / cfg.picker_speed_mps
        pick_time = n_pick_lines * cfg.pick_time_sec
        cycle_time = travel_time + pick_time + cfg.setup_time_sec
        results.append(
            RouteResult(
                batch_id=bi,
                n_orders=len(batch["order_id"].unique()),
                n_locations=len(pick_locations),
                distance_m=distance,
                travel_time_sec=travel_time,
                pick_time_sec=pick_time,
                cycle_time_sec=cycle_time,
                route=tuple(route),
                legs=tuple(legs),
            )
        )
    return results


def summarize(results: list[RouteResult], cfg: SimulationConfig | None = None) -> dict:
    if not results:
        return {
            "n_batches": 0,
            "total_orders": 0,
            "total_pick_lines": 0,
            "total_distance_m": 0.0,
            "avg_distance_m": 0.0,
            "total_route_time_sec": 0.0,
            "avg_cycle_time_sec": 0.0,
            "avg_time_per_order_sec": 0.0,
            "total_travel_time_sec": 0.0,
            "total_pick_time_sec": 0.0,
            "total_setup_time_sec": 0.0,
            "throughput_orders_per_hr": 0.0,
        }
    pick_time_sec = cfg.pick_time_sec if cfg is not None else 8.0
    setup_time_sec = cfg.setup_time_sec if cfg is not None else 10.0
    total_dist = sum(r.distance_m for r in results)
    total_cycle = sum(r.cycle_time_sec for r in results)
    total_orders = sum(r.n_orders for r in results)
    total_travel = sum(r.travel_time_sec for r in results)
    total_pick = sum(r.pick_time_sec for r in results)
    n_pick_lines = int(round(total_pick / pick_time_sec)) if pick_time_sec else 0
    return {
        "n_batches": len(results),
        "total_orders": total_orders,
        "total_pick_lines": n_pick_lines,
        "total_distance_m": round(total_dist, 2),
        "avg_distance_m": round(total_dist / len(results), 2),
        "total_route_time_sec": round(total_cycle, 2),
        "avg_cycle_time_sec": round(total_cycle / len(results), 2),
        "avg_time_per_order_sec": round(total_cycle / total_orders, 2) if total_orders else 0.0,
        "total_travel_time_sec": round(total_travel, 2),
        "total_pick_time_sec": round(total_pick, 2),
        "total_setup_time_sec": round(len(results) * setup_time_sec, 2),
        "throughput_orders_per_hr": round(total_orders / (total_cycle / 3600), 2) if total_cycle > 0 else 0.0,
    }
