"""Optimization objective (demand-weighted travel + proximity-based affinity + ergonomics).

minimize  alpha*D + beta*A + gamma*E + delta*U

- D (travel): demand-weighted distance-to-dispatch. Demand is the SKU's real demand
  frequency, NOT quantity — quantity is fixed real inventory, never fabricated.
- A (affinity): penalty for distance between co-ordered SKU pairs. Because one SKU per bin is
  enforced, affinity rewards placing co-ordered SKUs in NEARBY bins (not co-located).
- E (ergonomics): penalize high-demand SKUs not in the golden zone.
- U (utilization): penalize under-filled bins.

Safety/feasibility and inventory conservation are HARD constraints enforced by the validator,
never part of this objective.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.entities import SKU, StorageAssignment, Warehouse

GOLDEN_SHELF_LEVELS = {2}


@dataclass
class Objective:
    alpha: float = 1.0
    beta: float = 0.4
    gamma: float = 0.3
    delta: float = 0.2

    def evaluate(
        self,
        assignments: list[StorageAssignment],
        skus: dict[str, SKU],
        warehouse: Warehouse,
        demand: dict[str, float] | None = None,
        affinity_penalty: dict[frozenset[str], float] | None = None,
    ) -> float:
        """Lower is better."""
        if not assignments:
            return 0.0
        demand = demand or {}
        D = _travel_cost(assignments, skus, warehouse, demand)
        E = _ergonomics(assignments, skus, warehouse, demand)
        U = _utilization(assignments, skus, warehouse)
        A = _affinity(assignments, skus, warehouse, affinity_penalty) if affinity_penalty else 0.0
        return self.alpha * D + self.beta * A + self.gamma * E + self.delta * U


def _travel_cost(assignments, skus, warehouse, demand) -> float:
    total = 0.0
    for a in assignments:
        sku = skus.get(a.sku_id)
        loc = warehouse.locations.get(a.location_id)
        if sku is None or loc is None:
            continue
        w = demand.get(a.sku_id, 1.0)
        dist = loc.distance_to_dispatch_m if loc.distance_to_dispatch_m > 0 else (abs(loc.x - 20.4) + abs(loc.y - 0.5))
        total += w * dist
    return total


def _ergonomics(assignments, skus, warehouse, demand) -> float:
    """Penalize high-demand SKUs stored outside the golden shelf levels (near dispatch)."""
    total = 0.0
    for a in assignments:
        loc = warehouse.locations.get(a.location_id)
        if loc is None:
            continue
        w = demand.get(a.sku_id, 1.0)
        if loc.shelf_level not in GOLDEN_SHELF_LEVELS:
            total += w * 0.2  # mild ergonomic penalty for non-golden shelf
    return total


def _utilization(assignments, skus, warehouse) -> float:
    """Lower is better: penalize empty/wasted bin volume."""
    cap = {lid: loc.usable_volume_cm3 for lid, loc in warehouse.locations.items()}
    used = {lid: 0.0 for lid in cap}
    for a in assignments:
        sku = skus.get(a.sku_id)
        if sku is None or a.location_id not in used:
            continue
        used[a.location_id] += sku.unit_volume_cm3 * a.assigned_quantity
    total = 0.0
    for lid in used:
        if cap[lid] <= 0:
            continue
        total += 1.0 - min(1.0, used[lid] / cap[lid])
    return total


def _affinity(assignments, skus, warehouse, affinity_penalty) -> float:
    """Penalty = sum over co-ordered pairs of (penalty * distance between their bins).

    Higher-lift / higher-coorder pairs carry higher penalty weight, so the optimizer pulls
    them into adjacent bins. Length scale normalized by a representative distance (50 m).
    """
    if not assignments:
        return 0.0
    bin_of_sku: dict[str, str] = {}
    for a in assignments:
        bin_of_sku.setdefault(a.sku_id, a.location_id)
    scale = 50.0
    total = 0.0
    for (a, b), penalty in affinity_penalty.items():
        la = bin_of_sku.get(a)
        lb = bin_of_sku.get(b)
        if la is None or lb is None or la == lb:
            continue
        d = _bin_distance(warehouse, la, lb)
        total += penalty * (d / scale)
    return total


def _bin_distance(warehouse, la, lb) -> float:
    loc_a = warehouse.locations.get(la)
    loc_b = warehouse.locations.get(lb)
    if loc_a is None or loc_b is None:
        return 50.0
    return abs(loc_a.x - loc_b.x) + abs(loc_a.y - loc_b.y) + abs(loc_a.z - loc_b.z)


def build_affinity_penalty(affinity_df, co_count_floor: int = 2, lift_floor: float = 1.0) -> dict[frozenset[str], float]:
    """From the canonical sku_affinity table, build penalty weight per SKU pair.

    penalty = co_count (higher co-occurrence -> stronger pull to be adjacent).
    Filters out very rare / weak pairs.
    """
    penalty = {}
    for _, r in affinity_df.iterrows():
        if r.get("co_count", 0) < co_count_floor:
            continue
        if r.get("lift", 0) < lift_floor:
            continue
        pair = frozenset((r["sku_a"], r["sku_b"]))
        # keep the max co_count for the unordered pair
        penalty[pair] = max(penalty.get(pair, 0), int(r["co_count"]))
    return penalty