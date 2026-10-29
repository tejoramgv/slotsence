"""Phase-5 allocation optimizer (inventory-conserving).

Greedy feasible starter + simulated-annealing local search. Key correctness properties:

1. INVENTORY CONSERVATION: each SKU's real on-hand quantity must be placed across its bins —
   never fabricated, never dropped. If one on-hand exceeds a single bin's capacity/volume, the
   SKU spans multiple bins until the whole quantity is stored.
2. Every candidate (starter and each neighbor) is re-validated with the full Phase-4 validator
   (which now enforces zone/hazard/handling/heavy-height/structural/inventory/completeness).
3. Objective is demand-weighted travel + proximity-based affinity + ergonomics + utilization.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.models.entities import AssignmentType, SKU, StorageAssignment, Warehouse
from src.validation.validator import Validator

from .objective import Objective

GOLDEN_SHELF_LEVELS = {2}


@dataclass
class OptimizationLog:
    initial_score: float | None = None
    final_score: float | None = None
    n_moves_evaluated: int = 0
    n_moves_accepted: int = 0
    runtime_sec: float = 0.0
    starter_score: float | None = None
    feasibility_status: str = "not_run"
    stopping_reason: str = ""
    n_assigned: int = 0
    inventory_units: int = 0

    def to_dict(self) -> dict:
        return {
            "initial_objective": self.initial_score,
            "starter_objective": self.starter_score,
            "final_objective": self.final_score,
            "moves_evaluated": self.n_moves_evaluated,
            "moves_accepted": self.n_moves_accepted,
            "runtime_sec": round(self.runtime_sec, 3),
            "feasibility_status": self.feasibility_status,
            "stopping_reason": self.stopping_reason,
            "n_assigned": self.n_assigned,
            "inventory_units": self.inventory_units,
            "globally_optimal": False,
        }


@dataclass
class OptimizerConfig:
    seed: int = 7
    max_iterations: int = 600
    anneal_start: float = 5.0
    anneal_end: float = 0.01


class SlottingOptimizer:
    def __init__(
        self,
        warehouse: Warehouse,
        skus: dict[str, SKU],
        abc_classes: dict[str, str],
        validator: Validator,
        inventory: dict[str, int],
        affinity_penalty: dict[frozenset[str], float] | None = None,
        objective: Objective | None = None,
        demand: dict[str, float] | None = None,
        cfg: OptimizerConfig | None = None,
        initial_assignments: list[StorageAssignment] | None = None,
    ):
        self.warehouse = warehouse
        self.skus = skus
        self.abc = abc_classes
        self.validator = validator
        self.inventory = inventory
        self.affinity = affinity_penalty or {}
        self.objective = objective or Objective()
        self.demand = demand or {}
        self.cfg = cfg or OptimizerConfig()
        self.initial_assignments = initial_assignments or []
        self.rng = random.Random(self.cfg.seed)
        self.log = OptimizationLog()
        self._feasible_cache: dict[str, list] = {}

    def _velocity(self, sku_id: str) -> float:
        return self.demand.get(sku_id, 0.0)

    def _feasible_bins_for(self, sku: SKU) -> list:
        """Bins a SKU's unit can be placed in (per-dimension fit + zone/hazard/handling + heavy height)."""
        cached = self._feasible_cache.get(sku.sku_id)
        if cached is not None:
            return cached
        out = []
        for loc in self.warehouse.locations.values():
            if not (
                sku.width_cm <= loc.width_cm
                and sku.height_cm <= loc.height_cm
                and sku.depth_cm <= loc.depth_cm
            ):
                continue
            if sku.hazard_class.value not in loc.allowed_hazard:
                continue
            if sku.handling_type.value not in loc.allowed_handling:
                continue
            if sku.handling_type.value == "heavy" and loc.shelf_level > 2:
                continue
            out.append(loc)
        self._feasible_cache[sku.sku_id] = out
        return out

    def _nearest_feasible(self, sku: SKU, prefer_golden: bool = True):
        """Shortlist feasible bins near dispatch, golden shelves first, with zone edge preference."""
        feasible = self._feasible_bins_for(sku)
        # Prefer zone edges (bins at the outer boundaries of zones)
        # Zone edges are at the extremes of each zone's y-coordinate range
        def _zone_edge_priority(loc):
            # Determine which zone this bin belongs to
            zone_idx = None
            for i in range(len(self.warehouse.zones)):
                # Zone y-bands are spaced at regular intervals
                zone_y_offset = i * (self.warehouse.cfg.aisles_per_zone * 2 * self.warehouse.cfg.aisle_spacing_m + 2 * self.warehouse.cfg.rack_depth_m)
                # Check if loc.y is within this zone's band
                zone_width = self.warehouse.cfg.aisles_per_zone * 2 * self.warehouse.cfg.aisle_spacing_m + 2 * self.warehouse.cfg.rack_depth_m
                if abs(loc.y - zone_y_offset) < zone_width / 2:
                    zone_idx = i
                    break
            # Edge bins are those near the outer boundaries of their zone
            if zone_idx is not None:
                zone_y_offset = zone_idx * (self.warehouse.cfg.aisles_per_zone * 2 * self.warehouse.cfg.aisle_spacing_m + 2 * self.warehouse.cfg.rack_depth_m)
                zone_width = self.warehouse.cfg.aisles_per_zone * 2 * self.warehouse.cfg.aisle_spacing_m + 2 * self.warehouse.cfg.rack_depth_m
                distance_from_zone_center = abs(loc.y - zone_y_offset - zone_width / 2)
                # Bins closer to the zone edge (farther from center) get higher priority
            return loc.distance_to_dispatch_m if loc.distance_to_dispatch_m > 0 else (abs(loc.x - 20.4) + abs(loc.y - 0.5))
        # Sort by zone edge priority first, then by distance from dispatch
        def _dist(l):
            return l.distance_to_dispatch_m if l.distance_to_dispatch_m > 0 else (abs(l.x - 20.4) + abs(l.y - 0.5))
        return sorted(feasible, key=lambda l: (_zone_edge_priority(l), _dist(l)))

    # -- greedy inventory-conserving starter ----------------------------------
    def greedy_starter(self) -> list[StorageAssignment]:
        """Place each SKU's real on-hand quantity, splitting across bins as needed.

        SKUs are processed in velocity-descending order so fast movers grab golden near-dock
        bins first. The starter tracks bin volume, bin weight, shelf weight, and rack weight
        so it respects the same structural limits as the validator. If a SKU cannot be fully
        placed, it is left under-stored and the validator reports it; nothing is fabricated.
        """
        locations = list(self.warehouse.locations.values())
        residual_bin_weight = {loc.location_id: loc.max_weight_kg for loc in locations}
        residual_volume = {loc.location_id: loc.usable_volume_cm3 for loc in locations}
        shelf_cap: dict[tuple[str, int], float] = {}
        rack_cap: dict[str, float] = {}
        for loc in locations:
            key = (loc.rack_id, loc.shelf_level)
            if loc.shelf_max_weight_kg is not None:
                shelf_cap[key] = min(shelf_cap.get(key, loc.shelf_max_weight_kg), loc.shelf_max_weight_kg)
            if loc.rack_max_weight_kg is not None:
                rack_cap[loc.rack_id] = min(rack_cap.get(loc.rack_id, loc.rack_max_weight_kg), loc.rack_max_weight_kg)
        shelf_load: dict[tuple[str, int], float] = {key: 0.0 for key in shelf_cap}
        rack_load: dict[str, float] = {key: 0.0 for key in rack_cap}
        occupied_by_bin: dict[str, str] = {}  # bin -> sku_id (single-SKU policy)
        assignments: list[StorageAssignment] = []

        sku_order = sorted(
            [s for s in self.skus.values() if self.inventory.get(s.sku_id, 0) > 0],
            key=lambda s: (-self._velocity(s.sku_id), s.sku_id),
        )

        for sku in sku_order:
            remaining = self.inventory[sku.sku_id]
            for loc in self._nearest_feasible(sku):
                if remaining <= 0:
                    break
                if occupied_by_bin.get(loc.location_id) not in (None, sku.sku_id):
                    continue
                shelf_key = (loc.rack_id, loc.shelf_level)
                qty = _split_quantity(
                    sku,
                    loc,
                    remaining,
                    residual_bin_weight[loc.location_id],
                    residual_volume[loc.location_id],
                    shelf_cap.get(shelf_key, float("inf")) - shelf_load.get(shelf_key, 0.0),
                    rack_cap.get(loc.rack_id, float("inf")) - rack_load.get(loc.rack_id, 0.0),
                )
                if qty <= 0:
                    continue
                occupied_by_bin[loc.location_id] = sku.sku_id
                weight = sku.unit_weight_kg * qty
                residual_bin_weight[loc.location_id] -= weight
                residual_volume[loc.location_id] -= sku.unit_volume_cm3 * qty
                shelf_load[shelf_key] = shelf_load.get(shelf_key, 0.0) + weight
                rack_load[loc.rack_id] = rack_load.get(loc.rack_id, 0.0) + weight
                assignments.append(
                    StorageAssignment(sku.sku_id, loc.location_id, qty, AssignmentType.OPTIMIZED)
                )
                remaining -= qty
            # remaining > 0 means the SKU cannot be fully placed; the validator reports it.
        return assignments

    # -- local search / annealing ------------------------------------------------
    def _refine(self, assignments) -> list[StorageAssignment]:
        """Try improving moves; re-validate each candidate. Moves preserve inventory totals.

        The annealing walk may accept a worse candidate, so the best feasible solution seen
        is tracked separately and returned.
        """
        current = assignments
        current_score = self.objective.evaluate(current, self.skus, self.warehouse, self.demand, self.affinity)
        best = current
        best_score = current_score
        T = self.cfg.anneal_start
        n_moves = 0
        accepted = 0

        for _ in range(self.cfg.max_iterations):
            n_moves += 1
            cand = self._random_neighbor(current)
            if cand is not None and self.validator.validate(cand, record_checks=False).is_feasible:
                score = self.objective.evaluate(cand, self.skus, self.warehouse, self.demand, self.affinity)
                if score < current_score or self.rng.random() < _anneal_accept(T, score - current_score):
                    current, current_score = cand, score
                    accepted += 1
                    if score < best_score:
                        best, best_score = cand, score
            T = max(self.cfg.anneal_end, T * 0.995)

        self.log.n_moves_evaluated = n_moves
        self.log.n_moves_accepted = accepted
        return best

    def _random_neighbor(self, assignments):
        """Swap two records' bins, or move one record to another feasible bin.

        Both operations keep per-SKU totals fixed, so inventory conservation is invariantly
        preserved; the validator still checks it.
        """
        if len(assignments) < 2:
            return None
        cand = [StorageAssignment(x.sku_id, x.location_id, x.assigned_quantity, x.assignment_type) for x in assignments]
        idx_a, idx_b = self.rng.sample(range(len(assignments)), 2)
        a, b = assignments[idx_a], assignments[idx_b]
        if self.rng.random() < 0.5:
            # swap bins between two (SKU, quantity) records
            cand[idx_a] = StorageAssignment(a.sku_id, b.location_id, a.assigned_quantity, a.assignment_type)
            cand[idx_b] = StorageAssignment(b.sku_id, a.location_id, b.assigned_quantity, b.assignment_type)
        else:
            # move record a to a new feasible bin for its SKU
            sku = self.skus.get(a.sku_id)
            if sku is None:
                return None
            feasible = self._feasible_bins_for(sku)
            if not feasible:
                return None
            target = self.rng.choice(feasible)
            cand[idx_a] = StorageAssignment(a.sku_id, target.location_id, a.assigned_quantity, a.assignment_type)
        return cand

    # -- public ------------------------------------------------------------------
    def optimize(self) -> tuple[list[StorageAssignment], OptimizationLog]:
        self.log = OptimizationLog()
        t0 = time.time()

        # Baseline is the reference layout and the safety net: it is feasible by construction.
        baseline = self.initial_assignments
        baseline_score = None
        baseline_ok = False
        if baseline:
            baseline_score = self.objective.evaluate(baseline, self.skus, self.warehouse, self.demand, self.affinity)
            baseline_ok = self.validator.validate(baseline, record_checks=False).is_feasible

        greedy = self.greedy_starter()
        greedy_score = self.objective.evaluate(greedy, self.skus, self.warehouse, self.demand, self.affinity)
        greedy_ok = self.validator.validate(greedy, record_checks=False).is_feasible

        if greedy_ok and (baseline_score is None or greedy_score <= baseline_score):
            starter, starter_score = greedy, greedy_score
        elif baseline_ok:
            starter, starter_score = baseline, baseline_score
        elif greedy_ok:
            starter, starter_score = greedy, greedy_score
        else:
            starter, starter_score = greedy, greedy_score

        self.log.initial_score = baseline_score if baseline_score is not None else starter_score
        self.log.starter_score = starter_score

        refined = self._refine(starter)
        refined_result = self.validator.validate(refined)
        refined_score = self.objective.evaluate(refined, self.skus, self.warehouse, self.demand, self.affinity)

        if refined_result.is_feasible and refined_score <= starter_score:
            final, final_result, final_score = refined, refined_result, refined_score
        elif starter_score is not None:
            final, final_score = starter, starter_score
            final_result = self.validator.validate(final)
        else:
            final, final_result, final_score = refined, refined_result, refined_score

        self.log.final_score = final_score
        self.log.runtime_sec = time.time() - t0
        self.log.n_assigned = len(final)
        self.log.inventory_units = sum(self.inventory.values())
        if not final_result.is_feasible:
            self.log.feasibility_status = "INFEASIBLE"
            self.log.stopping_reason = f"{len(final_result.violations)} violations after refinement"
        else:
            self.log.feasibility_status = "FEASIBLE"
            self.log.stopping_reason = (
                "max iterations reached (annealing)"
                if final is refined
                else "returned best feasible starter (refinement did not improve)"
            )
        return final, self.log


def _split_quantity(
    sku: SKU,
    loc,
    wanted: int,
    residual_bin_weight: float | None = None,
    residual_volume: float | None = None,
    residual_shelf_weight: float | None = None,
    residual_rack_weight: float | None = None,
) -> int:
    """Max units that fit loc and its structural parents, but <= wanted."""
    bin_weight = loc.max_weight_kg if residual_bin_weight is None else residual_bin_weight
    volume = loc.usable_volume_cm3 if residual_volume is None else residual_volume
    shelf_weight = float("inf") if residual_shelf_weight is None else residual_shelf_weight
    rack_weight = float("inf") if residual_rack_weight is None else residual_rack_weight
    limits = [wanted]
    if sku.unit_weight_kg > 0:
        limits.extend([
            int(bin_weight / sku.unit_weight_kg),
            int(shelf_weight / sku.unit_weight_kg),
            int(rack_weight / sku.unit_weight_kg),
        ])
    if sku.unit_volume_cm3 > 0:
        limits.append(int(volume / sku.unit_volume_cm3))
    return max(0, min(limits))


def _anneal_accept(T: float, delta: float) -> float:
    # accept worsening moves with prob exp(delta/T) where delta > 0
    if delta <= 0:
        return 1.0
    if T <= 0:
        return 0.0
    return min(1.0, 2.718281828459045 ** (-delta / T))