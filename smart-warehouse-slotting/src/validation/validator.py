"""Phase-4: standalone hard-constraint validation engine (canonical ruleset).

Enforces:
  SP  dimensional fit (per-dimension)
  REF referential integrity
  HZ  hazard/zone compatibility (hazardous SKUs only in the hazard zone)
  HD  handling/zone compatibility (heavy only in heavy zone; fragile excluded from heavy
      and hazard zones)
  ER  heavy shelf-height (levels 0-2 only)
  QTY positive assigned quantity
  AS  single-SKU-per-bin (a bin holds at most one SKU type)
  COMP assignment completeness (every SKU with inventory has a home)
  INV inventory conservation (sum of assigned quantity == on-hand for every SKU)
  LD  per-bin cumulative load
  VD  per-bin cumulative volume
  SL  per-shelf structural load (keyed by rack_id + shelf_level)
  RK  per-rack structural load
  CP  co-storage category compatibility (multi-SKU bins via compatibility matrix)

Every rule evaluation produces a structured ``RuleCheck`` (rule id, pass/fail, affected
SKU/location, observed value, allowed limit, explanation). Failed checks also become a
``Violation``. ``validate`` returns ALL checks and ALL violations, not just the first.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from src.models.entities import Location, SKU, StorageAssignment, Warehouse

HEAVY_MAX_SHELF_LEVEL = 2  # levels 0..3; heavy goods only on reinforced lower shelves

# Coarse grouping used by the dashboard to separate spatial / structural / safety rules.
RULE_GROUP_BY_PREFIX = {
    "SP": "Spatial & volumetric",
    "VD": "Spatial & volumetric",
    "LD": "Structural load",
    "SL": "Structural load",
    "RK": "Structural load",
    "HZ": "Hazard & safety",
    "HD": "Hazard & safety",
    "ER": "Hazard & safety",
    "CP": "Hazard & safety",
    "AS": "Hazard & safety",
    "REF": "Integrity",
    "QTY": "Integrity",
    "COMP": "Integrity",
    "INV": "Integrity",
    "DU": "Integrity",
}


def rule_group(rule_id: str) -> str:
    """Map a rule id (e.g. ``SL-01``) to its dashboard group."""
    prefix = rule_id.split("-")[0]
    return RULE_GROUP_BY_PREFIX.get(prefix, "Integrity")


@dataclass
class Violation:
    rule_id: str
    rule_name: str
    sku_id: str | None
    location_id: str | None
    severity: str
    reason: str
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "sku_id": self.sku_id,
            "location_id": self.location_id,
            "severity": self.severity,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class RuleCheck:
    """One evaluated rule instance, pass or fail, with the values it compared."""

    rule_id: str
    rule_name: str
    passed: bool
    severity: str
    sku_id: str | None = None
    location_id: str | None = None
    observed: object | None = None
    limit: object | None = None
    unit: str = ""
    explanation: str = ""

    @property
    def group(self) -> str:
        return rule_group(self.rule_id)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_group": self.group,
            "passed": self.passed,
            "severity": self.severity,
            "sku_id": self.sku_id or "",
            "location_id": self.location_id or "",
            "observed": "" if self.observed is None else self.observed,
            "limit": "" if self.limit is None else self.limit,
            "unit": self.unit,
            "explanation": self.explanation,
        }


@dataclass
class ValidationResult:
    violations: list[Violation] = field(default_factory=list)
    checks_run: int = 0
    checks: list[RuleCheck] = field(default_factory=list)
    record_checks: bool = True  # candidate screening can skip materialising pass rows

    @property
    def is_feasible(self) -> bool:
        return len(self.violations) == 0

    @property
    def hard_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "high"]

    @property
    def failed_checks(self) -> list[RuleCheck]:
        return [c for c in self.checks if not c.passed]

    def add(self, violation: Violation) -> None:
        self.violations.append(violation)

    def note_check(self) -> None:
        self.checks_run += 1

    def record(
        self,
        rule_id: str,
        rule_name: str,
        passed: bool,
        *,
        sku_id: str | None = None,
        location_id: str | None = None,
        observed: object | None = None,
        limit: object | None = None,
        unit: str = "",
        explanation: str = "",
        severity: str = "high",
    ) -> None:
        """Record one evaluated rule instance and, when it fails, a violation."""
        self.checks_run += 1
        if self.record_checks:
            self.checks.append(
                RuleCheck(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    passed=passed,
                    severity=severity if not passed else "info",
                    sku_id=sku_id,
                    location_id=location_id,
                    observed=observed,
                    limit=limit,
                    unit=unit,
                    explanation=explanation,
                )
            )
        if not passed:
            self.violations.append(
                Violation(rule_id, rule_name, sku_id, location_id, severity, explanation)
            )


class Validator:
    def __init__(
        self,
        skus: dict[str, SKU],
        warehouse: Warehouse,
        compatibility_rules: list | None = None,
        inventory: dict[str, int] | None = None,
        record_checks: bool = True,
    ):
        self.skus = skus
        self.warehouse = warehouse
        self.rules = compatibility_rules or []
        self.inventory = inventory or {}
        self.record_checks = record_checks

    def _sku(self, sku_id: str) -> SKU | None:
        return self.skus.get(sku_id)

    def _loc(self, location_id: str) -> Location | None:
        return self.warehouse.locations.get(location_id)

    def validate(
        self, assignments: list[StorageAssignment], record_checks: bool | None = None
    ) -> ValidationResult:
        res = ValidationResult(
            record_checks=self.record_checks if record_checks is None else record_checks
        )
        by_location: dict[str, list[StorageAssignment]] = defaultdict(list)
        for a in assignments:
            by_location[a.location_id].append(a)

        for a in assignments:
            self._check_dimensional(res, a)
            self._check_referential(res, a)
            self._check_hazard_zone(res, a)
            self._check_handling_zone(res, a)
            self._check_heavy_shelf_height(res, a)
            self._check_positive_quantity(res, a)
        for loc_id, group in by_location.items():
            self._check_location_weight(res, loc_id, group)
            self._check_volume_residual(res, loc_id, group)
            self._check_single_sku_per_bin(res, loc_id, group)
            self._check_co_storage_compatibility(res, loc_id, group)
        self._check_structural_limits(res, by_location)
        self._check_assignee_uniqueness(res, by_location)
        self._check_completeness(res, assignments)
        self._check_inventory_conservation(res, assignments)
        return res

    # -- dimensional ----------------------------------------------------------
    def _check_dimensional(self, res: ValidationResult, a) -> None:
        sku = self._sku(a.sku_id)
        loc = self._loc(a.location_id)
        if sku is None or loc is None:
            return
        fits = (
            sku.width_cm <= loc.width_cm
            and sku.height_cm <= loc.height_cm
            and sku.depth_cm <= loc.depth_cm
        )
        res.record(
            "SP-01",
            "dimensional_fit",
            fits,
            sku_id=a.sku_id,
            location_id=a.location_id,
            observed=f"{sku.width_cm:g}x{sku.height_cm:g}x{sku.depth_cm:g}",
            limit=f"{loc.width_cm:g}x{loc.height_cm:g}x{loc.depth_cm:g}",
            unit="cm (WxHxD)",
            explanation=(
                "SKU bounding box fits the bin"
                if fits
                else f"SKU dims exceed location {a.location_id}"
            ),
        )

    # -- referential ----------------------------------------------------------
    def _check_referential(self, res: ValidationResult, a) -> None:
        sku_ok = a.sku_id in self.skus
        loc_ok = a.location_id in self.warehouse.locations
        res.record(
            "REF-01",
            "referential_sku",
            sku_ok,
            sku_id=a.sku_id,
            location_id=a.location_id,
            observed=a.sku_id,
            limit="known SKU id",
            explanation="SKU exists" if sku_ok else f"unknown SKU {a.sku_id}",
        )
        res.record(
            "REF-02",
            "referential_location",
            loc_ok,
            sku_id=a.sku_id,
            location_id=a.location_id,
            observed=a.location_id,
            limit="known location id",
            explanation="location exists" if loc_ok else f"unknown location {a.location_id}",
        )

    # -- hazard zone ----------------------------------------------------------
    def _check_hazard_zone(self, res: ValidationResult, a) -> None:
        sku = self._sku(a.sku_id)
        loc = self._loc(a.location_id)
        if sku is None or loc is None:
            return
        allowed = sku.hazard_class.value in loc.allowed_hazard
        res.record(
            "HZ-01",
            "hazard_zone",
            allowed,
            sku_id=a.sku_id,
            location_id=a.location_id,
            observed=sku.hazard_class.value,
            limit="|".join(loc.allowed_hazard),
            explanation=(
                "hazard class permitted in this zone"
                if allowed
                else f"hazard {sku.hazard_class.value} not allowed at {a.location_id} (zone_type={loc.zone_type})"
            ),
        )

    # -- handling / zone --------------------------------------------------------
    def _check_handling_zone(self, res: ValidationResult, a) -> None:
        sku = self._sku(a.sku_id)
        loc = self._loc(a.location_id)
        if sku is None or loc is None:
            return
        allowed = sku.handling_type.value in loc.allowed_handling
        res.record(
            "HD-01",
            "handling_zone",
            allowed,
            sku_id=a.sku_id,
            location_id=a.location_id,
            observed=sku.handling_type.value,
            limit="|".join(loc.allowed_handling),
            explanation=(
                "handling type permitted in this zone"
                if allowed
                else f"handling {sku.handling_type.value} not allowed at {a.location_id} (zone_type={loc.zone_type})"
            ),
        )

    # -- heavy shelf height ----------------------------------------------------
    def _check_heavy_shelf_height(self, res: ValidationResult, a) -> None:
        sku = self._sku(a.sku_id)
        loc = self._loc(a.location_id)
        if sku is None or loc is None:
            return
        ok = not (
            sku.handling_type.value == "heavy" and loc.shelf_level > HEAVY_MAX_SHELF_LEVEL
        )
        res.record(
            "ER-01",
            "heavy_shelf_height",
            ok,
            sku_id=a.sku_id,
            location_id=a.location_id,
            observed=loc.shelf_level,
            limit=HEAVY_MAX_SHELF_LEVEL,
            unit="shelf level",
            explanation=(
                "shelf height permitted for this handling type"
                if ok
                else f"heavy SKU at shelf level {loc.shelf_level} > max {HEAVY_MAX_SHELF_LEVEL}"
            ),
        )

    # -- positive quantity -----------------------------------------------------
    def _check_positive_quantity(self, res: ValidationResult, a) -> None:
        ok = a.assigned_quantity > 0
        res.record(
            "QTY-01",
            "positive_quantity",
            ok,
            sku_id=a.sku_id,
            location_id=a.location_id,
            observed=a.assigned_quantity,
            limit="> 0",
            unit="units",
            explanation=(
                "assigned quantity is positive"
                if ok
                else f"assigned quantity {a.assigned_quantity} must be > 0"
            ),
        )

    # -- per-bin load ------------------------------------------------------------
    def _check_location_weight(self, res: ValidationResult, loc_id, group) -> None:
        loc = self._loc(loc_id)
        if loc is None:
            return
        total = sum(self._sku(a.sku_id).unit_weight_kg * a.assigned_quantity for a in group if self._sku(a.sku_id))
        ok = total <= loc.max_weight_kg
        res.record(
            "LD-01",
            "location_weight",
            ok,
            location_id=loc_id,
            observed=round(total, 3),
            limit=loc.max_weight_kg,
            unit="kg",
            explanation=(
                "bin load within limit"
                if ok
                else f"load {total:.1f}kg > per-bin limit {loc.max_weight_kg:.1f}kg"
            ),
        )

    # -- per-bin volume ----------------------------------------------------------
    def _check_volume_residual(self, res: ValidationResult, loc_id, group) -> None:
        loc = self._loc(loc_id)
        if loc is None:
            return
        total = sum(self._sku(a.sku_id).unit_volume_cm3 * a.assigned_quantity for a in group if self._sku(a.sku_id))
        ok = total <= loc.usable_volume_cm3
        res.record(
            "VD-01",
            "volume_capacity",
            ok,
            location_id=loc_id,
            observed=round(total, 2),
            limit=loc.usable_volume_cm3,
            unit="cm3",
            explanation=(
                "stored volume within usable bin volume"
                if ok
                else f"stored volume {total:.0f}cm3 > {loc.usable_volume_cm3:.0f}cm3"
            ),
        )

    # -- single SKU per bin -------------------------------------------------------
    def _check_single_sku_per_bin(self, res: ValidationResult, loc_id, group) -> None:
        sku_ids = {a.sku_id for a in group if self._sku(a.sku_id)}
        ok = len(sku_ids) <= 1
        res.record(
            "AS-01",
            "single_sku_per_bin",
            ok,
            location_id=loc_id,
            observed=len(sku_ids),
            limit=1,
            unit="SKUs per bin",
            explanation=(
                "single-SKU bin policy respected"
                if ok
                else f"multiple SKUs {sorted(sku_ids)} in bin {loc_id}"
            ),
        )

    # -- co-storage compatibility (for the rare multi-SKU case) ------------------
    def _check_co_storage_compatibility(self, res: ValidationResult, loc_id, group) -> None:
        skus = [self._sku(a.sku_id) for a in group if self._sku(a.sku_id)]
        pairs = [(skus[i], skus[j]) for i in range(len(skus)) for j in range(i + 1, len(skus))]
        if not pairs:
            return
        for a, b in pairs:
            ok = not self._conflicts(a, b)
            res.record(
                "CP-01",
                "co_storage_conflict",
                ok,
                sku_id=a.sku_id,
                location_id=loc_id,
                observed=f"{a.category_id}+{b.category_id}",
                limit="compatible categories",
                explanation=(
                    "categories compatible"
                    if ok
                    else f"{a.sku_id} and {b.sku_id} incompatible to co-store"
                ),
            )

    def _conflicts(self, a: SKU, b: SKU) -> bool:
        for rule in self.rules:
            if {rule.category_a, rule.category_b} == {a.category_id, b.category_id}:
                return not rule.allowed
        return False  # single-SKU bins; unknown pairs between different bins are fine

    # -- structural (shelf + rack) limits ------------------------------------------
    def _check_structural_limits(self, res: ValidationResult, by_location) -> None:
        # Key shelf loads by (rack_id, shelf_level) and rack loads by rack_id. Use the
        # tightest limit on each shelf/rack; a missing limit means "not modelled".
        shelf_load: dict[tuple[str, int], float] = defaultdict(float)
        rack_load: dict[str, float] = defaultdict(float)
        shelf_limit: dict[tuple[str, int], float] = {}
        rack_limit: dict[str, float] = {}
        for loc_id, group in by_location.items():
            loc = self._loc(loc_id)
            if loc is None:
                continue
            total = sum(
                self._sku(a.sku_id).unit_weight_kg * a.assigned_quantity
                for a in group
                if self._sku(a.sku_id)
            )
            key = (loc.rack_id, loc.shelf_level)
            shelf_load[key] += total
            rack_load[loc.rack_id] += total
            if loc.shelf_max_weight_kg is not None:
                shelf_limit[key] = min(
                    shelf_limit.get(key, loc.shelf_max_weight_kg), loc.shelf_max_weight_kg
                )
            if loc.rack_max_weight_kg is not None:
                rack_limit[loc.rack_id] = min(
                    rack_limit.get(loc.rack_id, loc.rack_max_weight_kg), loc.rack_max_weight_kg
                )

        for key, load in shelf_load.items():
            limit = shelf_limit.get(key)
            if limit is None:
                continue
            ok = load <= limit + 1e-6
            res.record(
                "SL-01",
                "shelf_load",
                ok,
                location_id=f"{key[0]} shelf {key[1]}",
                observed=round(load, 3),
                limit=limit,
                unit="kg",
                explanation=(
                    "shelf load within structural limit"
                    if ok
                    else f"shelf {key} load {load:.1f}kg > limit {limit:.1f}kg"
                ),
            )
        for rack, load in rack_load.items():
            limit = rack_limit.get(rack)
            if limit is None:
                continue
            ok = load <= limit + 1e-6
            res.record(
                "RK-01",
                "rack_load",
                ok,
                location_id=rack,
                observed=round(load, 3),
                limit=limit,
                unit="kg",
                explanation=(
                    "rack load within structural limit"
                    if ok
                    else f"rack {rack} load {load:.1f}kg > limit {limit:.1f}kg"
                ),
            )

    # -- SKU assigned to at most one bin (consistency; a SKU may span bins in the
    #    canonical dataset, so this is reported as a warning, not hard).  --
    def _check_assignee_uniqueness(self, res: ValidationResult, by_location) -> None:
        # Multi-bin is legitimate in the canonical schema; only flag a SKU assigned twice
        # to the SAME bin.
        seen_pairs = set()
        for loc_id, group in by_location.items():
            for a in group:
                if (a.sku_id, loc_id) not in seen_pairs:
                    seen_pairs.add((a.sku_id, loc_id))
                else:
                    res.record(
                        "DU-01",
                        "duplicate_pin",
                        False,
                        sku_id=a.sku_id,
                        location_id=loc_id,
                        observed=f"{a.sku_id}@{loc_id}",
                        limit="1 assignment per SKU/bin",
                        explanation=f"SKU {a.sku_id} assigned twice to bin {loc_id}",
                    )

    # -- assignment completeness ------------------------------------------------
    def _check_completeness(self, res: ValidationResult, assignments) -> None:
        assigned_skus = {a.sku_id for a in assignments}
        expected = set(self.inventory.keys()) if self.inventory else set(self.skus.keys())
        missing = expected - assigned_skus
        ok = not missing
        res.record(
            "COMP-01",
            "assignment_completeness",
            ok,
            observed=len(missing),
            limit=0,
            unit="SKUs without a location",
            explanation=(
                "every SKU with inventory has a location"
                if ok
                else f"{len(missing)} SKUs with inventory have no assignment (e.g. {sorted(missing)[:5]})"
            ),
        )

    # -- inventory conservation ---------------------------------------------------
    def _check_inventory_conservation(self, res: ValidationResult, assignments) -> None:
        if not self.inventory:
            return
        assigned_total: dict[str, int] = defaultdict(int)
        for a in assignments:
            if self._sku(a.sku_id) is not None and a.assigned_quantity > 0:
                assigned_total[a.sku_id] += a.assigned_quantity
        for sku_id, placed in assigned_total.items():
            if sku_id not in self.inventory:
                res.record(
                    "INV-01",
                    "inventory_conservation_over",
                    False,
                    sku_id=sku_id,
                    observed=placed,
                    limit=0,
                    unit="units, on-hand = 0",
                    explanation=f"assigned {placed} units of a SKU with no on-hand inventory",
                )
        for sku_id, on_hand in self.inventory.items():
            placed = assigned_total.get(sku_id, 0)
            if placed > on_hand:
                res.record(
                    "INV-01",
                    "inventory_conservation_over",
                    False,
                    sku_id=sku_id,
                    observed=placed,
                    limit=on_hand,
                    unit="units",
                    explanation=f"assigned {placed} > on-hand {on_hand} (over-stored by {placed - on_hand})",
                )
            elif placed < on_hand:
                res.record(
                    "INV-02",
                    "inventory_conservation_under",
                    False,
                    sku_id=sku_id,
                    observed=placed,
                    limit=on_hand,
                    unit="units",
                    severity="medium",
                    explanation=f"assigned {placed} < on-hand {on_hand} (under-stored by {on_hand - placed})",
                )
            else:
                res.record(
                    "INV-03",
                    "inventory_conservation_match",
                    True,
                    sku_id=sku_id,
                    observed=placed,
                    limit=on_hand,
                    unit="units",
                    explanation="assigned quantity equals on-hand inventory",
                )


def validate_assignment(
    assignments: list[StorageAssignment],
    skus: dict[str, SKU],
    warehouse: Warehouse,
    compatibility_rules: list | None = None,
    inventory: dict[str, int] | None = None,
) -> ValidationResult:
    v = Validator(skus, warehouse, compatibility_rules, inventory)
    return v.validate(assignments)


def violations_to_dicts(result: ValidationResult) -> list[dict]:
    return [v.to_dict() for v in result.violations]


def audit_assignments(assignments: list[StorageAssignment], result: ValidationResult) -> Iterator[dict]:
    """Yield violation rows, or one explicit PASS row per assignment when clean.

    A clean baseline therefore produces a non-empty audit instead of a header-only file;
    a failing layout still lists every violation with its rule and reason.
    """
    if not assignments:
        return
    if result.violations:
        for v in result.violations:
            yield {
                "sku_id": v.sku_id or "",
                "location_id": v.location_id or "",
                "rule_id": v.rule_id,
                "rule_name": v.rule_name,
                "severity": v.severity,
                "passed": "False",
                "reason": v.reason,
            }
        return
    for a in assignments:
        yield {
            "sku_id": a.sku_id,
            "location_id": a.location_id,
            "rule_id": "PASS",
            "rule_name": "all_checks",
            "severity": "info",
            "passed": "True",
            "reason": "no violations",
        }
