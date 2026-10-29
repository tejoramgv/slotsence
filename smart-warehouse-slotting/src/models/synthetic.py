"""Synthetic SKU generation + feasible baseline assignment.

Phase-1 scope: enough SKUs and a seeded *feasible* baseline fixture so later phases can
be tested. A larger, provenance-tracked generator is used in the Phase-2 dataset pipeline.
"""

from __future__ import annotations

import random
from pathlib import Path

from .config import Config
from .entities import (
    AssignmentType,
    Category,
    HandlingType,
    HazardClass,
    SKU,
    StorageAssignment,
    Warehouse,
)
from .io import write_assignments_csv, write_categories_csv, write_locations_csv, write_skus_csv

# Category -> (name, storage_requirements)
CATEGORY_DEFS = [
    ("CAT001", "Accessories & Cables", "small, light"),
    ("CAT002", "Books & Media", "medium, boxed"),
    ("CAT003", "Household Goods", "medium, boxed"),
    ("CAT004", "Storage Containers", "large, bulky"),
    ("CAT005", "Appliances", "large, bulky"),
    ("CAT006", "Tools & Metal Parts", "heavy, dense"),
    ("CAT007", "Packaged Bulk Goods", "heavy, dense"),
    ("CAT008", "Glassware", "fragile"),
    ("CAT009", "Electronics & Sensitive", "fragile"),
    ("CAT010", "Restricted / Hazardous", "restricted"),
]

# Category id -> (with_cm, height_cm, depth_cm) representative dimension ranges.
_DIM_RANGES = {
    "CAT001": (5, 15, 2, 8, 2, 8),
    "CAT002": (15, 25, 15, 25, 2, 8),
    "CAT003": (15, 30, 10, 25, 5, 20),
    "CAT004": (25, 45, 20, 40, 20, 40),
    "CAT005": (30, 50, 30, 50, 25, 50),
    "CAT006": (8, 25, 5, 20, 5, 20),
    "CAT007": (20, 40, 15, 35, 15, 35),
    "CAT008": (10, 20, 10, 20, 5, 15),
    "CAT009": (5, 30, 2, 15, 5, 25),
    "CAT010": (15, 35, 10, 30, 10, 30),
}

_HANDLING = {
    "CAT001": HandlingType.STANDARD,
    "CAT002": HandlingType.STANDARD,
    "CAT003": HandlingType.STANDARD,
    "CAT004": HandlingType.STANDARD,
    "CAT005": HandlingType.STANDARD,
    "CAT006": HandlingType.HEAVY,
    "CAT007": HandlingType.HEAVY,
    "CAT008": HandlingType.FRAGILE,
    "CAT009": HandlingType.FRAGILE,
    "CAT010": HandlingType.STANDARD,
}

_HAZARD = {
    "CAT001": HazardClass.NONE,
    "CAT002": HazardClass.NONE,
    "CAT003": HazardClass.NONE,
    "CAT004": HazardClass.NONE,
    "CAT005": HazardClass.NONE,
    "CAT006": HazardClass.NONE,
    "CAT007": HazardClass.NONE,
    "CAT008": HazardClass.NONE,
    "CAT009": HazardClass.NONE,
    "CAT010": HazardClass.FLAMMABLE,
}

# Weight is class-relative: heavier for dense categories for a given volume.
_WEIGHT_PER_L = {
    "CAT001": 0.4,
    "CAT002": 0.7,
    "CAT003": 0.6,
    "CAT004": 0.4,
    "CAT005": 0.5,
    "CAT006": 1.8,
    "CAT007": 1.2,
    "CAT008": 0.9,
    "CAT009": 0.8,
    "CAT010": 1.0,
}


def _fit_bin(sku: SKU, warehouse: Warehouse) -> bool:
    """Can this SKU (single unit) fit some bin's per-dimension bounds?"""
    for loc in warehouse.locations.values():
        if (
            sku.width_cm <= loc.width_cm
            and sku.height_cm <= loc.height_cm
            and sku.depth_cm <= loc.depth_cm
        ):
            return True
    return False


def generate_categories() -> list[Category]:
    return [Category(cid, name, req) for cid, name, req in CATEGORY_DEFS]


def generate_skus(config: Config, seed: int, data_source: str = "synthetic") -> list[SKU]:
    """Generate a diverse, physically plausible SKU master."""
    rng = random.Random(seed)
    skus: list[SKU] = []
    for i in range(config.dataset.n_skus):
        cat_id = CATEGORY_DEFS[i % len(CATEGORY_DEFS)][0]
        # _DIM_RANGES layout is (w_lo, w_hi, h_lo, h_hi, d_lo, d_hi)
        wr = _DIM_RANGES[cat_id]
        width = rng.uniform(wr[0], wr[1])
        height = rng.uniform(wr[2], wr[3])
        depth = rng.uniform(wr[4], wr[5])
        vol_l = (width * height * depth) / 1000.0  # cm3 -> L
        weight = vol_l * _WEIGHT_PER_L[cat_id] * rng.uniform(0.7, 1.3)
        handling = _HANDLING[cat_id]
        hazard = _HAZARD[cat_id]
        # Restrict fragile/restricted SKUs to those that fit a bin, else skip.
        sku = SKU(
            sku_id=f"SKU{i + 1:06d}",
            product_name=f"Product {i + 1} ({cat_id})",
            category_id=cat_id,
            width_cm=round(width, 1),
            height_cm=round(height, 1),
            depth_cm=round(depth, 1),
            unit_weight_kg=round(weight, 3),
            handling_type=handling,
            hazard_class=hazard,
            data_source=data_source,
        )
        skus.append(sku)
    return skus


def generate_feasible_baseline(skus: list[SKU], warehouse: Warehouse, seed: int) -> list[StorageAssignment]:
    """Assign each SKU to a *feasible* location, seeded, respecting capacity.

    Single-location-per-SKU policy for this fixture. Locations are filtered by per-dimension
    fit and weight; a SKU that fits nothing is left unassigned (recorded, not forced).
    """
    rng = random.Random(seed)
    locations = sorted(warehouse.locations.values(), key=lambda l: l.location_id)
    # Track residual weight per location to keep loads within limits.
    residual = {loc.location_id: loc.max_weight_kg for loc in locations}
    assignments: list[StorageAssignment] = []
    unassigned: list[str] = []

    for sku in skus:
        feasible = [
            loc
            for loc in locations
            if (
                sku.width_cm <= loc.width_cm
                and sku.height_cm <= loc.height_cm
                and sku.depth_cm <= loc.depth_cm
            )
            and residual[loc.location_id] >= sku.unit_weight_kg
            and sku.hazard_class.value in loc.allowed_hazard
            and sku.handling_type.value in loc.allowed_handling
        ]
        if not feasible:
            unassigned.append(sku.sku_id)
            continue
        chosen = rng.choice(feasible)
        residual[chosen.location_id] -= sku.unit_weight_kg
        assignments.append(
            StorageAssignment(
                sku_id=sku.sku_id,
                location_id=chosen.location_id,
                assigned_quantity=1,
                assignment_type=AssignmentType.BASELINE,
            )
        )
    return assignments


def build_phase1_fixture(config: Config, output_dir: Path, seed: int | None = None) -> dict:
    """Build warehouse + SKUs + categories + feasible baseline and write them out.

    Returns a dict of key counts for smoke-test assertions.
    """
    seed = config.dataset.seed if seed is None else seed
    output_dir.mkdir(parents=True, exist_ok=True)

    warehouse = _build_warehouse(config)
    categories = generate_categories()
    skus = generate_skus(config, seed)
    baseline = generate_feasible_baseline(skus, warehouse, seed)

    write_locations_csv(warehouse, output_dir / "locations.csv")
    write_categories_csv(categories, output_dir / "categories.csv")
    write_skus_csv(skus, output_dir / "products.csv")
    write_assignments_csv(baseline, output_dir / "storage_assignments_baseline.csv")

    return {
        "locations": len(warehouse.locations),
        "skus": len(skus),
        "baseline_assigned": len(baseline),
        "baseline_unassigned": len(skus) - len(baseline),
    }


def _build_warehouse(config: Config):
    from .topology import WarehouseBuilder

    return WarehouseBuilder(config).build()