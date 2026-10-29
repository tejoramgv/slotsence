"""Build validator/optimizer inputs from the canonical dataset (dataset/processed).

Canonical schema notes:
- locations have zone_type (general/small_parts/bulk/heavy/hazard) which drives handling &
  hazard eligibility, plus per-shelf and per-rack structural limits.
- inventory carries location_id (multi-bin current stock snapshot).
- orders/order_items carry scenario_id (three alternative histories, not additive).
- storage_assignments_baseline is multi-bin and feasible.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.entities import (
    AccessibilityZone,
    AssignmentType,
    CompatibilityRule,
    HandlingType,
    HazardClass,
    Location,
    SKU,
    StorageAssignment,
    Warehouse,
)


def load_warehouse_from_locations(df: pd.DataFrame, warehouse_id: str = "WH_01") -> Warehouse:
    wh = Warehouse(warehouse_id=warehouse_id, warehouse_name="canonical", layout_version="1.0")
    for _, r in df.iterrows():
        wh.locations[r["location_id"]] = Location(
            location_id=r["location_id"],
            zone_id=r.get("zone_id", ""),
            aisle_id=str(r.get("aisle_id", "")),
            rack_id=str(r.get("rack_id", "")),
            bay=int(r["bay"]),
            shelf_level=int(r["shelf_level"]),
            x=float(r["x_m"]),
            y=float(r["y_m"]),
            z=float(r["z_m"]),
            width_cm=float(r["width_cm"]),
            height_cm=float(r["height_cm"]),
            depth_cm=float(r["depth_cm"]),
            max_weight_kg=float(r["max_weight_kg"]),
            usable_volume_cm3=float(r["usable_volume_cm3"]),
            ergonomic_zone=AccessibilityZone(r["ergonomic_zone"]) if r.get("ergonomic_zone") in AccessibilityZone._value2member_map_ else AccessibilityZone.RESERVE,
            allowed_handling=tuple(str(r["allowed_handling"]).split("|")),
            allowed_hazard=tuple(str(r["allowed_hazard"]).split("|")),
            distance_to_dispatch_m=float(r["distance_to_dispatch_m"]) if "distance_to_dispatch_m" in r and pd.notna(r["distance_to_dispatch_m"]) else 0.0,
            zone_type=r.get("zone_type", ""),
            shelf_max_weight_kg=float(r["shelf_max_weight_kg"]) if r.get("shelf_max_weight_kg") not in (None, "", "nan") else None,
            rack_max_weight_kg=float(r["rack_max_weight_kg"]) if r.get("rack_max_weight_kg") not in (None, "", "nan") else None,
        )
    return wh


def load_skus(df: pd.DataFrame) -> dict[str, SKU]:
    out = {}
    for _, r in df.iterrows():
        out[r["sku_id"]] = SKU(
            sku_id=r["sku_id"],
            product_name=r["product_name"],
            category_id=r["category_id"],
            width_cm=float(r["width_cm"]),
            height_cm=float(r["height_cm"]),
            depth_cm=float(r["depth_cm"]),
            unit_weight_kg=float(r["unit_weight_kg"]),
            handling_type=HandlingType(r["handling_type"]),
            hazard_class=HazardClass(r["hazard_class"]) if r["hazard_class"] in HazardClass._value2member_map_ else HazardClass.UNKNOWN,
            data_source=r.get("data_source", "synthetic"),
        )
    return out


def load_compatibility_rules(df: pd.DataFrame) -> list[CompatibilityRule]:
    out = []
    for _, r in df.iterrows():
        out.append(
            CompatibilityRule(
                rule_id=r["rule_id"],
                category_a=r["category_a"],
                category_b=r["category_b"],
                allowed=bool(r["allowed"]),
                severity=r["severity"],
                rule_reason=r["rule_reason"],
            )
        )
    return out


def load_assignments(df: pd.DataFrame, atype: AssignmentType = AssignmentType.BASELINE) -> list[StorageAssignment]:
    out = []
    for _, r in df.iterrows():
        out.append(
            StorageAssignment(
                sku_id=r["sku_id"],
                location_id=r["location_id"],
                assigned_quantity=int(r["assigned_quantity"]),
                assignment_type=AssignmentType(r["assignment_type"]) if "assignment_type" in r and pd.notna(r.get("assignment_type")) else atype,
            )
        )
    return out


def load_inventory(df: pd.DataFrame) -> dict[str, int]:
    """sku_id -> total on-hand, aggregated across bins. Used for conservation."""
    return df.groupby("sku_id")["quantity_on_hand"].sum().to_dict()


def load_shipping(processed: Path):
    """Load everything needed to validate the baseline from the canonical dataset."""
    locations = pd.read_csv(processed / "locations.csv")
    products = pd.read_csv(processed / "products.csv")
    rules = pd.read_csv(processed / "compatibility_rules.csv")
    baseline = pd.read_csv(processed / "storage_assignments_baseline.csv")
    inventory = pd.read_csv(processed / "inventory.csv")
    wh = load_warehouse_from_locations(locations)
    skus = load_skus(products)
    comp = load_compatibility_rules(rules)
    assignments = load_assignments(baseline)
    inv = load_inventory(inventory)
    return wh, skus, comp, assignments, inv