"""Persistence helpers: serialize domain entities to/from CSV.

CSV is the canonical interchange format for the project (per the execution plan).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .entities import (
    AssignmentType,
    Category,
    Inventory,
    Location,
    Order,
    OrderItem,
    SKU,
    StorageAssignment,
    Warehouse,
)


def write_locations_csv(warehouse: Warehouse, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for loc in warehouse.locations.values():
        rows.append(
            {
                "location_id": loc.location_id,
                "warehouse_id": warehouse.warehouse_id,
                "zone_id": loc.zone_id,
                "aisle_id": loc.aisle_id,
                "rack_id": loc.rack_id,
                "bay": loc.bay,
                "shelf_level": loc.shelf_level,
                "x_m": loc.x,
                "y_m": loc.y,
                "z_m": loc.z,
                "width_cm": loc.width_cm,
                "height_cm": loc.height_cm,
                "depth_cm": loc.depth_cm,
                "max_weight_kg": loc.max_weight_kg,
                "usable_volume_cm3": loc.usable_volume_cm3,
                "ergonomic_zone": loc.ergonomic_zone.value,
                "allowed_handling": "|".join(loc.allowed_handling),
                "allowed_hazard": "|".join(loc.allowed_hazard),
                "distance_to_dispatch_m": loc.distance_to_dispatch_m,
            }
        )
    _write_csv(path, rows)


def write_skus_csv(skus: list[SKU], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in skus:
        rows.append(
            {
                "sku_id": s.sku_id,
                "product_name": s.product_name,
                "category_id": s.category_id,
                "width_cm": s.width_cm,
                "height_cm": s.height_cm,
                "depth_cm": s.depth_cm,
                "unit_weight_kg": s.unit_weight_kg,
                "unit_volume_cm3": s.unit_volume_cm3,
                "handling_type": s.handling_type.value,
                "hazard_class": s.hazard_class.value,
                "data_source": s.data_source,
            }
        )
    _write_csv(path, rows)


def write_categories_csv(categories: list[Category], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        path,
        [
            {
                "category_id": c.category_id,
                "category_name": c.category_name,
                "storage_requirements": c.storage_requirements,
            }
            for c in categories
        ],
    )


def write_assignments_csv(assignments: list[StorageAssignment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        path,
        [
            {
                "sku_id": a.sku_id,
                "location_id": a.location_id,
                "assigned_quantity": a.assigned_quantity,
                "assignment_type": a.assignment_type.value,
            }
            for a in assignments
        ],
    )


def read_assignments_csv(path: Path) -> list[StorageAssignment]:
    out = []
    for row in _read_csv(path):
        out.append(
            StorageAssignment(
                sku_id=row["sku_id"],
                location_id=row["location_id"],
                assigned_quantity=int(row["assigned_quantity"]),
                assignment_type=AssignmentType(row["assignment_type"]),
            )
        )
    return out


def write_orders_csv(orders: list[Order], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        path,
        [
            {"order_id": o.order_id, "order_timestamp": o.timestamp, "order_status": o.status}
            for o in orders
        ],
    )


def write_order_items_csv(items: list[OrderItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        path,
        [
            {"order_id": it.order_id, "sku_id": it.sku_id, "quantity": it.quantity}
            for it in items
        ],
    )


def write_inventory_csv(inventory: list[Inventory], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        path,
        [
            {"sku_id": inv.sku_id, "location_id": inv.location_id, "quantity_on_hand": inv.quantity_on_hand}
            for inv in inventory
        ],
    )


def write_metadata_csv(path: Path, rows: list[dict]) -> None:
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))