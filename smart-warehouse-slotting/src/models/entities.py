"""Core warehouse domain entities.

Canonical units:
  - dimensions: centimetres (cm)
  - weight: kilograms (kg)
  - volume: cubic centimetres (cm3)
  - coordinates / travel distance: metres (m)
  - timestamps: UTC
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class HandlingType(str, Enum):
    STANDARD = "standard"
    FRAGILE = "fragile"
    HEAVY = "heavy"
    TEMPERATURE_CONTROLLED = "temperature_controlled"


class HazardClass(str, Enum):
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"
    FLAMMABLE = "flammable"
    CORROSIVE = "corrosive"
    TOXIC = "toxic"
    COMPRESSED_GAS = "compressed_gas"


class AccessibilityZone(str, Enum):
    GOLDEN = "golden"
    SECONDARY = "secondary"
    RESERVE = "reserve"


class AssignmentType(str, Enum):
    BASELINE = "baseline"
    OPTIMIZED = "optimized"


def sku_volume_cm3(width_cm: float, height_cm: float, depth_cm: float) -> float:
    """Bounding-box volume for a rectangular product."""
    return width_cm * height_cm * depth_cm


@dataclass
class Category:
    category_id: str
    category_name: str
    storage_requirements: str = ""


@dataclass
class SKU:
    sku_id: str
    product_name: str
    category_id: str
    width_cm: float
    height_cm: float
    depth_cm: float
    unit_weight_kg: float
    handling_type: HandlingType = HandlingType.STANDARD
    hazard_class: HazardClass = HazardClass.NONE
    data_source: str = "synthetic"

    @property
    def unit_volume_cm3(self) -> float:
        return sku_volume_cm3(self.width_cm, self.height_cm, self.depth_cm)


@dataclass
class Location:
    """A physical storage bin / pick-face at a (x, y, z) coordinate."""

    location_id: str
    zone_id: str
    aisle_id: str
    rack_id: str
    bay: int
    shelf_level: int
    x: float  # metres
    y: float  # metres
    z: float  # metres (shelf height)
    width_cm: float
    height_cm: float
    depth_cm: float
    max_weight_kg: float
    usable_volume_cm3: float
    ergonomic_zone: AccessibilityZone = AccessibilityZone.RESERVE
    allowed_handling: tuple[str, ...] = ("standard",)
    allowed_hazard: tuple[str, ...] = ("NONE",)
    distance_to_dispatch_m: float = 0.0
    zone_type: str = "general"
    shelf_max_weight_kg: float | None = None
    rack_max_weight_kg: float | None = None

    @property
    def volume_cm3(self) -> float:
        return self.usable_volume_cm3


@dataclass
class Warehouse:
    warehouse_id: str
    warehouse_name: str
    layout_version: str
    locations: dict[str, Location] = field(default_factory=dict)
    zones: list[str] = field(default_factory=list)
    dispatch_point: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CompatibilityRule:
    rule_id: str
    category_a: str
    category_b: str
    allowed: bool
    severity: str = "high"
    rule_reason: str = ""


@dataclass
class Inventory:
    sku_id: str
    location_id: str
    quantity_on_hand: int


@dataclass
class Order:
    order_id: str
    timestamp: str
    status: str = "completed"


@dataclass
class OrderItem:
    order_id: str
    sku_id: str
    quantity: int


@dataclass
class StorageAssignment:
    sku_id: str
    location_id: str
    assigned_quantity: int
    assignment_type: AssignmentType