"""Canonical schema for the benchmark dataset.

Single source of truth for column order, data types, units, and meanings. The generator,
data dictionary, and validator all read this registry so they cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str  # str | int | float | bool | timestamp
    unit: str = ""
    description: str = ""


TABLES: dict[str, tuple[Field, ...]] = {
    "warehouses": (
        Field("warehouse_id", "str", "", "unique warehouse identifier"),
        Field("warehouse_name", "str", "", "display name"),
        Field("layout_version", "str", "", "topology / layout version label"),
        Field("data_source", "str", "", "measured | sourced | inferred | synthetic | derived"),
        Field("data_quality_status", "str", "", "valid | warning | invalid"),
    ),
    "categories": (
        Field("category_id", "str", "", "unique category identifier"),
        Field("category_name", "str", "", "display name"),
        Field("storage_requirements", "str", "", "handling / storage notes"),
        Field("data_source", "str", "", "provenance of the row"),
    ),
    "products": (
        Field("sku_id", "str", "", "unique SKU identifier"),
        Field("product_name", "str", "", "display name"),
        Field("category_id", "str", "", "FK to categories.category_id"),
        Field("width_cm", "float", "cm", "per-unit bounding-box width"),
        Field("height_cm", "float", "cm", "per-unit bounding-box height"),
        Field("depth_cm", "float", "cm", "per-unit bounding-box depth"),
        Field("unit_volume_cm3", "float", "cm3", "width x height x depth"),
        Field("unit_weight_kg", "float", "kg", "per-unit weight"),
        Field("handling_type", "str", "", "standard | fragile | heavy | temperature_controlled"),
        Field("hazard_class", "str", "", "NONE | UNKNOWN | flammable | corrosive | toxic | compressed_gas"),
        Field("data_source", "str", "", "measured | sourced | inferred | synthetic | derived"),
        Field("data_quality_status", "str", "", "valid | warning | invalid"),
        Field("physical_data_source", "str", "", "source of the dimensions and weight"),
        Field("physical_confidence", "str", "", "high | medium | low"),
    ),
    "locations": (
        Field("location_id", "str", "", "stable bin id (independent of the stored SKU)"),
        Field("warehouse_id", "str", "", "FK to warehouses.warehouse_id"),
        Field("zone_id", "str", "", "zone identifier"),
        Field("zone_type", "str", "", "general | small_parts | bulk | heavy | hazard"),
        Field("aisle_id", "str", "", "aisle identifier"),
        Field("rack_id", "str", "", "rack identifier"),
        Field("bay", "int", "", "0-based bay index within the rack"),
        Field("shelf_level", "int", "", "0-based shelf level"),
        Field("x_m", "float", "m", "x coordinate"),
        Field("y_m", "float", "m", "y coordinate"),
        Field("z_m", "float", "m", "shelf height"),
        Field("width_cm", "float", "cm", "internal bin width"),
        Field("height_cm", "float", "cm", "internal bin height"),
        Field("depth_cm", "float", "cm", "internal bin depth"),
        Field("max_weight_kg", "float", "kg", "permissible load for this bin"),
        Field("usable_volume_cm3", "float", "cm3", "internal bin volume"),
        Field("shelf_max_weight_kg", "float", "kg", "permissible load for the whole shelf"),
        Field("rack_max_weight_kg", "float", "kg", "permissible load for the whole rack"),
        Field("ergonomic_zone", "str", "", "golden | secondary | reserve"),
        Field("allowed_handling", "str", "", "|-joined handling types allowed in this bin"),
        Field("allowed_hazard", "str", "", "|-joined hazard classes allowed in this bin"),
        Field("distance_to_dispatch_m", "float", "m", "shortest walk along the aisle graph to the depot"),
        Field("data_source", "str", "", "provenance of the row"),
        Field("data_quality_status", "str", "", "valid | warning | invalid"),
    ),
    "inventory": (
        Field("sku_id", "str", "", "FK to products.sku_id"),
        Field("location_id", "str", "", "FK to locations.location_id"),
        Field("quantity_on_hand", "int", "units", "current stock in this bin"),
        Field("data_source", "str", "", "provenance of the row"),
        Field("data_quality_status", "str", "", "valid | warning | invalid"),
    ),
    "compatibility_rules": (
        Field("rule_id", "str", "", "unique rule identifier"),
        Field("category_a", "str", "", "FK to categories.category_id"),
        Field("category_b", "str", "", "FK to categories.category_id"),
        Field("allowed", "bool", "", "true when the pair may share storage areas"),
        Field("severity", "str", "", "hard | soft | info"),
        Field("rule_reason", "str", "", "plain-language rationale"),
        Field("source", "str", "", "provenance of the rule"),
    ),
    "orders": (
        Field("order_id", "str", "", "unique order identifier"),
        Field("scenario_id", "str", "", "A | B | C (alternative demand histories)"),
        Field("order_timestamp", "timestamp", "UTC", "ISO-8601 UTC order time"),
        Field("order_status", "str", "", "completed"),
        Field("data_source", "str", "", "provenance of the row"),
        Field("data_quality_status", "str", "", "valid | warning | invalid"),
    ),
    "order_items": (
        Field("order_id", "str", "", "FK to orders.order_id"),
        Field("sku_id", "str", "", "FK to products.sku_id"),
        Field("quantity", "int", "units", "units ordered (> 0)"),
        Field("data_source", "str", "", "provenance of the row"),
    ),
    "storage_assignments_baseline": (
        Field("sku_id", "str", "", "FK to products.sku_id"),
        Field("location_id", "str", "", "FK to locations.location_id"),
        Field("assigned_quantity", "int", "units", "units stored in this bin"),
        Field("assignment_type", "str", "", "baseline (optimized layouts are written by later phases)"),
        Field("data_source", "str", "", "provenance of the row"),
    ),
    "location_graph": (
        Field("from_node_id", "str", "", "start node id"),
        Field("from_node_type", "str", "", "location | aisle_column | aisle_entrance | zone_hub | depot"),
        Field("to_node_id", "str", "", "end node id"),
        Field("to_node_type", "str", "", "location | aisle_column | aisle_entrance | zone_hub | depot"),
        Field("distance_m", "float", "m", "walking distance along the aisle network"),
        Field("data_source", "str", "", "provenance of the row"),
    ),
    "demand_summary": (
        Field("scenario_id", "str", "", "A | B | C"),
        Field("sku_id", "str", "", "FK to products.sku_id"),
        Field("demand_frequency", "int", "orders", "distinct orders containing the SKU"),
        Field("total_units", "int", "units", "total units ordered in the scenario"),
        Field("rank", "int", "", "demand rank (1 = highest)"),
        Field("cumulative_share", "float", "share", "cumulative share of demand frequency"),
        Field("abc_class", "str", "", "A | B | C"),
        Field("data_source", "str", "", "derived from the scenario order history"),
    ),
    "sku_affinity": (
        Field("scenario_id", "str", "", "A | B | C"),
        Field("sku_a", "str", "", "FK to products.sku_id (lexicographically smaller)"),
        Field("sku_b", "str", "", "FK to products.sku_id (lexicographically larger)"),
        Field("co_count", "int", "orders", "orders containing both SKUs"),
        Field("support", "float", "share", "co_count / number of orders"),
        Field("confidence_a_to_b", "float", "share", "co_count / orders containing sku_a"),
        Field("confidence_b_to_a", "float", "share", "co_count / orders containing sku_b"),
        Field("lift", "float", "ratio", "support / (share_a * share_b)"),
        Field("data_source", "str", "", "derived from the scenario order history"),
    ),
}

TABLE_NOTES: dict[str, tuple[str, str]] = {
    "warehouses": ("Warehouse master", "warehouse_id"),
    "categories": ("Product categories and storage notes", "category_id"),
    "products": ("Diverse SKU master with physical attributes", "sku_id"),
    "locations": ("Physical bins with geometry, capacity, and compatibility", "location_id"),
    "inventory": ("Current stock snapshot per bin", "sku_id + location_id"),
    "compatibility_rules": ("Category-pair storage compatibility", "rule_id"),
    "orders": ("Order headers for three alternative demand scenarios", "order_id"),
    "order_items": ("Order line items", "order_id + sku_id"),
    "storage_assignments_baseline": ("Feasible current layout (never overwritten)", "sku_id + location_id"),
    "location_graph": ("Directed aisle-network graph used for travel distance", "from_node_id + to_node_id"),
    "demand_summary": ("Per-SKU demand and ABC classification by scenario", "scenario_id + sku_id"),
    "sku_affinity": ("Per-scenario SKU-pair co-occurrence statistics", "scenario_id + sku_a + sku_b"),
}

HANDLING_TYPES = ("standard", "fragile", "heavy", "temperature_controlled")
HAZARD_CLASSES = ("NONE", "UNKNOWN", "flammable", "corrosive", "toxic", "compressed_gas")
ERGONOMIC_ZONES = ("golden", "secondary", "reserve")
ABC_CLASSES = ("A", "B", "C")
SCENARIO_IDS = ("A", "B", "C")
PROVENANCE_VALUES = ("measured", "sourced", "inferred", "synthetic", "derived")
QUALITY_VALUES = ("valid", "warning", "invalid")
NODE_TYPES = ("location", "aisle_column", "aisle_entrance", "zone_hub", "depot")
