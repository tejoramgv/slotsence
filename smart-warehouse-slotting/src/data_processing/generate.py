"""Deterministic generator for the generalized warehouse benchmark dataset.

The generator builds one coherent relational snapshot: a diverse SKU master, a zoned
warehouse, three alternative demand histories, a feasible baseline layout, and an
aisle-network distance graph. Every row is labelled synthetic or derived in the CSV
provenance columns. The same seed produces identical processed files.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Product classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CategorySpec:
    category_id: str
    name: str
    storage_requirements: str
    share: float
    physical_class: str
    handling: str
    hazards: tuple[str, ...]
    base_dims_cm: tuple[float, float, float]
    size_range: tuple[float, float]
    density_range: tuple[float, float]
    max_unit_weight_kg: float
    sample_names: tuple[str, ...]


CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec(
        "CAT001", "Accessories & Cables", "small, light", 25.0, "small", "standard",
        ("NONE",), (10.0, 6.0, 5.0), (0.5, 1.6), (0.15, 0.5), 2.0,
        ("USB-C Cable", "Phone Case", "AA Batteries", "Desk Organizer", "Charging Adapter"),
    ),
    CategorySpec(
        "CAT002", "Books & Media", "medium, boxed", 15.0, "medium", "standard",
        ("NONE",), (21.0, 15.0, 4.0), (0.6, 1.4), (0.5, 0.9), 4.0,
        ("Hardcover Notebook", "Paperback Novel", "Cookbook", "Art Album", "Study Guide"),
    ),
    CategorySpec(
        "CAT003", "Household Goods", "medium, boxed", 15.0, "medium", "standard",
        ("NONE",), (24.0, 18.0, 14.0), (0.6, 1.4), (0.3, 0.7), 15.0,
        ("Ceramic Mug", "Storage Basket", "LED Lamp", "Bath Towel Set", "Kitchen Scale"),
    ),
    CategorySpec(
        "CAT004", "Storage Containers", "large, bulky", 7.5, "bulky", "standard",
        ("NONE",), (45.0, 35.0, 35.0), (0.7, 1.4), (0.15, 0.4), 100.0,
        ("Storage Bin", "Plastic Crate", "Wardrobe Box", "Stackable Container", "Moving Carton"),
    ),
    CategorySpec(
        "CAT005", "Appliances", "large, bulky", 7.5, "bulky", "standard",
        ("NONE",), (55.0, 45.0, 45.0), (0.7, 1.3), (0.2, 0.5), 160.0,
        ("Microwave Oven", "Air Fryer", "Vacuum Cleaner", "Coffee Machine", "Blender"),
    ),
    CategorySpec(
        "CAT006", "Tools & Metal Parts", "heavy, dense", 7.5, "heavy", "heavy",
        ("NONE",), (18.0, 12.0, 10.0), (0.6, 1.5), (1.5, 3.0), 35.0,
        ("Wrench Set", "Drill Bit Kit", "Steel Brackets", "Hammer", "Screwdriver Set"),
    ),
    CategorySpec(
        "CAT007", "Packaged Bulk Goods", "heavy, dense", 7.5, "heavy", "heavy",
        ("NONE",), (25.0, 20.0, 18.0), (0.6, 1.4), (0.7, 1.2), 40.0,
        ("Detergent Pack", "Rice Bag", "Flour Sack", "Canned Goods Case", "Bottled Water Pack"),
    ),
    CategorySpec(
        "CAT008", "Glassware", "fragile", 5.0, "fragile", "fragile",
        ("NONE",), (15.0, 15.0, 10.0), (0.6, 1.4), (0.6, 1.5), 12.0,
        ("Wine Glasses", "Glass Vase", "Ceramic Plates", "Drinking Glass Set", "Glass Jar"),
    ),
    CategorySpec(
        "CAT009", "Electronics & Sensitive", "fragile", 5.0, "fragile", "fragile",
        ("NONE",), (20.0, 8.0, 15.0), (0.6, 1.4), (0.4, 1.2), 10.0,
        ("Wireless Headphones", "Smart Watch", "Tablet", "Camera Lens", "Portable Speaker"),
    ),
    CategorySpec(
        "CAT010", "Restricted / Hazardous", "restricted", 5.0, "hazard", "standard",
        ("flammable", "corrosive", "toxic"), (22.0, 16.0, 14.0), (0.6, 1.3), (0.9, 1.8), 30.0,
        ("Solvent Cleaner", "Bleach Bottle", "Aerosol Spray", "Battery Pack", "Adhesive Can"),
    ),
)

SPEC_BY_ID: dict[str, CategorySpec] = {s.category_id: s for s in CATEGORY_SPECS}
CLASS_BY_CATEGORY: dict[str, str] = {s.category_id: s.physical_class for s in CATEGORY_SPECS}

CLASS_DEMAND_FACTOR: dict[str, float] = {
    "small": 1.30,
    "medium": 1.15,
    "bulky": 0.45,
    "heavy": 0.75,
    "fragile": 0.90,
    "hazard": 0.35,
}

PACK_PRIORITY: dict[str, int] = {
    "bulky": 0,
    "heavy": 1,
    "hazard": 2,
    "fragile": 3,
    "medium": 4,
    "small": 5,
}


def _allocate_counts(n_skus: int, shares: Iterable[float]) -> list[int]:
    """Largest-remainder allocation so the category counts sum exactly to n_skus."""
    shares = list(shares)
    raw = [n_skus * share / 100.0 for share in shares]
    counts = [int(math.floor(value)) for value in raw]
    remainder = n_skus - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def generate_categories() -> list[dict[str, Any]]:
    return [
        {
            "category_id": spec.category_id,
            "category_name": spec.name,
            "storage_requirements": spec.storage_requirements,
            "data_source": "synthetic",
        }
        for spec in CATEGORY_SPECS
    ]


def generate_products(n_skus: int, seed: int) -> list[dict[str, Any]]:
    """Generate a physically plausible, class-correlated SKU master."""
    rng = random.Random(seed)
    counts = _allocate_counts(n_skus, [spec.share for spec in CATEGORY_SPECS])
    products: list[dict[str, Any]] = []
    sku_number = 0

    for spec, count in zip(CATEGORY_SPECS, counts):
        for local_index in range(count):
            sku_number += 1
            size = rng.uniform(*spec.size_range)
            width = spec.base_dims_cm[0] * size * rng.uniform(0.92, 1.08)
            height = spec.base_dims_cm[1] * size * rng.uniform(0.92, 1.08)
            depth = spec.base_dims_cm[2] * size * rng.uniform(0.92, 1.08)
            width, height, depth = (round(max(0.1, v), 1) for v in (width, height, depth))
            volume_cm3 = round(width * height * depth, 2)
            density = rng.uniform(*spec.density_range)
            weight = round(volume_cm3 / 1000.0 * density, 3)
            weight = max(0.01, min(weight, spec.max_unit_weight_kg))

            hazard = spec.hazards[local_index % len(spec.hazards)]
            base_name = spec.sample_names[local_index % len(spec.sample_names)]
            products.append(
                {
                    "sku_id": f"SKU{sku_number:06d}",
                    "product_name": f"{base_name} {local_index + 1:04d}",
                    "category_id": spec.category_id,
                    "width_cm": width,
                    "height_cm": height,
                    "depth_cm": depth,
                    "unit_volume_cm3": volume_cm3,
                    "unit_weight_kg": weight,
                    "handling_type": spec.handling,
                    "hazard_class": hazard,
                    "data_source": "synthetic",
                    "data_quality_status": "valid",
                    "physical_data_source": "synthetic_class_rule",
                    "physical_confidence": "low",
                }
            )
    return products


# ---------------------------------------------------------------------------
# Warehouse topology + distance graph
# ---------------------------------------------------------------------------

def _build_warehouse_3x3(
    wh: dict[str, Any], zones: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    warehouse_id = wh["warehouse_id"]
    outer_margin = 3.0
    zone_size = 9.6
    aisle_width = float(wh.get("aisle_width_m", 3.0))
    rack_depth = float(wh.get("rack_depth_m", 1.0))
    total_w = 2 * outer_margin + 3 * zone_size + 2 * aisle_width  # 40.8m

    bin_classes = wh["bin_classes"]
    zone_rules = wh["zone_rules"]
    golden_levels = set(int(level) for level in wh.get("golden_levels", [1, 2]))
    shelf_height = float(wh.get("shelf_height_m", 0.5))

    x_corrs = [
        round(outer_margin / 2.0, 3),  # 1.5
        round(outer_margin + zone_size + aisle_width / 2.0, 3),  # 14.1
        round(outer_margin + 2 * zone_size + 1.5 * aisle_width, 3),  # 26.7
        round(outer_margin + 3 * zone_size + 2 * aisle_width + outer_margin / 2.0, 3),  # 39.3
    ]
    y_corrs = [
        round(outer_margin / 2.0, 3),  # 1.5
        round(outer_margin + zone_size + aisle_width / 2.0, 3),  # 14.1
        round(outer_margin + 2 * zone_size + 1.5 * aisle_width, 3),  # 26.7
        round(outer_margin + 3 * zone_size + 2 * aisle_width + outer_margin / 2.0, 3),  # 39.3
    ]

    locations: list[dict[str, Any]] = []
    location_column: dict[str, str] = {}
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, float]] = []

    depot_x = round(total_w / 2.0, 3)
    depot_y = 0.5
    nodes["DEPOT"] = {"node_type": "depot", "x_m": depot_x, "y_m": depot_y}
    depot_corr = "CORR_DEPOT_ENTRY"
    nodes[depot_corr] = {"node_type": "aisle_entrance", "x_m": depot_x, "y_m": y_corrs[0]}
    edges.append(("DEPOT", depot_corr, round(abs(y_corrs[0] - depot_y), 3)))

    for i, xc in enumerate(x_corrs):
        for j, yc in enumerate(y_corrs):
            nodes[f"INT_{i}_{j}"] = {"node_type": "zone_hub", "x_m": xc, "y_m": yc}

    for i in range(len(x_corrs)):
        for j in range(len(y_corrs) - 1):
            edges.append((f"INT_{i}_{j}", f"INT_{i}_{j+1}", round(abs(y_corrs[j+1] - y_corrs[j]), 3)))
    for j in range(len(y_corrs)):
        for i in range(len(x_corrs) - 1):
            edges.append((f"INT_{i}_{j}", f"INT_{i+1}_{j}", round(abs(x_corrs[i+1] - x_corrs[i]), 3)))

    edges.append((depot_corr, "INT_1_0", round(abs(depot_x - x_corrs[1]), 3)))
    edges.append((depot_corr, "INT_2_0", round(abs(x_corrs[2] - depot_x), 3)))

    col_nodes: dict[tuple[float, float], str] = {}
    loc_id_counter = 0
    bay_len = zone_size / 8.0
    bin_len = bay_len / 2.0

    for idx, zone in enumerate(zones):
        zid = zone["zone_id"]
        ztype = zone["zone_type"]
        bclass = zone["bin_class"]
        col = idx % 3
        row = idx // 3
        zx0 = outer_margin + col * (zone_size + aisle_width)
        zy0 = outer_margin + row * (zone_size + aisle_width)
        zx1 = zx0 + zone_size
        zy1 = zy0 + zone_size

        bc = bin_classes[bclass]
        bin_w_cm, bin_h_cm, bin_d_cm = (float(v) for v in bc["dims_cm"])
        shelf_weights = [float(v) for v in bc["shelf_weight_kg"]]
        handling_rule = list(zone_rules[ztype]["allowed_handling"])
        hazard_rule = list(zone_rules[ztype]["allowed_hazard"])
        rack_max_weight = round(float(wh.get("rack_limit_factor", 0.85)) * sum(shelf_weights) * 2 * 8, 1)

        sides = [
            ("South", 1, 1),
            ("North", 1, 2),
            ("West", 2, 1),
            ("East", 2, 2),
        ]
        for side, aisle_num, rack_num in sides:
            rack_id = f"R{idx+1}-{aisle_num}-{rack_num}"
            aisle_id = f"A{idx+1}-{aisle_num}"

            for bay in range(8):
                for shelf_level in range(4):
                    for bin_idx in range(2):
                        loc_id_counter += 1
                        loc_id = f"LOC{loc_id_counter:06d}"
                        z_m = round((shelf_level + 0.5) * shelf_height, 3)

                        if side == "South":
                            x_m = round(zx0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            y_m = round(zy0 + rack_depth / 2.0, 3)
                            cx, cy = x_m, y_corrs[row]
                        elif side == "North":
                            x_m = round(zx0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            y_m = round(zy1 - rack_depth / 2.0, 3)
                            cx, cy = x_m, y_corrs[row + 1]
                        elif side == "West":
                            x_m = round(zx0 + rack_depth / 2.0, 3)
                            y_m = round(zy0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            cx, cy = x_corrs[col], y_m
                        else:  # East
                            x_m = round(zx1 - rack_depth / 2.0, 3)
                            y_m = round(zy0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            cx, cy = x_corrs[col + 1], y_m

                        if shelf_level in golden_levels:
                            ergonomic = "golden"
                        elif shelf_level == 0:
                            ergonomic = "secondary"
                        else:
                            ergonomic = "reserve"

                        allowed_handling = list(handling_rule)
                        if ztype == "heavy" and shelf_level > 2:
                            allowed_handling = [h for h in allowed_handling if h != "heavy"]

                        locations.append({
                            "location_id": loc_id,
                            "warehouse_id": warehouse_id,
                            "zone_id": zid,
                            "zone_type": ztype,
                            "aisle_id": aisle_id,
                            "rack_id": rack_id,
                            "bay": bay,
                            "shelf_level": shelf_level,
                            "x_m": x_m,
                            "y_m": y_m,
                            "z_m": z_m,
                            "width_cm": bin_w_cm,
                            "height_cm": bin_h_cm,
                            "depth_cm": bin_d_cm,
                            "max_weight_kg": shelf_weights[shelf_level],
                            "usable_volume_cm3": round(bin_w_cm * bin_h_cm * bin_d_cm, 2),
                            "shelf_max_weight_kg": shelf_weights[shelf_level] * 4.0,
                            "rack_max_weight_kg": rack_max_weight,
                            "ergonomic_zone": ergonomic,
                            "allowed_handling": "|".join(allowed_handling),
                            "allowed_hazard": "|".join(hazard_rule),
                            "data_source": "synthetic",
                            "data_quality_status": "valid",
                            "_handling": frozenset(allowed_handling),
                            "_hazard": frozenset(hazard_rule),
                            "_residual_weight": shelf_weights[shelf_level],
                        })

                        ckey = (cx, cy)
                        if ckey not in col_nodes:
                            cid = f"ACOL_{len(col_nodes)+1:04d}"
                            col_nodes[ckey] = cid
                            nodes[cid] = {"node_type": "aisle_column", "x_m": cx, "y_m": cy}

                            connected = False
                            for j, yc in enumerate(y_corrs):
                                if abs(cy - yc) < 1e-4:
                                    for i in range(len(x_corrs) - 1):
                                        if x_corrs[i] <= cx <= x_corrs[i + 1]:
                                            edges.append((cid, f"INT_{i}_{j}", round(abs(cx - x_corrs[i]), 3)))
                                            edges.append((cid, f"INT_{i+1}_{j}", round(abs(x_corrs[i + 1] - cx), 3)))
                                            connected = True
                                            break
                                    break
                            if not connected:
                                for i, xc in enumerate(x_corrs):
                                    if abs(cx - xc) < 1e-4:
                                        for j in range(len(y_corrs) - 1):
                                            if y_corrs[j] <= cy <= y_corrs[j + 1]:
                                                edges.append((cid, f"INT_{i}_{j}", round(abs(cy - y_corrs[j]), 3)))
                                                edges.append((cid, f"INT_{i}_{j+1}", round(abs(y_corrs[j + 1] - cy), 3)))
                                                connected = True
                                                break
                                        break

                        cid = col_nodes[ckey]
                        location_column[loc_id] = cid
                        nodes[loc_id] = {"node_type": "location", "x_m": x_m, "y_m": y_m}
                        lat_dist = round(aisle_width / 2.0 + rack_depth / 2.0, 3)
                        edges.append((loc_id, cid, lat_dist))

    graph_rows = _build_graph_rows(nodes, edges)
    return locations, graph_rows, location_column


def build_warehouse(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Build bins, the aisle-network graph, and each bin's shortest walk to the depot."""
    wh = config["warehouse"]
    warehouse_id = wh["warehouse_id"]
    aisle_width = float(wh["aisle_width_m"])
    rack_depth = float(wh["rack_depth_m"])
    shelf_height = float(wh["shelf_height_m"])
    zone_gap = float(wh["zone_gap_m"])
    aisle_pitch = 2.0 * rack_depth + aisle_width

    zones: list[dict[str, Any]] = []
    zone_number = 0
    for entry in wh["zone_layout"]:
        for _ in range(int(entry["count"])):
            zone_number += 1
            zones.append(
                {
                    "zone_id": f"Z{zone_number}",
                    "zone_type": entry["zone_type"],
                    "bin_class": entry["bin_class"],
                }
            )

    if len(zones) == 9:
        return _build_warehouse_3x3(wh, zones)

    y_cursor = 0.0
    for zone in zones:
        zone_height = int(wh["n_aisles_per_zone"]) * aisle_pitch
        zone["y0"] = y_cursor
        zone["y_center"] = y_cursor + zone_height / 2.0
        y_cursor += zone_height + zone_gap

    depot_x, depot_y = (float(v) for v in wh["dispatch_point_m"])

    # Graph nodes: location bins, aisle columns, aisle entrances, zone hubs, depot.
    nodes: dict[str, dict[str, Any]] = {
        "DEPOT": {"node_type": "depot", "x_m": depot_x, "y_m": depot_y}
    }
    edges: list[tuple[str, str, float]] = []
    locations: list[dict[str, Any]] = []
    location_column: dict[str, str] = {}
    location_number = 0

    bin_classes = wh["bin_classes"]
    zone_rules = wh["zone_rules"]
    golden_levels = set(int(level) for level in wh["golden_levels"])
    bays = int(wh["n_bays_per_rack"])
    shelves = int(wh["n_shelves"])
    bins_per_shelf = int(wh["n_bins_per_shelf"])

    for zone in zones:
        zone_id = zone["zone_id"]
        zone_number = int(zone_id[1:])
        zone_type = zone["zone_type"]
        bin_class = zone["bin_class"]
        bin_width_cm, bin_height_cm, bin_depth_cm = (float(v) for v in bin_classes[bin_class]["dims_cm"])
        bin_width_m = bin_width_cm / 100.0
        shelf_weights = [float(v) for v in bin_classes[bin_class]["shelf_weight_kg"]]
        handling_rule = list(zone_rules[zone_type]["allowed_handling"])
        hazard_rule = list(zone_rules[zone_type]["allowed_hazard"])
        rack_max_weight = round(
            float(wh["rack_limit_factor"])
            * sum(shelf_weights)
            * bins_per_shelf
            * bays,
            1,
        )
        hub_id = f"ZHUB-{zone_id}"
        nodes[hub_id] = {"node_type": "zone_hub", "x_m": 0.0, "y_m": zone["y_center"]}
        edges.append((hub_id, "DEPOT", abs(zone["y_center"] - depot_y)))

        for aisle_index in range(int(wh["n_aisles_per_zone"])):
            aisle_id = f"A{zone_number}-{aisle_index + 1}"
            aisle_y = zone["y0"] + (aisle_index + 0.5) * aisle_pitch
            entrance_id = f"AENT-{zone_id}-{aisle_id.split('-')[1]}"
            nodes[entrance_id] = {"node_type": "aisle_entrance", "x_m": 0.0, "y_m": aisle_y}
            edges.append((entrance_id, hub_id, abs(aisle_y - zone["y_center"])))
            columns: dict[float, str] = {}

            for rack_index in range(int(wh["n_racks_per_aisle"])):
                side = -1.0 if rack_index % 2 == 0 else 1.0
                segment = rack_index // 2
                rack_x0 = segment * bays * bins_per_shelf * bin_width_m
                rack_y = aisle_y + side * (aisle_width / 2.0 + rack_depth / 2.0)
                rack_id = f"R{zone_number}-{aisle_index + 1}-{rack_index + 1}"

                for bay in range(bays):
                    for shelf_level in range(shelves):
                        for bin_index in range(bins_per_shelf):
                            location_number += 1
                            location_id = f"LOC{location_number:06d}"
                            x = rack_x0 + (bay * bins_per_shelf + bin_index + 0.5) * bin_width_m
                            z = (shelf_level + 0.5) * shelf_height
                            column_key = round(x, 3)
                            if column_key not in columns:
                                column_id = f"ACOL-{zone_id}-{aisle_id.split('-')[1]}-{len(columns) + 1:02d}"
                                columns[column_key] = column_id
                                nodes[column_id] = {
                                    "node_type": "aisle_column",
                                    "x_m": column_key,
                                    "y_m": aisle_y,
                                }
                            column_id = columns[column_key]

                            allowed_handling = list(handling_rule)
                            if zone_type == "heavy" and shelf_level > 2:
                                allowed_handling = [h for h in allowed_handling if h != "heavy"]

                            if shelf_level in golden_levels:
                                ergonomic_zone = "golden"
                            elif shelf_level == 0:
                                ergonomic_zone = "secondary"
                            else:
                                ergonomic_zone = "reserve"

                            locations.append(
                                {
                                    "location_id": location_id,
                                    "warehouse_id": warehouse_id,
                                    "zone_id": zone_id,
                                    "zone_type": zone_type,
                                    "aisle_id": aisle_id,
                                    "rack_id": rack_id,
                                    "bay": bay,
                                    "shelf_level": shelf_level,
                                    "x_m": round(x, 3),
                                    "y_m": round(rack_y, 3),
                                    "z_m": round(z, 3),
                                    "width_cm": bin_width_cm,
                                    "height_cm": bin_height_cm,
                                    "depth_cm": bin_depth_cm,
                                    "max_weight_kg": shelf_weights[shelf_level],
                                    "usable_volume_cm3": round(bin_width_cm * bin_height_cm * bin_depth_cm, 2),
                                    "shelf_max_weight_kg": shelf_weights[shelf_level],
                                    "rack_max_weight_kg": rack_max_weight,
                                    "ergonomic_zone": ergonomic_zone,
                                    "allowed_handling": "|".join(allowed_handling),
                                    "allowed_hazard": "|".join(hazard_rule),
                                    "distance_to_dispatch_m": 0.0,
                                    "data_source": "synthetic",
                                    "data_quality_status": "valid",
                                    "_handling": frozenset(allowed_handling),
                                    "_hazard": frozenset(hazard_rule),
                                    "_residual_weight": shelf_weights[shelf_level],
                                }
                            )
                            location_column[location_id] = column_id
                            nodes[location_id] = {
                                "node_type": "location",
                                "x_m": round(x, 3),
                                "y_m": aisle_y,
                            }

            ordered_columns = sorted((key, value) for key, value in columns.items())
            previous_id = entrance_id
            previous_x = 0.0
            for column_x, column_id in ordered_columns:
                edges.append((previous_id, column_id, column_x - previous_x))
                previous_id = column_id
                previous_x = column_x

    for location in locations:
        lateral = aisle_width / 2.0 + rack_depth / 2.0
        edges.append((location["location_id"], location_column[location["location_id"]], lateral))

    graph_rows = _build_graph_rows(nodes, edges)
    distances = _shortest_paths(nodes, edges, "DEPOT")
    for location in locations:
        distance = distances.get(location["location_id"])
        if distance is None:
            raise RuntimeError(f"location {location['location_id']} is unreachable from the depot")
        location["distance_to_dispatch_m"] = round(distance, 3)

    return locations, graph_rows, location_column


def _build_graph_rows(
    nodes: dict[str, dict[str, Any]], edges: list[tuple[str, str, float]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right, distance in edges:
        rows.append(
            {
                "from_node_id": left,
                "from_node_type": nodes[left]["node_type"],
                "to_node_id": right,
                "to_node_type": nodes[right]["node_type"],
                "distance_m": round(distance, 3),
                "data_source": "derived",
            }
        )
        rows.append(
            {
                "from_node_id": right,
                "from_node_type": nodes[right]["node_type"],
                "to_node_id": left,
                "to_node_type": nodes[left]["node_type"],
                "distance_m": round(distance, 3),
                "data_source": "derived",
            }
        )
    rows.sort(key=lambda row: (row["from_node_id"], row["to_node_id"]))
    return rows


def _shortest_paths(
    nodes: dict[str, dict[str, Any]],
    edges: list[tuple[str, str, float]],
    source: str,
) -> dict[str, float]:
    adjacency: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in nodes}
    for left, right, distance in edges:
        adjacency[left].append((right, distance))
        adjacency[right].append((left, distance))
    distances = {node_id: float("inf") for node_id in nodes}
    distances[source] = 0.0
    heap: list[tuple[float, str]] = [(0.0, source)]
    while heap:
        current_distance, node_id = heapq.heappop(heap)
        if current_distance > distances[node_id]:
            continue
        for neighbour, weight in adjacency[node_id]:
            candidate = current_distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                heapq.heappush(heap, (candidate, neighbour))
    return distances


# ---------------------------------------------------------------------------
# Compatibility rules
# ---------------------------------------------------------------------------

def generate_compatibility_rules() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = list(CATEGORY_SPECS)
    rule_number = 0
    fragile_ids = {spec.category_id for spec in specs if spec.physical_class == "fragile"}
    heavy_ids = {spec.category_id for spec in specs if spec.physical_class == "heavy"}
    hazard_id = next(spec.category_id for spec in specs if spec.physical_class == "hazard")

    for index_a, spec_a in enumerate(specs):
        for spec_b in specs[index_a:]:
            rule_number += 1
            pair = {spec_a.category_id, spec_b.category_id}
            if hazard_id in pair and pair != {hazard_id}:
                allowed = False
                severity = "hard"
                reason = "Hazardous products must be separated from non-hazardous goods."
            elif pair & fragile_ids and pair & heavy_ids:
                allowed = False
                severity = "hard"
                reason = "Fragile items are excluded from heavy, high-impact storage zones."
            else:
                allowed = True
                severity = "info"
                reason = "No category-level restriction; location handling and hazard rules still apply."
            rows.append(
                {
                    "rule_id": f"RULE{rule_number:03d}",
                    "category_a": spec_a.category_id,
                    "category_b": spec_b.category_id,
                    "allowed": "true" if allowed else "false",
                    "severity": severity,
                    "rule_reason": reason,
                    "source": "synthetic_policy",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def _weighted_sample_without_replacement(
    rng: random.Random, population_size: int, weights: list[float], count: int
) -> list[int]:
    chosen: set[int] = set()
    attempts = 0
    limit = max(100, count * 200)
    while len(chosen) < count and attempts < limit:
        chosen.add(rng.choices(range(population_size), weights=weights, k=1)[0])
        attempts += 1
    if len(chosen) < count:
        for index in range(population_size):
            if index not in chosen:
                chosen.add(index)
            if len(chosen) == count:
                break
    return list(chosen)


def _build_clusters(products: list[dict[str, Any]], cluster_size: int, rng: random.Random) -> list[list[int]]:
    by_category: dict[str, list[int]] = {}
    for index, product in enumerate(products):
        by_category.setdefault(product["category_id"], []).append(index)
    clusters: list[list[int]] = []
    for category_id in sorted(by_category):
        indices = by_category[category_id][:]
        rng.shuffle(indices)
        for start in range(0, len(indices), cluster_size):
            chunk = indices[start : start + cluster_size]
            if len(chunk) >= 2:
                clusters.append(chunk)
    return clusters


def _scenario_weights(
    scenario: dict[str, Any],
    products: list[dict[str, Any]],
    seed: int,
) -> tuple[list[float], list[list[int]] | None]:
    rng = random.Random(seed)
    weights = [CLASS_DEMAND_FACTOR[CLASS_BY_CATEGORY[p["category_id"]]] for p in products]
    kind = scenario["kind"]
    clusters: list[list[int]] | None = None

    if kind == "concentrated":
        order = list(range(len(products)))
        rng.shuffle(order)
        alpha = float(scenario.get("zipf_alpha", 1.15))
        ranked = [1.0 / ((rank + 10.0) ** alpha) for rank in range(len(products))]
        weights = [0.0] * len(products)
        for rank, product_index in enumerate(order):
            weights[product_index] = ranked[rank] * CLASS_DEMAND_FACTOR[
                CLASS_BY_CATEGORY[products[product_index]["category_id"]]
            ]
    elif kind == "balanced":
        # Broad demand: every SKU is close to equally likely, with small noise.
        weights = [rng.uniform(0.8, 1.2) for _ in products]
    elif kind == "affinity":
        weights = [weight * rng.uniform(0.85, 1.15) for weight in weights]
        clusters = _build_clusters(products, int(scenario.get("cluster_size", 10)), rng)
    else:
        raise ValueError(f"unknown demand scenario kind: {kind}")

    return weights, clusters


def generate_orders(
    config: dict[str, Any],
    products: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    window_start = datetime.strptime(config["order_window"]["start_date"], "%Y-%m-%d")
    window_days = int(config["order_window"]["days"])
    target_lines = int(config["n_order_lines_per_scenario"])
    day_weights = [
        0.7 if (window_start + timedelta(days=day)).weekday() >= 5 else 1.0
        for day in range(window_days)
    ]

    all_orders: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(config["demand"]["scenarios"]):
        scenario_id = scenario["scenario_id"]
        rng = random.Random(seed + 1000 + scenario_index)
        weights, clusters = _scenario_weights(scenario, products, seed + 2000 + scenario_index)
        cluster_share = float(scenario.get("cluster_order_share", 0.75))
        max_items = int(scenario.get("max_items_per_order", 6))
        item_weights = [5, 15, 25, 25, 20, 10][:max_items]
        quantity_values = [1, 2, 3, 4]
        quantity_weights = [0.60, 0.25, 0.10, 0.05]

        raw_orders: list[tuple[datetime, list[tuple[int, int]]]] = []
        remaining = target_lines
        while remaining > 0:
            item_count = min(rng.choices(range(1, max_items + 1), weights=item_weights, k=1)[0], remaining)
            chosen: list[int] = []
            if clusters and rng.random() < cluster_share:
                cluster = clusters[rng.choices(range(len(clusters)), k=1)[0]]
                take = min(item_count, len(cluster))
                chosen.extend(rng.sample(cluster, take))
            if len(chosen) < item_count:
                extra = _weighted_sample_without_replacement(
                    rng, len(products), weights, item_count - len(chosen)
                )
                for index in extra:
                    if index not in chosen:
                        chosen.append(index)
            chosen = chosen[:item_count]
            lines = [
                (index, rng.choices(quantity_values, weights=quantity_weights, k=1)[0])
                for index in chosen
            ]
            remaining -= len(lines)

            day = rng.choices(range(window_days), weights=day_weights, k=1)[0]
            hour = rng.choices(range(6, 22), weights=[1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1], k=1)[0]
            minute = rng.randrange(60)
            second = rng.randrange(60)
            timestamp = window_start + timedelta(days=day, hours=hour, minutes=minute, seconds=second)
            raw_orders.append((timestamp, lines))

        raw_orders.sort(key=lambda entry: entry[0])
        for order_index, (timestamp, lines) in enumerate(raw_orders, start=1):
            order_id = f"ORD{scenario_id}{order_index:06d}"
            all_orders.append(
                {
                    "order_id": order_id,
                    "scenario_id": scenario_id,
                    "order_timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "order_status": "completed",
                    "data_source": "synthetic",
                    "data_quality_status": "valid",
                }
            )
            for product_index, quantity in sorted(lines, key=lambda entry: products[entry[0]]["sku_id"]):
                all_items.append(
                    {
                        "order_id": order_id,
                        "sku_id": products[product_index]["sku_id"],
                        "quantity": quantity,
                        "data_source": "synthetic",
                    }
                )

    all_orders.sort(key=lambda row: (row["scenario_id"], row["order_id"]))
    all_items.sort(key=lambda row: (row["order_id"], row["sku_id"]))
    return all_orders, all_items


# ---------------------------------------------------------------------------
# Inventory + feasible baseline
# ---------------------------------------------------------------------------

def _fits(product: dict[str, Any], location: dict[str, Any]) -> bool:
    return (
        product["width_cm"] <= location["width_cm"]
        and product["height_cm"] <= location["height_cm"]
        and product["depth_cm"] <= location["depth_cm"]
        and product["handling_type"] in location["_handling"]
        and product["hazard_class"] in location["_hazard"]
    )


def _static_capacity(product: dict[str, Any], location: dict[str, Any]) -> int:
    if not _fits(product, location):
        return 0
    volume_units = int(math.floor(location["usable_volume_cm3"] / product["unit_volume_cm3"]))
    weight_units = int(math.floor(location["max_weight_kg"] / product["unit_weight_kg"]))
    return min(volume_units, weight_units)


def build_inventory_and_baseline(
    config: dict[str, Any],
    products: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    order_items: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Size inventory from scenario A, then pack it into feasible bins.

    The baseline is deliberately unoptimized (random feasible placement) but fully feasible:
    every SKU is assigned, and bin/shelf/rack weight, volume, handling, and hazard limits
    are respected. Capacity shortfalls raise instead of leaving SKUs unassigned.
    """
    window_days = int(config["order_window"]["days"])
    inventory_cfg = config["inventory"]
    max_bins_by_class = inventory_cfg["max_bins_per_class"]
    scenario_a_orders = {row["order_id"] for row in orders if row["scenario_id"] == "A"}
    units_by_sku: dict[str, int] = {product["sku_id"]: 0 for product in products}
    for item in order_items:
        if item["order_id"] in scenario_a_orders:
            units_by_sku[item["sku_id"]] += int(item["quantity"])

    product_by_sku = {product["sku_id"]: product for product in products}
    feasible_pairs: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for product in products:
        pairs = [
            (location, capacity)
            for location in locations
            if (capacity := _static_capacity(product, location)) >= 1
        ]
        if not pairs:
            raise RuntimeError(
                f"{product['sku_id']} does not fit any compatible location "
                f"({product['handling_type']}, {product['hazard_class']})"
            )
        feasible_pairs[product["sku_id"]] = pairs

    quantities: dict[str, int] = {}
    for product in products:
        sku_id = product["sku_id"]
        # Use the smallest compatible bin so the cap holds no matter which bins the
        # random baseline picks; otherwise a bulk-bin capacity could overflow standard bins.
        conservative_capacity = min(capacity for _, capacity in feasible_pairs[sku_id])
        max_bins = int(max_bins_by_class[CLASS_BY_CATEGORY[product["category_id"]]])
        demand_based = math.ceil(
            units_by_sku[sku_id] / window_days * float(inventory_cfg["cover_days"])
            * (1.0 + float(inventory_cfg["safety_factor"]))
        )
        target = max(demand_based, int(inventory_cfg["min_units"]))
        quantities[sku_id] = max(1, min(target, conservative_capacity * max_bins))

    rng = random.Random(seed + 5000)
    residual_bin = {location["location_id"]: location["max_weight_kg"] for location in locations}
    residual_shelf = {
        (location["rack_id"], location["shelf_level"]): location["shelf_max_weight_kg"]
        for location in locations
    }
    residual_rack = {location["rack_id"]: location["rack_max_weight_kg"] for location in locations}
    occupied: set[str] = set()
    assignments: list[dict[str, Any]] = []

    ordered_products = sorted(
        products,
        key=lambda product: (
            PACK_PRIORITY[CLASS_BY_CATEGORY[product["category_id"]]],
            -product["unit_volume_cm3"],
            product["sku_id"],
        ),
    )

    for product in ordered_products:
        sku_id = product["sku_id"]
        remaining = quantities[sku_id]
        while remaining > 0:
            candidates: list[tuple[dict[str, Any], int]] = []
            for location, _ in feasible_pairs[sku_id]:
                location_id = location["location_id"]
                if location_id in occupied:
                    continue
                bin_capacity = min(
                    int(math.floor(location["usable_volume_cm3"] / product["unit_volume_cm3"])),
                    int(math.floor(residual_bin[location_id] / product["unit_weight_kg"])),
                    int(math.floor(residual_shelf[(location["rack_id"], location["shelf_level"])] / product["unit_weight_kg"])),
                    int(math.floor(residual_rack[location["rack_id"]] / product["unit_weight_kg"])),
                )
                if bin_capacity >= 1:
                    candidates.append((location, bin_capacity))
            if not candidates:
                raise RuntimeError(
                    f"capacity exhausted while placing {sku_id}: {remaining} units left; "
                    f"increase zones, bin sizes, or reduce inventory cover"
                )
            location, bin_capacity = rng.choice(candidates)
            take = min(remaining, bin_capacity)
            location_id = location["location_id"]
            occupied.add(location_id)
            residual_bin[location_id] -= take * product["unit_weight_kg"]
            residual_shelf[(location["rack_id"], location["shelf_level"])] -= take * product["unit_weight_kg"]
            residual_rack[location["rack_id"]] -= take * product["unit_weight_kg"]
            assignments.append(
                {
                    "sku_id": sku_id,
                    "location_id": location_id,
                    "assigned_quantity": take,
                    "assignment_type": "baseline",
                    "data_source": "synthetic",
                }
            )
            remaining -= take

    assignments.sort(key=lambda row: (row["sku_id"], row["location_id"]))
    inventory = [
        {
            "sku_id": row["sku_id"],
            "location_id": row["location_id"],
            "quantity_on_hand": row["assigned_quantity"],
            "data_source": "derived",
            "data_quality_status": "valid",
        }
        for row in assignments
    ]

    for sku_id, quantity in quantities.items():
        assigned = sum(row["assigned_quantity"] for row in assignments if row["sku_id"] == sku_id)
        if assigned != quantity:
            raise RuntimeError(f"inventory balance failed for {sku_id}: {assigned} != {quantity}")

    return assignments, inventory


def inventory_totals(
    products: list[dict[str, Any]], assignments: list[dict[str, Any]]
) -> dict[str, int]:
    totals = {product["sku_id"]: 0 for product in products}
    for row in assignments:
        totals[row["sku_id"]] += int(row["assigned_quantity"])
    return totals
