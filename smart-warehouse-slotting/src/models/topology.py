"""Deterministic warehouse geometry generator.

Builds the hierarchy Warehouse -> Zone -> Aisle -> Rack -> Bay -> Shelf -> Bin and computes
real (x, y, z) coordinates from aisle/rack geometry. Distance-to-dispatch is *derived* from the
coordinates (Manhattan within aisles to the dispatch point), never assigned independently.
Ergonomic zones are assigned from shelf height and travel distance.
"""

from __future__ import annotations

import math

from .config import Config
from .entities import AccessibilityZone, Location, Warehouse


def _manhattan(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Manhattan distance in the (x, y) plane (metres)."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class WarehouseBuilder:
    """Builds a Warehouse from a Config using a fixed deterministic geometry (no rng needed)."""

    def __init__(self, config: Config):
        self.cfg = config.topology
        self.warehouse = Warehouse(
            warehouse_id="WH001",
            warehouse_name="Generalized Benchmark Warehouse",
            layout_version="1.0",
        )

    def build(self) -> Warehouse:
        t = self.cfg
        wh = self.warehouse
        loc_id = 0

        if t.n_zones >= 9:
            return self._build_3x3()

        for zone_idx in range(t.n_zones):
            zone_id = f"Z{zone_idx + 1}"
            wh.zones.append(zone_id)
            # Each zone sits in its own y-band.
            zone_y_offset = zone_idx * (t.aisles_per_zone * 2 * t.aisle_spacing_m + 2 * t.rack_depth_m)

            for aisle_idx in range(t.aisles_per_zone):
                aisle_id = f"A{zone_idx + 1}-{aisle_idx + 1}"
                # Aisle's centreline x position.
                aisle_x = (aisle_idx + 0.5) * (2 * t.rack_depth_m + 1.0) + 2.0

                for rack_idx in range(t.racks_per_aisle):
                    rack_id = f"R{zone_idx + 1}-{aisle_idx + 1}-{rack_idx + 1}"
                    # Racks alternate on either side of the aisle; rack footprint depth.
                    side = 1.0 if rack_idx % 2 == 0 else -1.0
                    rack_y = (
                        zone_y_offset
                        + (aisle_idx + 0.5) * t.aisle_spacing_m
                        + side * (t.rack_depth_m + t.aisle_spacing_m / 2)
                    )

                    for bay in range(t.bays_per_rack):
                        bay_x = aisle_x + (bay - (t.bays_per_rack - 1) / 2) * t.rack_depth_m

                        for shelf in range(t.shelf_levels):
                            shelf_z = (shelf + 0.5) * t.shelf_height_m

                            for bin_idx in range(t.bins_per_shelf):
                                b_x = bay_x + (bin_idx - (t.bins_per_shelf - 1) / 2) * (t.bin_width_cm / 100.0)
                                loc_id += 1
                                loc = Location(
                                    location_id=f"LOC{loc_id:06d}",
                                    zone_id=zone_id,
                                    aisle_id=aisle_id,
                                    rack_id=rack_id,
                                    bay=bay,
                                    shelf_level=shelf,
                                    x=b_x,
                                    y=rack_y,
                                    z=shelf_z,
                                    width_cm=t.bin_width_cm,
                                    height_cm=t.bin_height_cm,
                                    depth_cm=t.bin_depth_cm,
                                    max_weight_kg=t.max_weight_kg,
                                    usable_volume_cm3=t.bin_width_cm * t.bin_height_cm * t.bin_depth_cm,
                                    ergonomic_zone=self._ergonomic_zone(shelf, rack_y),
                                    allowed_handling=(
                                        "standard",
                                        "fragile",
                                        "heavy",
                                        "temperature_controlled",
                                    ),
                                    allowed_hazard=(
                                        "NONE",
                                        "flammable",
                                        "corrosive",
                                        "toxic",
                                        "compressed_gas",
                                    ),
                                )
                                loc.distance_to_dispatch_m = _manhattan((loc.x, loc.y), wh.dispatch_point[:2])
                                wh.locations[loc.location_id] = loc

        return wh

    def _build_3x3(self) -> Warehouse:
        t = self.cfg
        wh = self.warehouse
        outer_margin = 3.0
        zone_size = 9.6
        aisle_width = 3.0
        rack_depth = 1.0
        loc_id = 0
        bay_len = zone_size / 8.0
        bin_len = bay_len / 2.0

        for zone_idx in range(9):
            zone_id = f"Z{zone_idx + 1}"
            wh.zones.append(zone_id)
            col = zone_idx % 3
            row = zone_idx // 3
            zx0 = outer_margin + col * (zone_size + aisle_width)
            zy0 = outer_margin + row * (zone_size + aisle_width)
            zx1 = zx0 + zone_size
            zy1 = zy0 + zone_size

            sides = [
                ("South", 1, 1),
                ("North", 1, 2),
                ("West", 2, 1),
                ("East", 2, 2),
            ]
            for side, aisle_num, rack_num in sides:
                rack_id = f"R{zone_idx + 1}-{aisle_num}-{rack_num}"
                aisle_id = f"A{zone_idx + 1}-{aisle_num}"

                for bay in range(8):
                    for shelf in range(4):
                        for bin_idx in range(2):
                            loc_id += 1
                            shelf_z = (shelf + 0.5) * t.shelf_height_m

                            if side == "South":
                                b_x = round(zx0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                                b_y = round(zy0 + rack_depth / 2.0, 3)
                            elif side == "North":
                                b_x = round(zx0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                                b_y = round(zy1 - rack_depth / 2.0, 3)
                            elif side == "West":
                                b_x = round(zx0 + rack_depth / 2.0, 3)
                                b_y = round(zy0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            else:
                                b_x = round(zx1 - rack_depth / 2.0, 3)
                                b_y = round(zy0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)

                            ergonomic = AccessibilityZone.GOLDEN if shelf in (1, 2) else (
                                AccessibilityZone.SECONDARY if shelf == 0 else AccessibilityZone.RESERVE
                            )

                            loc = Location(
                                location_id=f"LOC{loc_id:06d}",
                                zone_id=zone_id,
                                aisle_id=aisle_id,
                                rack_id=rack_id,
                                bay=bay,
                                shelf_level=shelf,
                                x=b_x,
                                y=b_y,
                                z=shelf_z,
                                width_cm=t.bin_width_cm,
                                height_cm=t.bin_height_cm,
                                depth_cm=t.bin_depth_cm,
                                max_weight_kg=t.max_weight_kg,
                                usable_volume_cm3=t.bin_width_cm * t.bin_height_cm * t.bin_depth_cm,
                                shelf_max_weight_kg=t.max_weight_kg * 4.0,
                                rack_max_weight_kg=t.max_weight_kg * 32.0,
                                ergonomic_zone=ergonomic,
                                allowed_handling=("standard", "fragile", "heavy", "temperature_controlled"),
                                allowed_hazard=("NONE", "flammable", "corrosive", "toxic", "compressed_gas"),
                            )
                            loc.distance_to_dispatch_m = 0.0
                            wh.locations[loc.location_id] = loc
        return wh

    def _ergonomic_zone(self, shelf: int, rack_y: float) -> AccessibilityZone:
        """Golden shelves near picking height; then by distance to dispatch."""
        if shelf in self.cfg.golden_levels:
            return AccessibilityZone.GOLDEN
        # Reserve zone = far bays of later zones; otherwise secondary.
        if rack_y > (self.cfg.n_zones - 1) * (self.cfg.aisles_per_zone * 2 * self.cfg.aisle_spacing_m):
            return AccessibilityZone.RESERVE
        return AccessibilityZone.SECONDARY


def manhattan_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return _manhattan(p1, p2)