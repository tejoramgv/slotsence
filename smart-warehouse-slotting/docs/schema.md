# Warehouse Data Schema (Phase 1)

> This file documents the Phase-1 fixture. The canonical benchmark dataset lives in
> `../dataset/processed/`; field-level definitions are in `../dataset/metadata/data_dictionary.csv`
> and design/limitation notes in `../dataset/metadata/schema_documentation.md`.

Canonical units: cm (dimensions/volume-composition), kg (weight), cm3 (volume), m (coordinates &
distance), UTC (timestamps). Volumes are given as cm3 explicitly.

## locations.csv

| field | type | meaning |
|---|---|---|
| location_id | str | stable unique id (independent of stored SKU) |
| warehouse_id | str | owning warehouse |
| zone_id / aisle_id / rack_id | str | hierarchy ids |
| bay | int | bay index within rack |
| shelf_level | int | 0-based shelf level |
| x_m, y_m, z_m | float | coordinates (m); z = shelf height |
| width_cm, height_cm, depth_cm | float | internal bin dims |
| max_weight_kg | float | permissible load for the location |
| usable_volume_cm3 | float | bin usable volume |
| ergonomic_zone | str | golden / secondary / reserve |
| allowed_handling | str | `\|`-joined handling types |
| allowed_hazard | str | `\|`-joined hazard classes |
| distance_to_dispatch_m | float | Manhattan distance to dispatch, derived from coordinates |

## products.csv (SKU master)

| field | type | meaning |
|---|---|---|
| sku_id | str | unique SKU id |
| product_name | str | display name |
| category_id | str | FK to categories |
| width_cm / height_cm / depth_cm | float | per-unit dims |
| unit_weight_kg | float | per-unit weight |
| unit_volume_cm3 | float | width×height×depth |
| handling_type | str | standard / fragile / heavy / temperature_controlled |
| hazard_class | str | explicit NONE / UNKNOWN / flammable / corrosive / toxic / compressed_gas |
| data_source | str | provenance (synthetic, sourced, measured, inferred) |

## categories.csv

| field | type | meaning |
|---|---|---|
| category_id | str | unique category id |
| category_name | str | display name |
| storage_requirements | str | handling/storage notes |

## storage_assignments_*.csv

| field | type | meaning |
|---|---|---|
| sku_id | str | FK to products |
| location_id | str | FK to locations |
| assigned_quantity | int | inventory quantity assigned |
| assignment_type | str | `baseline` or `optimized` (kept separate, baseline never overwritten) |

## Derived volume / weight rules

- `unit_volume_cm3 = width_cm * height_cm * depth_cm`
- `total stored volume = quantity * unit_volume_cm3`
- `total stored weight = quantity * unit_weight_kg`

Assumption: each unit occupies its full bounding-box volume; no packing overlap. Usable space may
be lower in practice.