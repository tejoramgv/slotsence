"""Orchestration for the benchmark dataset build.

Generates the tables, derived demand artifacts, metadata, and validation reports in one
deterministic run. The output directory defaults to ``ai_syndicate/dataset`` next to the
project repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .derive import build_affinity, build_demand_summary
from .generate import (
    build_inventory_and_baseline,
    build_warehouse,
    generate_categories,
    generate_compatibility_rules,
    generate_orders,
    generate_products,
)
from .schema import TABLES, TABLE_NOTES, Field
from .validate import run_validation, write_validation_reports

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.json"
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "dataset"
DATASET_VERSION = "1.0.0"


def load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return json.loads(json.dumps(config))
    path = Path(config)
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset(
    config: str | Path | dict[str, Any] = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    cfg = load_config(config)
    if seed is not None:
        cfg["seed"] = int(seed)
    seed = int(cfg["seed"])
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()

    (output / "raw").mkdir(parents=True, exist_ok=True)
    (output / "processed").mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)

    categories = generate_categories()
    products = generate_products(int(cfg["n_skus"]), seed + 1)
    locations, graph_rows, _ = build_warehouse(cfg)
    rules = generate_compatibility_rules()
    orders, order_items = generate_orders(cfg, products, seed + 2)
    assignments, inventory = build_inventory_and_baseline(
        cfg, products, locations, orders, order_items, seed + 3
    )
    demand = build_demand_summary(products, orders, order_items)
    affinity = build_affinity(products, orders, order_items)
    warehouses = [
        {
            "warehouse_id": cfg["warehouse"]["warehouse_id"],
            "warehouse_name": cfg["warehouse"]["warehouse_name"],
            "layout_version": cfg["warehouse"]["layout_version"],
            "data_source": "synthetic",
            "data_quality_status": "valid",
        }
    ]

    tables: dict[str, list[dict[str, Any]]] = {
        "warehouses": warehouses,
        "categories": categories,
        "products": products,
        "locations": locations,
        "inventory": inventory,
        "compatibility_rules": rules,
        "orders": orders,
        "order_items": order_items,
        "storage_assignments_baseline": assignments,
        "location_graph": graph_rows,
        "demand_summary": demand,
        "sku_affinity": affinity,
    }

    for table, rows in tables.items():
        _write_table(output / "processed" / f"{table}.csv", table, rows)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_data_dictionary(output, tables)
    _write_source_manifest(output, cfg, tables, generated_at)
    _write_schema_documentation(output, cfg, tables)
    _write_raw_readme(output)
    _write_dataset_readme(output, cfg, tables)
    _write_manifest(output, cfg, tables, generated_at)

    validation = run_validation(output)
    write_validation_reports(output, validation)

    return {
        "output": str(output),
        "seed": seed,
        "counts": {table: len(rows) for table, rows in tables.items()},
        "validation": {
            "ok": validation.ok,
            "hard_errors": validation.hard_errors,
            "warnings": validation.warnings,
        },
    }


def _write_table(path: Path, table: str, rows: list[dict[str, Any]]) -> None:
    fields = [field.name for field in TABLES[table]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _format_value(row.get(name, "")) for name in fields})


def _format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return value
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    output: Path,
    cfg: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    generated_at: str,
) -> None:
    files: dict[str, Any] = {}
    for table, rows in tables.items():
        relative = f"processed/{table}.csv"
        path = output / relative
        files[relative] = {"rows": len(rows), "sha256": _sha256(path)}
    manifest = {
        "dataset_name": "smart_warehouse_benchmark",
        "dataset_version": DATASET_VERSION,
        "seed": int(cfg["seed"]),
        "generated_at": generated_at,
        "python_version": platform.python_version(),
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "config": cfg,
        "counts": {table: len(rows) for table, rows in tables.items()},
        "files": files,
    }
    (output / "metadata" / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _write_data_dictionary(output: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    path = output / "metadata" / "data_dictionary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "table", "table_purpose", "primary_key", "field", "data_type", "unit", "description",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for table, fields in TABLES.items():
            purpose, primary_key = TABLE_NOTES[table]
            for field in fields:
                writer.writerow(_dictionary_row(table, purpose, primary_key, field))


def _dictionary_row(table: str, purpose: str, primary_key: str, field: Field) -> dict[str, str]:
    return {
        "table": table,
        "table_purpose": purpose,
        "primary_key": primary_key,
        "field": field.name,
        "data_type": field.dtype,
        "unit": field.unit,
        "description": field.description,
    }


def _write_source_manifest(
    output: Path,
    cfg: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    generated_at: str,
) -> None:
    total_records = sum(len(rows) for rows in tables.values())
    rows = [
        {
            "source_id": "synthetic_benchmark_v1",
            "source_name": "Synthetic benchmark generator v1",
            "download_date": generated_at[:10],
            "license": "Project-internal synthetic data",
            "file_name": "processed/*.csv",
            "record_count": total_records,
            "primary_key": "see data_dictionary.csv",
            "notes": f"Generated with seed {cfg['seed']}; physical attributes are synthetic.",
            "status": "generated",
        },
        {
            "source_id": "warehouse_mendeley_v1",
            "source_name": "Footwear Warehouse Dataset (Mendeley Data)",
            "download_date": "",
            "license": "Check DOI terms (https://doi.org/10.17632/pf2w725pw3.1)",
            "file_name": "",
            "record_count": 0,
            "primary_key": "",
            "notes": "Not acquired: real footwear operations, but package dimensions/weights are incomplete.",
            "status": "not_acquired",
        },
        {
            "source_id": "amazon_ucsd_v1",
            "source_name": "Amazon product metadata (UCSD)",
            "download_date": "",
            "license": "Check source terms (https://cseweb.ucsd.edu/~jmcauley/datasets.html)",
            "file_name": "",
            "record_count": 0,
            "primary_key": "",
            "notes": "Not acquired: metadata lacks reliable physical dimensions and weights.",
            "status": "not_acquired",
        },
        {
            "source_id": "hamburg_order_picking_v1",
            "source_name": "Artificial Generated Order Picking Data (University of Hamburg)",
            "download_date": "",
            "license": "Check source terms (https://www.fdr.uni-hamburg.de/record/1645)",
            "file_name": "",
            "record_count": 0,
            "primary_key": "",
            "notes": "Not acquired: supplementary synthetic picking patterns; not needed for the controlled benchmark.",
            "status": "not_acquired",
        },
    ]
    path = output / "metadata" / "source_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_schema_documentation(
    output: Path,
    cfg: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Benchmark Dataset Schema",
        "",
        f"- Dataset version: {DATASET_VERSION}",
        f"- Seed: {cfg['seed']}",
        f"- SKUs: {len(tables['products'])}",
        f"- Categories: {len(tables['categories'])}",
        f"- Locations: {len(tables['locations'])}",
        f"- Orders: {len(tables['orders'])}",
        f"- Order lines: {len(tables['order_items'])}",
        "",
        "## Canonical units",
        "",
        "Dimensions cm, weight kg, volume cm3, coordinates/distance m, timestamps UTC.",
        "",
        "## Tables",
        "",
        "| table | purpose | primary key | rows |",
        "|---|---|---|---|",
    ]
    for table, rows in tables.items():
        purpose, primary_key = TABLE_NOTES[table]
        lines.append(f"| {table} | {purpose} | {primary_key} | {len(rows)} |")
    lines.extend(
        [
            "",
            "Full field definitions and units: `metadata/data_dictionary.csv`.",
            "",
            "## Relationships",
            "",
            "- `products.category_id` -> `categories.category_id`.",
            "- `locations.warehouse_id` -> `warehouses.warehouse_id`.",
            "- `order_items` is the order/SKU junction; `(order_id, sku_id)` is unique.",
            "- `inventory` and `storage_assignments_baseline` both reference `products` and `locations`.",
            "- `demand_summary` and `sku_affinity` are derived from the order history.",
            "- `location_graph` is a directed, symmetric aisle network; `DEPOT` is the sink/source.",
            "",
            "## Design decisions",
            "",
            "- Three demand scenarios (A concentrated, B balanced, C affinity) cover the same 180-day",
            "  window. They are alternative histories, not additive; do not sum them.",
            "- Inventory is sized on Scenario A with 30 days of cover and a safety factor, then capped",
            "  by physical bin capacity. It is a snapshot, not a forecast.",
            "- The baseline is a seeded random feasible packing: unoptimized by design, but every SKU",
            "  is assigned and every bin, shelf, and rack limit is respected.",
            "- One SKU per bin is allowed; a SKU may occupy several bins.",
            "- `sku_affinity` keeps pairs seen in at least two orders. Very rare pairs can show",
            "  extreme lift; rank by `co_count` or `support` when rare-pair noise matters.",
            "- Volumes are full bounding-box volumes; packing clearance and palletisation are not modelled.",
            "- Hazardous goods are confined to the hazard zone; heavy goods to reinforced lower",
            "  shelves (levels 0-2); fragile goods are excluded from the heavy/high-impact zone.",
            "- `temperature_controlled` is part of the handling vocabulary but no cold-chain zone exists",
            "  in this benchmark.",
            "",
            "## Location graph",
            "",
            "Node types: `location` (bin pick point), `aisle_column` (a pick column on the aisle",
            "centerline), `aisle_entrance`, `zone_hub`, and `depot`. Locations connect laterally to",
            "their column, columns chain along the aisle, aisles connect to the zone hub, and zone",
            "hubs connect to `DEPOT`. Distances are walking distances, not straight lines through racks.",
            "`locations.distance_to_dispatch_m` is the shortest-path distance in this graph.",
            "",
            "## Provenance",
            "",
            "- `data_source`: measured, sourced, inferred, synthetic, or derived.",
            "- `data_quality_status`: valid, warning, or invalid.",
            "- Product physical attributes carry `physical_data_source=synthetic_class_rule` and",
            "  `physical_confidence=low`.",
            "- No real source data was acquired for this version; `raw/` contains only a note.",
            "",
            "## Reproducibility",
            "",
            "The same config and seed reproduce identical processed CSVs. File SHA-256 hashes and the",
            "full config are recorded in `metadata/generation_manifest.json`. Rebuild with:",
            "",
            "```bash",
            "python app/build_dataset.py --config configs/dataset.json",
            "python app/build_dataset.py --validate-only",
            "```",
            "",
            "## Limitations",
            "",
            "- All records are synthetic; the dataset is a controlled engineering benchmark, not a",
            "  claim about a specific real warehouse.",
            "- Demand is generated by documented rules rather than measured transactions.",
            "- Category compatibility is simplified to category pairs; hazard segregation inside the",
            "  hazard zone is not modelled at the bin level.",
            "",
        ]
    )
    (output / "metadata" / "schema_documentation.md").write_text("\n".join(lines), encoding="utf-8")


def _write_raw_readme(output: Path) -> None:
    (output / "raw" / "README.md").write_text(
        "# Raw sources\n\n"
        "No raw source files were acquired for this dataset version. The benchmark is fully\n"
        "synthetic; `metadata/source_manifest.csv` lists the real sources that were considered and\n"
        "why they were not ingested. If sources are added later, keep their originals unchanged in\n"
        "this directory and record them in the source manifest.\n",
        encoding="utf-8",
    )


def _write_dataset_readme(
    output: Path,
    cfg: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Smart Warehouse Benchmark Dataset",
        "",
        "A coherent, synthetic warehouse dataset for the slotting and optimization engine.",
        "",
        f"- Seed: `{cfg['seed']}`",
        f"- SKUs: {len(tables['products'])} across {len(tables['categories'])} categories",
        f"- Locations: {len(tables['locations'])}",
        f"- Orders: {len(tables['orders'])}; order lines: {len(tables['order_items'])}",
        f"- Demand scenarios: {len({row['scenario_id'] for row in tables['orders']})}",
        "",
        "## Contents",
        "",
        "- `processed/` — the relational tables (CSV).",
        "- `metadata/` — source manifest, data dictionary, schema notes, generation manifest.",
        "- `reports/` — validation report and machine-readable error log.",
        "- `raw/` — empty in this version; see `raw/README.md`.",
        "",
        "## Regenerate",
        "",
        "From `smart-warehouse-slotting/`:",
        "",
        "```bash",
        "python app/build_dataset.py --config configs/dataset.json",
        "python app/build_dataset.py --validate-only",
        "```",
        "",
        "## Notes",
        "",
        "- Scenarios A, B, and C are alternative demand histories. Never add their order lines together.",
        "- Inventory is sized on Scenario A and capped by physical capacity.",
        "- The baseline is feasible but deliberately unoptimized.",
        "- All data is synthetic or derived; see `metadata/schema_documentation.md`.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
