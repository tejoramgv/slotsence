"""Phase-1 smoke test: config -> build warehouse -> seeded baseline -> valid CSV outputs."""

import csv
from pathlib import Path

import pytest

from src.models.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "small_warehouse.json"


def _load_config() -> Config:
    return Config.from_json(CONFIG)


def test_config_loads() -> None:
    cfg = _load_config()
    assert cfg.topology.n_zones > 0
    assert cfg.dataset.n_skus > 0
    assert cfg.optimization.alpha >= 0


def test_config_rejects_bad_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        Config.from_json(bad)


def _build(tmp_path: Path):
    from src.models.synthetic import build_phase1_fixture

    cfg = _load_config()
    out = tmp_path / "out"
    counts = build_phase1_fixture(cfg, out, seed=cfg.dataset.seed)

    # Determinism
    out2 = tmp_path / "out2"
    counts2 = build_phase1_fixture(cfg, out2, seed=cfg.dataset.seed)
    assert (out / "locations.csv").read_bytes() == (out2 / "locations.csv").read_bytes()
    assert (out / "storage_assignments_baseline.csv").read_bytes() == (
        out2 / "storage_assignments_baseline.csv"
    ).read_bytes()

    return out, counts


def test_locations_csv(tmp_path: Path) -> None:
    out, counts = _build(tmp_path)
    rows = list(csv.DictReader((out / "locations.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == counts["locations"]
    # required columns present
    for col in (
        "location_id",
        "zone_id",
        "x_m",
        "y_m",
        "z_m",
        "width_cm",
        "height_cm",
        "depth_cm",
        "max_weight_kg",
        "usable_volume_cm3",
        "ergonomic_zone",
        "distance_to_dispatch_m",
    ):
        assert col in rows[0], f"missing column: {col}"
    # distances derived from geometry: min is >= 0; dispatch (0,0) -> x=2.0 area
    assert all(float(r["distance_to_dispatch_m"]) >= 0 for r in rows)


def test_products_csv(tmp_path: Path) -> None:
    out, counts = _build(tmp_path)
    rows = list(csv.DictReader((out / "products.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == counts["skus"]
    assert all(float(r["unit_volume_cm3"]) > 0 for r in rows)
    assert all(float(r["unit_weight_kg"]) > 0 for r in rows)
    # hazard_class is explicit
    assert all(r["hazard_class"] in ("NONE", "UNKNOWN", "flammable", "corrosive", "toxic", "compressed_gas") for r in rows)


def test_baseline_csv(tmp_path: Path) -> None:
    out, counts = _build(tmp_path)
    rows = list(csv.DictReader((out / "storage_assignments_baseline.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == counts["baseline_assigned"]
    assert all(r["assignment_type"] == "baseline" for r in rows)


def test_baseline_feasible_and_weight_ok(tmp_path: Path) -> None:
    from src.models.entities import AssignmentType
    from src.models.synthetic import generate_skus

    cfg = _load_config()
    skus = generate_skus(cfg, cfg.dataset.seed)
    sku_by_id = {s.sku_id: s for s in skus}
    out, _ = _build(tmp_path)
    rows = list(csv.DictReader((out / "storage_assignments_baseline.csv").open(newline="", encoding="utf-8")))
    loc_by_id = {
        r["location_id"]: r for r in csv.DictReader((out / "locations.csv").open(newline="", encoding="utf-8"))
    }
    # Feasibility: every assignment's SKU fits its location's per-dimension bounds.
    for r in rows:
        sku = sku_by_id[r["sku_id"]]
        loc = loc_by_id[r["location_id"]]
        assert sku.width_cm <= float(loc["width_cm"])
        assert sku.height_cm <= float(loc["height_cm"])
        assert sku.depth_cm <= float(loc["depth_cm"])
        assert sku.unit_weight_kg <= float(loc["max_weight_kg"])
        assert sku.hazard_class.value in loc["allowed_hazard"].split("|")