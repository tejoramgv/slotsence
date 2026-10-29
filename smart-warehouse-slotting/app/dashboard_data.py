"""Data access + derived computations for the warehouse dashboard.

Everything the UI shows is computed from the generated artifacts in ``../dataset/processed``
and ``outputs/`` — no KPI, improvement, or safety result is hardcoded. Functions are plain
pandas/numpy so they can be unit-tested without Streamlit.

Canonical definitions
---------------------
Travel-distance reduction (%) = (baseline distance - optimized distance) / baseline distance x 100
Route-time reduction (%)      = (baseline time - optimized time) / baseline time x 100
Cube utilization (%)          = occupied inventory volume / usable storage volume x 100
Utilization change (pp)       = optimized utilization (%) - baseline utilization (%)
Assignment feasibility (%)    = valid assignments / required assignments x 100
Golden-zone demand share (%)  = demand-weighted units on golden shelf levels / all placed units x 100

``None`` is returned for a change whose denominator is zero; the UI renders "not defined".
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT.parent / "dataset" / "processed"
OUT = PROJECT_ROOT / "outputs"
ANALYSIS = OUT / "analysis"
EVAL = OUT / "evaluation"
OPT = OUT / "optimized"
CONFIG = PROJECT_ROOT / "configs" / "dataset.json"

CLASS_ORDER = ["A", "B", "C", "unassigned"]
CLASS_COLORS = {
    "A": "#c0392b",  # fast movers — highest pick frequency
    "B": "#d68910",
    "C": "#2471a3",
    "unassigned": "#b3b6b7",
}
CLASS_DESCRIPTIONS = {
    "A": "Class A — top 80% of cumulative demand",
    "B": "Class B — next 15% of cumulative demand",
    "C": "Class C — remaining 5% of cumulative demand",
    "unassigned": "Unassigned — no SKU stored (or no assignment data)",
}

KPI_LABELS = {
    "total_travel_distance_m": "Total pick-path travel distance",
    "avg_travel_distance_m": "Average path per person",
    "avg_distance_per_order_m": "Path per order (per person)",
    "total_route_time_sec": "Route completion time per person",
    "avg_route_time_sec": "Average route time per person",
    "avg_time_per_order_sec": "Pick-cycle time per order",
    "throughput_orders_per_hr": "Throughput",
    "cube_utilization_pct": "Volumetric cube utilization",
    "golden_zone_demand_share_pct": "Golden-zone demand share",
    "class_a_skus_outside_golden": "Class A SKUs with no golden-shelf stock",
    "affinity_separation_m": "Affinity separation (weighted aisle distance)",
    "constraint_violations": "Constraint violations (hard)",
    "valid_assignments": "Valid assignments",
    "required_assignments": "Required assignments",
    "assignment_feasibility_pct": "Assignment feasibility",
    "rule_check_pass_pct": "Rule-check pass rate",
}


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _first_csv(*paths: Path) -> pd.DataFrame | None:
    """Return the first existing, non-empty CSV from the given candidates."""
    for path in paths:
        frame = _read_csv(path)
        if frame is not None and not frame.empty:
            return frame
    return None


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def load_all() -> dict:
    """Load every artifact the dashboard can use. Missing files stay ``None``."""
    demand = _read_csv(DATA_DIR / "demand_summary.csv")
    demand_a = None
    if demand is not None and "scenario_id" in demand.columns:
        demand_a = demand[demand["scenario_id"] == "A"].reset_index(drop=True)
    return {
        "locations": _read_csv(DATA_DIR / "locations.csv"),
        "products": _read_csv(DATA_DIR / "products.csv"),
        "inventory": _read_csv(DATA_DIR / "inventory.csv"),
        "warehouses": _read_csv(DATA_DIR / "warehouses.csv"),
        "compatibility": _read_csv(DATA_DIR / "compatibility_rules.csv"),
        "demand": demand_a,
        "demand_all": demand,
        "affinity_raw": _read_csv(DATA_DIR / "sku_affinity.csv"),
        "baseline": _read_csv(DATA_DIR / "storage_assignments_baseline.csv"),
        "abc": _read_csv(ANALYSIS / "abc_classification.csv"),
        "affinity": _read_csv(ANALYSIS / "affinity_matrix.csv"),
        "clusters": _read_csv(ANALYSIS / "affinity_clusters.csv"),
        "analysis_summary": _read_json(ANALYSIS / "analysis_summary.json"),
        "optimized": _read_csv(OPT / "optimized_assignments.csv"),
        "opt_summary": _read_json(OPT / "optimization_summary.json"),
        "run_history": _read_json(OPT / "run_history.json"),
        "metrics": _read_csv(EVAL / "optimization_metrics.csv"),
        "routes": _read_csv(EVAL / "pick_routes.csv"),
        "route_stops_baseline": _read_csv(EVAL / "pick_route_stops_baseline.csv"),
        "route_stops_optimized": _read_csv(EVAL / "pick_route_stops_optimized.csv"),
        "checks_baseline": _first_csv(
            EVAL / "validation_checks_baseline.csv", OUT / "validation_checks_baseline.csv"
        ),
        "checks_optimized": _read_csv(EVAL / "validation_checks_optimized.csv"),
        "eval_summary": _read_json(EVAL / "evaluation_summary.json"),
        "validation_status": _read_json(OUT / "validation_status.json"),
        "dataset_validation": _read_json(OUT / "dataset_validation.json"),
        "config": _read_json(CONFIG),
    }


# ----------------------------------------------------------------------------- formulas
def pct_reduction(baseline: float | None, optimized: float | None) -> float | None:
    """(baseline - optimized) / baseline x 100. ``None`` when baseline is 0 or missing."""
    if baseline is None or optimized is None:
        return None
    try:
        b = float(baseline)
        o = float(optimized)
    except (TypeError, ValueError):
        return None
    if b == 0:
        return None
    return (b - o) / b * 100.0


def pct_increase(baseline: float | None, optimized: float | None) -> float | None:
    """(optimized - baseline) / baseline x 100 for higher-is-better metrics."""
    if baseline is None or optimized is None:
        return None
    try:
        b = float(baseline)
        o = float(optimized)
    except (TypeError, ValueError):
        return None
    if b == 0:
        return None
    return (o - b) / b * 100.0


def percentage_points(baseline: float | None, optimized: float | None) -> float | None:
    if baseline is None or optimized is None:
        return None
    try:
        return float(optimized) - float(baseline)
    except (TypeError, ValueError):
        return None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return float(numerator) / float(denominator) * 100.0
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # NaN -> None


def _metric_lookup(metrics: pd.DataFrame | None) -> dict:
    if metrics is None or metrics.empty or "metric" not in metrics.columns:
        return {}
    out = {}
    for _, row in metrics.iterrows():
        out[str(row["metric"])] = {
            "baseline": _to_float(row.get("baseline")),
            "optimized": _to_float(row.get("optimized")),
            "raw_optimized": row.get("optimized"),
            "raw_baseline": row.get("baseline"),
            "unit": row.get("unit"),
            "change_kind": row.get("change_kind"),
            "direction": row.get("direction"),
            "note": row.get("note"),
        }
    return out


def kpi_rows(metrics: pd.DataFrame | None) -> list[dict]:
    """Baseline/optimized/delta rows for the KPI cards, derived from the metrics output."""
    lookup = _metric_lookup(metrics)
    rows: list[dict] = []
    for key, label in KPI_LABELS.items():
        entry = lookup.get(key)
        if entry is None:
            rows.append({
                "key": key,
                "label": label,
                "available": False,
                "baseline": None,
                "optimized": None,
                "delta": None,
                "change_pct": None,
                "change_pp": None,
                "unit": "",
                "direction": "",
                "definition": "",
            })
            continue
        b, o = entry["baseline"], entry["optimized"]
        kind = entry["change_kind"] or ""
        rows.append({
            "key": key,
            "label": label,
            "available": True,
            "baseline": b,
            "optimized": o,
            "delta": (o - b) if (b is not None and o is not None) else None,
            "change_pct": (
                pct_reduction(b, o) if kind == "reduction"
                else pct_increase(b, o) if kind == "improvement"
                else None
            ),
            "change_pp": percentage_points(b, o) if kind in ("points", "pct") else None,
            "unit": entry["unit"] or "",
            "direction": entry["direction"] or "",
            "change_kind": kind,
            "definition": entry["note"] or "",
        })
    return rows


def kpi_row(metrics: pd.DataFrame | None, key: str) -> dict | None:
    for row in kpi_rows(metrics):
        if row["key"] == key:
            return row if row["available"] else None
    return None


def chart_frame(metrics: pd.DataFrame | None, keys: list[str]) -> pd.DataFrame:
    """Long frame (metric label, layout, value) for grouped comparison charts."""
    lookup = _metric_lookup(metrics)
    records = []
    for key in keys:
        entry = lookup.get(key)
        if entry is None:
            continue
        label = KPI_LABELS.get(key, key)
        for layout in ("baseline", "optimized"):
            value = entry[layout]
            if value is None:
                continue
            records.append({"metric": label, "layout": layout.title(), "value": value, "unit": entry["unit"]})
    return pd.DataFrame(records)


# ----------------------------------------------------------------------------- ABC
def abc_rule_text(config: dict | None) -> str:
    a_share = 0.80
    b_share = 0.95
    if isinstance(config, dict):
        demand_cfg = config.get("demand") or {}
        abc_cfg = demand_cfg.get("abc") if isinstance(demand_cfg, dict) else None
        if isinstance(abc_cfg, dict):
            a_share = float(abc_cfg.get("a_cumulative_share", a_share))
            b_share = float(abc_cfg.get("b_cumulative_share", b_share))
        # benchmark.json stores the thresholds outside dataset.json
        bench = _read_json(PROJECT_ROOT / "configs" / "benchmark.json")
        if isinstance(bench, dict) and isinstance(bench.get("demand"), dict):
            a_share = float(bench["demand"].get("abc_a_cumulative_share", a_share))
            b_share = float(bench["demand"].get("abc_b_cumulative_share", b_share))
    return (
        f"ABC rule: SKUs are ranked by descending demand frequency (scenario A). "
        f"Class A = SKUs whose cumulative demand share is <= {a_share:.0%}; "
        f"Class B = cumulative share <= {b_share:.0%}; Class C = the remainder. "
        f"Thresholds come from the project configuration, not from the UI."
    )


def abc_summary(abc: pd.DataFrame | None) -> pd.DataFrame | None:
    if abc is None or abc.empty:
        return None
    df = abc.copy()
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
    total_demand = df["demand"].sum()
    grouped = df.groupby("abc_class", dropna=False).agg(
        skus=("sku_id", "nunique"), demand=("demand", "sum")
    ).reindex([c for c in CLASS_ORDER if c in set(df["abc_class"])]).reset_index()
    grouped["share_of_skus_pct"] = (grouped["skus"] / len(df) * 100).round(2)
    grouped["share_of_demand_pct"] = (
        (grouped["demand"] / total_demand * 100).round(2) if total_demand else 0.0
    )
    return grouped


def abc_class_map(abc: pd.DataFrame | None) -> dict[str, str]:
    if abc is None or abc.empty:
        return {}
    return dict(zip(abc["sku_id"], abc["abc_class"]))


# ----------------------------------------------------------------------------- layout state
def _failed_mask(df: pd.DataFrame) -> pd.Series:
    passed = df["passed"].astype(str).str.strip().str.lower()
    return ~passed.isin(["true", "1", "pass", "passed"])


def _checks_map(checks: pd.DataFrame | None) -> tuple[dict, dict]:
    """Return (failures_by_location, failures_by_sku) from validation checks."""
    by_loc: dict[str, set] = defaultdict(set)
    by_sku: dict[str, set] = defaultdict(set)
    if checks is None or checks.empty:
        return by_loc, by_sku
    failed = checks[_failed_mask(checks)]
    for row in failed.itertuples(index=False):
        rule_id = getattr(row, "rule_id", "")
        loc = str(getattr(row, "location_id", "") or "")
        sku = str(getattr(row, "sku_id", "") or "")
        if loc:
            by_loc[loc].add(rule_id)
        if sku:
            by_sku[sku].add(rule_id)
    return by_loc, by_sku


def location_state(data: dict, layout: str = "optimized") -> pd.DataFrame | None:
    """One row per storage location with assignment, utilization and constraint status.

    ``layout`` is ``baseline`` or ``optimized``. Returns ``None`` when topology is missing;
    when that layout's assignment file is missing, locations are returned as unassigned.
    """
    locations = data.get("locations")
    if locations is None or locations.empty:
        return None
    frame = locations.copy()
    assignments = data.get(layout)
    products = data.get("products")
    abc = data.get("abc")

    if assignments is not None and not assignments.empty:
        grouped = (
            assignments.groupby(["location_id", "sku_id"], as_index=False)["assigned_quantity"].sum()
        )
        frame = frame.merge(grouped, on="location_id", how="left")
    else:
        frame["sku_id"] = pd.NA
        frame["assigned_quantity"] = pd.NA

    if products is not None and not products.empty:
        product_cols = [
            "sku_id", "product_name", "category_id", "handling_type", "hazard_class",
            "width_cm", "height_cm", "depth_cm", "unit_volume_cm3", "unit_weight_kg",
        ]
        existing = [c for c in product_cols if c in products.columns]
        sku_dims = products[existing].rename(columns={
            # SKU geometry is kept separate from the bin geometry of the location row
            "width_cm": "sku_width_cm",
            "height_cm": "sku_height_cm",
            "depth_cm": "sku_depth_cm",
        })
        frame = frame.merge(sku_dims, on="sku_id", how="left")

    class_map = abc_class_map(abc)
    frame["abc_class"] = frame["sku_id"].map(class_map).fillna("unassigned")
    frame["assigned"] = frame["sku_id"].notna()

    qty = pd.to_numeric(frame.get("assigned_quantity"), errors="coerce").fillna(0.0)
    unit_volume = pd.to_numeric(frame.get("unit_volume_cm3"), errors="coerce").fillna(0.0)
    unit_weight = pd.to_numeric(frame.get("unit_weight_kg"), errors="coerce").fillna(0.0)
    usable = pd.to_numeric(frame["usable_volume_cm3"], errors="coerce").fillna(0.0)
    bin_limit = pd.to_numeric(frame["max_weight_kg"], errors="coerce").fillna(0.0)

    frame["occupied_volume_cm3"] = unit_volume * qty
    frame["load_kg"] = unit_weight * qty
    frame["bin_volume_util_pct"] = (frame["occupied_volume_cm3"] / usable.where(usable > 0) * 100).round(2)
    frame["bin_weight_util_pct"] = (frame["load_kg"] / bin_limit.where(bin_limit > 0) * 100).round(2)

    shelf_key = ["rack_id", "shelf_level"]
    frame["shelf_load_kg"] = frame.groupby(shelf_key)["load_kg"].transform("sum")
    shelf_cap = frame.groupby(shelf_key)["shelf_max_weight_kg"].transform("min")
    frame["shelf_max_weight_kg"] = shelf_cap
    frame["shelf_load_util_pct"] = (
        frame["shelf_load_kg"] / pd.to_numeric(shelf_cap, errors="coerce").where(lambda s: s > 0) * 100
    ).round(2)
    frame["rack_load_kg"] = frame.groupby("rack_id")["load_kg"].transform("sum")
    rack_cap = frame.groupby("rack_id")["rack_max_weight_kg"].transform("min")
    frame["rack_max_weight_kg"] = rack_cap
    frame["rack_load_util_pct"] = (
        frame["rack_load_kg"] / pd.to_numeric(rack_cap, errors="coerce").where(lambda s: s > 0) * 100
    ).round(2)

    # Hazard / handling / heavy-height eligibility for the stored SKU.
    def _allowed(value, allowed: str) -> bool:
        if pd.isna(value):
            return True
        return str(value) in str(allowed).split("|")

    frame["hazard_ok"] = [
        _allowed(h, a) for h, a in zip(frame["hazard_class"], frame["allowed_hazard"])
    ]
    frame["handling_ok"] = [
        _allowed(h, a) for h, a in zip(frame["handling_type"], frame["allowed_handling"])
    ]
    frame["heavy_height_ok"] = ~(
        (frame["handling_type"].astype(str) == "heavy") & (frame["shelf_level"].astype(int) > 2)
    )
    frame["golden_zone"] = frame["ergonomic_zone"].astype(str).str.lower() == "golden"

    checks = data.get(f"checks_{layout}")
    by_loc, by_sku = _checks_map(checks)
    if checks is None:
        status = "UNKNOWN"
        detail = ""
        rule_text = "not validated"
    else:
        statuses = []
        details = []
        for row in frame.itertuples(index=False):
            rules = set(by_loc.get(str(row.location_id), set()))
            if not pd.isna(getattr(row, "sku_id", None)):
                rules |= by_sku.get(str(row.sku_id), set())
            if rules:
                statuses.append("FAIL")
                details.append(",".join(sorted(rules)))
            else:
                statuses.append("OK")
                details.append("")
        frame["constraint_status"] = statuses
        frame["failed_rules"] = details
        status = None
        detail = ""
        rule_text = None
    if status is not None:
        frame["constraint_status"] = status
        frame["failed_rules"] = detail

    # Baseline-vs-optimized comparison flag (which bins changed SKU).
    baseline = data.get("baseline")
    optimized = data.get("optimized")
    if baseline is not None and optimized is not None:
        base_map = dict(zip(baseline["location_id"], baseline["sku_id"]))
        opt_map = dict(zip(optimized["location_id"], optimized["sku_id"]))
        frame["baseline_sku"] = frame["location_id"].map(base_map)
        frame["optimized_sku"] = frame["location_id"].map(opt_map)
        frame["changed"] = [
            (str(b) if pd.notna(b) else "") != (str(o) if pd.notna(o) else "")
            for b, o in zip(frame["baseline_sku"], frame["optimized_sku"])
        ]
    else:
        frame["baseline_sku"] = pd.NA
        frame["optimized_sku"] = pd.NA
        frame["changed"] = False

    frame["sku_id"] = frame["sku_id"].where(frame["sku_id"].notna(), pd.NA)
    return frame


def filter_locations(
    frame: pd.DataFrame,
    abc_classes: list[str] | None = None,
    zone: str | None = None,
    aisle: str | None = None,
    rack: str | None = None,
    shelf: int | None = None,
    hazard: str | None = None,
    min_utilization: float | None = None,
    constraint_status: str | None = None,
    assigned_only: bool = False,
) -> pd.DataFrame:
    """Apply the layout filters. A ``None``/empty argument means "no restriction"."""
    out = frame
    if abc_classes:
        out = out[out["abc_class"].isin(abc_classes)]
    if zone:
        out = out[out["zone_id"].astype(str) == str(zone)]
    if aisle:
        out = out[out["aisle_id"].astype(str) == str(aisle)]
    if rack:
        out = out[out["rack_id"].astype(str) == str(rack)]
    if shelf is not None:
        out = out[out["shelf_level"].astype(int) == int(shelf)]
    if hazard:
        if hazard == "NONE":
            out = out[out["hazard_class"].astype(str) == "NONE"]
        else:
            out = out[out["hazard_class"].astype(str) == hazard]
    if min_utilization is not None:
        util = pd.to_numeric(out["bin_volume_util_pct"], errors="coerce").fillna(0.0)
        out = out[util >= float(min_utilization)]
    if constraint_status in ("OK", "FAIL", "UNKNOWN"):
        out = out[out["constraint_status"] == constraint_status]
    if assigned_only:
        out = out[out["assigned"]]
    return out


def rack_geometry(locations: pd.DataFrame) -> pd.DataFrame:
    """Physical rack footprints derived from the bin coordinates (x/y extents)."""
    half_w = pd.to_numeric(locations["width_cm"], errors="coerce").fillna(0.0) / 200.0
    half_d = pd.to_numeric(locations["depth_cm"], errors="coerce").fillna(0.0) / 200.0
    temp = locations.assign(_hw=half_w, _hd=half_d)
    # Per-row extents then group (footprint from the bin coordinates themselves).
    temp["x_low"] = temp["x_m"] - temp["_hw"]
    temp["x_high"] = temp["x_m"] + temp["_hw"]
    temp["y_low"] = temp["y_m"] - temp["_hd"]
    temp["y_high"] = temp["y_m"] + temp["_hd"]
    geom = temp.groupby(["rack_id", "zone_id", "aisle_id"], as_index=False).agg(
        x_min=("x_low", "min"), x_max=("x_high", "max"),
        y_min=("y_low", "min"), y_max=("y_high", "max"),
        bins=("location_id", "nunique"),
        shelves=("shelf_level", "nunique"),
    )
    return geom.sort_values(["zone_id", "aisle_id", "rack_id"]).reset_index(drop=True)


def aisle_geometry(locations: pd.DataFrame) -> pd.DataFrame:
    temp = locations.assign(
        x_low=locations["x_m"] - pd.to_numeric(locations["width_cm"], errors="coerce").fillna(0) / 200.0,
        x_high=locations["x_m"] + pd.to_numeric(locations["width_cm"], errors="coerce").fillna(0) / 200.0,
        y_low=locations["y_m"] - pd.to_numeric(locations["depth_cm"], errors="coerce").fillna(0) / 200.0,
        y_high=locations["y_m"] + pd.to_numeric(locations["depth_cm"], errors="coerce").fillna(0) / 200.0,
    )
    return (
        temp.groupby(["zone_id", "aisle_id"], as_index=False)
        .agg(x_min=("x_low", "min"), x_max=("x_high", "max"),
             y_min=("y_low", "min"), y_max=("y_high", "max"),
             zones=("zone_id", "nunique"))
        .sort_values(["zone_id", "aisle_id"])
        .reset_index(drop=True)
    )


def dispatch_point(data: dict) -> tuple[float, float] | None:
    """Configured dispatch/I-O point (metres) from the warehouse config, if present."""
    config = data.get("config")
    if not isinstance(config, dict):
        return None
    point = (config.get("warehouse") or {}).get("dispatch_point_m")
    if not point or len(point) < 2:
        return None
    return float(point[0]), float(point[1])


def topology_summary(locations: pd.DataFrame | None) -> dict:
    if locations is None or locations.empty:
        return {}
    return {
        "zones": int(locations["zone_id"].nunique()),
        "aisles": int(locations["aisle_id"].nunique()),
        "racks": int(locations["rack_id"].nunique()),
        "bays_per_rack": int(locations.groupby("rack_id")["bay"].nunique().max()),
        "shelf_levels": int(locations["shelf_level"].nunique()),
        "bins": int(locations["location_id"].nunique()),
        "usable_volume_m3": round(float(locations["usable_volume_cm3"].sum()) / 1_000_000, 2),
        "golden_levels": sorted(
            int(v) for v in locations.loc[
                locations["ergonomic_zone"].astype(str).str.lower() == "golden", "shelf_level"
            ].unique()
        ),
    }


# ----------------------------------------------------------------------------- SKUs
def sku_table(data: dict, layout: str = "optimized") -> pd.DataFrame | None:
    """SKU-level demand, ABC, and assigned-location context."""
    abc = data.get("abc")
    if abc is None or abc.empty:
        return None
    df = abc.copy()
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
    products = data.get("products")
    if products is not None:
        cols = [c for c in [
            "sku_id", "product_name", "category_id", "handling_type", "hazard_class",
            "width_cm", "height_cm", "depth_cm", "unit_volume_cm3", "unit_weight_kg",
        ] if c in products.columns]
        df = df.merge(products[cols], on="sku_id", how="left")
    inventory = data.get("inventory")
    if inventory is not None:
        on_hand = inventory.groupby("sku_id")["quantity_on_hand"].sum()
        df["on_hand"] = df["sku_id"].map(on_hand).fillna(0).astype(int)
    assignments = data.get(layout)
    locations = data.get("locations")
    if assignments is not None and not assignments.empty:
        primary = (
            assignments.sort_values("location_id")
            .groupby("sku_id", as_index=False)
            .agg(assigned_qty=("assigned_quantity", "sum"), location_id=("location_id", "first"))
        )
        df = df.merge(primary, on="sku_id", how="left")
        if locations is not None:
            loc_idx = locations.set_index("location_id")
            df["zone_id"] = df["location_id"].map(loc_idx["zone_id"])
            df["shelf_level"] = df["location_id"].map(loc_idx["shelf_level"])
            df["ergonomic_zone"] = df["location_id"].map(loc_idx["ergonomic_zone"])
    else:
        df["assigned_qty"] = pd.NA
        df["location_id"] = pd.NA
        df["zone_id"] = pd.NA
        df["shelf_level"] = pd.NA
        df["ergonomic_zone"] = pd.NA
    df["golden_zone"] = df["ergonomic_zone"].astype(str).str.lower() == "golden"
    df["assigned_qty"] = pd.to_numeric(df["assigned_qty"], errors="coerce")
    return df


def unassigned_skus(data: dict, layout: str = "optimized") -> pd.DataFrame | None:
    """SKUs with on-hand inventory that are not fully placed in the given layout."""
    inventory = data.get("inventory")
    assignments = data.get(layout)
    if inventory is None or assignments is None:
        return None
    required = inventory.groupby("sku_id")["quantity_on_hand"].sum()
    placed = assignments.groupby("sku_id")["assigned_quantity"].sum()
    frame = pd.DataFrame({"required": required})
    frame["placed"] = frame.index.map(placed).fillna(0).astype(int)
    frame = frame.reset_index()
    short = frame[frame["placed"] < frame["required"]].copy()
    short["shortfall"] = short["required"] - short["placed"]
    products = data.get("products")
    if products is not None:
        short = short.merge(products[["sku_id", "product_name", "handling_type", "hazard_class"]], on="sku_id", how="left")
    return short.sort_values("shortfall", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- validation
def safety_status(data: dict, layout: str = "optimized") -> dict:
    """Safety verdict backed by the validation that actually ran for this layout."""
    checks = data.get(f"checks_{layout}")
    metrics = _metric_lookup(data.get("metrics"))
    if checks is None or checks.empty:
        return {
            "status": "UNAVAILABLE",
            "checks_run": 0,
            "failed_checks": 0,
            "hard_violations": None,
            "rules_failed": [],
            "reason": "No validation run is available for this layout.",
        }
    failed = checks[_failed_mask(checks)]
    hard = failed[failed["severity"].astype(str).str.lower() == "high"] if "severity" in failed else failed
    metric_violations = metrics.get("constraint_violations", {}).get("optimized" if layout == "optimized" else "baseline")
    rules = sorted(set(failed["rule_id"].astype(str))) if not failed.empty else []
    if failed.empty and (metric_violations in (None, 0.0)):
        status = "PASS"
        reason = "All configured hard constraints passed."
    else:
        status = "FAIL"
        reason = (
            f"{len(failed)} rule check(s) failed across {len(rules)} rule(s); "
            f"{len(hard)} are high severity."
        )
    return {
        "status": status,
        "checks_run": int(len(checks)),
        "failed_checks": int(len(failed)),
        "hard_violations": int(len(hard)),
        "rules_failed": rules,
        "reason": reason,
    }


def validation_by_rule(checks: pd.DataFrame | None) -> pd.DataFrame | None:
    if checks is None or checks.empty:
        return None
    frame = checks.copy()
    frame["passed_bool"] = ~_failed_mask(frame)
    grouped = frame.groupby(["rule_group", "rule_id", "rule_name"], dropna=False).agg(
        checks=("passed_bool", "size"), passed=("passed_bool", "sum")
    ).reset_index()
    grouped["failed"] = grouped["checks"] - grouped["passed"]
    grouped["pass_rate_pct"] = (grouped["passed"] / grouped["checks"] * 100).round(2)
    return grouped.sort_values(["failed", "rule_id"], ascending=[False, True]).reset_index(drop=True)


# ----------------------------------------------------------------------------- routes
def route_batches(data: dict) -> pd.DataFrame | None:
    routes = data.get("routes")
    if routes is None or routes.empty:
        return None
    frame = routes.copy()
    frame["batch_id"] = frame["batch_id"].astype(int)
    return frame.sort_values(["layout", "batch_id"]).reset_index(drop=True)


def route_stops(data: dict, layout: str, batch_id: int) -> pd.DataFrame | None:
    stops = data.get(f"route_stops_{layout}")
    if stops is None or stops.empty:
        return None
    frame = stops[stops["batch_id"].astype(int) == int(batch_id)].sort_values("sequence").copy()
    return frame if not frame.empty else None


def route_summary_row(data: dict, layout: str, batch_id: int) -> dict | None:
    batches = route_batches(data)
    if batches is None:
        return None
    row = batches[(batches["layout"] == layout) & (batches["batch_id"] == int(batch_id))]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def stops_with_coordinates(data: dict, layout: str, batch_id: int) -> pd.DataFrame | None:
    """Route stops joined to topology coordinates; the depot uses the configured dispatch point."""
    stops = route_stops(data, layout, batch_id)
    locations = data.get("locations")
    if stops is None or locations is None:
        return None
    idx = locations.set_index("location_id")
    dispatch = dispatch_point(data)
    rows = []
    for row in stops.itertuples(index=False):
        loc_id = str(row.location_id)
        if loc_id in idx.index:
            loc = idx.loc[loc_id]
            rows.append({
                **row._asdict(),
                "x_m": float(loc["x_m"]),
                "y_m": float(loc["y_m"]),
                "shelf_level": int(loc["shelf_level"]),
                "zone_id": str(loc["zone_id"]),
            })
        elif dispatch is not None:
            rows.append({
                **row._asdict(),
                "x_m": dispatch[0],
                "y_m": dispatch[1],
                "shelf_level": None,
                "zone_id": "dispatch",
            })
    return pd.DataFrame(rows) if rows else None


# ----------------------------------------------------------------------------- affinity
def affinity_table(data: dict, min_co_count: int = 2, top: int | None = None) -> pd.DataFrame | None:
    affinity = data.get("affinity")
    if affinity is None or affinity.empty:
        return None
    df = affinity.copy()
    if "co_count" in df:
        df = df[pd.to_numeric(df["co_count"], errors="coerce").fillna(0) >= min_co_count]
    df = df.sort_values("lift", ascending=False)
    if top:
        df = df.head(int(top))
    return df.reset_index(drop=True)


def affinity_heatmap(data: dict, top_skus: int = 20) -> tuple[pd.DataFrame | None, list[str]]:
    affinity = data.get("affinity")
    if affinity is None or affinity.empty:
        return None, []
    df = affinity.copy()
    df["co_count"] = pd.to_numeric(df["co_count"], errors="coerce").fillna(0)
    totals = pd.concat([
        df.groupby("sku_a")["co_count"].sum(),
        df.groupby("sku_b")["co_count"].sum(),
    ], axis=1).fillna(0).sum(axis=1).sort_values(ascending=False)
    skus = list(totals.head(top_skus).index)
    if not skus:
        return None, []
    subset = df[df["sku_a"].isin(skus) & df["sku_b"].isin(skus)]
    matrix = pd.DataFrame(0.0, index=skus, columns=skus)
    for row in subset.itertuples(index=False):
        matrix.loc[row.sku_a, row.sku_b] = row.co_count
        matrix.loc[row.sku_b, row.sku_a] = row.co_count
    return matrix, skus


def pair_location_detail(data: dict, sku_a: str, sku_b: str, layout: str = "optimized") -> dict:
    """Assigned bins for two SKUs plus the aisle-network distance between their bins."""
    assignments = data.get(layout)
    locations = data.get("locations")
    if assignments is None or locations is None:
        return {}
    idx = locations.set_index("location_id")
    out = {}
    for label, sku in (("a", sku_a), ("b", sku_b)):
        rows = assignments[assignments["sku_id"] == sku]
        bins = sorted(rows["location_id"].astype(str).tolist())
        out[label] = {
            "sku_id": sku,
            "bins": bins,
            "primary": bins[0] if bins else None,
            "shelf_level": int(idx.loc[bins[0], "shelf_level"]) if bins else None,
            "zone_id": str(idx.loc[bins[0], "zone_id"]) if bins else None,
            "aisle_id": str(idx.loc[bins[0], "aisle_id"]) if bins else None,
        }
    if out["a"]["primary"] and out["b"]["primary"]:
        out["network_distance_m"] = network_distance(out["a"]["primary"], out["b"]["primary"])
        out["same_aisle"] = out["a"]["aisle_id"] is not None and out["a"]["aisle_id"] == out["b"]["aisle_id"]
    else:
        out["network_distance_m"] = None
        out["same_aisle"] = None
    return out


@lru_cache(maxsize=1)
def _graph():
    from src.routing.graph import build_graph

    path = DATA_DIR / "location_graph.csv"
    return build_graph(path) if path.exists() else None


def network_distance(loc_a: str, loc_b: str) -> float | None:
    graph = _graph()
    if graph is None:
        return None
    distance = graph.shortest(str(loc_a), str(loc_b))
    return None if distance == float("inf") else round(float(distance), 3)


def get_3x3_matrix_zones() -> list[dict]:
    """Returns the geometry and metadata of all nine zones in a 3x3 matrix in a box."""
    outer_margin = 3.0
    zone_size = 9.6
    aisle_width = 3.0
    zones = []
    zone_types = [
        ("Z1", "General", "#2980b9"),
        ("Z2", "General", "#2980b9"),
        ("Z3", "General", "#2980b9"),
        ("Z4", "Small Parts", "#8e44ad"),
        ("Z5", "Bulk", "#16a085"),
        ("Z6", "Bulk", "#16a085"),
        ("Z7", "Heavy", "#d35400"),
        ("Z8", "Heavy", "#d35400"),
        ("Z9", "Hazard", "#c0392b"),
    ]
    for idx, (zid, ztype, color) in enumerate(zone_types):
        col = idx % 3
        row = idx // 3
        x0 = outer_margin + col * (zone_size + aisle_width)
        y0 = outer_margin + row * (zone_size + aisle_width)
        zones.append({
            "zone_id": zid,
            "zone_type": ztype,
            "color": color,
            "col": col,
            "row": row,
            "x0": round(x0, 2),
            "y0": round(y0, 2),
            "x1": round(x0 + zone_size, 2),
            "y1": round(y0 + zone_size, 2),
            "x_center": round(x0 + zone_size / 2.0, 2),
            "y_center": round(y0 + zone_size / 2.0, 2),
        })
    return zones


def get_3x3_zone_racks() -> pd.DataFrame:
    """Returns the geometry of the four edge racks on each of the nine zones."""
    outer_margin = 3.0
    zone_size = 9.6
    aisle_width = 3.0
    rack_depth = 1.0
    racks = []
    sides = [
        ("South", "South Edge (Bottom)"),
        ("North", "North Edge (Top)"),
        ("West", "West Edge (Left)"),
        ("East", "East Edge (Right)"),
    ]
    for idx in range(9):
        zid = f"Z{idx + 1}"
        col = idx % 3
        row = idx // 3
        zx0 = outer_margin + col * (zone_size + aisle_width)
        zy0 = outer_margin + row * (zone_size + aisle_width)
        zx1 = zx0 + zone_size
        zy1 = zy0 + zone_size
        for side, label in sides:
            rack_id = f"R{idx + 1}-{side}"
            if side == "South":
                x0, x1 = zx0, zx1
                y0, y1 = zy0, zy0 + rack_depth
            elif side == "North":
                x0, x1 = zx0, zx1
                y0, y1 = zy1 - rack_depth, zy1
            elif side == "West":
                x0, x1 = zx0, zx0 + rack_depth
                y0, y1 = zy0, zy1
            else:
                x0, x1 = zx1 - rack_depth, zx1
                y0, y1 = zy0, zy1
            racks.append({
                "zone_id": zid,
                "side": side,
                "label": label,
                "rack_id": rack_id,
                "x_min": round(x0, 2),
                "x_max": round(x1, 2),
                "y_min": round(y0, 2),
                "y_max": round(y1, 2),
            })
    return pd.DataFrame(racks)


def get_corridor_grid() -> tuple[list[float], list[float]]:
    """Returns the four horizontal and four vertical corridor centerlines between/around zones."""
    outer_margin = 3.0
    zone_size = 9.6
    aisle_width = 3.0
    x_corrs = [
        outer_margin / 2.0,
        outer_margin + zone_size + aisle_width / 2.0,
        outer_margin + 2 * zone_size + 1.5 * aisle_width,
        outer_margin + 3 * zone_size + 2 * aisle_width + outer_margin / 2.0,
    ]
    y_corrs = [
        outer_margin / 2.0,
        outer_margin + zone_size + aisle_width / 2.0,
        outer_margin + 2 * zone_size + 1.5 * aisle_width,
        outer_margin + 3 * zone_size + 2 * aisle_width + outer_margin / 2.0,
    ]
    return [round(v, 2) for v in x_corrs], [round(v, 2) for v in y_corrs]


def ruler_path(loc_a: str, loc_b: str, locations: pd.DataFrame | None = None) -> dict:
    """Calculates the exact path and distance of the invisible ruler between two locations.

    The path strictly follows the perimeter and inter-zone corridors, never cutting through zones.
    """
    graph = _graph()
    if graph is None:
        return {"distance": None, "path": [], "xs": [], "ys": []}
    path, dist = graph.shortest_path(str(loc_a), str(loc_b))
    if dist == float("inf"):
        return {"distance": None, "path": [], "xs": [], "ys": []}

    loc_lookup = {}
    if locations is not None and not locations.empty:
        for r in locations.itertuples(index=False):
            loc_lookup[str(r.location_id)] = (float(r.x_m), float(r.y_m))

    x_c, y_c = get_corridor_grid()
    xs: list[float] = []
    ys: list[float] = []

    for node in path:
        if node in loc_lookup:
            xs.append(loc_lookup[node][0])
            ys.append(loc_lookup[node][1])
        elif node == "DEPOT":
            xs.append(20.4)
            ys.append(0.5)
        elif node == "CORR_DEPOT_ENTRY":
            xs.append(20.4)
            ys.append(y_c[0])
        elif node.startswith("INT_"):
            parts = node.split("_")
            i, j = int(parts[1]), int(parts[2])
            xs.append(x_c[i])
            ys.append(y_c[j])
        elif node.startswith("ACOL_"):
            # Connect along adjacent corridor
            if xs:
                xs.append(xs[-1])
                ys.append(ys[-1])

    return {
        "distance": round(dist, 2),
        "path": path,
        "xs": xs,
        "ys": ys,
    }



# ----------------------------------------------------------------------------- availability
def data_availability(data: dict, layout: str = "optimized") -> dict[str, tuple[bool, str]]:
    """Which sections have real backend data, with the reason when they do not."""
    checks = data.get(f"checks_{layout}")
    stops = data.get(f"route_stops_{layout}")
    runs = data.get("run_history")
    return {
        "overview": (
            data.get("metrics") is not None,
            "Run `python app/main.py run-all` to generate baseline/optimized metrics.",
        ),
        "layout": (
            data.get("locations") is not None,
            "Canonical topology (locations.csv) not found.",
        ),
        "assignments": (
            data.get(layout) is not None,
            f"No {layout} assignments file found.",
        ),
        "demand": (
            data.get("abc") is not None,
            "Run `python app/main.py run-analysis` to generate ABC/demand outputs.",
        ),
        "affinity": (
            data.get("affinity") is not None,
            "No association-analysis output found (outputs/analysis/affinity_matrix.csv).",
        ),
        "routes": (
            data.get("routes") is not None and stops is not None,
            "Route stop detail not generated; run `python app/main.py simulate`.",
        ),
        "constraints": (
            checks is not None,
            f"No validation checks were exported for the {layout} layout.",
        ),
        "ergonomics": (
            data.get("metrics") is not None and data.get("abc") is not None,
            "Golden-zone metrics require evaluation metrics and ABC output.",
        ),
        "optimization": (
            data.get("opt_summary") is not None,
            "No optimization summary found; run `python app/main.py optimize`.",
        ),
        "run_history": (
            bool(runs),
            "No completed optimization runs recorded yet.",
        ),
        "replenishment": (
            False,
            "Not modeled: the dataset has no replenishment policy, lead times, or reorder points.",
        ),
    }


def metric_definition(data: dict) -> dict:
    summary = data.get("eval_summary") or {}
    return {
        "assumptions": summary.get("assumptions") or {},
        "notes": summary.get("notes") or {},
        "route_breakdown": summary.get("route_breakdown") or {},
    }


def report_payload(data: dict, layout: str = "optimized") -> dict:
    """Structured report containing exactly the results shown in the interface."""
    summary = data.get("eval_summary") or {}
    metrics = data.get("metrics")
    comparison = []
    if metrics is not None:
        for row in kpi_rows(metrics):
            if row["available"]:
                comparison.append(row)
    short = unassigned_skus(data, layout)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methodology": {
            "abac_rule": abc_rule_text(data.get("config")),
            "evaluation": (
                "Baseline and optimized layouts are evaluated on the same order workload, "
                "warehouse geometry, routing method, picker speed, pick time and setup time. "
                "Both layouts hold the same on-hand inventory."
            ),
            "constraints": (
                "Safety and physical feasibility are hard constraints enforced by the validator; "
                "they are never traded against the weighted objective."
            ),
            "route_time": "travel time (distance / picker speed) + pick lines x pick time + setup per batch",
        },
        "assumptions": summary.get("assumptions") or {},
        "results": {
            "baseline": summary.get("baseline") or {},
            "optimized": summary.get("optimized") or {},
            "comparison": comparison,
            "route_breakdown": summary.get("route_breakdown") or {},
        },
        "safety": {
            "baseline": safety_status(data, "baseline"),
            "optimized": safety_status(data, "optimized"),
        },
        "optimization": data.get("opt_summary") or {},
        "unassigned_skus": [] if short is None else short.head(200).to_dict(orient="records"),
        "limitations": [
            "Cube utilization is a volume-based approximation, not an exact 3-D packing calculation.",
            "Replenishment performance is not modeled (no replenishment policy or lead-time inputs).",
            "Optimization is a greedy + simulated-annealing heuristic; globally_optimal is false.",
            "Route legs are drawn as straight connectors; distances come from the aisle-network graph.",
        ],
    }
