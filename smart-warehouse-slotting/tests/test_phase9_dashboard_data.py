"""Dashboard data-mapping and formula tests.

These cover the acceptance-critical pieces of the UI redesign: documented evaluation
formulas, KPI mapping from real outputs, ABC/layout mapping, safety verdicts backed by
actual validation, route-stop integrity, exports matching the screen, and explicit
unavailable/failure states.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT.parent / "dataset" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"

pytestmark = pytest.mark.skipif(
    not (PROCESSED.exists() and (OUTPUTS / "evaluation" / "optimization_metrics.csv").exists()),
    reason="canonical dataset / evaluation outputs not generated",
)


@pytest.fixture(scope="module")
def data():
    from app import dashboard_data as dd

    return dd.load_all()


# --------------------------------------------------------------------------- formulas
def test_distance_and_time_reduction_formulas():
    from app import dashboard_data as dd

    assert dd.pct_reduction(100.0, 75.0) == pytest.approx(25.0)
    assert dd.pct_reduction(0.0, 10.0) is None, "zero baseline must not divide"
    assert dd.pct_reduction(None, 10.0) is None
    assert dd.pct_increase(10.0, 12.0) == pytest.approx(20.0)
    assert dd.percentage_points(15.42, 15.42) == 0.0
    assert dd.percentage_points(10.0, 12.5) == pytest.approx(2.5)
    assert dd.ratio(5.0, 0.0) is None


def test_metric_rows_match_evaluation_csv(data):
    from app import dashboard_data as dd

    metrics = data["metrics"].set_index("metric")
    row = dd.kpi_row(data["metrics"], "total_travel_distance_m")
    csv = metrics.loc["total_travel_distance_m"]
    assert row["baseline"] == pytest.approx(float(csv["baseline"]))
    assert row["optimized"] == pytest.approx(float(csv["optimized"]))
    # reduction is the positive magnitude of the CSV's signed percent change
    assert row["change_pct"] == pytest.approx(-float(csv["percent_change"]), abs=0.01)

    route = dd.kpi_row(data["metrics"], "avg_route_time_sec")
    csv_route = metrics.loc["avg_route_time_sec"]
    assert route["change_pct"] == pytest.approx(-float(csv_route["percent_change"]), abs=0.01)

    cube = dd.kpi_row(data["metrics"], "cube_utilization_pct")
    assert cube["change_pp"] == pytest.approx(float(metrics.loc["cube_utilization_pct", "absolute_diff"]))
    assert cube["change_pct"] is None, "utilization must be reported in percentage points"


def test_feasibility_numerator_and_denominator(data):
    from app import dashboard_data as dd

    valid = dd.kpi_row(data["metrics"], "valid_assignments")
    required = dd.kpi_row(data["metrics"], "required_assignments")
    feasibility = dd.kpi_row(data["metrics"], "assignment_feasibility_pct")
    assert required["optimized"] == data["eval_summary"]["skus_with_inventory"]
    assert feasibility["optimized"] == pytest.approx(
        valid["optimized"] / required["optimized"] * 100.0
    )

    metrics = data["metrics"].set_index("metric")
    assert valid["optimized"] == float(metrics.loc["valid_assignments", "optimized"])


def test_zero_baseline_is_reported_not_divided():
    from app import dashboard_data as dd
    from app.dashboard import change_label

    frame = pd.DataFrame([{
        "metric": "total_travel_distance_m", "baseline": 0.0, "optimized": 10.0,
        "unit": "m", "change_kind": "reduction", "direction": "lower is better", "note": "",
    }])
    row = dd.kpi_row(frame, "total_travel_distance_m")
    assert row is not None
    assert row["change_pct"] is None
    assert change_label(row).startswith("not defined")


def test_missing_metrics_render_as_unavailable():
    from app import dashboard_data as dd

    rows = dd.kpi_rows(None)
    assert rows and all(not row["available"] for row in rows)
    assert dd.kpi_row(None, "total_travel_distance_m") is None
    assert dd.kpi_row(pd.DataFrame(columns=["metric", "baseline", "optimized"]), "throughput_orders_per_hr") is None


# --------------------------------------------------------------------------- safety
def test_safety_pass_is_backed_by_real_validation(data):
    from app import dashboard_data as dd

    verdict = dd.safety_status(data, "optimized")
    assert verdict["status"] == "PASS"
    checks = data["checks_optimized"]
    assert verdict["checks_run"] == len(checks)
    assert verdict["failed_checks"] == 0
    assert verdict["rules_failed"] == []
    assert data["opt_summary"]["n_violations"] == 0


def test_missing_validation_is_not_reported_as_pass(data):
    from app import dashboard_data as dd

    verdict = dd.safety_status({**data, "checks_optimized": None}, "optimized")
    assert verdict["status"] == "UNAVAILABLE"
    assert verdict["checks_run"] == 0
    assert "No validation" in verdict["reason"]


def test_failed_check_makes_safety_fail(data):
    from app import dashboard_data as dd

    checks = data["checks_optimized"].copy()
    checks.loc[checks.index[0], "passed"] = False
    checks.loc[checks.index[0], "severity"] = "high"
    verdict = dd.safety_status({**data, "checks_optimized": checks}, "optimized")
    assert verdict["status"] == "FAIL"
    assert verdict["hard_violations"] == 1
    assert verdict["rules_failed"]


# --------------------------------------------------------------------------- layout mapping
def test_location_state_maps_abc_classes_without_inventing_them(data):
    from app import dashboard_data as dd

    state = dd.location_state(data, "optimized")
    assert state is not None
    assert len(state) == len(data["locations"]), "one row per physical location"
    known = set(data["abc"]["abc_class"])
    assigned = state[state["assigned"]]
    assert set(assigned["abc_class"]).issubset(known)
    assert (state.loc[~state["assigned"], "abc_class"] == "unassigned").all()
    assert state["assigned"].sum() == data["optimized"]["location_id"].nunique()
    # a clean optimized layout validates every bin and no utilizations are fabricated
    assert set(state["constraint_status"].unique()) == {"OK"}
    assert (state["bin_volume_util_pct"].fillna(0) >= 0).all()


def test_layout_filters_apply(data):
    from app import dashboard_data as dd

    state = dd.location_state(data, "optimized")
    only_a = dd.filter_locations(state, abc_classes=["A"], zone="Z1")
    assert not only_a.empty
    assert set(only_a["abc_class"]) == {"A"}
    assert set(only_a["zone_id"].astype(str)) == {"Z1"}
    shelf_two = dd.filter_locations(state, shelf=2, assigned_only=True)
    assert set(shelf_two["shelf_level"].astype(int)) == {2}
    assert shelf_two["assigned"].all()


def test_missing_assignments_do_not_fabricate_skus(data):
    from app import dashboard_data as dd

    state = dd.location_state({**data, "optimized": None}, "optimized")
    assert state is not None
    assert not state["assigned"].any()
    assert set(state["abc_class"].unique()) == {"unassigned"}
    assert state["sku_id"].isna().all()


def test_unassigned_sku_shortfall_is_explicit():
    from app import dashboard_data as dd

    inventory = pd.DataFrame({"sku_id": ["S1", "S2"], "location_id": ["L1", "L2"], "quantity_on_hand": [5, 4]})
    assignments = pd.DataFrame({"sku_id": ["S1"], "location_id": ["L1"], "assigned_quantity": [3]})
    short = dd.unassigned_skus({"inventory": inventory, "optimized": assignments}, "optimized")
    assert short is not None and len(short) == 2
    by_sku = short.set_index("sku_id")
    assert by_sku.loc["S1", "shortfall"] == 2
    assert by_sku.loc["S2", "shortfall"] == 4


def test_abc_summary_uses_configured_rule(data):
    from app import dashboard_data as dd

    summary = dd.abc_summary(data["abc"])
    assert summary is not None
    assert int(summary["skus"].sum()) == len(data["abc"])
    assert set(summary["abc_class"]) == {"A", "B", "C"}
    a_row = summary[summary["abc_class"] == "A"].iloc[0]
    assert 70.0 <= a_row["share_of_demand_pct"] <= 85.0, "A class should cover the configured ~80% of demand"
    assert "cumulative" in dd.abc_rule_text(data.get("config")).lower()


# --------------------------------------------------------------------------- routes
def test_route_stops_are_ordered_and_sum_to_simulated_distance(data):
    from app import dashboard_data as dd

    routes = dd.route_batches(data)
    assert routes is not None
    base_batches = set(routes.loc[routes["layout"] == "baseline", "batch_id"])
    opt_batches = set(routes.loc[routes["layout"] == "optimized", "batch_id"])
    assert base_batches == opt_batches, "both layouts must simulate the same batches"

    for layout in ("baseline", "optimized"):
        stops = data[f"route_stops_{layout}"]
        assert not stops.empty
        # every batch starts and ends at the depot
        for _batch, group in stops.groupby("batch_id"):
            ordered = group.sort_values("sequence")
            assert ordered.iloc[0]["location_id"] == "DEPOT"
            assert ordered.iloc[-1]["location_id"] == "DEPOT"
            assert list(ordered["sequence"]) == sorted(ordered["sequence"])
        totals = stops.groupby("batch_id")["leg_distance_m"].sum()
        expected = routes[routes["layout"] == layout].set_index("batch_id")["distance_m"]
        joined = totals.to_frame("legs").join(expected)
        assert (joined["legs"] - joined["distance_m"]).abs().max() < 0.5


def test_route_stop_coordinates_use_configured_dispatch(data):
    from app import dashboard_data as dd

    stops = dd.stops_with_coordinates(data, "optimized", 0)
    assert stops is not None
    dispatch = dd.dispatch_point(data)
    assert dispatch is not None
    first = stops.iloc[0]
    assert first["location_id"] == "DEPOT"
    assert float(first["x_m"]) == dispatch[0] and float(first["y_m"]) == dispatch[1]


def test_routes_unavailable_state_without_stop_detail(data):
    from app import dashboard_data as dd

    availability = dd.data_availability({**data, "route_stops_optimized": None})
    assert availability["routes"][0] is False
    assert "simulate" in availability["routes"][1]


# --------------------------------------------------------------------------- exports
def test_report_matches_dashboard_values(data):
    from app import dashboard_data as dd

    report = dd.report_payload(data, "optimized")
    comparison = {row["key"]: row for row in report["results"]["comparison"]}
    for key in ("total_travel_distance_m", "avg_route_time_sec", "cube_utilization_pct", "assignment_feasibility_pct"):
        ui_row = dd.kpi_row(data["metrics"], key)
        assert comparison[key]["optimized"] == pytest.approx(ui_row["optimized"])
        assert comparison[key]["baseline"] == pytest.approx(ui_row["baseline"])
    assert report["safety"]["optimized"]["status"] == dd.safety_status(data, "optimized")["status"]
    assert report["safety"]["baseline"]["status"] == dd.safety_status(data, "baseline")["status"]
    assert report["assumptions"]["picker_speed_mps"] > 0
    assert report["unassigned_skus"] == []
    # report must serialise to JSON for download
    json.dumps(report, default=str)


def test_validation_checks_export_has_evidence_columns(data):
    from app import dashboard_data as dd

    checks = data["checks_optimized"]
    assert {"rule_id", "rule_group", "passed", "observed", "limit", "explanation"}.issubset(checks.columns)
    assert checks["passed"].astype(str).str.lower().eq("true").all()
    summary = dd.validation_by_rule(checks)
    assert summary is not None and summary["failed"].sum() == 0
    # every configured rule id that ran is present in the audit
    rule_ids = set(summary["rule_id"])
    assert {"SP-01", "VD-01", "LD-01", "SL-01", "RK-01", "HZ-01", "HD-01", "ER-01", "COMP-01", "INV-03"}.issubset(rule_ids)


def test_validator_records_observed_and_limit_for_failures():
    from src.models.entities import AssignmentType, StorageAssignment
    from src.validation.validator import validate_assignment

    try:
        from tests.test_phase4_validator import _fixtures
    except ImportError:  # pragma: no cover - direct file run
        pytest.skip("phase-4 fixtures unavailable")
    wh, skus, rules = _fixtures()
    res = validate_assignment([StorageAssignment("S2", "L1", 1, AssignmentType.OPTIMIZED)], skus, wh, rules)
    failures = [c for c in res.checks if not c.passed]
    spatial = [c for c in failures if c.rule_id == "SP-01"]
    assert spatial, "oversized SKU must fail dimensional fit"
    check = spatial[0]
    assert check.observed and check.limit and check.explanation
    assert check.group == "Spatial & volumetric"
    assert res.violations and res.violations[0].rule_id == "SP-01"


def test_availability_reports_unmodeled_replenishment(data):
    from app import dashboard_data as dd

    availability = dd.data_availability(data)
    assert availability["replenishment"][0] is False
    assert "Not modeled" in availability["replenishment"][1]
    assert availability["overview"][0] and availability["routes"][0] and availability["constraints"][0]


def test_heatmap_modes_build_from_real_layout_state(data):
    """Every warehouse-visualization mode (ABC, utilization, load, safety, route overlay)."""
    from app import dashboard as dash
    from app import dashboard_data as dd

    state = dd.location_state(data, "optimized")
    zone_frame = state[state["zone_id"].astype(str) == "Z1"]
    frame = dd.filter_locations(state, zone="Z1", shelf=2)
    assert not frame.empty
    racks = dd.rack_geometry(zone_frame)
    aisles = dd.aisle_geometry(zone_frame)
    dispatch = dd.dispatch_point(data)

    for mode in [
        "ABC class", "Volume utilization", "Structural load — shelf",
        "Structural load — rack", "Constraint status", "Assignment change",
    ]:
        figure = dash.plan_figure(frame, racks, aisles, dispatch, None, mode, "test")
        assert figure.data, f"no traces drawn for mode {mode}"

    stops = dd.stops_with_coordinates(data, "optimized", 0)
    assert stops is not None
    figure = dash.plan_figure(
        frame, racks, aisles, dispatch, None, "ABC class", "test",
        zone_box=(0.0, 6.0, 0.0, 20.0), cross_box=(17.0, 18.0),
        route_overlay=stops[stops["zone_id"].astype(str) == "Z1"], show_ids=True,
    )
    names = {trace.name for trace in figure.data}
    assert "Route sequence" in names and "Pick stops" in names
    # structural figures (boundary/cross-aisle) are real shapes, not placeholders
    assert len(figure.layout.shapes) >= len(racks) + 2


def test_overview_and_elevation_figures_use_topology(data):
    from app import dashboard as dash
    from app import dashboard_data as dd

    overview = dash.overview_figure(data["locations"], dd.dispatch_point(data))
    assert len(overview.layout.shapes) >= 1, "warehouse boundary must be drawn"
    assert overview.data, "dispatch marker must be drawn"

    state = dd.location_state(data, "optimized")
    rack_frame = state[(state["zone_id"].astype(str) == "Z1") & (state["rack_id"].astype(str) == "R1-1-1")]
    elevation = dash.elevation_figure(rack_frame, None, [1, 2], "ABC class", "test")
    assert elevation.data
    golden_shapes = [s for s in elevation.layout.shapes if s.fillcolor == "#fef5e7"]
    assert len(golden_shapes) >= 1, "golden shelf band must be highlighted"
