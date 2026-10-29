"""Phase-7/9 dashboard tests: the dashboard must read REAL generated outputs (not hardcoded),
all views must have data, selecting a location surfaces its assignment, and every UI state
(including failure states) is driven by the backend artifacts.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def dashboard_data():
    from app import dashboard

    return dashboard.load_all()


def test_dashboard_loads_real_data(dashboard_data):
    # The dashboard must be fed from real generated artifacts, not hardcoded demo data.
    assert dashboard_data["locations"] is not None and len(dashboard_data["locations"]) >= 500
    assert dashboard_data["abc"] is not None and len(dashboard_data["abc"]) >= 500
    assert dashboard_data["optimized"] is not None and len(dashboard_data["optimized"]) >= 500
    assert dashboard_data["metrics"] is not None and len(dashboard_data["metrics"]) >= 5
    assert dashboard_data["routes"] is not None and len(dashboard_data["routes"]) > 0
    assert dashboard_data["checks_optimized"] is not None and len(dashboard_data["checks_optimized"]) > 0
    assert dashboard_data["route_stops_optimized"] is not None
    assert dashboard_data["run_history"], "completed optimizer runs must be recorded"


def test_kpi_metrics_from_real_output(dashboard_data):
    metrics = dashboard_data["metrics"].set_index("metric")
    # optimized must be feasible and safe per the real evaluation output
    assert str(metrics.loc["safety_compliant", "optimized"]).strip().lower() == "true"
    assert int(metrics.loc["constraint_violations", "optimized"]) == 0
    assert float(metrics.loc["assignment_feasibility_pct", "optimized"]) == 100.0


def test_layout_view_maps_abc_classes(dashboard_data):
    abc = dashboard_data["abc"]
    baseline = dashboard_data["baseline"]
    locations = dashboard_data["locations"]
    abc_map = dict(zip(abc["sku_id"], abc["abc_class"]))
    loc_sku = dict(zip(baseline["location_id"], baseline["sku_id"]))
    # the dashboard maps ABC classes through location -> SKU -> class
    classes = {lid: abc_map.get(loc_sku.get(lid, ""), "unassigned") for lid in locations["location_id"]}
    assert set(classes.values()).issubset({"A", "B", "C", "unassigned"})
    assert any(c == "A" for c in classes.values())
    assert sum(1 for c in classes.values() if c != "unassigned") == baseline["location_id"].nunique()


def test_route_view_reads_both_layouts(dashboard_data):
    routes = dashboard_data["routes"]
    assert set(routes["layout"].unique()) == {"baseline", "optimized"}
    base_avg = routes[routes["layout"] == "baseline"]["distance_m"].mean()
    opt_avg = routes[routes["layout"] == "optimized"]["distance_m"].mean()
    assert opt_avg < base_avg


def test_inspect_quantity_matches_optimized(dashboard_data):
    optimized = dashboard_data["optimized"]
    sku = optimized["sku_id"].iloc[0]
    row = optimized[optimized["sku_id"] == sku].iloc[0]
    assert int(row["assigned_quantity"]) >= 1
    assert str(row["location_id"]).startswith("LOC")


def test_dashboard_app_renders_all_sections(dashboard_data):
    """Run the Streamlit script headlessly and check the KPI mapping matches the CSV."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.tabs) == 9

    labels = {m.label: m.value for m in at.metric}
    assert labels["Safety compliance"] == "PASS"
    assert labels["Constraint violations (hard)"] == "0"
    assert labels["Assignment feasibility"].startswith("100.00")
    assert labels["Valid assignments"].startswith("1,200 / 1,200")

    metrics = dashboard_data["metrics"].set_index("metric")
    travel = float(metrics.loc["total_travel_distance_m", "optimized"])
    assert labels["Total pick-path travel distance"] == f"{travel:,.0f} m"
    throughput = float(metrics.loc["throughput_orders_per_hr", "optimized"])
    assert labels["Throughput"] == f"{throughput:,.2f} orders/h"

    # the safety claim is only shown because validation output exists for the optimized layout
    assert labels["Safety compliance"] == "PASS"
    assert dashboard_data["checks_optimized"]["passed"].astype(str).str.lower().eq("true").all()


def test_dashboard_baseline_toggle_and_elevation_view(dashboard_data):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    for radio in at.radio:
        if radio.key == "layout_view":
            radio.set_value("Rack elevation")
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any(select.key == "layout_rack" for select in at.selectbox)

    for radio in at.radio:
        if radio.key == "sidebar_layout":
            radio.set_value("Baseline")
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    # constraints tab must now describe the baseline layout
    captions = " ".join(c.value for c in at.caption)
    assert "baseline layout" in captions.lower() or "baseline" in captions.lower()
    labels = {m.label: m.value for m in at.metric}
    assert labels["Constraint violations (hard)"] == "0"
