"""Phase-5 tests: optimizer feasibility, inventory conservation, improvement over the
baseline, determinism, and output validation via the Phase-4 validator."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.config import Config
from src.optimization.pipeline import run_phase5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT.parent / "dataset" / "processed"


@pytest.fixture(scope="module")
def optimized(tmp_path_factory):
    if not PROCESSED.exists():
        pytest.skip("canonical dataset not generated")
    cfg = Config.from_json(PROJECT_ROOT / "configs" / "benchmark.json")
    out = tmp_path_factory.mktemp("opt")
    summary = run_phase5(PROCESSED, cfg, out)
    return out, summary


def test_optimizer_feasible(optimized):
    out, summary = optimized
    assert summary["feasibility"] is True
    assert summary["n_violations"] == 0
    assert summary["units_placed"] == summary["inventory_units"]
    assert summary["n_assignments"] >= summary["n_skus_with_inventory"]


def test_optimizer_conserves_inventory_and_uses_multi_bin(optimized):
    out, summary = optimized
    inventory = pd.read_csv(PROCESSED / "inventory.csv")
    assignments = pd.read_csv(out / "optimized_assignments.csv")
    assert assignments["assigned_quantity"].sum() == inventory["quantity_on_hand"].sum()
    per_sku = assignments.groupby("sku_id")["assigned_quantity"].sum()
    on_hand = inventory.groupby("sku_id")["quantity_on_hand"].sum()
    assert per_sku.sort_index().equals(on_hand.sort_index())
    assert (assignments.groupby("sku_id").size() > 1).any(), "expected at least one multi-bin SKU"


def test_optimizer_assignments_pass_phase4_validator(optimized):
    out, summary = optimized
    from src.models.entities import AssignmentType, StorageAssignment
    from src.validation.loader import load_shipping
    from src.validation.validator import validate_assignment

    wh, skus, comp, _baseline, inventory = load_shipping(PROCESSED)
    opt_df = pd.read_csv(out / "optimized_assignments.csv")
    assignments = [
        StorageAssignment(r.sku_id, r.location_id, int(r.assigned_quantity), AssignmentType(r.assignment_type))
        for r in opt_df.itertuples(index=False)
    ]
    res = validate_assignment(assignments, skus, wh, comp, inventory)
    assert res.is_feasible, f"optimized output has violations: {[v.reason for v in res.violations[:5]]}"


def test_optimizer_improves_over_baseline(optimized):
    out, summary = optimized
    assert summary["baseline_feasibility"] is True
    assert summary["final_objective"] < summary["baseline_objective"]
    assert summary["final_objective"] <= summary["starter_objective"]
    assert summary["objective_improvement_pct"] > 0


def test_optimizer_summary_json(optimized):
    out, summary = optimized
    p = out / "optimization_summary.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert "globally_optimal" in data and data["globally_optimal"] is False
    assert data["algorithm"] != ""
    assert data["runtime_sec"] >= 0


def test_optimizer_deterministic():
    cfg = Config.from_json(PROJECT_ROOT / "configs" / "benchmark.json")
    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        s1 = run_phase5(PROCESSED, cfg, Path(d1))
        s2 = run_phase5(PROCESSED, cfg, Path(d2))
        # seeded optimizer should reproduce final objective and assignment count
        assert s1["final_objective"] == s2["final_objective"]
        df1 = pd.read_csv(Path(d1) / "optimized_assignments.csv").sort_values(["sku_id", "location_id"])
        df2 = pd.read_csv(Path(d2) / "optimized_assignments.csv").sort_values(["sku_id", "location_id"])
        assert df1["location_id"].tolist() == df2["location_id"].tolist()
