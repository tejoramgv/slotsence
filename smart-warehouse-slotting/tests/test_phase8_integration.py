"""Phase-8 integration / hardening tests: end-to-end pipeline, edge cases, malformed input,
and the delivery acceptance checks."""

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT.parent / "dataset" / "processed"


def test_four_required_artifacts_exist():
    required = [
        "outputs/baseline_assignments.csv",
        "outputs/constraint_audit.csv",
        "outputs/optimized/optimized_assignments.csv",
        "outputs/optimized/optimization_summary.json",
    ]
    for rel in required:
        assert (PROJECT_ROOT / rel).exists(), f"missing required artifact: {rel}"


def test_constraint_audit_is_not_header_only():
    audit = pd.read_csv(PROJECT_ROOT / "outputs" / "constraint_audit.csv")
    assert len(audit) > 0
    assert set(audit["passed"].astype(str).str.lower().unique()) == {"true"}


def test_optimized_output_is_feasible_via_summary():
    summary = json.loads((PROJECT_ROOT / "outputs" / "optimized" / "optimization_summary.json").read_text())
    assert summary["feasibility"] is True
    assert summary["n_violations"] == 0
    assert summary["globally_optimal"] is False
    assert summary["units_placed"] == summary["inventory_units"]
    assert summary["final_objective"] <= summary["baseline_objective"]


def test_malformed_config_rejected(tmp_path):
    from src.models.config import Config

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(ValueError):
        Config.from_json(bad)


def test_validator_rejects_referential_junk():
    from src.models.entities import AssignmentType, StorageAssignment
    from src.validation.loader import load_skus, load_warehouse_from_locations
    from src.validation.validator import validate_assignment

    locations = pd.read_csv(PROCESSED / "locations.csv")
    products = pd.read_csv(PROCESSED / "products.csv")
    wh = load_warehouse_from_locations(locations)
    skus = load_skus(products)
    # Assignment referencing a nonexistent location and SKU must be flagged.
    bad = [
        StorageAssignment("SKU000001", "NONEXISTENT", 1, AssignmentType.OPTIMIZED),
        StorageAssignment("UNKNOWN_SKU", "LOC000001", 1, AssignmentType.OPTIMIZED),
    ]
    res = validate_assignment(bad, skus, wh)
    assert any(v.rule_id == "REF-02" for v in res.violations)
    assert any(v.rule_id == "REF-01" for v in res.violations)


def test_end_to_end_pipeline_is_reproducible():
    """Re-running the optimizer with the same seed reproduces the final objective."""
    from src.models.config import Config
    from src.optimization.pipeline import run_phase5

    cfg = Config.from_json(PROJECT_ROOT / "configs" / "benchmark.json")
    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        s1 = run_phase5(PROCESSED, cfg, Path(d1))
        s2 = run_phase5(PROCESSED, cfg, Path(d2))
        assert s1["final_objective"] == s2["final_objective"]
