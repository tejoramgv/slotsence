"""CLI entry point for the Smart Warehouse Slotting & Optimization Engine.

Phase 1: builds a small warehouse + SKUs + feasible baseline fixture and writes outputs.
Later phases extend this pipeline (analysis -> validate -> optimize -> simulate -> evaluate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Canonical dataset (single source of truth): ../dataset/processed
CANONICAL_PROCESSED = PROJECT_ROOT.parent / "dataset" / "processed"

from src.analytics.pipeline import run_phase3
from src.data_processing.pipeline import DEFAULT_OUTPUT, build_dataset
from src.data_processing.validate import run_validation
from src.models.config import Config
from src.models.io import write_assignments_csv
from src.models.synthetic import build_phase1_fixture
from src.optimization.pipeline import run_phase5
from src.simulation.pipeline import run_phase6
from src.evaluation.metrics import assignment_feasibility
from src.validation.audit import write_constraint_audit, write_validation_checks
from src.validation.loader import load_shipping
from src.validation.validator import validate_assignment


def build_phase1(args: argparse.Namespace) -> None:
    config = Config.from_json(Path(args.config))
    out = PROJECT_ROOT / args.output
    counts = build_phase1_fixture(config, out, seed=args.seed)
    print(json.dumps(counts, indent=2))
    summary = PROJECT_ROOT / "outputs" / "phase1_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(counts, indent=2), encoding="utf-8")


def build_dataset_cmd(args: argparse.Namespace) -> None:
    summary = build_dataset(args.config, DEFAULT_OUTPUT, seed=args.seed)
    print(json.dumps(summary, indent=2))
    out = PROJECT_ROOT / "outputs" / "dataset_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    cfg = Config.from_json(Path(args.config))
    processed = CANONICAL_PROCESSED
    out = PROJECT_ROOT / args.output
    summary = run_phase3(
        processed,
        out,
        config_abc={
            "a_share": cfg.demand.abc_a_cumulative_share,
            "b_share": cfg.demand.abc_b_cumulative_share,
            "scenario": "A",
            "affinity_scenario": "C",
        },
    )
    print(json.dumps(summary, indent=2))


def _write_dataset_validation(processed: Path, out: Path) -> None:
    """Run the dataset-level validator and persist its machine-readable report."""
    result = run_validation(processed.parent)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"ok": result.ok, "hard_errors": result.hard_errors, "stats": result.stats, "errors": result.errors}
    (out / "dataset_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_validate(args: argparse.Namespace) -> None:
    """Phase 4: validate the baseline and export the constraint audit."""
    processed = Path(args.processed)
    out = PROJECT_ROOT / args.output
    wh, skus, comp, assignments, inventory = load_shipping(processed)
    res = validate_assignment(assignments, skus, wh, comp, inventory)
    write_constraint_audit(assignments, res, out / "constraint_audit.csv")
    write_validation_checks(res, out / "validation_checks_baseline.csv")
    write_assignments_csv(assignments, out / "baseline_assignments.csv")
    _write_dataset_validation(processed, out)
    slotting: dict[str, list[tuple[str, int]]] = {}
    for a in assignments:
        slotting.setdefault(a.sku_id, []).append((a.location_id, a.assigned_quantity))
    feasibility = assignment_feasibility(slotting, inventory, res)
    summary = {
        "dataset": str(processed),
        "n_assignments": len(assignments),
        "n_violations": len(res.violations),
        "n_hard_violations": len(res.hard_violations),
        "checks_run": res.checks_run,
        "feasibility_pct": feasibility["assignment_feasibility_pct"],
        "rule_check_pass_pct": round(100.0 - 100.0 * len(res.violations) / max(1, res.checks_run), 2),
        "valid_assignments": feasibility["valid_assignments"],
        "required_assignments": feasibility["required_assignments"],
        "feasible": res.is_feasible,
        "safety_compliant": len(res.hard_violations) == 0,
    }
    (out / "validation_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not res.is_feasible:
        for v in res.violations[:8]:
            print(f"  [{v.rule_id}] {v.reason}")


def run_optimize(args: argparse.Namespace) -> None:
    """Phase 5: run the allocation optimizer and write outputs."""
    config = Config.from_json(Path(args.config))
    processed = Path(args.processed)
    out = PROJECT_ROOT / args.output
    summary = run_phase5(processed, config, out)
    print(json.dumps(summary, indent=2))


def run_simulate(args: argparse.Namespace) -> None:
    """Phase 6: baseline vs optimized route simulation + evaluation."""
    processed = Path(args.processed)
    out = PROJECT_ROOT / args.output
    result = run_phase6(
        processed,
        scenario=args.scenario,
        out=out,
        optimized_path=PROJECT_ROOT / args.optimized if args.optimized else None,
    )
    print(json.dumps(result, indent=2))


def run_all(args: argparse.Namespace) -> None:
    """Phase 8: run the full pipeline end-to-end from the canonical dataset."""
    config = Config.from_json(Path(args.config))
    processed = Path(args.processed)
    out_root = PROJECT_ROOT / "outputs"

    print("== validate ==")
    wh, skus, comp, baseline, inventory = load_shipping(processed)
    res = validate_assignment(baseline, skus, wh, comp, inventory)
    write_constraint_audit(baseline, res, out_root / "constraint_audit.csv")
    write_validation_checks(res, out_root / "validation_checks_baseline.csv")
    write_assignments_csv(baseline, out_root / "baseline_assignments.csv")
    _write_dataset_validation(processed, out_root)
    print(f"   baseline feasible={res.is_feasible} violations={len(res.violations)} checks={res.checks_run}")
    print("== analysis ==")
    run_phase3(
        processed,
        out_root / "analysis",
        config_abc={
            "a_share": config.demand.abc_a_cumulative_share,
            "b_share": config.demand.abc_b_cumulative_share,
            "scenario": "A",
            "affinity_scenario": "C",
        },
    )
    print("== optimize ==")
    run_phase5(processed, config, out_root / "optimized")
    print("== simulate ==")
    run_phase6(processed, scenario=args.scenario, out=out_root / "evaluation")
    print("== done ==")


def main() -> None:
    parser = argparse.ArgumentParser(prog="smart-warehouse-slotting")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("build-phase1")
    p1.add_argument("--config", default="configs/small_warehouse.json")
    p1.add_argument("--output", default="outputs/phase1")
    p1.add_argument("--seed", type=int, default=None)
    p1.set_defaults(func=build_phase1)

    p2 = sub.add_parser("build-dataset")
    p2.add_argument("--config", default="configs/dataset.json")
    p2.add_argument("--seed", type=int, default=None)
    p2.set_defaults(func=build_dataset_cmd)

    p3 = sub.add_parser("run-analysis")
    p3.add_argument("--config", default="configs/benchmark.json")
    p3.add_argument("--output", default="outputs/analysis")
    p3.set_defaults(func=run_analysis)

    p4 = sub.add_parser("validate")
    p4.add_argument("--processed", default=str(CANONICAL_PROCESSED))
    p4.add_argument("--output", default="outputs")
    p4.set_defaults(func=run_validate)

    p5 = sub.add_parser("optimize")
    p5.add_argument("--config", default="configs/benchmark.json")
    p5.add_argument("--processed", default=str(CANONICAL_PROCESSED))
    p5.add_argument("--output", default="outputs/optimized")
    p5.set_defaults(func=run_optimize)

    p6 = sub.add_parser("simulate")
    p6.add_argument("--processed", default=str(CANONICAL_PROCESSED))
    p6.add_argument("--scenario", default="A")
    p6.add_argument("--optimized", default=None, help="path to optimized_assignments.csv")
    p6.add_argument("--output", default="outputs/evaluation")
    p6.set_defaults(func=run_simulate)

    pall = sub.add_parser("run-all")
    pall.add_argument("--config", default="configs/benchmark.json")
    pall.add_argument("--processed", default=str(CANONICAL_PROCESSED))
    pall.add_argument("--seed", type=int, default=None)
    pall.add_argument("--scenario", default="A")
    pall.set_defaults(func=run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()