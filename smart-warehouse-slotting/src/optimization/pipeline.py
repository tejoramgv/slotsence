"""Phase-5 orchestration over the canonical dataset (dataset/processed)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.config import Config
from src.models.io import write_assignments_csv
from src.validation.loader import (
    load_assignments,
    load_compatibility_rules,
    load_inventory,
    load_skus,
    load_warehouse_from_locations,
)
from src.validation.validator import Validator

from .objective import Objective, build_affinity_penalty
from .optimizer import OptimizerConfig, SlottingOptimizer


def run_phase5(processed: Path, config: Config, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    locations = pd.read_csv(processed / "locations.csv")
    products = pd.read_csv(processed / "products.csv")
    rules = pd.read_csv(processed / "compatibility_rules.csv")
    inventory = pd.read_csv(processed / "inventory.csv")
    baseline_rows = pd.read_csv(processed / "storage_assignments_baseline.csv")
    demand = pd.read_csv(processed / "demand_summary.csv")
    affinity = pd.read_csv(processed / "sku_affinity.csv")

    wh = load_warehouse_from_locations(locations)
    skus = load_skus(products)
    comp = load_compatibility_rules(rules)
    inv = load_inventory(inventory)
    baseline = load_assignments(baseline_rows)

    # ABC + demand from the canonical demand_summary (scenario A)
    dA = demand[demand["scenario_id"] == "A"]
    abc_classes = dict(zip(dA["sku_id"], dA["abc_class"]))
    demand_map = dict(zip(dA["sku_id"], dA["demand_frequency"].astype(float)))

    # Affinity penalty from canonical sku_affinity (scenario C)
    affC = affinity[affinity["scenario_id"] == "C"]
    affinity_penalty = build_affinity_penalty(affC)

    validator = Validator(skus, wh, comp, inventory=inv)
    objective = Objective(
        alpha=config.optimization.alpha,
        beta=config.optimization.beta,
        gamma=config.optimization.gamma,
        delta=config.optimization.delta,
    )
    opt_cfg = OptimizerConfig(seed=config.dataset.seed)

    optimizer = SlottingOptimizer(
        warehouse=wh,
        skus=skus,
        abc_classes=abc_classes,
        validator=validator,
        inventory=inv,
        affinity_penalty=affinity_penalty,
        objective=objective,
        demand=demand_map,
        cfg=opt_cfg,
        initial_assignments=baseline,
    )
    optimized, log = optimizer.optimize()
    result = validator.validate(optimized)

    write_assignments_csv(optimized, out / "optimized_assignments.csv")

    baseline_result = validator.validate(baseline)
    summary = {
        "dataset": str(processed),
        "algorithm": "greedy (velocity-ordered, inventory-conserving) + simulated annealing",
        "objective_weights": {"alpha": objective.alpha, "beta": objective.beta, "gamma": objective.gamma, "delta": objective.delta},
        "n_assignments": len(optimized),
        "n_skus_with_inventory": len(inv),
        "inventory_units": log.inventory_units,
        "units_placed": sum(a.assigned_quantity for a in optimized),
        "feasibility": result.is_feasible,
        "n_violations": len(result.violations),
        "baseline_feasibility": baseline_result.is_feasible,
        "baseline_objective": log.initial_score,
        "starter_objective": log.starter_score,
        "final_objective": log.final_score,
        "objective_improvement_pct": (
            round((log.initial_score - log.final_score) / log.initial_score * 100, 3)
            if log.initial_score
            else None
        ),
        "moves_evaluated": log.n_moves_evaluated,
        "moves_accepted": log.n_moves_accepted,
        "runtime_sec": round(log.runtime_sec, 3),
        "stopping_reason": log.stopping_reason,
        "globally_optimal": False,
    }
    (out / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not result.is_feasible:
        top = [v.reason for v in result.violations[:8]]
        summary["top_violations"] = top
        (out / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _append_run_history(out, summary, config)
    return summary


def _append_run_history(out: Path, summary: dict, config: Config, keep: int = 20) -> None:
    """Record real completed runs so the dashboard can compare them (newest last)."""
    path = out / "run_history.json"
    history: list[dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except (json.JSONDecodeError, ValueError):
            history = []
    entry = {
        "run_id": len(history) + 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": summary.get("algorithm"),
        "objective_weights": summary.get("objective_weights"),
        "seed": config.dataset.seed,
        "feasibility": summary.get("feasibility"),
        "n_violations": summary.get("n_violations"),
        "units_placed": summary.get("units_placed"),
        "inventory_units": summary.get("inventory_units"),
        "final_objective": summary.get("final_objective"),
        "objective_improvement_pct": summary.get("objective_improvement_pct"),
        "moves_evaluated": summary.get("moves_evaluated"),
        "moves_accepted": summary.get("moves_accepted"),
        "runtime_sec": summary.get("runtime_sec"),
        "stopping_reason": summary.get("stopping_reason"),
        "globally_optimal": summary.get("globally_optimal"),
    }
    history.append(entry)
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history[-keep:], indent=2), encoding="utf-8")