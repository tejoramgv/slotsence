"""Phase-3 orchestrator: produce all analysis outputs from the Phase-2 processed data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .abc import abc_classify_summary
from .affinity import build_affinity
from .demand import build_demand_summary
from .volume_profile import build_volume_profiles


def run_phase3(processed: Path, out: Path, config_abc: dict | None = None) -> dict:
    """Run ABC + demand + affinity + volume profiling. Writes outputs/ CSVs. Returns summary."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_abc or {"a_share": 0.80, "b_share": 0.95, "scenario": "A"}

    products = pd.read_csv(processed / "products.csv")
    all_skus = pd.Series(products["sku_id"])

    # Demand + ABC
    demand = build_demand_summary(processed, all_skus, scenario=cfg["scenario"], out=out / "demand_summary.csv")
    abc = abc_classify_summary(demand, a_share=cfg["a_share"], b_share=cfg["b_share"])
    abc.to_csv(out / "abc_classification.csv", index=False)

    # Volume profiles
    vp = build_volume_profiles(processed, products, out=out / "volume_profiles.csv")

    # Affinity (scenario C by default)
    pairs, clusters = build_affinity(
        processed,
        scenario=cfg.get("affinity_scenario", "C"),
        matrix_out=out / "affinity_matrix.csv",
        cluster_out=out / "affinity_clusters.csv",
    )

    summary = {
        "abc_counts": abc["abc_class"].value_counts().to_dict(),
        "demand_rows": int(len(demand)),
        "volume_profile_rows": int(len(vp)),
        "affinity_pairs": int(len(pairs)),
        "affinity_clusters": int(len(clusters)),
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary