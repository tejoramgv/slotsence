# Smart Warehouse Slotting & Optimization Engine

End-to-end warehouse slotting system that assigns SKUs to physical storage locations while
satisfying spatial, weight, safety, demand, and ergonomic constraints, and demonstrates
measurable improvement over a feasible baseline layout.

## Problem context

Modern fulfillment centers lose productivity to inefficient SKU placement. This engine maps SKUs
to bins/racks/shelf levels based on historical demand, SKU velocity (ABC), order affinity, 3D
dimensions, weight bounds, category/hazard compatibility, and ergonomic shelf-height rules — then
proves improvement over the unoptimized baseline with quantitative metrics under identical demand,
geometry, and routing.

## Repository structure

```
smart-warehouse-slotting/
  app/main.py                 CLI (build phase1 / dataset / analysis / validate / optimize / simulate / run-all)
  app/dashboard.py            Streamlit + Plotly dashboard (Phase 7)
  src/models/                 domain entities, config, topology, synthetic data, IO
  src/data_processing/        benchmark dataset generator + validator + EDA (Phase 2)
  src/analytics/              demand/ABC/affinity/volume analysis (Phase 3)
  src/validation/             constraint validator (Phase 4)
  src/optimization/           allocation optimizer (Phase 5)
  src/routing/ + simulation/  aisle-network route simulation (Phase 6)
  src/evaluation/             baseline-vs-optimized metrics (Phase 6)
  tests/                      pytest suites
  configs/                    JSON config files
  outputs/                    generated CSVs / JSON / charts
  docs/                       technical documentation
```

The canonical dataset lives **outside** this package at `../dataset/` (`processed/`, `metadata/`,
`reports/`). The former `smart-warehouse-slotting/data/` tree was a legacy fixture and has been
removed; all code and tests read `../dataset/processed`.

## Canonical units

- Dimensions: centimetres (cm)
- Weight: kilograms (kg)
- Volume: cubic centimetres (cm3)
- Coordinates & travel distance: metres (m)
- Timestamps: UTC

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Run

**Phase 1** — small fixture warehouse + SKUs + feasible baseline:

```bash
python app/main.py build-phase1 --config configs/small_warehouse.json
```

**Phase 2** — build (or rebuild) the canonical benchmark dataset into `../dataset/`:

```bash
python app/main.py build-dataset --config configs/dataset.json
python -m src.data_processing.eda                # EDA charts + report -> outputs/eda/
```

**Phase 3** — demand / ABC / affinity / volume analysis:

```bash
python app/main.py run-analysis
```

Writes `outputs/analysis/{abc_classification,demand_summary,affinity_matrix,affinity_clusters,volume_profiles}.csv`.

**Phases 4–6 and the full pipeline:**

```bash
python app/main.py validate                  # baseline audit -> outputs/constraint_audit.csv
python app/main.py optimize                  # -> outputs/optimized/
python app/main.py simulate                  # -> outputs/evaluation/
python app/main.py run-all                   # validate + analysis + optimize + simulate
```

**Phase 7** — dashboard:

```bash
streamlit run app/dashboard.py
```

Nine tabs: Overview (baseline vs optimized KPIs, safety verdict, solver status), Warehouse Layout
(plan + rack elevation built from the topology), Demand & ABC, Order Affinity, Pick Routes,
Constraints & Safety (per-check audit with observed values and limits), Ergonomics,
Optimization controls, and Exports. `app/dashboard_data.py` holds the data mapping and documented
formulas; every value comes from the generated artifacts below.

Additional evaluation artifacts: `outputs/evaluation/validation_checks_{baseline,optimized}.csv`
(every rule evaluation with observed value/limit), `pick_route_stops_{baseline,optimized}.csv`
(ordered pick sequences with leg distances), and `outputs/optimized/run_history.json`
(completed optimizer runs).

## Tests

```bash
python -m pytest tests/ -q
```

## Phase status

- **Phase 1 (domain model)** — COMPLETE. Entities, config, seeded topology, feasible baseline
  fixture, smoke tests. See `docs/schema.md`.
- **Phase 2 (dataset + EDA)** — COMPLETE. Seeded synthetic benchmark: 1,200 SKUs across 10
  categories, 2,304 locations, 1,425 inventory rows (6,375 units), three alternative demand
  histories (60,000 order lines total), feasible baseline, 55 compatibility rules, aisle-network
  location graph, provenance columns, validation report, and EDA.
- **Phase 3 (demand/affinity/velocity)** — COMPLETE. ABC classification, demand summary,
  affinity matrix + clusters, volume profiles; reads the scenario-tagged canonical order tables.
- **Phase 4 (constraint validator)** — COMPLETE. Hard rules (dimensional fit, bin load/volume,
  shelf/rack structure, hazard/handling zones, heavy shelf height, single-SKU-per-bin, co-storage,
  multi-bin inventory conservation, completeness) with per-rule reasons. The canonical baseline
  passes with 0 violations over 2,850 checks.
- **Phase 5 (allocation optimizer)** — COMPLETE. Inventory-conserving greedy starter + simulated
  annealing from the baseline; every candidate is re-validated. All 6,375 units placed across
  1,534 assignments, 0 violations, objective 2,336,184 → 1,670,432 (−28.5%).
- **Phase 6 (simulation + evaluation)** — COMPLETE. Dijkstra distances on `location_graph.csv`,
  nearest-neighbor routing, identical workload for both layouts. **Result: −24.07% total travel
  distance, +27.02% throughput, 0 pp cube-utilization change (same inventory), 0 violations for
  both layouts.**
- **Phase 7 (dashboard)** — COMPLETE. Nine-section operational dashboard reading the canonical
  dataset and real outputs: warehouse plan/elevation by ABC class, Pareto/ABC analysis, affinity
  pairs and clusters, ordered pick routes, per-rule constraint audit, ergonomics, optimizer
  controls, and CSV/JSON exports with explicit unavailable/failure states.
- **Phase 8 (integration + docs)** — COMPLETE. `run-all` end-to-end pipeline, four required
  artifacts, edge-case tests, `docs/TECHNICAL_REPORT.md`.

Results above are from the checked-in `outputs/` generated by `run-all` (seed 42).
