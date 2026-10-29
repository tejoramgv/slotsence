# SlotSense — Smart Warehouse Slotting & Optimization

SlotSense assigns SKUs to warehouse bins and shelf levels under hard spatial, structural,
hazard, handling, and ergonomic constraints, then proves a measurable travel-time improvement
over a feasible baseline layout.

The engine includes a synthetic benchmark dataset, ABC/affinity demand analytics, a
constraint validator, an inventory-conserving optimizer, aisle-network route simulation,
baseline-vs-optimized evaluation, and a Streamlit dashboard.

## Repository layout

```
dataset/                       Canonical benchmark dataset (processed/, metadata/, reports/)
smart-warehouse-slotting/      Engine source, tests, configs, docs, generated outputs
prompts/                       Phase-by-phase build instructions used for the project
*.txt                          Project summary, execution plan, dataset roadmap
PS_Track2.pdf                  Problem statement
```

The engine and dataset are siblings so the canonical relative path
`smart-warehouse-slotting/../dataset/processed` works unchanged.

## Quick start

```powershell
cd smart-warehouse-slotting
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python app/main.py run-all        # validate -> analysis -> optimize -> simulate
streamlit run app/dashboard.py    # interactive dashboard at http://localhost:8501
python -m pytest tests/ -q        # full test suite
```

## Measured results (seed 42)

| Metric | Baseline | Optimized |
|---|---|---|
| Total travel distance | 1,707,167 m | 1,296,259 m (**−24.07%**) |
| Throughput | 13.14 orders/hr | 16.69 orders/hr (**+27.02%**) |
| Constraint violations | 0 | 0 |
| Assignment feasibility | 100% | 100% |

Optimizer objective: 2,336,184 → 1,670,432 (**−28.5%**) with all 6,375 inventory units placed.

See `smart-warehouse-slotting/README.md` and
`smart-warehouse-slotting/docs/TECHNICAL_REPORT.md` for the full design and results.
