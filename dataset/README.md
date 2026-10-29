# Smart Warehouse Benchmark Dataset

A coherent, synthetic warehouse dataset for the slotting and optimization engine.

- Seed: `42`
- SKUs: 1200 across 10 categories
- Locations: 2304
- Orders: 16244; order lines: 60000
- Demand scenarios: 3

## Contents

- `processed/` — the relational tables (CSV).
- `metadata/` — source manifest, data dictionary, schema notes, generation manifest.
- `reports/` — validation report and machine-readable error log.
- `raw/` — empty in this version; see `raw/README.md`.

## Regenerate

From `smart-warehouse-slotting/`:

```bash
python app/build_dataset.py --config configs/dataset.json
python app/build_dataset.py --validate-only
```

## Notes

- Scenarios A, B, and C are alternative demand histories. Never add their order lines together.
- Inventory is sized on Scenario A and capped by physical capacity.
- The baseline is feasible but deliberately unoptimized.
- All data is synthetic or derived; see `metadata/schema_documentation.md`.
