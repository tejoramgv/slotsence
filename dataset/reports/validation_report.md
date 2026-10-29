# Dataset Validation Report

- Status: **PASS**
- Hard errors: **0**
- Warnings: **0**

## Row counts

| table | rows |
|---|---|
| categories | 10 |
| compatibility_rules | 55 |
| demand_summary | 3600 |
| inventory | 1425 |
| location_graph | 5274 |
| locations | 2304 |
| order_items | 60000 |
| orders | 16244 |
| products | 1200 |
| sku_affinity | 10130 |
| storage_assignments_baseline | 1425 |
| warehouses | 1 |

## Capacity

- Locations: 2304 (1425 used, 879 empty)
- Volume utilization: 15.42%
- Weight utilization: 14.82%
- Occupied volume: 66,411,409 cm3 of 430,592,000 cm3
- Stored weight: 28,071.9 kg of 189,440.0 kg

## Demand scenarios

| scenario | orders | lines |
|---|---|---|
| A | 5415 | 20000 |
| B | 5420 | 20000 |
| C | 5409 | 20000 |

## ABC classification

| scenario | SKUs | A | B | C |
|---|---|---|---|---|
| A | 1200 | 294 | 444 | 462 |
| B | 1200 | 856 | 235 | 109 |
| C | 1200 | 861 | 236 | 103 |

## Checks

| check | severity | count |
|---|---|---|

## Assumptions

- All rows are synthetic or derived; no measured warehouse data is present.
- Volume is the full bounding-box volume; packing clearance is not modelled.
- One SKU per bin; a SKU may occupy several bins.
- Demand scenarios are alternative histories, not additive.
