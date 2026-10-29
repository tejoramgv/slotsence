"""Phase-2 dataset tests: reproducibility, integrity, demand scenarios, corruption handling."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from src.data_processing.pipeline import build_dataset
from src.data_processing.schema import TABLES
from src.data_processing.validate import run_validation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "dataset.json"


def _reduced_config() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg["n_skus"] = 120
    cfg["n_order_lines_per_scenario"] = 1000
    return cfg


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset") / "data"
    summary = build_dataset(_reduced_config(), out)
    return out, summary


def test_build_is_reproducible(built, tmp_path: Path) -> None:
    out, _ = built
    repeat = tmp_path / "repeat"
    build_dataset(_reduced_config(), repeat)
    for table in TABLES:
        name = f"{table}.csv"
        assert _sha256(out / "processed" / name) == _sha256(repeat / "processed" / name), name


def test_validation_passes(built) -> None:
    out, summary = built
    result = run_validation(out)
    assert result.ok
    assert result.hard_errors == 0
    assert result.stats["row_counts"] == {
        table: len(_read_csv(out / "processed" / f"{table}.csv")) for table in TABLES
    }
    assert summary["validation"]["ok"] is True


def test_scale_and_scenarios(built) -> None:
    out, summary = built
    counts = summary["counts"]
    assert counts["products"] == 120
    assert 500 <= counts["locations"] <= 2500
    assert counts["categories"] >= 8
    assert counts["demand_summary"] == 3 * counts["products"]

    orders = _read_csv(out / "processed" / "orders.csv")
    items = _read_csv(out / "processed" / "order_items.csv")
    scenario_by_order = {row["order_id"]: row["scenario_id"] for row in orders}
    lines = Counter(scenario_by_order[item["order_id"]] for item in items)
    assert set(scenario_by_order.values()) == {"A", "B", "C"}
    assert lines == {"A": 1000, "B": 1000, "C": 1000}


def test_abc_partition(built) -> None:
    out, _ = built
    demand = _read_csv(out / "processed" / "demand_summary.csv")
    for scenario in ("A", "B", "C"):
        rows = sorted(
            (row for row in demand if row["scenario_id"] == scenario),
            key=lambda row: int(row["rank"]),
        )
        assert len(rows) == 120
        assert [int(row["rank"]) for row in rows] == list(range(1, 121))
        assert {row["abc_class"] for row in rows} <= {"A", "B", "C"}
        sequence = [row["abc_class"] for row in rows]
        assert sequence == sorted(sequence)
        class_max = {
            cls: max(float(row["cumulative_share"]) for row in rows if row["abc_class"] == cls)
            for cls in ("A", "B", "C")
        }
        assert class_max["A"] >= 0.80 - 1e-6
        assert class_max["B"] >= 0.95 - 1e-6
        assert abs(float(rows[-1]["cumulative_share"]) - 1.0) < 1e-5


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def test_scenario_c_has_strongest_affinity(built) -> None:
    out, _ = built
    affinity = _read_csv(out / "processed" / "sku_affinity.csv")

    def pairs(co_count: int = 3, scenario: str = "C") -> list[dict[str, str]]:
        return [
            row
            for row in affinity
            if row["scenario_id"] == scenario and int(row["co_count"]) >= co_count
        ]

    strong_c = pairs()
    assert len(strong_c) >= 20
    assert _median([float(row["lift"]) for row in strong_c]) > 2.0
    assert _median([float(row["confidence_a_to_b"]) for row in strong_c]) > 0.3

    strong_a = pairs(scenario="A")
    assert strong_a
    assert _median([float(row["lift"]) for row in strong_c]) > _median(
        [float(row["lift"]) for row in strong_a]
    )
    assert _median([float(row["confidence_a_to_b"]) for row in strong_c]) > _median(
        [float(row["confidence_a_to_b"]) for row in strong_a]
    )


# ---------------------------------------------------------------------------
# Corrupt one table at a time and require the validator to report it.
# ---------------------------------------------------------------------------

def _copy_dataset(out: Path, tmp_path: Path) -> Path:
    target = tmp_path / "dataset"
    shutil.copytree(out, target)
    return target


def _rewrite(path: Path, mutate) -> None:
    rows = _read_csv(path)
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _has_error(result, table: str, check: str) -> bool:
    return any(error["table"] == table and error["check"] == check for error in result.errors)


def test_validator_flags_negative_quantity(built, tmp_path: Path) -> None:
    out, _ = built
    target = _copy_dataset(out, tmp_path)
    _rewrite(target / "processed" / "order_items.csv", lambda rows: rows[0].update(quantity="-3"))
    result = run_validation(target)
    assert not result.ok
    assert _has_error(result, "order_items", "value_range")


def test_validator_flags_duplicate_location(built, tmp_path: Path) -> None:
    out, _ = built
    target = _copy_dataset(out, tmp_path)

    def duplicate(rows):
        rows[1]["location_id"] = rows[0]["location_id"]

    _rewrite(target / "processed" / "locations.csv", duplicate)
    result = run_validation(target)
    assert not result.ok
    assert _has_error(result, "locations", "unique_key")


def test_validator_flags_invalid_timestamp(built, tmp_path: Path) -> None:
    out, _ = built
    target = _copy_dataset(out, tmp_path)
    _rewrite(
        target / "processed" / "orders.csv",
        lambda rows: rows[0].update(order_timestamp="not-a-timestamp"),
    )
    result = run_validation(target)
    assert not result.ok
    assert _has_error(result, "orders", "timestamp")


def test_validator_flags_dangling_sku(built, tmp_path: Path) -> None:
    out, _ = built
    target = _copy_dataset(out, tmp_path)
    _rewrite(
        target / "processed" / "order_items.csv",
        lambda rows: rows[0].update(sku_id="SKU999999"),
    )
    result = run_validation(target)
    assert not result.ok
    assert _has_error(result, "order_items", "foreign_key")


def test_validator_flags_missing_assignment(built, tmp_path: Path) -> None:
    out, _ = built
    target = _copy_dataset(out, tmp_path)

    def drop_sku(rows):
        sku_id = rows[0]["sku_id"]
        rows[:] = [row for row in rows if row["sku_id"] != sku_id]

    _rewrite(target / "processed" / "storage_assignments_baseline.csv", drop_sku)
    result = run_validation(target)
    assert not result.ok
    assert _has_error(result, "products", "assignment_complete")
