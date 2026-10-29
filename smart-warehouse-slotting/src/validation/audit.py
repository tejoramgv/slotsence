"""Constraint audit export helpers.

Two exports are produced from the same ``ValidationResult``:

- ``constraint_audit.csv`` — verdict rows (one PASS row per assignment when clean, or one
  row per violation when not). Kept for backwards compatibility with earlier phases.
- ``validation_checks.csv`` — every evaluated rule instance with its observed value,
  allowed limit, unit, and explanation. This is the audit the dashboard displays and
  exports, so pass/fail detail always matches the validation that ran.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .validator import ValidationResult, audit_assignments

CHECK_FIELDS = [
    "rule_id",
    "rule_name",
    "rule_group",
    "passed",
    "severity",
    "sku_id",
    "location_id",
    "observed",
    "limit",
    "unit",
    "explanation",
]


def write_constraint_audit(
    assignments,
    result: ValidationResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sku_id", "location_id", "rule_id", "rule_name", "severity", "passed", "reason"],
        )
        writer.writeheader()
        for row in audit_assignments(assignments, result):
            writer.writerow(row)


def write_validation_checks(result: ValidationResult, path: Path) -> None:
    """Write every rule evaluation (pass and fail) with observed/limit values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHECK_FIELDS)
        writer.writeheader()
        for check in result.checks:
            writer.writerow(check.to_dict())
