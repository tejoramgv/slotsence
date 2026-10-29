"""Build or validate the benchmark dataset.

Examples:
    python app/build_dataset.py
    python app/build_dataset.py --config configs/dataset.json --seed 42
    python app/build_dataset.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.pipeline import DEFAULT_CONFIG, DEFAULT_OUTPUT, build_dataset
from src.data_processing.validate import run_validation, write_validation_reports


def main() -> None:
    parser = argparse.ArgumentParser(prog="build-dataset")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        output = Path(args.output).resolve()
        result = run_validation(output)
        write_validation_reports(output, result)
        summary = {
            "output": str(output),
            "ok": result.ok,
            "hard_errors": result.hard_errors,
            "warnings": result.warnings,
        }
        print(json.dumps(summary, indent=2))
        raise SystemExit(0 if result.ok else 1)

    summary = build_dataset(args.config, args.output, args.seed)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["validation"]["ok"] else 1)


if __name__ == "__main__":
    main()
