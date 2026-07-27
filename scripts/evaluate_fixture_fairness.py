from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from parm_bench.choice_matching import matches_choice
from parm_bench.dataset import load_cases, validate_cases


FAIRNESS_EVALUATOR_VERSION = "fixture_fairness_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("predictions")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--finalize-dataset",
        action="store_true",
        help="Record a passing fairness gate in construction provenance.",
    )
    args = parser.parse_args()
    cases = load_cases(args.dataset_dir)
    validate_cases(cases, include_profile=False)
    case_by_id = {case["case_id"]: case for case in cases}
    predictions = {
        row["case_id"]: row for row in _read_jsonl(Path(args.predictions))
    }
    if set(predictions) != set(case_by_id):
        missing = sorted(set(case_by_id) - set(predictions))
        extra = sorted(set(predictions) - set(case_by_id))
        raise ValueError(
            f"prediction IDs differ; missing={missing}, extra={extra}"
        )
    rows: list[dict[str, Any]] = []
    by_base: dict[str, list[bool]] = {}
    for case in cases:
        prediction = predictions[case["case_id"]]
        output_choice = case["decisions"]["output_only"]["choice"]
        memory_choice = case["decisions"]["memory_conditioned"]["choice"]
        desired = (
            memory_choice
            if case["variant"] == "memory-included"
            else output_choice
        )
        alternative = (
            output_choice
            if desired == memory_choice
            else memory_choice
        )
        desired_present = matches_choice(prediction["response_text"], desired)
        alternative_present = (
            matches_choice(prediction["response_text"], alternative)
            if alternative != desired
            else False
        )
        passed = desired_present and not alternative_present
        by_base.setdefault(case["base_case_id"], []).append(passed)
        rows.append(
            {
                "case_id": case["case_id"],
                "base_case_id": case["base_case_id"],
                "variant": case["variant"],
                "desired_choice": desired,
                "alternative_choice": alternative,
                "response_text": prediction["response_text"],
                "passed": passed,
                "provider_response_id": prediction.get("provider_response_id"),
                "resolved_model": prediction.get("resolved_model"),
            }
        )
    passed_scenarios = sum(all(values) for values in by_base.values())
    report = {
        "evaluator_version": FAIRNESS_EVALUATOR_VERSION,
        "dataset_dir": str(Path(args.dataset_dir)),
        "predictions": str(Path(args.predictions)),
        "cases": len(cases),
        "passed_cases": sum(row["passed"] for row in rows),
        "scenarios": len(by_base),
        "passed_scenarios": passed_scenarios,
        "failed_scenarios": sorted(
            base_case_id
            for base_case_id, values in by_base.items()
            if not all(values)
        ),
        "rows": rows,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.finalize_dataset:
        if passed_scenarios != len(by_base):
            raise ValueError("cannot finalize a dataset with failed scenarios")
        _finalize_dataset(
            Path(args.dataset_dir),
            report_path=path,
            predictions_path=Path(args.predictions),
            report=report,
        )
        validate_cases(cases)
    print(
        f"fixture fairness: {passed_scenarios}/{len(by_base)} scenarios, "
        f"{report['passed_cases']}/{len(cases)} cases"
    )
    return 0 if passed_scenarios == len(by_base) else 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finalize_dataset(
    dataset_dir: Path,
    *,
    report_path: Path,
    predictions_path: Path,
    report: dict[str, Any],
) -> None:
    construction_path = dataset_dir / "construction_records.jsonl"
    manifest_path = dataset_dir / "dataset_manifest.json"
    construction_rows = _read_jsonl(construction_path)
    report_rows = {
        row["base_case_id"]: row for row in report["rows"]
    }
    for row in construction_rows:
        base_case_id = row["base_case_id"]
        scenario_rows = [
            item
            for item in report["rows"]
            if item["base_case_id"] == base_case_id
        ]
        if len(scenario_rows) != 3 or not all(
            item["passed"] for item in scenario_rows
        ):
            raise ValueError(
                f"{base_case_id}: fairness report is not a passing triplet"
            )
        acceptance = row["acceptance"]
        acceptance["status"] = "accepted"
        acceptance["fairness_status"] = "passed"
        acceptance["fairness_evaluator_version"] = FAIRNESS_EVALUATOR_VERSION
        acceptance["fairness_case_ids"] = sorted(
            item["case_id"] for item in scenario_rows
        )
    construction_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in construction_rows
        ),
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["construction_records"]["sha256"] = _sha256(construction_path)
    manifest["fairness_gate"] = {
        "evaluator_version": FAIRNESS_EVALUATOR_VERSION,
        "path": _relative_path(dataset_dir, report_path),
        "sha256": _sha256(report_path),
        "predictions_path": _relative_path(dataset_dir, predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "passed_cases": report["passed_cases"],
        "passed_scenarios": report["passed_scenarios"],
        "resolved_models": sorted(
            {
                str(row["resolved_model"])
                for row in report_rows.values()
                if row.get("resolved_model")
            }
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _relative_path(root: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve()).replace("\\", "/")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
