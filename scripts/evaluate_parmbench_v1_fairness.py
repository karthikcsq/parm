"""Score the `no_memory` fixture-fairness sweep for `benchmark_parmbench_v1`.

Acceptance criteria 5, 6, and 7 of the construction contract are established by
one run of the `no_memory` baseline over all three variants of every scenario.
The required pattern is:

- positive: the output-only choice, because the cue alone must not give the
  memory away;
- cue-ablated: the output-only choice, because removing the affordance leaves
  ordinary visible evidence in charge; and
- memory-included: the memory-conditioned choice, because a model handed the
  personal fact must be able to act on it.

A scenario passes only when all three variants land on their required choice
and on nothing else. Anything short of that is reported with the variant that
broke and a diagnosis, which is what a repair pass reads.

Usage:

    $env:PYTHONPATH = 'src'
    & 'C:\\Users\\karth\\anaconda3\\python.exe' \\
      scripts\\evaluate_parmbench_v1_fairness.py data\\benchmark_parmbench_v1 \\
      data\\benchmark-results\\parmbench-v1-fairness\\no-memory.jsonl \\
      --out data\\benchmark-results\\parmbench-v1-fairness\\fairness.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from parm_bench.choice_matching import matches_choice
from parm_bench.dataset import load_cases, validate_cases


FAIRNESS_EVALUATOR_VERSION = "parmbench_fairness_v1"

REQUIRED_ROLE = {
    "positive": "output_only",
    "cue-ablated": "output_only",
    "memory-included": "memory_conditioned",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("predictions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--table")
    parser.add_argument(
        "--finalize-dataset",
        action="store_true",
        help="Record the fairness gate in dataset_manifest.json.",
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset_dir)
    validate_cases(cases, include_profile=False)
    predictions = {
        row["case_id"]: row for row in _read_jsonl(Path(args.predictions))
    }
    case_by_id = {case["case_id"]: case for case in cases}
    missing = sorted(set(case_by_id) - set(predictions))
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} cases: {missing[:5]}")

    # The control arm declares its memory-conditioned choice as the output-only
    # winner, so the scenario's real target is read off the positive arm and
    # shared across the triplet.
    targets = {
        case["base_case_id"]: case["decisions"]["memory_conditioned"]["choice"]
        for case in cases
        if case["variant"] == "positive"
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(
            _score_case(
                case,
                predictions[case["case_id"]],
                targets[case["base_case_id"]],
            )
        )

    scenarios: dict[str, dict[str, Any]] = {}
    for row in rows:
        scenario = scenarios.setdefault(
            row["base_case_id"],
            {"base_case_id": row["base_case_id"], "variants": {}},
        )
        scenario["variants"][row["variant"]] = row
    for scenario in scenarios.values():
        failures = [
            variant
            for variant in ("positive", "cue-ablated", "memory-included")
            if not scenario["variants"][variant]["passed"]
        ]
        scenario["passed"] = not failures
        scenario["failed_variants"] = failures
        scenario["diagnosis"] = [
            scenario["variants"][variant]["diagnosis"] for variant in failures
        ]

    passed_scenarios = sum(1 for item in scenarios.values() if item["passed"])
    by_variant = {
        variant: {
            "cases": sum(1 for row in rows if row["variant"] == variant),
            "passed": sum(
                1 for row in rows if row["variant"] == variant and row["passed"]
            ),
        }
        for variant in ("positive", "cue-ablated", "memory-included")
    }
    report = {
        "evaluator_version": FAIRNESS_EVALUATOR_VERSION,
        "dataset_dir": str(Path(args.dataset_dir)),
        "predictions": str(Path(args.predictions)),
        "cases": len(rows),
        "passed_cases": sum(1 for row in rows if row["passed"]),
        "scenarios": len(scenarios),
        "passed_scenarios": passed_scenarios,
        "by_variant": by_variant,
        "diagnosis_counts": dict(
            sorted(
                Counter(
                    row["diagnosis"] for row in rows if not row["passed"]
                ).items()
            )
        ),
        "failed_scenarios": [
            {
                "base_case_id": item["base_case_id"],
                "failed_variants": item["failed_variants"],
                "diagnosis": item["diagnosis"],
            }
            for item in sorted(scenarios.values(), key=lambda x: x["base_case_id"])
            if not item["passed"]
        ],
        "scenario_table": [
            {
                "base_case_id": item["base_case_id"],
                "passed": item["passed"],
                "output_only_choice": item["variants"]["positive"]["desired_choice"],
                "memory_conditioned_choice": item["variants"]["memory-included"][
                    "desired_choice"
                ],
                **{
                    f"{variant}_passed": item["variants"][variant]["passed"]
                    for variant in ("positive", "cue-ablated", "memory-included")
                },
                **{
                    f"{variant}_diagnosis": item["variants"][variant]["diagnosis"]
                    for variant in ("positive", "cue-ablated", "memory-included")
                },
            }
            for item in sorted(scenarios.values(), key=lambda x: x["base_case_id"])
        ],
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.table:
        _write_table(Path(args.table), report["scenario_table"])
    if args.finalize_dataset:
        if passed_scenarios != len(scenarios):
            raise ValueError("cannot finalize a dataset with failed scenarios")
        _finalize_dataset(
            Path(args.dataset_dir),
            report_path=out_path,
            predictions_path=Path(args.predictions),
            report=report,
            rows=rows,
        )
        validate_cases(load_cases(args.dataset_dir))

    print(
        f"parmbench fairness: {passed_scenarios}/{len(scenarios)} scenarios, "
        f"{report['passed_cases']}/{len(rows)} cases"
    )
    for variant, counts in by_variant.items():
        print(f"  {variant}: {counts['passed']}/{counts['cases']}")
    return 0 if passed_scenarios == len(scenarios) else 1


def _score_case(
    case: dict[str, Any], prediction: dict[str, Any], target: str
) -> dict[str, Any]:
    variant = case["variant"]
    winner = case["decisions"]["output_only"]["choice"]
    desired = winner if REQUIRED_ROLE[variant] == "output_only" else target
    response = prediction.get("response_text", "")
    chose_winner = matches_choice(response, winner)
    chose_target = target != winner and matches_choice(response, target)
    chose_desired = chose_winner if desired == winner else chose_target
    chose_other = chose_target if desired == winner else chose_winner
    passed = bool(chose_desired and not chose_other)
    return {
        "case_id": case["case_id"],
        "base_case_id": case["base_case_id"],
        "variant": variant,
        "desired_choice": desired,
        "output_only_choice": winner,
        "memory_conditioned_choice": target,
        "chose_output_only": bool(chose_winner),
        "chose_memory_conditioned": bool(chose_target),
        "response_text": response,
        "passed": passed,
        "diagnosis": _diagnose(variant, passed, bool(chose_winner), bool(chose_target)),
        "provider_response_id": prediction.get("provider_response_id"),
        "resolved_model": prediction.get("resolved_model"),
        "truncated": bool(prediction.get("truncated")),
    }


def _diagnose(
    variant: str, passed: bool, chose_winner: bool, chose_target: bool
) -> str:
    if passed:
        return "ok"
    if variant == "positive":
        if chose_target:
            return "cue_too_strong_without_memory"
        return "positive_chose_neither_declared_option"
    if variant == "cue-ablated":
        if chose_target:
            return "ordinary_evidence_too_weak_in_control"
        return "control_chose_neither_declared_option"
    if chose_winner and not chose_target:
        return "ceiling_memory_not_actionable"
    if chose_winner and chose_target:
        return "ceiling_named_both_options"
    return "ceiling_chose_neither_declared_option"


def _write_table(path: Path, table: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| scenario | positive | cue-ablated | memory-included | scenario |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    def mark(value: bool) -> str:
        return "pass" if value else "FAIL"

    lines = [
        f"| {row['base_case_id']} | {mark(row['positive_passed'])} "
        f"| {mark(row['cue-ablated_passed'])} "
        f"| {mark(row['memory-included_passed'])} "
        f"| {mark(row['passed'])} |"
        for row in table
    ]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _finalize_dataset(
    dataset_dir: Path,
    *,
    report_path: Path,
    predictions_path: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fairness_gate"] = {
        "evaluator_version": FAIRNESS_EVALUATOR_VERSION,
        "baseline": "no_memory",
        "path": _relative_path(dataset_dir, report_path),
        "sha256": _sha256(report_path),
        "predictions_path": _relative_path(dataset_dir, predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "response_cache": _relative_path(
            dataset_dir, dataset_dir.parent / "response-caches" / "parmbench-v1-fairness"
        ),
        "cases": report["cases"],
        "passed_cases": report["passed_cases"],
        "scenarios": report["scenarios"],
        "passed_scenarios": report["passed_scenarios"],
        "resolved_models": sorted(
            {str(row["resolved_model"]) for row in rows if row.get("resolved_model")}
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _relative_path(root: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve()).replace("\\", "/")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
