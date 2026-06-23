from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .baselines import available_baselines
from .dataset import DatasetValidationError, load_cases, validate_cases, write_cases
from .scoring import score_predictions
from .synthetic import generate_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parm-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate the synthetic V1 dataset.")
    generate_parser.add_argument("dataset_dir")
    generate_parser.add_argument("--count", type=int, default=50)

    validate_parser = subparsers.add_parser("validate", help="Validate a dataset directory.")
    validate_parser.add_argument("dataset_dir")

    inspect_parser = subparsers.add_parser("inspect", help="Pretty-print one benchmark case for review.")
    inspect_parser.add_argument("dataset_dir")
    inspect_parser.add_argument("--case", dest="case_id", help="Case ID to inspect. Defaults to the first case.")

    run_parser = subparsers.add_parser("run", help="Run a baseline over a dataset.")
    run_parser.add_argument("dataset_dir")
    run_parser.add_argument("--baseline", required=True, choices=sorted(available_baselines()))
    run_parser.add_argument("--out", required=True)

    score_parser = subparsers.add_parser("score", help="Score baseline predictions.")
    score_parser.add_argument("results_jsonl")
    score_parser.add_argument("--gold", required=True)
    score_parser.add_argument("--out")

    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            return _generate(args.dataset_dir, args.count)
        if args.command == "validate":
            return _validate(args.dataset_dir)
        if args.command == "inspect":
            return _inspect(args.dataset_dir, args.case_id)
        if args.command == "run":
            return _run(args.dataset_dir, args.baseline, args.out)
        if args.command == "score":
            return _score(args.results_jsonl, args.gold, args.out)
    except DatasetValidationError as exc:
        for issue in exc.issues:
            print(issue, file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _generate(dataset_dir: str, count: int) -> int:
    cases = generate_cases(count)
    validate_cases(cases)
    cases_path = write_cases(cases, dataset_dir)
    print(f"Wrote {len(cases)} cases to {cases_path}")
    return 0


def _validate(dataset_dir: str) -> int:
    cases = load_cases(dataset_dir)
    validate_cases(cases)
    print(f"Validated {len(cases)} cases")
    return 0


def _inspect(dataset_dir: str, case_id: str | None) -> int:
    cases = load_cases(dataset_dir)
    validate_cases(cases)
    case = _find_case(cases, case_id)
    if case is None:
        print(f"Case not found: {case_id}", file=sys.stderr)
        return 1
    print(_format_case(case))
    return 0


def _run(dataset_dir: str, baseline: str, out: str) -> int:
    cases = load_cases(dataset_dir)
    validate_cases(cases)
    predict = available_baselines()[baseline]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            prediction = predict(case)
            prediction["baseline"] = baseline
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")
    print(f"Wrote {len(cases)} predictions to {out_path}")
    return 0


def _find_case(cases: list[dict[str, Any]], case_id: str | None) -> dict[str, Any] | None:
    if case_id is None:
        return cases[0] if cases else None
    for case in cases:
        if case["case_id"] == case_id:
            return case
    return None


def _format_case(case: dict[str, Any]) -> str:
    expected = case["expected"]
    lines = [
        f"Case: {case['case_id']}",
        f"Goal family: {case['goal_family']}",
        f"Trigger source: {case['trigger_source']}",
        f"Join depth: {case['join_depth']}",
        f"Distractor type: {case['distractor_type']}",
        f"Actionability: {case['actionability']}",
        "",
        "User goal:",
        f"  {case['user_goal']}",
        "",
        "Assistant output:",
        f"  {case['assistant_output']}",
        "",
        "Tool response:",
        f"  {json.dumps(case['tool_response'], sort_keys=True)}",
        "",
        "Trigger entity:",
        f"  {case['trigger_entity']['node_id']} | {case['trigger_entity']['label']}",
        "",
        "Graph nodes:",
    ]
    for node in case["graph"]["nodes"]:
        lines.append(f"  {node['id']} | {node['type']} | {node['label']}")

    lines.extend(["", "Graph edges:"])
    for edge in case["graph"]["edges"]:
        validity = "valid" if edge.get("valid", True) else "invalid"
        lines.append(
            f"  {edge['id']} | {edge['source']} -[{edge['relation']}]-> {edge['target']} | {validity}"
        )

    lines.extend(
        [
            "",
            "Distractors:",
            f"  {', '.join(case.get('distractors', [])) or '(none)'}",
            "",
            "Gold path:",
            f"  {' -> '.join(expected['gold_path'])}",
            "",
            "Expected suggestion:",
            f"  {expected['suggestion']}",
            "",
            "Action keywords:",
            f"  {', '.join(expected.get('action_keywords', []))}",
        ]
    )
    return "\n".join(lines)


def _score(results_jsonl: str, gold: str, out: str | None) -> int:
    cases = load_cases(gold)
    validate_cases(cases)
    predictions = _load_jsonl(Path(results_jsonl))
    metrics = score_predictions(cases, predictions)
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote metrics to {out_path}")
    else:
        print(payload)
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
