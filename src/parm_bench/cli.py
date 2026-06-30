from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .amara import normalize_amara
from .backends import GBrainBackend
from .baselines import BaselineNotImplementedError, get_baseline
from .dataset import DatasetValidationError, load_cases, validate_cases
from .scoring import score_predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parm-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-amara")
    prepare.add_argument("--source", default="data/amara-life-v1/source")
    prepare.add_argument("--out", default=".gbrain-local/corpus/amara-life-v1")

    validate = commands.add_parser("validate")
    validate.add_argument("dataset_dir")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("dataset_dir")
    inspect.add_argument("--case", dest="case_id")

    run = commands.add_parser("run")
    run.add_argument("dataset_dir")
    run.add_argument("--baseline", required=True)
    run.add_argument("--memory-backend", choices=("gbrain",), default="gbrain")
    run.add_argument("--out", required=True)

    score = commands.add_parser("score")
    score.add_argument("results_jsonl")
    score.add_argument("--gold", required=True)
    score.add_argument("--out")

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-amara":
            count = normalize_amara(args.source, args.out)
            print(f"Prepared {count} Amara pages in {args.out}")
            return 0
        if args.command == "validate":
            cases = load_cases(args.dataset_dir)
            validate_cases(cases)
            print(f"Validated {len(cases)} cases")
            return 0
        if args.command == "inspect":
            return _inspect(args.dataset_dir, args.case_id)
        if args.command == "run":
            return _run(args.dataset_dir, args.baseline, args.memory_backend, args.out)
        if args.command == "score":
            return _score(args.results_jsonl, args.gold, args.out)
    except DatasetValidationError as exc:
        for issue in exc.issues:
            print(issue, file=sys.stderr)
        return 1
    except BaselineNotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


def _inspect(dataset_dir: str, case_id: str | None) -> int:
    cases = load_cases(dataset_dir)
    validate_cases(cases)
    case = next(
        (item for item in cases if case_id is None or item["case_id"] == case_id),
        None,
    )
    if case is None:
        print(f"Case not found: {case_id}", file=sys.stderr)
        return 1
    print(json.dumps({key: value for key, value in case.items() if not key.startswith("_") and key != "observation_text"}, indent=2))
    print(f"\nObservation tokens are validated; content: {case['observation']['content_path']}")
    return 0


def _run(dataset_dir: str, baseline: str, backend_name: str, out: str) -> int:
    cases = load_cases(dataset_dir)
    validate_cases(cases)
    implementation = get_baseline(baseline)
    backend = GBrainBackend()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            row = implementation.run(case, backend)
            row["baseline"] = baseline
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(cases)} predictions to {path}")
    return 0


def _score(results_path: str, gold: str, out: str | None) -> int:
    cases = load_cases(gold)
    validate_cases(cases)
    predictions = _load_jsonl(Path(results_path))
    metrics = score_predictions(cases, predictions)
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote metrics to {path}")
    else:
        print(payload)
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
