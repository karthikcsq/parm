"""Summarize natural Email + Calendar v2 semantic-outcome rates.

Reads ``<condition>.sample*.metrics.json`` files produced by ``workflow score``.
It reports the positive, cue-ablated control, and memory-included oracle rates
per scenario and in aggregate.  These are descriptive sample rates, not a
pass/fail gate: compare fresh independent samples against a threshold
preregistered before making a retrieval-effect claim.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


INTERPRETATION = (
    "Rate comparisons require fresh independent samples and an explicit "
    "preregistered threshold before claims; do not treat a fixed 3/3 result "
    "as a gate."
)
VARIANTS = {"positive": "positive", "cue-ablated": "control", "memory-included": "oracle"}


def rate(rows: list[bool]) -> dict[str, int | float]:
    successes = sum(rows)
    samples = len(rows)
    return {"successes": successes, "samples": samples, "rate": successes / samples if samples else 0.0}


def build_report(
    runs: dict[str, list[dict[str, Any]]], *, baseline: str = "no_memory", parm: str = "parm"
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, list[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for condition, samples in runs.items():
        for sample in samples:
            for row in sample.get("rows", []):
                label = VARIANTS.get(str(row.get("variant")))
                if label:
                    grouped[str(row.get("base_case_id"))][condition][label].append(bool(row.get("decisive_success")))

    per_triplet: dict[str, dict[str, dict[str, Any]]] = {}
    aggregate_rows: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for scenario, conditions in sorted(grouped.items()):
        per_triplet[scenario] = {}
        for condition, variants in sorted(conditions.items()):
            summary: dict[str, Any] = {name: rate(variants[name]) for name in ("positive", "control", "oracle")}
            summary["cue_triggered_lift"] = summary["positive"]["rate"] - summary["control"]["rate"]
            per_triplet[scenario][condition] = summary
            for name, values in variants.items():
                aggregate_rows[condition][name].extend(values)

    aggregate: dict[str, Any] = {}
    for condition, variants in sorted(aggregate_rows.items()):
        summary: dict[str, Any] = {name: rate(variants[name]) for name in ("positive", "control", "oracle")}
        summary["cue_triggered_lift"] = summary["positive"]["rate"] - summary["control"]["rate"]
        aggregate[condition] = summary
    if baseline in aggregate and parm in aggregate:
        aggregate["parm_minus_no_memory_positive"] = (
            aggregate[parm]["positive"]["rate"] - aggregate[baseline]["positive"]["rate"]
        )
        aggregate["parm_minus_no_memory_cue_triggered_lift"] = (
            aggregate[parm]["cue_triggered_lift"] - aggregate[baseline]["cue_triggered_lift"]
        )
    return {"interpretation": INTERPRETATION, "per_triplet": per_triplet, "aggregate": aggregate}


def read_runs(root: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("*.sample*.metrics.json")):
        condition = path.name.split(".sample", 1)[0]
        runs.setdefault(condition, []).append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="directory containing <condition>.sample*.metrics.json")
    parser.add_argument("--baseline", default="no_memory")
    parser.add_argument("--parm", default="parm")
    parser.add_argument("--out", help="optional JSON output path; otherwise print JSON")
    args = parser.parse_args()
    report = build_report(read_runs(Path(args.results)), baseline=args.baseline, parm=args.parm)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
