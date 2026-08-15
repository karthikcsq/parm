"""Render the PARMBench Workflows comparison ladder as a Markdown table.

Reports each condition as a count over independent samples rather than a single
pass/fail, because one agent run does not distinguish a policy effect from
run-to-run variance. A condition that passes twice out of three is reported as
2/3, not as "passes".
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "benchmark-results" / "workflows-v1"
ORDER = (
    "no_memory",
    "input_rag",
    "naive_output_rag",
    "all_entity_output_rag",
    "prompted_memory_tool",
    "parm",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="tier-100")
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument(
        "--dataset",
        help="optional dataset label to include in the rendered report",
    )
    parser.add_argument("--detail", action="store_true")
    arguments = parser.parse_args()
    root = Path(arguments.results) / arguments.tier

    conditions = []
    for name in ORDER:
        samples = sorted(root.glob(f"{name}.sample*.metrics.json"))
        if not samples:
            continue
        conditions.append(
            (name, [json.loads(path.read_text(encoding="utf-8")) for path in samples])
        )
    if not conditions:
        print(f"no results under {root}")
        return 1

    if arguments.dataset:
        print(f"Dataset: {arguments.dataset}\n")

    total = len(conditions[0][1])
    scenarios = sorted(
        {
            str(row.get("base_case_id", "unknown"))
            for _name, runs in conditions
            for run in runs
            for row in run["rows"]
        }
    )
    for scenario in scenarios:
        print(f"### {scenario}\n")
        print(f"{arguments.tier}, {total} samples per condition\n")
        print(
            "| Condition | Positive | Control | Ceiling | Gold | Spurious | "
            "Timely | Memory tokens |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name, runs in conditions:
            positives = [_variant(run, "positive", scenario) for run in runs]
            controls = [_variant(run, "cue-ablated", scenario) for run in runs]
            ceilings = [_variant(run, "memory-included", scenario) for run in runs]
            print(
                f"| `{name}` "
                f"| {_count(positives, 'decisive_success')}/{len(runs)} "
                f"| {_count(controls, 'decisive_success')}/{len(runs)} "
                f"| {_count(ceilings, 'decisive_success')}/{len(runs)} "
                f"| {_mean(positives, 'gold_admitted_count'):.1f} "
                f"| {_mean(positives, 'spurious_admitted_count'):.1f} "
                f"| {_count(positives, 'timely_gold_admission')}/{len(runs)} "
                f"| {_mean(positives, 'injected_memory_tokens'):,.0f} |"
            )
        print()

    print(
        "Positive and control are counts of samples whose decisive assertions "
        "all passed. Gold, spurious, and memory tokens are means over the "
        "positive samples. Memory tokens are per trajectory and cover every "
        "scenario the run touched, so they are not attributable to one "
        "scenario when several ran together."
    )

    if arguments.detail:
        print("\nPer-sample detail:")
        for name, runs in conditions:
            for index, run in enumerate(runs):
                for row in sorted(
                    run["rows"],
                    key=lambda item: (item.get("base_case_id", ""), item["variant"]),
                ):
                    failures = ", ".join(row["decisive_failures"]) or "-"
                    scenario = str(row.get("base_case_id", "")).rsplit("-", 2)[-1]
                    print(
                        f"  {name:<22} s{index} {scenario:<11} {row['variant']:<16} "
                        f"decisive={row['decisive_success']!s:<5} "
                        f"cue={row['cue_step']} act={row['decisive_action_step']} "
                        f"gold={row['first_gold_admission_step']} "
                        f"adm={row['admitted_count']} "
                        f"tok={row['injected_memory_tokens']} "
                        f"steps={row['step_count']} stop={row['stopped_reason']} "
                        f"failed=[{failures}]"
                    )
    return 0


def _variant(run: dict, variant: str, scenario: str | None = None) -> dict:
    for row in run["rows"]:
        if row["variant"] != variant:
            continue
        if scenario is not None and str(row.get("base_case_id")) != scenario:
            continue
        return row
    return {}


def _count(rows: list[dict], key: str) -> int:
    return sum(1 for row in rows if row.get(key))


def _mean(rows: list[dict], key: str) -> float:
    values = [row.get(key, 0) for row in rows]
    return statistics.mean(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
