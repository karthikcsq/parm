"""Render the PARMBench Workflows comparison ladder as a Markdown table."""

from __future__ import annotations

import json
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


def main() -> None:
    rows = []
    for name in ORDER:
        path = RESULTS / f"{name}.metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        per_case = {row["case_id"]: row for row in metrics["rows"]}
        rows.append((name, metrics, per_case))

    print("| Condition | Positive decision | Control decision | Gold admitted | "
          "Spurious | Timely | Workflow |")
    print("| --- | --- | --- | ---: | ---: | --- | ---: |")
    for name, metrics, per_case in rows:
        positive = _row(per_case, "positive")
        control = _row(per_case, "cue-ablated")
        print(
            f"| `{name}` "
            f"| {_mark(positive['decisive_success'])} "
            f"| {_mark(control['decisive_success'])} "
            f"| {positive['gold_admitted_count']} "
            f"| {positive['spurious_admitted_count']}/{control['spurious_admitted_count']} "
            f"| {_mark(positive['timely_gold_admission'])} "
            f"| {metrics['workflow_completion_rate']:.2f} |"
        )

    print()
    print("Per-case detail:")
    for name, _metrics, per_case in rows:
        for case_id, row in sorted(per_case.items()):
            failures = ", ".join(row["decisive_failures"]) or "-"
            print(
                f"  {name:<22} {row['variant']:<16} "
                f"decisive={row['decisive_success']!s:<5} "
                f"cue_step={row['cue_step']} "
                f"action_step={row['decisive_action_step']} "
                f"gold_step={row['first_gold_admission_step']} "
                f"steps={row['step_count']} "
                f"stop={row['stopped_reason']} "
                f"failed=[{failures}]"
            )


def _row(per_case: dict, variant: str) -> dict:
    for row in per_case.values():
        if row["variant"] == variant:
            return row
    return {
        "decisive_success": False,
        "gold_admitted_count": 0,
        "spurious_admitted_count": 0,
        "timely_gold_admission": False,
    }


def _mark(value: bool) -> str:
    return "pass" if value else "fail"


if __name__ == "__main__":
    main()
