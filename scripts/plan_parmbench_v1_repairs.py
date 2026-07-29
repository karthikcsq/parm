"""Turn a fairness report into the tracked repair and drop decisions.

Every scenario that misses the criteria 5 to 7 pattern is either sent back to
the construction model for one more attempt or, once it has used its attempts,
dropped with the reason it kept failing. The result is written to
`data/parmbench-v1-supply/fairness_repairs.json`, which the builder reads, so
the repaired batch rebuilds from the tracked file rather than from a transcript.

Usage:

    $env:PYTHONPATH = 'src'
    & 'C:\\Users\\karth\\anaconda3\\python.exe' \\
      scripts\\plan_parmbench_v1_repairs.py \\
      data\\benchmark-results\\parmbench-v1-fairness\\fairness-pass-2.json \\
      --max-attempts 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPAIRS_PATH = ROOT / "data" / "parmbench-v1-supply" / "fairness_repairs.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--repairs-path", default=str(REPAIRS_PATH))
    args = parser.parse_args()

    path = Path(args.repairs_path)
    payload: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"repairs": {}, "drops": []}
    )
    repairs: dict[str, Any] = dict(payload.get("repairs", {}))
    drops: list[dict[str, str]] = list(payload.get("drops", []))
    dropped_ids = {item["base_case_id"] for item in drops}

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    failing = {
        item["base_case_id"]: item for item in report["failed_scenarios"]
    }
    passing = {
        row["base_case_id"]
        for row in report["scenario_table"]
        if row["passed"]
    }

    # A scenario that now passes keeps the attempt that fixed it and stops
    # being a candidate for further repair.
    added, escalated, newly_dropped = 0, 0, 0
    for base_case_id, item in sorted(failing.items()):
        if base_case_id in dropped_ids:
            continue
        attempt = int(repairs.get(base_case_id, {}).get("attempt", 0))
        if attempt >= args.max_attempts:
            repairs.pop(base_case_id, None)
            drops.append(
                {
                    "base_case_id": base_case_id,
                    "reason": (
                        "fairness_unrepaired_after_"
                        f"{attempt}_attempts: "
                        + ",".join(item["diagnosis"])
                    ),
                }
            )
            dropped_ids.add(base_case_id)
            newly_dropped += 1
            continue
        repairs[base_case_id] = {
            "attempt": attempt + 1,
            "failed_variants": item["failed_variants"],
        }
        if attempt:
            escalated += 1
        else:
            added += 1

    payload = {
        "repairs": dict(sorted(repairs.items())),
        "drops": sorted(drops, key=lambda item: item["base_case_id"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "passing_scenarios": len(passing),
                "first_attempt": added,
                "second_attempt": escalated,
                "newly_dropped": newly_dropped,
                "open_repairs": len(payload["repairs"]),
                "total_drops": len(payload["drops"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
