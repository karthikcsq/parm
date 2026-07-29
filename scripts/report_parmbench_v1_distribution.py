"""Report the axis distribution of `data/benchmark_parmbench_v1`.

Construction diversity has to be measured, not eyeballed: the contract's
criterion 9 is about repetition a reader will not notice. This prints one
table per axis over base scenarios, then runs the shared
`parm_bench.construction_checks` detectors over the loaded cases.

    $env:PYTHONPATH = 'src'
    & 'C:\\Users\\karth\\anaconda3\\python.exe' \\
      scripts\\report_parmbench_v1_distribution.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from parm_bench.construction_checks import construction_issues
from parm_bench.dataset import load_cases

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_parmbench_v1_benchmark import prompt_claim_overlap  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "benchmark_parmbench_v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def table(title: str, counts: Counter[Any]) -> str:
    width = max((len(str(key)) for key in counts), default=0)
    lines = [f"{title} ({len(counts)} distinct, {sum(counts.values())} scenarios)"]
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
        lines.append(f"  {str(key).ljust(width)}  {count}")
    return "\n".join(lines)


def decile(value: float) -> str:
    index = min(9, max(0, int(value * 10)))
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET))
    args = parser.parse_args()
    root = Path(args.dataset)

    records = load_jsonl(root / "construction_records.jsonl")
    cases = load_cases(root)
    positives = [case for case in cases if case["variant"] == "positive"]

    print(f"base scenarios: {len(records)}")
    print(f"cases: {len(cases)}")
    print(f"personas: {len({record['persona_id'] for record in records})}")
    print()
    print(table("capability", Counter(r["capability"] for r in records)))
    print()
    print(table("envelope style", Counter(r["axes"]["envelope_style"] for r in records)))
    print()
    print(table("task domain", Counter(r["axes"]["domain"] for r in records)))
    print()
    print(
        table(
            "ordinary evidence mechanism",
            Counter(r["axes"]["ordinary_evidence_mechanism"] for r in records),
        )
    )
    print()
    print(
        table(
            "lexical overlap mode",
            Counter(r["axes"]["lexical_overlap_mode"] for r in records),
        )
    )
    print()
    print(
        table(
            "observation kind", Counter(r["axes"]["observation_kind"] for r in records)
        )
    )
    print()
    print(
        table("numeric ratings", Counter(r["axes"]["ratings_allowed"] for r in records))
    )
    print()
    print(
        table(
            "abstention pressure",
            Counter(bool(r["abstention_pressure"]) for r in records),
        )
    )
    print()
    print(table("evidence grade", Counter(r["evidence_grade"]["grade"] for r in records)))
    print()
    print(
        table(
            "drafted fact kind",
            Counter(case["provenance"]["fact_kind"] for case in positives),
        )
    )
    print()
    print(
        table(
            "control replacement repaired",
            Counter(bool(r.get("neutral_clause_repaired")) for r in records),
        )
    )
    print()
    print(
        table(
            "cue position decile",
            Counter(decile(r["metrics"]["cue_char_fraction"]) for r in records),
        )
    )
    print()
    print(table("decoys per scenario", Counter(r["axes"]["decoy_count"] for r in records)))
    print()
    print(
        table(
            "memory distractors per scenario",
            Counter(r["axes"]["distractor_count"] for r in records),
        )
    )
    print()

    order: Counter[str] = Counter()
    for case in positives:
        text = case["observation_text"].casefold()
        output_index = text.find(
            case["decisions"]["output_only"]["choice"].casefold()
        )
        memory_index = text.find(
            case["decisions"]["memory_conditioned"]["choice"].casefold()
        )
        order["memory choice later" if memory_index > output_index else "memory choice earlier"] += 1
    print(table("option order", order))
    print()

    tokens = [record["metrics"]["token_count"] for record in records]
    print(
        "observation tokens: "
        f"min {min(tokens)}, median {int(statistics.median(tokens))}, max {max(tokens)}"
    )
    fractions = sorted(record["metrics"]["cue_char_fraction"] for record in records)
    print(
        "cue position: p10 "
        f"{fractions[len(fractions) // 10]:.2f}, p90 "
        f"{fractions[-(len(fractions) // 10) - 1]:.2f}"
    )
    rows_per_persona = Counter(record["persona_id"] for record in records)
    spread = Counter(rows_per_persona.values())
    print(
        "scenarios per persona: "
        + ", ".join(
            f"{count} persona(s) with {scenarios}"
            for scenarios, count in sorted(spread.items())
        )
    )
    print()

    claims = {record["base_case_id"]: record["claim"] for record in records}
    overlap = Counter(
        len(prompt_claim_overlap(case["prompt"], claims[case["base_case_id"]]))
        for case in positives
    )
    print(
        table("prompt words shared with the personal fact", overlap)
    )
    print()

    issues = construction_issues(cases)
    if issues:
        print(f"construction-signature issues: {len(issues)}")
        for case_id, message in issues[:20]:
            print(f"  {case_id}: {message}")
    else:
        print("construction-signature issues: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
