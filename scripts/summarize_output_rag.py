"""Summarize naive_output_rag grid results by flow x mode.

Each config is tested on 5 scenarios per variant:
  - positive: the memory cue is present, so the CORRECT answer is the
    memory-conditioned choice (the model has to use retrieved memory).
  - control (cue-ablated): the cue is removed, so the CORRECT answer is the
    plain default/endorsed choice visible in the observation.
  - ceiling (memory-included): the gold memory is injected into the prompt, so
    the CORRECT answer is the memory-conditioned choice without any retrieval.
    This is the upper bound: how often the model uses the right context when it
    is handed to it directly.

We report how many of the 5 the model got right in each variant.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parm_bench.dataset import load_cases  # noqa: E402

RESULTS = ROOT / "data" / "benchmark-results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_correct(variant: str, response: str, output_choice: str, memory_choice: str) -> bool:
    """Did the model give the correct answer for this variant?

    positive: correct = the memory-conditioned choice, and NOT the default
              (it had to override the visible default using memory).
    control : correct = the plain default choice (memory choice == default here).
    """
    resp = response.casefold()
    if variant in ("positive", "memory-included"):
        return memory_choice.casefold() in resp and output_choice.casefold() not in resp
    return output_choice.casefold() in resp


def main() -> None:
    cases = {c["case_id"]: c for c in load_cases(ROOT / "data" / "benchmark_v1")}
    files = sorted(glob.glob(str(RESULTS / "naive-output-*.jsonl")))

    print("Each config was tested on 5 scenarios per variant. Score = how many")
    print("of the 5 the model answered CORRECTLY.")
    print("  positive : cue present     -> correct = use the remembered answer")
    print("  control  : cue removed      -> correct = pick the plain default answer")
    print("  ceiling  : memory in prompt -> correct = use the injected answer (no retrieval)")
    print()
    print(f"{'flow':<24}{'mode':<10}{'positive':<12}{'control':<12}{'ceiling':<12}")
    print("-" * 82)

    grid: dict[tuple[str, str], dict] = {}
    for file in files:
        name = Path(file).stem.replace("naive-output-", "")
        flow, _, mode = name.rpartition("-")
        # [correct, total] per variant
        tally = {"positive": [0, 0], "cue-ablated": [0, 0], "memory-included": [0, 0]}
        for pred in load_jsonl(Path(file)):
            case = cases[pred["case_id"]]
            d = case["decisions"]
            bucket = tally[case["variant"]]
            bucket[1] += 1
            if is_correct(case["variant"], pred["response_text"], d["output_only"]["choice"], d["memory_conditioned"]["choice"]):
                bucket[0] += 1
        grid[(flow, mode)] = tally

    for (flow, mode), tally in sorted(grid.items()):
        pos = f"{tally['positive'][0]}/{tally['positive'][1]}"
        ctl = f"{tally['cue-ablated'][0]}/{tally['cue-ablated'][1]}"
        ceil = f"{tally['memory-included'][0]}/{tally['memory-included'][1]}"
        print(f"{flow:<24}{mode:<10}{pos:<12}{ctl:<12}{ceil:<12}")

    pos_correct = sum(t["positive"][0] for t in grid.values())
    pos_total = sum(t["positive"][1] for t in grid.values())
    ctl_correct = sum(t["cue-ablated"][0] for t in grid.values())
    ctl_total = sum(t["cue-ablated"][1] for t in grid.values())
    mi_correct = sum(t["memory-included"][0] for t in grid.values())
    mi_total = sum(t["memory-included"][1] for t in grid.values())
    print("-" * 82)
    print(
        f"{'TOTAL':<34}"
        f"{f'{pos_correct}/{pos_total}':<12}"
        f"{f'{ctl_correct}/{ctl_total}':<12}"
        f"{f'{mi_correct}/{mi_total}':<12}"
    )


if __name__ == "__main__":
    main()
