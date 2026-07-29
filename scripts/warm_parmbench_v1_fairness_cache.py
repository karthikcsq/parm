"""Populate a response cache for the `no_memory` fairness sweep in parallel.

`parm-bench run` walks its cases one at a time, which is the right shape for a
scored run but too slow for a 381-case sweep of 10k-token observations. This
helper issues the same `no_memory` requests concurrently against the same
`CachingLanguageModel`, so the entries land under the same request hashes. The
scored run is then produced by the CLI in the normal way; it hits the cache for
every warmed case and only reaches the API for whatever this helper skipped.

A truncated response is left uncached on purpose. The CLI records truncation as
an abstention, and that decision belongs to the run, not to the warmer.

Usage:

    $env:PYTHONPATH = 'src'
    & 'C:\\Users\\karth\\anaconda3\\python.exe' \\
      scripts\\warm_parmbench_v1_fairness_cache.py data\\benchmark_parmbench_v1 \\
      --cache data\\response-caches\\parmbench-v1-fairness
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from parm_bench.baselines import benchmark_input
from parm_bench.dataset import load_cases, validate_cases
from parm_bench.models import (
    FINAL_ANSWER_INSTRUCTIONS,
    CachingLanguageModel,
    ModelTruncationError,
    OpenAIResponsesModel,
    ResponsePolicy,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument(
        "--base-case-id",
        action="append",
        default=[],
        help="restrict warming to one or more scenarios (repeatable)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    cases = load_cases(args.dataset_dir)
    validate_cases(cases, include_profile=False)
    if args.base_case_id:
        wanted = set(args.base_case_id)
        cases = [case for case in cases if case["base_case_id"] in wanted]
    model = CachingLanguageModel(
        OpenAIResponsesModel(args.model), args.cache, ResponsePolicy.POPULATE
    )
    counts = {"cached": 0, "truncated": 0, "failed": 0}
    lock = threading.Lock()

    def warm(case: dict[str, Any]) -> None:
        entry = benchmark_input(case)
        outcome = "failed"
        # A 10k-token observation per call puts a wide pool straight into the
        # organisation's tokens-per-minute ceiling, so a rejected call waits
        # and comes back rather than leaving a hole in the cache.
        for attempt in range(args.retries + 1):
            try:
                model.generate(
                    prompt=entry.prompt,
                    observation_kind=entry.observation_kind,
                    observation_text=entry.observation_text,
                    instructions=FINAL_ANSWER_INSTRUCTIONS,
                )
            except ModelTruncationError:
                outcome = "truncated"
                break
            except Exception as error:  # noqa: BLE001 - left to the scored run
                if attempt == args.retries:
                    print(f"{case['case_id']}: {error}", file=sys.stderr)
                    outcome = "failed"
                    break
                time.sleep(min(60.0, 2.0 * 2**attempt) * (0.5 + random.random()))
            else:
                outcome = "cached"
                break
        with lock:
            counts[outcome] += 1
            done = sum(counts.values())
            if done % 25 == 0 or done == len(cases):
                print(f"{done}/{len(cases)} {counts}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(warm, cases))
    print(counts)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
