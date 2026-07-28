"""Construction-signature detectors for a finished PARMBench dataset.

The legacy mixed_v0 slice reused one rating pair, one pool of lead phrases, one
cue band, one envelope opening, and one answer role across all thirty base
scenarios. A system can learn that shape instead of the memory, so criterion 9
of the construction contract forbids it. These checks measure the repetition
across base scenarios and report it as dataset-level issues.

Every check operates on base scenarios, not cases: the three triplet variants
share one construction, so counting cases would triple every share.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


# Below this many base scenarios a single scenario already exceeds the share
# thresholds on its own, so the detectors cannot tell a template from a sample.
MIN_SCENARIOS_FOR_CHECKS = 10
# A decimal pair such as 9.8 alongside 8.8 repeating above this share of base
# scenarios is a rating signature rather than incidental numbers.
MAX_NUMERIC_PAIR_SCENARIO_SHARE = 0.20
# Any six-word run of observation prose shared above this share of base
# scenarios means the envelope, lead, or cue text is templated.
MAX_PHRASE_SCENARIO_SHARE = 0.15
PHRASE_NGRAM_SIZE = 6
# The banned phrases below named the answer role in the legacy generator, so a
# second scenario reusing one is already a construction signature.
MAX_BANNED_PHRASE_SCENARIOS = 1
# Cue offsets must span at least this much of the observation between the tenth
# and ninetieth percentile, otherwise every cue sits in one predictable band.
MIN_CUE_POSITION_INTERDECILE_RANGE = 0.30
# When the memory-conditioned choice falls on the same side of the output-only
# choice above this share of scenarios, the answer role never rotates.
MAX_OPTION_ROLE_SHARE = 0.80
# More than this many scenarios opening on an identical line means the
# observation envelope is drawn from a small fixed template pool.
MAX_SCENARIOS_PER_OPENING_LINE = 3
# Upper bound on how many offending items one detector names in its messages.
MAX_REPORTED_ITEMS = 5

# Terms that would tell a retrieval or admission prompt how the benchmark
# builds its answer. Criterion 10 forbids them in any model-visible
# instruction.
JUDGE_LEAKAGE_TERMS = (
    "lower-ranked",
    "ordinary evidence winner",
    "already-strongest",
    "target rank",
)

BANNED_SIGNATURE_PHRASES = (
    "narrower choice",
    "narrower case",
    "narrower than the default",
    "no one challenged",
    "consensus was",
)

DATASET_SCOPE = "dataset"

_DECIMAL_NUMBER = re.compile(r"\d+\.\d+")
_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class BaseScenario:
    base_case_id: str
    observation_text: str
    cue_text: str
    output_choice: str
    memory_choice: str


def construction_issues(cases: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Report construction-signature repetition across a dataset's scenarios.

    Each issue is a ``(case_id or "dataset", message)`` pair, matching the shape
    :mod:`parm_bench.dataset` turns into ``ValidationIssue`` rows.
    """

    scenarios = base_scenarios(cases)
    if len(scenarios) < MIN_SCENARIOS_FOR_CHECKS:
        return []
    issues: list[tuple[str, str]] = []
    issues.extend(_numeric_pair_issues(scenarios))
    issues.extend(_phrase_reuse_issues(scenarios))
    issues.extend(_banned_phrase_issues(scenarios))
    issues.extend(_cue_position_issues(scenarios))
    issues.extend(_option_role_issues(scenarios))
    issues.extend(_opening_line_issues(scenarios))
    return issues


def judge_leakage_violations(text: str) -> list[str]:
    """Return the answer-construction terms a prompt or instruction leaks.

    Exported for retrieval and admission prompts. It is deliberately not part
    of dataset validation: the leak lives in a system's instructions, not in
    the benchmark files.
    """

    lowered = str(text).casefold()
    return [term for term in JUDGE_LEAKAGE_TERMS if term in lowered]


def base_scenarios(cases: list[dict[str, Any]]) -> list[BaseScenario]:
    """Collapse a case list to one representative case per base scenario."""

    chosen: dict[str, dict[str, Any]] = {}
    for case in cases:
        base_case_id = str(case.get("base_case_id", case.get("case_id", "")))
        current = chosen.get(base_case_id)
        if current is None or (
            current.get("variant") != "positive"
            and case.get("variant") == "positive"
        ):
            chosen[base_case_id] = case
    scenarios = []
    for base_case_id, case in chosen.items():
        decisions = case.get("decisions", {})
        scenarios.append(
            BaseScenario(
                base_case_id=base_case_id,
                observation_text=str(case.get("observation_text", "")),
                cue_text=str(case.get("cue", {}).get("text", "")),
                output_choice=str(
                    decisions.get("output_only", {}).get("choice", "")
                ),
                memory_choice=str(
                    decisions.get("memory_conditioned", {}).get("choice", "")
                ),
            )
        )
    return scenarios


def _numeric_pair_issues(scenarios: list[BaseScenario]) -> list[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for scenario in scenarios:
        values = sorted(set(_DECIMAL_NUMBER.findall(scenario.observation_text)))
        for first in range(len(values)):
            for second in range(first + 1, len(values)):
                counts[(values[first], values[second])] += 1
    limit = MAX_NUMERIC_PAIR_SCENARIO_SHARE * len(scenarios)
    repeated = sorted(
        (item for item in counts.items() if item[1] > limit),
        key=lambda item: (-item[1], item[0]),
    )
    if not repeated:
        return []
    named = ", ".join(
        f"{pair[0]}/{pair[1]} in {count}" for pair, count in repeated[:MAX_REPORTED_ITEMS]
    )
    return [
        (
            DATASET_SCOPE,
            f"repeated numeric pair in more than "
            f"{MAX_NUMERIC_PAIR_SCENARIO_SHARE:.0%} of {len(scenarios)} base "
            f"scenarios: {named}",
        )
    ]


def _phrase_reuse_issues(scenarios: list[BaseScenario]) -> list[tuple[str, str]]:
    # Only observation prose is counted. The prompt template is shared by
    # construction and would flag every dataset.
    counts: Counter[str] = Counter()
    for scenario in scenarios:
        counts.update(_ngrams(scenario.observation_text))
    limit = MAX_PHRASE_SCENARIO_SHARE * len(scenarios)
    repeated = sorted(
        (item for item in counts.items() if item[1] > limit),
        key=lambda item: (-item[1], item[0]),
    )
    if not repeated:
        return []
    named = "; ".join(
        f"{phrase!r} in {count}" for phrase, count in repeated[:MAX_REPORTED_ITEMS]
    )
    return [
        (
            DATASET_SCOPE,
            f"{len(repeated)} {PHRASE_NGRAM_SIZE}-grams recur in more than "
            f"{MAX_PHRASE_SCENARIO_SHARE:.0%} of {len(scenarios)} base "
            f"scenarios: {named}",
        )
    ]


def _ngrams(text: str) -> set[str]:
    words = _WORD.findall(text.casefold())
    return {
        " ".join(words[index : index + PHRASE_NGRAM_SIZE])
        for index in range(len(words) - PHRASE_NGRAM_SIZE + 1)
    }


def _banned_phrase_issues(scenarios: list[BaseScenario]) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    for phrase in BANNED_SIGNATURE_PHRASES:
        carriers = [
            scenario.base_case_id
            for scenario in scenarios
            if phrase in scenario.observation_text.casefold()
        ]
        if len(carriers) <= MAX_BANNED_PHRASE_SCENARIOS:
            continue
        for base_case_id in sorted(carriers)[:MAX_REPORTED_ITEMS]:
            issues.append(
                (
                    base_case_id,
                    f"banned construction phrase {phrase!r} appears in "
                    f"{len(carriers)} base scenarios",
                )
            )
    return issues


def _cue_position_issues(scenarios: list[BaseScenario]) -> list[tuple[str, str]]:
    positions = []
    for scenario in scenarios:
        text = scenario.observation_text
        cue = scenario.cue_text
        if not cue:
            continue
        index = text.find(cue)
        if index < 0:
            continue
        positions.append(index / max(1, len(text) - len(cue)))
    if len(positions) < 2:
        return []
    spread = _percentile(positions, 0.9) - _percentile(positions, 0.1)
    if spread >= MIN_CUE_POSITION_INTERDECILE_RANGE:
        return []
    return [
        (
            DATASET_SCOPE,
            f"cue position varies by only {spread:.2f} between the tenth and "
            f"ninetieth percentile across {len(positions)} base scenarios; "
            f"expected at least {MIN_CUE_POSITION_INTERDECILE_RANGE:.2f}",
        )
    ]


def _option_role_issues(scenarios: list[BaseScenario]) -> list[tuple[str, str]]:
    later = 0
    earlier = 0
    for scenario in scenarios:
        text = scenario.observation_text.casefold()
        if not scenario.output_choice or not scenario.memory_choice:
            continue
        output_index = text.find(scenario.output_choice.casefold())
        memory_index = text.find(scenario.memory_choice.casefold())
        if output_index < 0 or memory_index < 0 or output_index == memory_index:
            continue
        if memory_index > output_index:
            later += 1
        else:
            earlier += 1
    total = later + earlier
    if total < 2:
        return []
    share = max(later, earlier) / total
    if share <= MAX_OPTION_ROLE_SHARE:
        return []
    direction = "after" if later >= earlier else "before"
    return [
        (
            DATASET_SCOPE,
            f"the memory-conditioned choice appears {direction} the output-only "
            f"choice in {share:.0%} of {total} base scenarios; expected at most "
            f"{MAX_OPTION_ROLE_SHARE:.0%}",
        )
    ]


def _opening_line_issues(scenarios: list[BaseScenario]) -> list[tuple[str, str]]:
    counts: Counter[str] = Counter()
    for scenario in scenarios:
        opening = _opening_line(scenario.observation_text)
        if opening:
            counts[opening] += 1
    repeated = sorted(
        (
            item
            for item in counts.items()
            if item[1] > MAX_SCENARIOS_PER_OPENING_LINE
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not repeated:
        return []
    named = "; ".join(
        f"{opening!r} in {count}" for opening, count in repeated[:MAX_REPORTED_ITEMS]
    )
    return [
        (
            DATASET_SCOPE,
            f"{len(repeated)} observation opening lines are shared by more than "
            f"{MAX_SCENARIOS_PER_OPENING_LINE} base scenarios: {named}",
        )
    ]


def _opening_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
