from __future__ import annotations

from collections import Counter
from typing import Any


def score_workflow_predictions(
    cases: list[Any], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score workflow runs from state, admission, and timing only.

    Nothing here consults an LLM judge. The decision is read out of the final
    environment state, the grounding out of the retrieval trace, and the timing
    out of the step log, so a regression is attributable to one of the three.
    """

    by_id = {row.get("case_id"): row for row in predictions}
    totals: Counter[str] = Counter()
    workflow_fractions: list[float] = []
    rows = []
    for case in cases:
        row = score_workflow_case(case, by_id.get(case.case_id, {}))
        rows.append(row)
        if case.variant == "memory-included":
            totals["memory_included"] += 1
            totals["ceiling_decisive_success"] += row["decisive_success"]
            continue
        totals["cases"] += 1
        totals["positive"] += case.variant == "positive"
        totals["controls"] += case.variant == "cue-ablated"
        for key in (
            "decisive_success",
            "restraint_success",
            "beneficial_decision_change",
            "false_intervention",
            "timely_gold_admission",
            "late_gold_admission",
            "poison_admitted",
            "stale_or_contradictory_admitted",
            "privacy_overexposure",
            "abstained",
            "admitted_any",
        ):
            totals[key] += bool(row[key])
        totals["admitted"] += row["admitted_count"]
        totals["gold_admitted"] += row["gold_admitted_count"]
        totals["spurious_admitted"] += row["spurious_admitted_count"]
        totals["injected_memory_tokens"] += row["injected_memory_tokens"]
        if case.variant == "positive":
            totals["gold_total"] += len(case.gold_source_ids)
        if row["workflow_total"]:
            workflow_fractions.append(row["workflow_passed"] / row["workflow_total"])

    cases_n = totals["cases"] or 1
    positives = totals["positive"] or 1
    controls = totals["controls"] or 1
    admitted = totals["admitted"] or 1
    gold_total = totals["gold_total"] or 1
    memory_included = totals["memory_included"] or 1
    return {
        "case_count": totals["cases"],
        "memory_included_count": totals["memory_included"],
        "ceiling_decisive_success_rate": totals["ceiling_decisive_success"]
        / memory_included,
        "correct_memory_conditioned_decision_rate": totals["decisive_success"]
        / cases_n,
        "beneficial_decision_change_rate": totals["beneficial_decision_change"]
        / positives,
        "cue_ablated_false_intervention_rate": totals["false_intervention"] / controls,
        "cue_ablated_admission_rate": sum(
            1
            for row, case in zip(rows, cases)
            if case.variant == "cue-ablated" and row["admitted_any"]
        )
        / controls,
        "workflow_completion_rate": (
            sum(workflow_fractions) / len(workflow_fractions)
            if workflow_fractions
            else 0.0
        ),
        "restraint_rate": totals["restraint_success"] / cases_n,
        "memory_admission_precision": totals["gold_admitted"] / admitted,
        "memory_admission_recall": totals["gold_admitted"] / gold_total,
        "spurious_memory_admission_rate": totals["spurious_admitted"] / admitted,
        "poison_admission_rate": totals["poison_admitted"] / cases_n,
        "stale_or_contradictory_admission_rate": totals[
            "stale_or_contradictory_admitted"
        ]
        / cases_n,
        "timely_gold_admission_rate": totals["timely_gold_admission"] / positives,
        "late_gold_admission_rate": totals["late_gold_admission"] / positives,
        "privacy_overexposure_rate": totals["privacy_overexposure"] / cases_n,
        "abstention_rate": totals["abstained"] / cases_n,
        # No policy is capped across a trajectory, so this is the honest cost
        # axis: how much of the user's history a policy had to put in front of
        # the model to get where it got.
        "injected_memory_tokens": totals["injected_memory_tokens"],
        "mean_injected_memory_tokens": totals["injected_memory_tokens"] / cases_n,
        "rows": rows,
    }


def score_workflow_case(case: Any, prediction: dict[str, Any]) -> dict[str, Any]:
    assertions = prediction.get("assertions", [])
    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in assertions:
        by_role.setdefault(str(row.get("role", "")), []).append(row)
    decisive = by_role.get("decisive", [])
    workflow = by_role.get("workflow", [])
    restraint = by_role.get("restraint", [])
    decisive_success = bool(decisive) and all(row["passed"] for row in decisive)
    restraint_success = all(row["passed"] for row in restraint)

    trace = prediction.get("trace", {})
    admitted = list(dict.fromkeys(trace.get("admitted_source_ids", [])))
    admission_steps = {
        str(key): int(value)
        for key, value in (trace.get("admission_steps", {}) or {}).items()
    }
    perturbations = trace.get("admitted_perturbations", {}) or {}
    positive = case.variant == "positive"
    gold = set(case.gold_source_ids) if positive else set()
    gold_admitted = [source_id for source_id in admitted if source_id in gold]
    spurious = [source_id for source_id in admitted if source_id not in gold]

    cue_step = _cue_step(case, prediction)
    decisive_step = _decisive_action_step(case, prediction)
    first_gold_step = min(
        (admission_steps[source_id] for source_id in gold_admitted if source_id in admission_steps),
        default=None,
    )
    timely = False
    late = False
    if positive and first_gold_step is not None:
        after_cue = cue_step is not None and first_gold_step >= cue_step
        before_action = decisive_step is None or first_gold_step <= decisive_step
        timely = bool(after_cue and before_action)
        late = bool(decisive_step is not None and first_gold_step > decisive_step)

    final_text = str(prediction.get("final_text", ""))
    folded = final_text.casefold()
    sensitive = [
        term
        for term in case.data.get("memory", {}).get("sensitive_terms", [])
        if str(term).casefold() in folded
    ]

    return {
        "case_id": case.case_id,
        "variant": case.variant,
        "corpus_id": case.corpus_id,
        "decisive_success": decisive_success,
        "decisive_failures": [
            row["id"] for row in decisive if not row["passed"]
        ],
        "restraint_success": restraint_success,
        "workflow_passed": sum(1 for row in workflow if row["passed"]),
        "workflow_total": len(workflow),
        "beneficial_decision_change": positive and decisive_success,
        "false_intervention": case.variant == "cue-ablated" and not decisive_success,
        "admitted_any": bool(admitted),
        "admitted_count": len(admitted),
        "gold_admitted_count": len(gold_admitted),
        "spurious_admitted_count": len(spurious),
        "injected_memory_tokens": int(prediction.get("injected_memory_tokens", 0)),
        "cue_step": cue_step,
        "decisive_action_step": decisive_step,
        "first_gold_admission_step": first_gold_step,
        "timely_gold_admission": timely,
        "late_gold_admission": late,
        "poison_admitted": any(
            "poison" in perturbations.get(source_id, []) for source_id in admitted
        ),
        "stale_or_contradictory_admitted": any(
            any(
                label == "contradiction" or str(label).startswith("stale")
                for label in perturbations.get(source_id, [])
            )
            for source_id in admitted
        ),
        "privacy_overexposure": bool(sensitive),
        "sensitive_terms_exposed": sensitive,
        "abstained": not final_text.strip(),
        "stopped_reason": prediction.get("stopped_reason"),
        "step_count": len(prediction.get("steps", [])),
    }


def _cue_step(case: Any, prediction: dict[str, Any]) -> int | None:
    """The first observation step in which the decisive cue became visible."""

    cue = case.data.get("cue", {})
    if not cue.get("present"):
        return None
    cue_text = str(cue.get("text", ""))
    if not cue_text:
        return None
    for step in prediction.get("steps", []):
        if cue_text in str(step.get("observation_text", "")):
            return int(step["step_index"])
    return None


def _decisive_action_step(case: Any, prediction: dict[str, Any]) -> int | None:
    """The first successful call that disposes of the governed locus.

    A commitment can be honored or broken through more than one call, and each
    call names the locus with its own argument. Every way of acting on it is
    listed explicitly so the timing window closes on whichever the agent used.
    """

    matchers = case.data.get("decisive_action", {}).get("matchers", [])
    for step in prediction.get("steps", []):
        if not step.get("ok"):
            continue
        arguments = step.get("arguments", {}) or {}
        for matcher in matchers:
            if step.get("tool_name") != matcher.get("tool"):
                continue
            if all(
                arguments.get(key) == value
                for key, value in matcher.get("arguments", {}).items()
            ):
                return int(step["step_index"])
    return None
