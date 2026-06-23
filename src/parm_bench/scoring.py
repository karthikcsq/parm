from __future__ import annotations

from collections import Counter
import re
import unicodedata
from typing import Any


DECISION_EFFECT_TERMS = {
    "avoid_or_reconsider_item": {
        "avoid",
        "reconsider",
        "warning",
        "risk",
        "check",
        "verify",
        "modify",
        "alternative",
        "change",
        "lighter",
        "shallower",
    },
    "improve_selected_item": {
        "prioritize",
        "ask",
        "intro",
        "follow up",
        "follow-up",
        "enrich",
        "revisit",
        "upgrade",
        "strengthen",
        "logistics",
        "certification",
    },
}

FAILURE_ORDER = [
    "format_failure",
    "abstained",
    "cue_detection_miss",
    "memory_fact_miss",
    "wrong_memory_fact",
    "wrong_affected_item",
    "invalid_or_stale_edge_used",
    "graph_traversal_miss",
    "ranking_or_selection_error",
    "actionability_miss",
    "decision_effect_mismatch",
    "overtrigger_noise",
    "scorer_gold_mismatch",
]


def score_predictions(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    cases_by_id = {case["case_id"]: case for case in cases}
    predictions_by_id = {prediction["case_id"]: prediction for prediction in predictions}

    rows: list[dict[str, Any]] = []
    totals = Counter()

    for case_id, case in cases_by_id.items():
        prediction = predictions_by_id.get(case_id, {"case_id": case_id, "suggestions": []})
        candidates, format_valid = _prediction_candidates(prediction)
        correctness = [_score_suggestion(case, suggestion) for suggestion in candidates]
        correct_count = sum(1 for item in correctness if item["correct"])
        useless_count = max(0, len(candidates) - correct_count)
        best = next((item for item in correctness if item["correct"]), None)
        row_failures = _row_failures(case, correctness, format_valid)

        trigger_found = any(item["trigger_found"] for item in correctness)
        memory_fact_found = any(item["memory_fact_found"] for item in correctness)
        affected_item_found = any(item["affected_item_found"] for item in correctness)
        action_found = any(item["action_found"] for item in correctness)
        decision_effect_found = any(item["decision_effect_found"] for item in correctness)
        path_applicable = any(item["path_applicable"] for item in correctness)
        path_correct = any(item["path_correct"] is True for item in correctness) if path_applicable else None

        totals["cases"] += 1
        totals["cases_with_correct"] += 1 if best else 0
        totals["suggestions"] += len(candidates)
        totals["correct_suggestions"] += correct_count
        totals["useless_suggestions"] += useless_count
        totals["format_valid"] += 1 if format_valid else 0
        totals["abstentions"] += 1 if not candidates and format_valid else 0
        totals["trigger_found"] += 1 if trigger_found else 0
        totals["memory_fact_found"] += 1 if memory_fact_found else 0
        totals["affected_item_found"] += 1 if affected_item_found else 0
        totals["action_found"] += 1 if action_found else 0
        totals["decision_effect_found"] += 1 if decision_effect_found else 0
        totals["path_applicable"] += 1 if path_applicable else 0
        totals["path_correct"] += 1 if path_correct is True else 0

        rows.append(
            {
                "case_id": case_id,
                "correct": bool(best),
                "suggestion_count": len(candidates),
                "useless_suggestion_count": useless_count,
                "format_valid": format_valid,
                "trigger_found": trigger_found,
                "memory_fact_found": memory_fact_found,
                "affected_item_found": affected_item_found,
                "action_found": action_found,
                "decision_effect_found": decision_effect_found,
                "path_correct": path_correct,
                "failed_stages": row_failures,
                "first_failure": row_failures[0] if row_failures else None,
            }
        )

    case_count = totals["cases"] or 1
    suggestion_count = totals["suggestions"] or 1
    path_count = totals["path_applicable"]
    return {
        "case_count": totals["cases"],
        "correct_surfaced_suggestion_rate": totals["cases_with_correct"] / case_count,
        "format_valid_rate": totals["format_valid"] / case_count,
        "precision": totals["correct_suggestions"] / suggestion_count,
        "recall": totals["cases_with_correct"] / case_count,
        "useless_intervention_rate": totals["useless_suggestions"] / suggestion_count,
        "abstention_rate": totals["abstentions"] / case_count,
        "path_correct_rate": totals["path_correct"] / path_count if path_count else None,
        "trigger_found_rate": totals["trigger_found"] / case_count,
        "memory_fact_found_rate": totals["memory_fact_found"] / case_count,
        "affected_item_found_rate": totals["affected_item_found"] / case_count,
        "action_found_rate": totals["action_found"] / case_count,
        "decision_effect_found_rate": totals["decision_effect_found"] / case_count,
        "total_suggestions": totals["suggestions"],
        "rows": rows,
    }


def _prediction_candidates(prediction: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    suggestions = prediction.get("suggestions")
    if suggestions is None:
        suggestions = []
    if not isinstance(suggestions, list):
        return [], False

    candidates = [item for item in suggestions if isinstance(item, dict)]
    format_valid = len(candidates) == len(suggestions)
    response_text = prediction.get("response_text")
    if response_text is not None:
        text = str(response_text).strip()
        if text:
            candidates.append({"text": text, "source": "response_text"})
        else:
            format_valid = False
    return candidates, format_valid


def _score_suggestion(case: dict[str, Any], suggestion: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    text = _normalize(str(suggestion.get("text", "")))
    trigger_label = _normalize(case["trigger_entity"]["label"])
    memory_label = _normalize(_node_label(case, expected["memory_fact_node_id"]))
    affected_item_label = _normalize(_node_label(case, expected.get("decision_effect", {}).get("item_node_id", "")))
    action_terms = [_normalize(term) for term in expected.get("action_keywords", [])]
    decision_effect = expected.get("decision_effect", {})
    path_applicable = any(field in suggestion for field in ("trigger_entity_id", "memory_fact_node_id", "path"))

    trigger_found = (
        suggestion.get("trigger_entity_id") == case["trigger_entity"]["node_id"]
        or _contains(text, trigger_label)
    )
    memory_fact_found = (
        suggestion.get("memory_fact_node_id") == expected["memory_fact_node_id"]
        or _contains(text, memory_label)
    )
    affected_item_found = _contains(text, affected_item_label)
    path_correct = suggestion.get("path") == expected["gold_path"] if path_applicable else None
    action_found = bool(action_terms) and any(_contains(text, term) for term in action_terms)
    decision_effect_found = _decision_effect_found(decision_effect, text)
    graph_trace_ok = path_correct is True if path_applicable else True

    return {
        "path": suggestion.get("path"),
        "memory_fact_node_id": suggestion.get("memory_fact_node_id"),
        "trigger_found": trigger_found,
        "memory_fact_found": memory_fact_found,
        "affected_item_found": affected_item_found,
        "path_applicable": path_applicable,
        "path_correct": path_correct,
        "action_found": action_found,
        "decision_effect_found": decision_effect_found,
        "correct": (
            trigger_found
            and memory_fact_found
            and affected_item_found
            and graph_trace_ok
            and action_found
            and decision_effect_found
        ),
    }


def _node_label(case: dict[str, Any], node_id: str) -> str:
    for node in case["graph"]["nodes"]:
        if node["id"] == node_id:
            return node["label"]
    return node_id


def _decision_effect_found(decision_effect: dict[str, Any], text: str) -> bool:
    effect_type = decision_effect.get("type")
    terms = DECISION_EFFECT_TERMS.get(effect_type)
    if not terms:
        return True
    return any(_contains(text, _normalize(term)) for term in terms)


def _row_failures(
    case: dict[str, Any],
    correctness: list[dict[str, Any]],
    format_valid: bool,
) -> list[str]:
    if not format_valid:
        return ["format_failure"]
    if not correctness:
        return ["abstained"]
    if any(item["correct"] for item in correctness):
        return ["overtrigger_noise"] if any(not item["correct"] for item in correctness) else []

    failures: list[str] = []
    if not any(item["trigger_found"] for item in correctness):
        failures.append("cue_detection_miss")
    memory_found = any(item["memory_fact_found"] for item in correctness)
    if not memory_found:
        failures.append("memory_fact_miss")
    elif any(
        item.get("memory_fact_node_id") not in (None, case["expected"]["memory_fact_node_id"])
        for item in correctness
    ):
        failures.append("wrong_memory_fact")
    if not any(item["affected_item_found"] for item in correctness):
        failures.append("wrong_affected_item")
    if any(_uses_invalid_edge(case, item) for item in correctness):
        failures.append("invalid_or_stale_edge_used")
    if any(item["path_applicable"] for item in correctness) and not any(item["path_correct"] is True for item in correctness):
        failures.append("graph_traversal_miss")
    if len(correctness) > 1:
        failures.append("ranking_or_selection_error")
    if not any(item["action_found"] for item in correctness):
        failures.append("actionability_miss")
    if not any(item["decision_effect_found"] for item in correctness):
        failures.append("decision_effect_mismatch")
    if _looks_like_scorer_gold_mismatch(correctness):
        failures.append("scorer_gold_mismatch")
    return [failure for failure in FAILURE_ORDER if failure in failures]


def _uses_invalid_edge(case: dict[str, Any], scored_suggestion: dict[str, Any]) -> bool:
    path = scored_suggestion.get("path")
    if not isinstance(path, list) or len(path) < 2:
        return False
    valid_pairs = {
        (edge.get("source"), edge.get("target"))
        for edge in case.get("graph", {}).get("edges", [])
        if edge.get("valid", True) is True
    }
    valid_pairs |= {(target, source) for source, target in valid_pairs}
    return any((left, right) not in valid_pairs for left, right in zip(path, path[1:]))


def _looks_like_scorer_gold_mismatch(correctness: list[dict[str, Any]]) -> bool:
    return any(
        item["trigger_found"]
        and item["memory_fact_found"]
        and item["affected_item_found"]
        and (not item["path_applicable"] or item["path_correct"] is True)
        and (not item["action_found"] or not item["decision_effect_found"])
        for item in correctness
    )


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains(text: str, term: str) -> bool:
    if not term:
        return False
    return f" {term} " in f" {text} "
