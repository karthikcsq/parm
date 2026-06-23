from __future__ import annotations

from collections import Counter
from typing import Any


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
        suggestions = prediction.get("suggestions") or []
        correctness = [_score_suggestion(case, suggestion) for suggestion in suggestions]
        correct_count = sum(1 for item in correctness if item["correct"])
        useless_count = max(0, len(suggestions) - correct_count)
        best = next((item for item in correctness if item["correct"]), None)

        trigger_found = any(item["trigger_found"] for item in correctness)
        memory_fact_found = any(item["memory_fact_found"] for item in correctness)
        path_correct = any(item["path_correct"] for item in correctness)

        totals["cases"] += 1
        totals["cases_with_correct"] += 1 if best else 0
        totals["suggestions"] += len(suggestions)
        totals["correct_suggestions"] += correct_count
        totals["useless_suggestions"] += useless_count
        totals["trigger_found"] += 1 if trigger_found else 0
        totals["memory_fact_found"] += 1 if memory_fact_found else 0
        totals["path_correct"] += 1 if path_correct else 0

        rows.append(
            {
                "case_id": case_id,
                "correct": bool(best),
                "suggestion_count": len(suggestions),
                "useless_suggestion_count": useless_count,
                "trigger_found": trigger_found,
                "memory_fact_found": memory_fact_found,
                "path_correct": path_correct,
            }
        )

    case_count = totals["cases"] or 1
    suggestion_count = totals["suggestions"] or 1
    return {
        "case_count": totals["cases"],
        "correct_surfaced_suggestion_rate": totals["cases_with_correct"] / case_count,
        "precision": totals["correct_suggestions"] / suggestion_count,
        "recall": totals["cases_with_correct"] / case_count,
        "useless_intervention_rate": totals["useless_suggestions"] / suggestion_count,
        "path_correct_rate": totals["path_correct"] / case_count,
        "trigger_found_rate": totals["trigger_found"] / case_count,
        "memory_fact_found_rate": totals["memory_fact_found"] / case_count,
        "total_suggestions": totals["suggestions"],
        "rows": rows,
    }


def _score_suggestion(case: dict[str, Any], suggestion: dict[str, Any]) -> dict[str, bool]:
    expected = case["expected"]
    text = str(suggestion.get("text", "")).lower()
    trigger_label = case["trigger_entity"]["label"].lower()
    memory_label = _node_label(case, expected["memory_fact_node_id"]).lower()
    action_terms = [term.lower() for term in expected.get("action_keywords", [])]
    decision_effect = expected.get("decision_effect", {})

    trigger_found = (
        suggestion.get("trigger_entity_id") == case["trigger_entity"]["node_id"]
        or trigger_label in text
    )
    memory_fact_found = (
        suggestion.get("memory_fact_node_id") == expected["memory_fact_node_id"]
        or memory_label in text
    )
    path_correct = suggestion.get("path") == expected["gold_path"]
    action_found = bool(action_terms) and any(term in text for term in action_terms)
    decision_effect_found = _decision_effect_found(decision_effect, text)

    return {
        "trigger_found": trigger_found,
        "memory_fact_found": memory_fact_found,
        "path_correct": path_correct,
        "action_found": action_found,
        "decision_effect_found": decision_effect_found,
        "correct": trigger_found and memory_fact_found and path_correct and action_found and decision_effect_found,
    }


def _node_label(case: dict[str, Any], node_id: str) -> str:
    for node in case["graph"]["nodes"]:
        if node["id"] == node_id:
            return node["label"]
    return node_id


def _decision_effect_found(decision_effect: dict[str, Any], text: str) -> bool:
    if decision_effect.get("type") != "avoid_or_reconsider_item":
        return True
    return any(
        phrase in text
        for phrase in [
            "avoid",
            "do not choose",
            "don't choose",
            "reconsider",
            "choose a different",
            "pick a different",
        ]
    )
