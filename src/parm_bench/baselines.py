from __future__ import annotations

from typing import Any, Callable


PredictionFn = Callable[[dict[str, Any]], dict[str, Any]]


def available_baselines() -> dict[str, PredictionFn]:
    return {
        "no_memory": no_memory,
        "input_rag": input_rag,
        "prompted_memory_tool": prompted_memory_tool,
        "parm_oracle_monitor": parm_oracle_monitor,
    }


def no_memory(case: dict[str, Any]) -> dict[str, Any]:
    return _prediction(case, suggestions=[], notes="No memory graph access.")


def input_rag(case: dict[str, Any]) -> dict[str, Any]:
    user_goal = case["user_goal"].lower()
    trigger_label = case["trigger_entity"]["label"].lower()
    if trigger_label not in user_goal:
        return _prediction(
            case,
            suggestions=[],
            notes="Trigger entity is absent from the initial prompt, so input-conditioned retrieval misses it.",
        )
    return _oracle_suggestion(case, notes="Trigger was present in the initial prompt.")


def prompted_memory_tool(case: dict[str, Any]) -> dict[str, Any]:
    user_goal = case["user_goal"].lower()
    obvious_cues = {"memory", "connection", "intro", "relationship", "contact"}
    if not obvious_cues.intersection(user_goal.split()):
        return _prediction(
            case,
            suggestions=[],
            notes="Prompt heuristic did not decide to call memory search.",
        )
    return input_rag(case)


def parm_oracle_monitor(case: dict[str, Any]) -> dict[str, Any]:
    return _oracle_suggestion(
        case,
        notes="Oracle monitor watches output/tool entities and uses gold graph linking.",
    )


def _oracle_suggestion(case: dict[str, Any], notes: str) -> dict[str, Any]:
    expected = case["expected"]
    suggestion = {
        "text": expected["suggestion"],
        "trigger_entity_id": case["trigger_entity"]["node_id"],
        "memory_fact_node_id": expected["memory_fact_node_id"],
        "path": expected["gold_path"],
    }
    return _prediction(case, suggestions=[suggestion], notes=notes)


def _prediction(
    case: dict[str, Any],
    suggestions: list[dict[str, Any]],
    notes: str,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "baseline": "",
        "suggestions": suggestions,
        "notes": notes,
    }
