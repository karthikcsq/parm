from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GOAL_FAMILIES = {
    "customer_discovery",
    "cooking_planning",
    "event_planning",
    "fitness_planning",
    "general_travel",
    "health_admin",
    "hiring",
    "home_ops",
    "interview_prep",
    "introductions",
    "learning_research",
    "opportunity_risk",
    "personal_finance",
    "travel_planning",
    "writing_help",
}
TRIGGER_SOURCES = {"generated_output", "tool_response"}
JOIN_DEPTHS = {1, 2, 3}
DISTRACTOR_TYPES = {
    "semantic",
    "graph_proximity",
    "stale_invalid_edge",
    "goal_irrelevant",
}
ACTIONABILITIES = {
    "enrichment",
    "follow_up_question",
    "intro",
    "prioritization",
    "warning",
}
REQUIRED_CASE_FIELDS = {
    "case_id",
    "goal_family",
    "trigger_source",
    "join_depth",
    "distractor_type",
    "actionability",
    "user_goal",
    "assistant_output",
    "tool_response",
    "graph",
    "trigger_entity",
    "expected",
}


@dataclass(frozen=True)
class ValidationIssue:
    case_id: str
    message: str

    def __str__(self) -> str:
        return f"{self.case_id}: {self.message}"


class DatasetValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("\n".join(str(issue) for issue in issues))


def load_cases(dataset_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cases.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return cases


def write_cases(cases: list[dict[str, Any]], dataset_dir: str | Path) -> Path:
    path = Path(dataset_dir)
    path.mkdir(parents=True, exist_ok=True)
    cases_path = path / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    return cases_path


def validate_cases(cases: list[dict[str, Any]]) -> None:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", f"<case {index}>"))
        missing = sorted(REQUIRED_CASE_FIELDS - set(case))
        for field in missing:
            issues.append(ValidationIssue(case_id, f"missing required field '{field}'"))

        if case_id in seen_ids:
            issues.append(ValidationIssue(case_id, "duplicate case_id"))
        seen_ids.add(case_id)

        _validate_taxonomy(case, case_id, issues)
        _validate_graph(case, case_id, issues)

    if issues:
        raise DatasetValidationError(issues)


def _validate_taxonomy(
    case: dict[str, Any],
    case_id: str,
    issues: list[ValidationIssue],
) -> None:
    if case.get("goal_family") not in GOAL_FAMILIES:
        issues.append(ValidationIssue(case_id, "invalid goal_family"))
    if case.get("trigger_source") not in TRIGGER_SOURCES:
        issues.append(ValidationIssue(case_id, "invalid trigger_source"))
    if case.get("join_depth") not in JOIN_DEPTHS:
        issues.append(ValidationIssue(case_id, "invalid join_depth"))
    if case.get("distractor_type") not in DISTRACTOR_TYPES:
        issues.append(ValidationIssue(case_id, "invalid distractor_type"))
    if case.get("actionability") not in ACTIONABILITIES:
        issues.append(ValidationIssue(case_id, "invalid actionability"))


def _validate_graph(
    case: dict[str, Any],
    case_id: str,
    issues: list[ValidationIssue],
) -> None:
    graph = case.get("graph")
    if not isinstance(graph, dict):
        issues.append(ValidationIssue(case_id, "graph must be an object"))
        return

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = {node.get("id") for node in nodes if isinstance(node, dict)}
    if None in node_ids:
        issues.append(ValidationIssue(case_id, "all graph nodes need an id"))
        node_ids.discard(None)

    for edge in edges:
        if not isinstance(edge, dict):
            issues.append(ValidationIssue(case_id, "all graph edges must be objects"))
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_ids or target not in node_ids:
            issues.append(
                ValidationIssue(
                    case_id,
                    f"edge '{edge.get('id', '<missing>')}' references unknown nodes",
                )
            )

    trigger_node = case.get("trigger_entity", {}).get("node_id")
    if trigger_node not in node_ids:
        issues.append(ValidationIssue(case_id, "trigger_entity.node_id is not in graph"))

    expected = case.get("expected", {})
    memory_node = expected.get("memory_fact_node_id")
    if memory_node not in node_ids:
        issues.append(ValidationIssue(case_id, "expected.memory_fact_node_id is not in graph"))

    gold_path = expected.get("gold_path")
    if not isinstance(gold_path, list) or len(gold_path) < 2:
        issues.append(ValidationIssue(case_id, "expected.gold_path must contain at least two node ids"))
        return
    if gold_path[0] != trigger_node:
        issues.append(ValidationIssue(case_id, "gold_path must start with trigger entity"))
    if gold_path[-1] != memory_node:
        issues.append(ValidationIssue(case_id, "gold_path must end with memory fact node"))
    for node_id in gold_path:
        if node_id not in node_ids:
            issues.append(ValidationIssue(case_id, f"gold_path references unknown node '{node_id}'"))

    valid_pairs = {
        (edge.get("source"), edge.get("target"))
        for edge in edges
        if edge.get("valid", True) is True
    }
    valid_pairs |= {(target, source) for source, target in valid_pairs}
    for left, right in zip(gold_path, gold_path[1:]):
        if (left, right) not in valid_pairs:
            issues.append(
                ValidationIssue(case_id, f"gold_path has no valid edge between '{left}' and '{right}'")
            )
