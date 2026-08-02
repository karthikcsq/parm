from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import get_environment, load_fixture


VARIANTS = {"positive", "cue-ablated", "memory-included"}
CUE_PRESENT_VARIANTS = {"positive", "memory-included"}
ASSERTION_ROLES = {"decisive", "workflow", "restraint"}
MINIMUM_DISTRACTORS = 3

REQUIRED_FIELDS = {
    "case_id",
    "base_case_id",
    "corpus_id",
    "corpus_tier",
    "variant",
    "goal",
    "environment",
    "cue",
    "memory",
    "distractors",
    "decisive_action",
    "assertions",
    "provenance",
}


@dataclass(frozen=True)
class WorkflowCaseIssue:
    case_id: str
    message: str

    def __str__(self) -> str:
        return f"{self.case_id}: {self.message}"


class WorkflowCaseValidationError(ValueError):
    def __init__(self, issues: list[WorkflowCaseIssue]):
        self.issues = issues
        super().__init__("\n".join(str(issue) for issue in issues))


@dataclass(frozen=True)
class WorkflowCase:
    """A validated workflow case with its resolved fixture and corpus roots."""

    data: dict[str, Any]
    dataset_root: Path
    corpus_root: Path

    @property
    def case_id(self) -> str:
        return str(self.data["case_id"])

    @property
    def base_case_id(self) -> str:
        return str(self.data["base_case_id"])

    @property
    def variant(self) -> str:
        return str(self.data["variant"])

    @property
    def corpus_id(self) -> str:
        return str(self.data["corpus_id"])

    @property
    def corpus_tier(self) -> str:
        """Smallest declared tier holding every gold record for this case.

        A scenario whose commitment was written above the first tier cannot be
        evaluated against that tier's index: the memory is simply absent, and
        the run would look like a retrieval failure rather than a missing
        fixture.
        """

        return str(self.data.get("corpus_tier", ""))

    @property
    def goal(self) -> str:
        return str(self.data["goal"])

    @property
    def memory_text(self) -> str:
        return str(self.data["memory"]["text"])

    @property
    def gold_source_ids(self) -> tuple[str, ...]:
        return tuple(self.data["memory"].get("gold_source_ids", []))

    def fixture(self) -> dict[str, Any]:
        return load_fixture(self.dataset_root / self.data["environment"]["fixture_path"])

    def build_environment(self) -> Any:
        """Return a freshly reset environment for this case.

        Each call reads the tracked fixture again, so a positive and its
        cue-ablated twin never share mutable state and may run concurrently.
        """

        return get_environment(self.data["environment"]["adapter"], self.fixture())


def load_workflow_cases(dataset_dir: str | Path) -> list[WorkflowCase]:
    root = Path(dataset_dir)
    path = root / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Workflow dataset file not found: {path}")
    manifest_path = root / "dataset_manifest.json"
    corpus_roots: dict[str, Path] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError(f"{manifest_path}: unsupported workflow manifest schema")
        for entry in manifest.get("corpora", []):
            corpus_roots[str(entry["corpus_id"])] = (
                root / str(entry["source_root"])
            ).resolve()
    cases: list[WorkflowCase] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        corpus_id = str(data.get("corpus_id", ""))
        cases.append(
            WorkflowCase(
                data=data,
                dataset_root=root.resolve(),
                corpus_root=corpus_roots.get(
                    corpus_id, (root / "corpora" / corpus_id / "source").resolve()
                ),
            )
        )
    return cases


def validate_workflow_cases(cases: list[WorkflowCase]) -> None:
    issues: list[WorkflowCaseIssue] = []
    seen: set[str] = set()
    bases: dict[str, dict[str, WorkflowCase]] = {}

    for case in cases:
        case_id = case.case_id
        if case_id in seen:
            issues.append(WorkflowCaseIssue(case_id, "duplicate case_id"))
        seen.add(case_id)
        for field in sorted(REQUIRED_FIELDS - set(case.data)):
            issues.append(
                WorkflowCaseIssue(case_id, f"missing required field '{field}'")
            )
        bases.setdefault(case.base_case_id, {})[case.variant] = case
        _validate_case(case, issues)

    for base_case_id, variants in bases.items():
        if set(variants) != VARIANTS:
            issues.append(
                WorkflowCaseIssue(
                    base_case_id,
                    "must contain positive, cue-ablated, and memory-included "
                    "variants",
                )
            )
            continue
        if len({case.corpus_id for case in variants.values()}) != 1:
            issues.append(
                WorkflowCaseIssue(
                    base_case_id, "all triplet variants must use the same corpus_id"
                )
            )
        _validate_triplet(base_case_id, variants, issues)

    if issues:
        raise WorkflowCaseValidationError(issues)


def _validate_case(case: WorkflowCase, issues: list[WorkflowCaseIssue]) -> None:
    case_id = case.case_id
    data = case.data
    variant = case.variant
    if variant not in VARIANTS:
        issues.append(WorkflowCaseIssue(case_id, "invalid variant"))
        return

    goal = case.goal
    if not goal.strip():
        issues.append(WorkflowCaseIssue(case_id, "goal must be non-empty"))
    folded_goal = goal.casefold()

    environment = data.get("environment", {})
    upstream = environment.get("upstream", {})
    for field in ("repository", "revision", "task_path", "license"):
        if not str(upstream.get(field, "")).strip():
            issues.append(
                WorkflowCaseIssue(
                    case_id, f"environment.upstream.{field} must be recorded"
                )
            )
    fixture_path = case.dataset_root / str(environment.get("fixture_path", ""))
    if not fixture_path.is_file():
        issues.append(
            WorkflowCaseIssue(case_id, f"missing fixture {environment.get('fixture_path')}")
        )
        return

    cue = data.get("cue", {})
    cue_text = str(cue.get("text", ""))
    if not cue_text.strip():
        issues.append(WorkflowCaseIssue(case_id, "cue.text must be non-empty"))
    if bool(cue.get("present")) != (variant in CUE_PRESENT_VARIANTS):
        issues.append(WorkflowCaseIssue(case_id, "cue.present disagrees with variant"))
    if cue_text and cue_text.casefold() in folded_goal:
        issues.append(WorkflowCaseIssue(case_id, "cue leaks into goal"))

    observation = _cue_location_observation(case, issues)
    if observation is not None and cue_text:
        present = cue_text in observation
        if variant in CUE_PRESENT_VARIANTS and not present:
            issues.append(
                WorkflowCaseIssue(
                    case_id, "gold cue is absent from the declared cue location"
                )
            )
        if variant == "cue-ablated" and present:
            issues.append(
                WorkflowCaseIssue(case_id, "ablated fixture still contains the cue")
            )

    memory = data.get("memory", {})
    memory_text = str(memory.get("text", ""))
    if not memory_text.strip():
        issues.append(WorkflowCaseIssue(case_id, "memory text must be readable prose"))
    if memory.get("corpus_id") != case.corpus_id:
        issues.append(WorkflowCaseIssue(case_id, "case and memory corpus_id disagree"))
    if variant == "memory-included":
        if memory_text and memory_text.casefold() not in folded_goal:
            issues.append(
                WorkflowCaseIssue(
                    case_id, "memory-included goal must contain the injected memory"
                )
            )
    elif memory_text and memory_text.casefold() in folded_goal:
        issues.append(WorkflowCaseIssue(case_id, "memory text leaks into goal"))

    sources = memory.get("sources", [])
    gold_ids = set(memory.get("gold_source_ids", []))
    if gold_ids != {str(source.get("source_id")) for source in sources}:
        issues.append(WorkflowCaseIssue(case_id, "gold source IDs and sources disagree"))
    for source in sources:
        _validate_source(case, source, issues, require_evidence=True)
    distractors = data.get("distractors", {}).get("sources", [])
    if len(distractors) < MINIMUM_DISTRACTORS:
        issues.append(
            WorkflowCaseIssue(
                case_id, f"needs at least {MINIMUM_DISTRACTORS} memory distractors"
            )
        )
    for source in distractors:
        _validate_source(case, source, issues, require_evidence=False)

    decisive = data.get("decisive_action", {})
    matchers = decisive.get("matchers", [])
    if not matchers:
        issues.append(
            WorkflowCaseIssue(
                case_id, "decisive_action.matchers must list at least one call"
            )
        )
    for matcher in matchers:
        if not str(matcher.get("tool", "")).strip():
            issues.append(
                WorkflowCaseIssue(case_id, "each decisive_action matcher needs a tool")
            )

    assertions = data.get("assertions", [])
    if not isinstance(assertions, list) or not assertions:
        issues.append(WorkflowCaseIssue(case_id, "assertions must be a non-empty list"))
        return
    assertion_ids: set[str] = set()
    decisive_count = 0
    for assertion in assertions:
        assertion_id = str(assertion.get("id", ""))
        if not assertion_id:
            issues.append(WorkflowCaseIssue(case_id, "each assertion needs an id"))
        if assertion_id in assertion_ids:
            issues.append(
                WorkflowCaseIssue(case_id, f"duplicate assertion id {assertion_id!r}")
            )
        assertion_ids.add(assertion_id)
        role = str(assertion.get("role", ""))
        if role not in ASSERTION_ROLES:
            issues.append(
                WorkflowCaseIssue(
                    case_id, f"assertion {assertion_id!r} has an invalid role {role!r}"
                )
            )
        decisive_count += role == "decisive"
        if not str(assertion.get("rationale", "")).strip():
            issues.append(
                WorkflowCaseIssue(
                    case_id, f"assertion {assertion_id!r} needs a rationale"
                )
            )
    if decisive_count == 0:
        issues.append(WorkflowCaseIssue(case_id, "needs at least one decisive assertion"))


def _validate_source(
    case: WorkflowCase,
    source: dict[str, Any],
    issues: list[WorkflowCaseIssue],
    *,
    require_evidence: bool,
) -> None:
    case_id = case.case_id
    source_id = str(source.get("source_id", ""))
    perturbations = [str(label) for label in source.get("perturbations", [])]
    if require_evidence and "poison" in perturbations:
        issues.append(WorkflowCaseIssue(case_id, "poison source cannot be gold"))
    path = case.corpus_root / str(source.get("path", ""))
    if not path.is_file():
        issues.append(
            WorkflowCaseIssue(case_id, f"missing memory source {source.get('path')}")
        )
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source.get("sha256"):
        issues.append(
            WorkflowCaseIssue(case_id, f"source hash mismatch {source.get('path')}")
        )
    if not require_evidence:
        return
    span = source.get("evidence_span")
    span_text = span.get("text") if isinstance(span, dict) else None
    if not isinstance(span_text, str) or not span_text.strip():
        issues.append(
            WorkflowCaseIssue(
                case_id, f"gold source {source_id} needs a non-empty evidence_span.text"
            )
        )
        return
    text = path.read_text(encoding="utf-8")
    if span_text not in text:
        issues.append(
            WorkflowCaseIssue(
                case_id,
                f"evidence_span for {source_id} is not verbatim in "
                f"{source.get('path')}",
            )
        )
    if span_text.casefold() in case.goal.casefold():
        issues.append(
            WorkflowCaseIssue(case_id, f"evidence_span for {source_id} leaks into goal")
        )


def _cue_location_observation(
    case: WorkflowCase, issues: list[WorkflowCaseIssue]
) -> str | None:
    """Render the declared cue location by actually calling the environment."""

    location = case.data.get("cue", {}).get("location", {})
    tool = location.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        issues.append(
            WorkflowCaseIssue(case.case_id, "cue.location.tool must name a tool")
        )
        return None
    try:
        environment = case.build_environment()
        result = environment.invoke(tool, dict(location.get("arguments", {})))
    except Exception as exc:  # adapter or fixture problem, not an agent problem
        issues.append(
            WorkflowCaseIssue(case.case_id, f"cue location is not reachable: {exc}")
        )
        return None
    if not result.ok:
        issues.append(
            WorkflowCaseIssue(
                case.case_id, f"cue location returned an error: {result.text}"
            )
        )
        return None
    return result.text


def _validate_triplet(
    base_case_id: str,
    variants: dict[str, WorkflowCase],
    issues: list[WorkflowCaseIssue],
) -> None:
    positive = variants["positive"]
    control = variants["cue-ablated"]
    ceiling = variants["memory-included"]

    if positive.gold_source_ids != ceiling.gold_source_ids:
        issues.append(
            WorkflowCaseIssue(
                base_case_id,
                "positive and memory-included must share the same gold sources",
            )
        )
    if control.data.get("expects_intervention", False):
        issues.append(
            WorkflowCaseIssue(
                base_case_id, "cue-ablated variant must not expect a memory intervention"
            )
        )
    if not positive.data.get("expects_intervention", False):
        issues.append(
            WorkflowCaseIssue(
                base_case_id, "positive variant must expect a memory intervention"
            )
        )
    positive_decisive = _decisive_signature(positive)
    control_decisive = _decisive_signature(control)
    if positive_decisive == control_decisive:
        issues.append(
            WorkflowCaseIssue(
                base_case_id,
                "positive and cue-ablated decisive assertions are identical, so "
                "memory cannot change the outcome",
            )
        )


def _decisive_signature(case: WorkflowCase) -> str:
    decisive = [
        {key: value for key, value in assertion.items() if key != "rationale"}
        for assertion in case.data.get("assertions", [])
        if assertion.get("role") == "decisive"
    ]
    return json.dumps(decisive, sort_keys=True)
