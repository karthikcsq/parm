from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from .amara import sha256_file


TOKENIZER = "cl100k_base"
MIN_CONTEXT_TOKENS = 8_000
MAX_CONTEXT_TOKENS = 12_000
VARIANTS = {"positive", "cue-ablated", "memory-included"}
CUE_PRESENT_VARIANTS = {"positive", "memory-included"}
OBSERVATION_KINDS = {"assistant_output", "tool_result"}
LISTING_PREFIXES = (
    "Session ",
    "Workshop ",
    "Lead Story ",
    "Item ",
    "Brief ",
    "Editor's Pick ",
    "Episode ",
    "New Release ",
    "Vendor ",
    "Candidate ",
    "Result ",
    "Listing ",
    "Demo ",
    "Webinar ",
    "Company ",
    "Case Study ",
    "Article ",
    "Automation ",
    "Feature ",
    "Film ",
    "Event ",
    "Essay ",
    "Chart ",
)
REQUIRED_FIELDS = {
    "case_id",
    "base_case_id",
    "corpus_id",
    "variant",
    "prompt",
    "observation",
    "cue",
    "memory",
    "decisions",
    "distractors",
    "provenance",
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
    root = Path(dataset_dir)
    path = root / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    corpus_roots = _load_corpus_roots(root)
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if "corpus_id" not in case:
                legacy_corpus_id = case.get("memory", {}).get("corpus_id")
                if legacy_corpus_id:
                    case["corpus_id"] = legacy_corpus_id
            case["_dataset_root"] = str(root.resolve())
            case["_corpus_roots"] = corpus_roots
            case["observation_text"] = observation_text(case, root)
            cases.append(case)
    return cases


def _load_corpus_roots(root: Path) -> dict[str, str]:
    path = root / "dataset_manifest.json"
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: expected an object")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported dataset manifest schema")
    corpora = manifest.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise ValueError(f"{path}: corpora must be a non-empty list")
    roots: dict[str, str] = {}
    for entry in corpora:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: corpus entries must be objects")
        corpus_id = entry.get("corpus_id")
        source_root = entry.get("source_root")
        if not isinstance(corpus_id, str) or not corpus_id.strip():
            raise ValueError(f"{path}: corpus_id must be non-empty")
        if corpus_id in roots:
            raise ValueError(f"{path}: duplicate corpus_id {corpus_id!r}")
        if not isinstance(source_root, str) or not source_root.strip():
            raise ValueError(f"{path}: source_root must be non-empty")
        roots[corpus_id] = str((root / source_root).resolve())
    return roots


def observation_text(case: dict[str, Any], dataset_root: str | Path) -> str:
    observation = case["observation"]
    path = Path(dataset_root) / observation["content_path"]
    text = path.read_text(encoding="utf-8")
    for replacement in observation.get("replacements", []):
        old = replacement["old"]
        if old not in text:
            raise ValueError(f"{case['case_id']}: replacement source text not found")
        text = text.replace(old, replacement["new"], 1)
    return text


def validate_cases(cases: list[dict[str, Any]]) -> None:
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    bases: dict[str, set[str]] = {}
    base_corpora: dict[str, set[str]] = {}
    encoding = tiktoken.get_encoding(TOKENIZER)

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", f"<case {index}>"))
        for field in sorted(REQUIRED_FIELDS - set(case)):
            issues.append(ValidationIssue(case_id, f"missing required field '{field}'"))
        if case_id in seen:
            issues.append(ValidationIssue(case_id, "duplicate case_id"))
        seen.add(case_id)
        base_case_id = str(case.get("base_case_id"))
        bases.setdefault(base_case_id, set()).add(
            str(case.get("variant"))
        )
        base_corpora.setdefault(base_case_id, set()).add(
            str(case.get("corpus_id"))
        )
        _validate_case(case, case_id, encoding, issues)

    for base_case_id, variants in bases.items():
        if variants != VARIANTS:
            issues.append(
                ValidationIssue(
                    base_case_id,
                    "must contain positive, cue-ablated, and memory-included "
                    "variants",
                )
            )
        if len(base_corpora[base_case_id]) != 1:
            issues.append(
                ValidationIssue(
                    base_case_id,
                    "all triplet variants must use the same corpus_id",
                )
            )
    if issues:
        raise DatasetValidationError(issues)


def _validate_case(
    case: dict[str, Any],
    case_id: str,
    encoding: Any,
    issues: list[ValidationIssue],
) -> None:
    variant = case.get("variant")
    if variant not in VARIANTS:
        issues.append(ValidationIssue(case_id, "invalid variant"))
    observation = case.get("observation", {})
    if observation.get("kind") not in OBSERVATION_KINDS:
        issues.append(ValidationIssue(case_id, "invalid observation.kind"))
    corpus_id = case.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        issues.append(ValidationIssue(case_id, "corpus_id must be non-empty"))

    text = str(case.get("observation_text", ""))
    token_count = len(encoding.encode(text))
    if not MIN_CONTEXT_TOKENS <= token_count <= MAX_CONTEXT_TOKENS:
        issues.append(
            ValidationIssue(
                case_id,
                f"observation has {token_count} tokens; expected "
                f"{MIN_CONTEXT_TOKENS}-{MAX_CONTEXT_TOKENS} using {TOKENIZER}",
            )
        )

    prompt = str(case.get("prompt", "")).casefold()
    cue = case.get("cue", {})
    cue_text = str(cue.get("text", ""))
    if cue_text.casefold() in prompt:
        issues.append(ValidationIssue(case_id, "cue leaks into prompt"))
    if bool(cue.get("present")) != (variant in CUE_PRESENT_VARIANTS):
        issues.append(ValidationIssue(case_id, "cue.present disagrees with variant"))
    if variant in CUE_PRESENT_VARIANTS and cue_text not in text:
        issues.append(ValidationIssue(case_id, "gold cue is absent from observation"))
    if variant == "cue-ablated" and cue_text in text:
        issues.append(ValidationIssue(case_id, "ablated observation still contains cue"))
    if variant == "memory-included":
        memory_text = str(case.get("memory", {}).get("text", ""))
        if not memory_text or memory_text.casefold() not in prompt:
            issues.append(
                ValidationIssue(
                    case_id,
                    "memory-included prompt must contain the injected memory",
                )
            )

    decisions = case.get("decisions", {})
    if decisions.get("answer_type") != "natural_language_choice":
        issues.append(
            ValidationIssue(case_id, "decisions.answer_type must be natural_language_choice")
        )
    output_choice = decisions.get("output_only", {}).get("choice")
    memory_choice = decisions.get("memory_conditioned", {}).get("choice")
    if "exactly one" not in prompt:
        issues.append(ValidationIssue(case_id, "prompt must request exactly one choice"))
    for label, choice in (
        ("output-only", output_choice),
        ("memory-conditioned", memory_choice),
    ):
        if not isinstance(choice, str) or not choice.strip():
            issues.append(ValidationIssue(case_id, f"{label} choice must be natural language"))
        elif text.casefold().count(choice.casefold()) != 1:
            issues.append(
                ValidationIssue(
                    case_id,
                    f"{label} choice must appear exactly once in the observation",
                )
            )
    if variant in CUE_PRESENT_VARIANTS and output_choice == memory_choice:
        issues.append(ValidationIssue(case_id, "positive case does not change decision"))
    if variant == "cue-ablated" and output_choice != memory_choice:
        issues.append(ValidationIssue(case_id, "control case changes decision"))

    visible_labels = [
        line.split(".", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith(LISTING_PREFIXES)
    ]
    distractor_sources = case.get("distractors", {}).get("sources", [])
    if len(visible_labels) < 25:
        issues.append(ValidationIssue(case_id, "needs at least 25 visible choices"))
    if len(visible_labels) != len({label.casefold() for label in visible_labels}):
        issues.append(
            ValidationIssue(case_id, "visible choice labels must be unique")
        )
    if len(distractor_sources) < 3:
        issues.append(ValidationIssue(case_id, "needs at least 3 memory distractors"))

    memory = case.get("memory", {})
    if memory.get("corpus_id") != corpus_id:
        issues.append(
            ValidationIssue(case_id, "case and memory corpus_id disagree")
        )
    if not str(memory.get("text", "")).strip():
        issues.append(ValidationIssue(case_id, "memory text must be readable prose"))
    source_ids = set(memory.get("gold_source_ids", []))
    sources = memory.get("sources", [])
    if source_ids != {source.get("source_id") for source in sources}:
        issues.append(ValidationIssue(case_id, "gold source IDs and sources disagree"))
    corpus_roots = case.get("_corpus_roots", {})
    configured_root = (
        corpus_roots.get(corpus_id)
        if isinstance(corpus_roots, dict)
        else None
    )
    corpus_root = (
        Path(configured_root)
        if configured_root is not None
        else (
            Path(case.get("_dataset_root", ".")).parent
            / str(corpus_id)
            / "source"
        )
    )
    for source in sources:
        if "poison" in source.get("perturbations", []):
            issues.append(ValidationIssue(case_id, "poison source cannot be gold"))
        path = corpus_root / source.get("path", "")
        if not path.exists():
            issues.append(
                ValidationIssue(case_id, f"missing memory source {source.get('path')}")
            )
        elif sha256_file(path) != source.get("sha256"):
            issues.append(
                ValidationIssue(case_id, f"source hash mismatch {source.get('path')}")
            )
        # A per-file corpus source must carry the frozen-index slug convention:
        # its source_id collection prefix equals the path's top-level directory
        # (e.g. notes/... not note/...). Bundle files map many IDs to one path
        # and are exempt.
        source_path = str(source.get("path", ""))
        source_id = str(source.get("source_id", ""))
        if source_path.endswith(".md") and "/" in source_path and "/" in source_id:
            path_prefix = source_path.split("/", 1)[0]
            id_prefix = source_id.split("/", 1)[0]
            if id_prefix != path_prefix:
                issues.append(
                    ValidationIssue(
                        case_id,
                        f"source_id prefix {id_prefix!r} does not match "
                        f"corpus directory {path_prefix!r} for {source_path}",
                    )
                )
