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
# Model-visible fields are exactly the four `baselines.benchmark_input` reads:
# prompt, observation.kind, observation_text, and corpus_id (plus case_id for
# bookkeeping). Everything else here is evaluator-only and must never reach a
# system under test: `memory` (text, gold_source_ids, sources and their
# evidence_span records), `cue`, `decisions`, `distractors`, and `provenance`.
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
class ValidationProfile:
    """Per-profile validation rules keyed by a manifest `validation_profile`.

    Generic rules run for every profile. The switches below only add or drop
    rules that a dataset family cannot share: the Listing-row envelope that
    Amara and the PersonaMem pilot use, the observation size band, the
    PersonaMem pilot's one-off construction audit, and the PARMBench evidence,
    opacity, isolation, and construction-signature gates.
    """

    name: str
    require_listing_rows: bool = True
    min_context_tokens: int = MIN_CONTEXT_TOKENS
    max_context_tokens: int = MAX_CONTEXT_TOKENS
    run_personamem_pilot_gate: bool = False
    require_evidence_spans: bool = False
    require_prompt_opacity: bool = False
    require_persona_isolation: bool = False
    run_construction_checks: bool = False


DEFAULT_VALIDATION_PROFILE = ValidationProfile(name="default")
VALIDATION_PROFILES: dict[str, ValidationProfile] = {
    "personamem_v2_v0": ValidationProfile(
        name="personamem_v2_v0",
        run_personamem_pilot_gate=True,
    ),
    "personamem_v2_mixed_v0": ValidationProfile(
        name="personamem_v2_mixed_v0",
        require_listing_rows=False,
        run_personamem_pilot_gate=True,
    ),
    "parmbench_v1": ValidationProfile(
        name="parmbench_v1",
        require_listing_rows=False,
        # A retrieval-agnostic schema varies the observation format, so the
        # legacy 8k-12k band is widened rather than dropped.
        min_context_tokens=6_000,
        max_context_tokens=16_000,
        require_evidence_spans=True,
        require_prompt_opacity=True,
        require_persona_isolation=True,
        run_construction_checks=True,
    ),
}


def validation_profile(name: Any) -> ValidationProfile:
    if not isinstance(name, str):
        return DEFAULT_VALIDATION_PROFILE
    return VALIDATION_PROFILES.get(name, DEFAULT_VALIDATION_PROFILE)


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
    corpus_roots, corpus_source_id_prefixes = _load_corpus_settings(root)
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
            case["_corpus_source_id_prefixes"] = corpus_source_id_prefixes
            case["observation_text"] = observation_text(case, root)
            cases.append(case)
    return cases


def _load_corpus_settings(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve each corpus source root and its optional source_id prefix."""

    path = root / "dataset_manifest.json"
    if not path.exists():
        return {}, {}
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
    source_id_prefixes: dict[str, str] = {}
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
        prefix = entry.get("source_id_prefix")
        if isinstance(prefix, str) and prefix.strip():
            source_id_prefixes[corpus_id] = prefix
    return roots, source_id_prefixes


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


def validate_cases(
    cases: list[dict[str, Any]], *, include_profile: bool = True
) -> None:
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    bases: dict[str, set[str]] = {}
    base_corpora: dict[str, set[str]] = {}
    encoding = tiktoken.get_encoding(TOKENIZER)
    dataset_root = Path(cases[0].get("_dataset_root", ".")) if cases else None
    manifest_path = (
        dataset_root / "dataset_manifest.json"
        if dataset_root is not None
        else None
    )
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path is not None and manifest_path.is_file()
        else {}
    )
    profile = validation_profile(manifest.get("validation_profile"))
    source_texts: dict[str, str] = {}

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
        _validate_case(
            case,
            case_id,
            encoding,
            issues,
            profile=profile,
            source_texts=source_texts,
        )

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
    if profile.run_construction_checks:
        from .construction_checks import construction_issues

        issues.extend(
            ValidationIssue(case_id, message)
            for case_id, message in construction_issues(cases)
        )
    if issues:
        raise DatasetValidationError(issues)
    if include_profile and manifest_path is not None and manifest_path.is_file():
        if profile.run_personamem_pilot_gate:
            from .personamem import personamem_pilot_validation_issues

            profile_issues = [
                ValidationIssue(case_id, message)
                for case_id, message in personamem_pilot_validation_issues(cases)
            ]
            if profile_issues:
                raise DatasetValidationError(profile_issues)


def _validate_case(
    case: dict[str, Any],
    case_id: str,
    encoding: Any,
    issues: list[ValidationIssue],
    *,
    profile: ValidationProfile = DEFAULT_VALIDATION_PROFILE,
    source_texts: dict[str, str] | None = None,
) -> None:
    require_listing_rows = profile.require_listing_rows
    if source_texts is None:
        source_texts = {}
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
    if not profile.min_context_tokens <= token_count <= profile.max_context_tokens:
        issues.append(
            ValidationIssue(
                case_id,
                f"observation has {token_count} tokens; expected "
                f"{profile.min_context_tokens}-{profile.max_context_tokens} "
                f"using {TOKENIZER}",
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
    if require_listing_rows and len(visible_labels) < 25:
        issues.append(ValidationIssue(case_id, "needs at least 25 visible choices"))
    if require_listing_rows and len(visible_labels) != len(
        {label.casefold() for label in visible_labels}
    ):
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
        if profile.require_evidence_spans:
            _validate_evidence_span(
                source, case_id, corpus_root, issues, source_texts
            )

    if profile.require_persona_isolation:
        _validate_persona_isolation(case, case_id, corpus_id, sources, issues)
    if profile.require_prompt_opacity:
        _validate_prompt_opacity(
            prompt,
            case_id,
            variant,
            str(memory.get("text", "")),
            memory_choice,
            sources,
            issues,
        )


def _validate_evidence_span(
    source: dict[str, Any],
    case_id: str,
    corpus_root: Path,
    issues: list[ValidationIssue],
    source_texts: dict[str, str],
) -> None:
    """Require a gold source to quote the raw span that supports the memory.

    Raw source spans are canonical benchmark truth, so the span must be present
    verbatim in the tracked source file rather than paraphrased by a builder.
    """

    source_id = str(source.get("source_id", ""))
    span = source.get("evidence_span")
    if not isinstance(span, dict):
        issues.append(
            ValidationIssue(
                case_id, f"gold source {source_id} is missing evidence_span"
            )
        )
        return
    span_text = span.get("text")
    if not isinstance(span_text, str) or not span_text.strip():
        issues.append(
            ValidationIssue(
                case_id,
                f"gold source {source_id} needs a non-empty evidence_span.text",
            )
        )
        return
    path = corpus_root / str(source.get("path", ""))
    if not path.exists():
        return
    key = str(path)
    if key not in source_texts:
        source_texts[key] = path.read_text(encoding="utf-8", errors="replace")
    if span_text not in source_texts[key]:
        issues.append(
            ValidationIssue(
                case_id,
                f"evidence_span for {source_id} is not verbatim in "
                f"{source.get('path')}",
            )
        )


def _validate_persona_isolation(
    case: dict[str, Any],
    case_id: str,
    corpus_id: Any,
    sources: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> None:
    """Keep a case inside the one persona history its provenance declares."""

    persona_id = case.get("provenance", {}).get("persona_id")
    persona_token = "" if persona_id is None else str(persona_id).strip()
    if not persona_token:
        issues.append(
            ValidationIssue(case_id, "provenance.persona_id must be set")
        )
    elif persona_token not in str(corpus_id):
        issues.append(
            ValidationIssue(
                case_id,
                f"corpus_id {corpus_id!r} does not name provenance persona "
                f"{persona_token!r}",
            )
        )
    prefixes = case.get("_corpus_source_id_prefixes", {})
    prefix = prefixes.get(corpus_id) if isinstance(prefixes, dict) else None
    if not prefix:
        issues.append(
            ValidationIssue(
                case_id,
                f"corpus {corpus_id!r} must declare source_id_prefix in the "
                "dataset manifest",
            )
        )
        return
    for source in sources:
        source_id = str(source.get("source_id", ""))
        if not source_id.startswith(prefix):
            issues.append(
                ValidationIssue(
                    case_id,
                    f"source_id {source_id!r} is outside corpus prefix "
                    f"{prefix!r}",
                )
            )


def _validate_prompt_opacity(
    prompt: str,
    case_id: str,
    variant: Any,
    memory_text: str,
    memory_choice: Any,
    sources: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> None:
    """Keep the ordinary prompt from revealing the answer or the memory query.

    The memory-included ceiling injects the memory text on purpose; nothing
    else may appear, including the raw evidence spans behind that memory.
    """

    if (
        variant != "memory-included"
        and memory_text.strip()
        and memory_text.casefold() in prompt
    ):
        issues.append(ValidationIssue(case_id, "memory text leaks into prompt"))
    if isinstance(memory_choice, str) and memory_choice.strip():
        if memory_choice.casefold() in prompt:
            issues.append(
                ValidationIssue(
                    case_id, "memory-conditioned choice leaks into prompt"
                )
            )
    for source in sources:
        span = source.get("evidence_span")
        span_text = span.get("text") if isinstance(span, dict) else None
        if not isinstance(span_text, str) or not span_text.strip():
            continue
        if span_text.casefold() in prompt:
            issues.append(
                ValidationIssue(
                    case_id,
                    f"evidence_span for {source.get('source_id')} leaks into "
                    "prompt",
                )
            )
