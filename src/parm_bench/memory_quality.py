"""Pre-construction memory-quality gate for candidate benchmark claims.

The evidence gate answers whether the user said it. This gate answers a
different question: is the fact worth retaining as memory about the user?
The v1 relevance audit found that many claims clear source support and still
decide nothing, because the drafter treated the user's own editing request, a
topic question, or a transient in-session state as a personal fact.

Deterministic pre-checks run first and reject a claim without any model call.
Claims that survive them are graded by a cached LLM rubric that sees the
claim, the verbatim evidence span, and the source turns.

Unlike the evidence gate, every deterministic reason here is hard. A question
can support the fact that the user asked it, so the evidence gate treats a
question-only span as a soft flag; no question on its own is a durable fact,
so this gate rejects.

The judge never sees the deterministic reasons, for the same reason the
evidence gate hides its flags: the two signals stay independent.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .relevance_taxonomy import (
    EDITING_REQUEST_AS_PREFERENCE,
    EPHEMERAL_EVENT_AS_DURABLE_MEMORY,
    MEMORY_QUALITY_REASONS,
    QUESTION_AS_PERSONAL_FACT,
)
from .service_tier import service_tier_kwargs


MEMORY_QUALITY_MODEL = "gpt-5-mini"
MEMORY_QUALITY_RUBRIC = "parmbench_memory_quality_v1"

MEMORY_QUALITY_CATEGORIES = (
    "preference",
    "constraint",
    "exclusion",
    "active_commitment",
    "stable_relationship",
    "owned_item",
    "accessibility_need",
    "concrete_schedule",
    "none",
)
DURABILITY_LEVELS = ("durable", "currently_operative", "ephemeral")
PASSING_DURABILITY = ("durable", "currently_operative")

# Every deterministic reason this gate raises stops the claim outright.
HARD_REJECTION_REASONS = (
    EDITING_REQUEST_AS_PREFERENCE,
    QUESTION_AS_PERSONAL_FACT,
    EPHEMERAL_EVENT_AS_DURABLE_MEMORY,
)

MAX_SENSITIVE_TERMS = 5

MEMORY_QUALITY_INSTRUCTIONS = """\
Judge whether a personal-fact claim about a user is worth retaining as
memory. Source support is already settled elsewhere: assume the user's own
words entail the claim. Your question is whether keeping this fact would help
an assistant choose better for this person later.

You are shown the claim, the verbatim span of the user's own words it came
from, and the source turns around that span. Judge the claim against those
three. Do not assume a hidden persona, profile, or benchmark label.

Pick the category that fits the fact best, or "none" when none fits:
- preference: a standing taste that ranks ordinary options.
- constraint: something the person has to work around.
- exclusion: something the person avoids or refuses.
- active_commitment: an ongoing project, obligation, or practice.
- stable_relationship: an enduring tie to a person, group, or organisation.
- owned_item: a thing the person owns that changes what suits them.
- accessibility_need: a physical or sensory need an option has to meet.
- concrete_schedule: a recurring or fixed time commitment.

Set durability:
- durable: the fact holds well beyond this conversation.
- currently_operative: the fact is live now and governs near-term choices,
  such as an ongoing project, a booked trip, or a current course of treatment.
- ephemeral: the fact is spent. A one-off anecdote or a completed event with
  no current commitment is ephemeral, and so is a passing mood, a resolved
  incident, or a state that exists only inside this conversation. A single
  narrated visit, purchase, or outing is one occasion, not a preference; mark
  it ephemeral instead of reading a taste into it.

Use a rejection reason only when it applies:
- topic_overlap_without_decision_relevance: the claim records interest in a
  subject but carries no choice implication. If knowing it would not make one
  ordinary option better than another for this person, reject it here. A broad
  interest is not a preference.
- ephemeral_event_as_durable_memory: the claim is worded as a standing fact
  while the span describes one finished occasion.
- question_as_personal_fact: the span is the user asking about something and
  the claim converts that curiosity into a taste, habit, or commitment.
- editing_request_as_preference: the span is the user asking for text to be
  written, rewritten, translated, or polished, and the claim turns that one
  request into a standing preference.
- medical_or_sensitive_overreach: the claim asserts a health, identity, or
  financial conclusion the raw span does not state. A mentioned symptom is not
  a diagnosis, a past procedure is not an ongoing condition, and a mentioned
  cost is not a financial situation. Any sensitive inference that reaches
  beyond the raw span belongs here.

Set memory_worthy to true only when the fact fits one of the listed
categories, its durability is durable or currently_operative, and no rejection
reason applies.

Set sensitive to true when the claim touches health, disability, religion,
sexuality, ethnicity, immigration status, or finances, whether or not the
claim is worth retaining. Put the short phrases from the claim that carry that
sensitivity in sensitive_terms, quoting the claim's own words, and leave
sensitive_terms empty when sensitive is false.

Keep the rationale to one or two sentences.
"""

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_FIRST_WORD_PATTERN = re.compile(r"[a-z']+")

_POLITE_PREFIX_PATTERN = re.compile(
    r"^(?:hi|hey|hello|ok|okay|so|also|and|then|quick one|one more thing|"
    r"thanks|thank you|good morning|good afternoon|good evening)"
    r"\b[\s,.:;!-]*"
)
_EDITING_VERB = (
    r"(?:re-?writ(?:e|ing)|re-?phras(?:e|ing)|re-?word(?:ing)?|"
    r"polish(?:ing)?|proof-?read(?:ing)?|edit(?:ing)?|revis(?:e|ing)|"
    r"refin(?:e|ing)|translat(?:e|ing)|summari[sz](?:e|ing)|"
    r"condens(?:e|ing)|shorten(?:ing)?|tighten(?:ing)?|clean up|tidy up|"
    r"punch up|make (?:this|it|that) (?:sound|read|flow)|"
    r"word (?:this|it|that)|phrase (?:this|it|that))"
)
_EDITING_IMPERATIVE_PATTERN = re.compile(
    r"^(?:please\s+)?" + _EDITING_VERB + r"\b"
)
_EDITING_REQUEST_FRAME_PATTERN = re.compile(
    r"\b(?:please|can you|could you|would you|will you|"
    r"i(?:'d|d| would) like you to|i want you to|i need you to|help me)\b"
    r"(?:\W+\w+){0,4}?\W+" + _EDITING_VERB + r"\b"
)

# Only wh-words open a question confidently enough to count a span with no
# terminal punctuation as interrogative. "Can you polish this" opens with an
# auxiliary and is an editing request first.
_INTERROGATIVE_OPENERS = frozenset(
    {"what", "why", "how", "when", "where", "which", "who", "whom", "whose"}
)

_IN_SESSION_VERB_PATTERN = re.compile(
    r"\b(?:is|are|am)\s+(?:currently\s+|right now\s+|just\s+|now\s+)?"
    r"(?:writing|drafting|composing|typing|sending|asking|inquiring|"
    r"requesting|wording|phrasing|chatting|talking|messaging|texting|"
    r"editing|revising|rewriting|rephrasing|translating|summari[sz]ing|"
    r"proofreading|polishing)\b"
)
_IN_SESSION_REFERENTS = (
    "this message",
    "this email",
    "this e-mail",
    "this note",
    "this paragraph",
    "this text",
    "this passage",
    "this draft",
    "this sentence",
    "this letter",
    "this post",
    "this reply",
    "this response",
    "this question",
    "this request",
    "this conversation",
    "this chat",
    "this thread",
    "this session",
    "right now",
    "at the moment",
    "currently",
    "just now",
)


class MemoryQualityCachePolicy(str, Enum):
    POPULATE = "populate"
    FROZEN = "frozen"


class MemoryQualityCacheMissError(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryQualityResult:
    """Retention verdict for one (claim, span, evidence) triple."""

    memory_worthy: bool
    category: str
    durability: str
    rationale: str
    rejection_reasons: tuple[str, ...] = ()
    sensitive: bool = False
    sensitive_terms: tuple[str, ...] = ()
    deterministic_rejections: tuple[str, ...] = ()
    rubric_version: str = MEMORY_QUALITY_RUBRIC
    model: str = MEMORY_QUALITY_MODEL
    resolved_model: str | None = None
    response_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_worthy": self.memory_worthy,
            "category": self.category,
            "durability": self.durability,
            "rationale": self.rationale,
            "rejection_reasons": list(self.rejection_reasons),
            "sensitive": self.sensitive,
            "sensitive_terms": list(self.sensitive_terms),
            "deterministic_rejections": list(self.deterministic_rejections),
            "rubric_version": self.rubric_version,
            "requested_model": self.model,
            "resolved_model": self.resolved_model,
            "response_id": self.response_id,
        }


class MemoryQualityJudge(Protocol):
    model_name: str
    rubric_version: str

    def judge(
        self,
        *,
        claim: str,
        evidence_span: str,
        evidence_turns: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]: ...


class CachedOpenAIMemoryQualityJudge:
    """Cached gpt-5-mini retention judge over the versioned rubric."""

    def __init__(
        self,
        cache_dir: str | Path,
        policy: MemoryQualityCachePolicy | str = (
            MemoryQualityCachePolicy.POPULATE
        ),
        *,
        model: str = MEMORY_QUALITY_MODEL,
        rubric_version: str = MEMORY_QUALITY_RUBRIC,
        instructions: str = MEMORY_QUALITY_INSTRUCTIONS,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.policy = MemoryQualityCachePolicy(policy)
        self.model_name = model
        self.rubric_version = rubric_version
        self.instructions = instructions
        self.client = client
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._used_cache_files: list[Path] = []

    def judge(
        self,
        *,
        claim: str,
        evidence_span: str,
        evidence_turns: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        input_text = render_memory_quality_request(
            claim, evidence_span, evidence_turns
        )
        request = {
            "rubric_version": self.rubric_version,
            "model": self.model_name,
            "input": input_text,
        }
        request_hash = _hash_request(request)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self._used_cache_files.append(cache_path)
            return {
                **cached["result"],
                "resolved_model": cached.get("resolved_model"),
                "response_id": cached.get("response_id"),
            }
        if self.policy is MemoryQualityCachePolicy.FROZEN:
            raise MemoryQualityCacheMissError(
                f"missing memory quality cache entry: {request_hash}"
            )

        response = self.client.responses.create(
            model=self.model_name,
            instructions=self.instructions,
            input=input_text,
            **service_tier_kwargs(),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parmbench_memory_quality",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "memory_worthy": {"type": "boolean"},
                            "category": {
                                "type": "string",
                                "enum": list(MEMORY_QUALITY_CATEGORIES),
                            },
                            "durability": {
                                "type": "string",
                                "enum": list(DURABILITY_LEVELS),
                            },
                            "rejection_reasons": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": list(MEMORY_QUALITY_REASONS),
                                },
                            },
                            "rationale": {"type": "string"},
                            "sensitive": {"type": "boolean"},
                            "sensitive_terms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": MAX_SENSITIVE_TERMS,
                            },
                        },
                        "required": [
                            "memory_worthy",
                            "category",
                            "durability",
                            "rejection_reasons",
                            "rationale",
                            "sensitive",
                            "sensitive_terms",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            store=False,
        )
        result = json.loads(response.output_text)
        _validate_verdict(result)
        resolved_model = response.model
        response_id = response.id
        _atomic_write_json(
            cache_path,
            {
                **request,
                "request_hash": request_hash,
                "result": result,
                "resolved_model": resolved_model,
                "response_id": response_id,
            },
        )
        self._used_cache_files.append(cache_path)
        return {
            **result,
            "resolved_model": resolved_model,
            "response_id": response_id,
        }

    @property
    def cache_hash(self) -> str | None:
        paths = sorted(set(self._used_cache_files))
        if not paths:
            return None
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()


def grade_memory_quality(
    claim: str,
    evidence_span: str,
    evidence_turns: Sequence[Mapping[str, Any]],
    *,
    model: str = MEMORY_QUALITY_MODEL,
    cache_dir: str | Path | None = None,
    policy: MemoryQualityCachePolicy | str = (
        MemoryQualityCachePolicy.POPULATE
    ),
    client: Any | None = None,
    judge: MemoryQualityJudge | None = None,
) -> MemoryQualityResult:
    """Grade whether `claim` is worth retaining as memory about the user.

    Deterministic pre-checks run first and short-circuit without a model call.
    A claim rejected there is reported as ephemeral and not memory-worthy,
    carrying the taxonomy reasons that stopped it.
    """

    rejections = deterministic_rejections(claim, evidence_span)
    if rejections:
        return MemoryQualityResult(
            memory_worthy=False,
            category="none",
            durability="ephemeral",
            rationale=(
                "Deterministic pre-check rejected the claim: "
                + ", ".join(rejections)
                + "."
            ),
            rejection_reasons=rejections,
            deterministic_rejections=rejections,
            model=model,
        )

    if judge is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required when no judge is supplied")
        judge = CachedOpenAIMemoryQualityJudge(
            cache_dir,
            policy,
            model=model,
            client=client,
        )
    verdict = judge.judge(
        claim=claim,
        evidence_span=evidence_span,
        evidence_turns=evidence_turns,
    )
    _validate_verdict(verdict)
    sensitive = bool(verdict.get("sensitive", False))
    terms = tuple(
        str(term).strip()
        for term in verdict.get("sensitive_terms", ())
        if str(term).strip()
    )
    return MemoryQualityResult(
        memory_worthy=bool(verdict["memory_worthy"]),
        category=str(verdict["category"]),
        durability=str(verdict["durability"]),
        rationale=str(verdict.get("rationale", "")),
        rejection_reasons=tuple(verdict.get("rejection_reasons", ())),
        sensitive=sensitive,
        sensitive_terms=terms[:MAX_SENSITIVE_TERMS] if sensitive else (),
        rubric_version=judge.rubric_version,
        model=judge.model_name,
        resolved_model=verdict.get("resolved_model"),
        response_id=verdict.get("response_id"),
    )


def passes_memory_quality(result: MemoryQualityResult) -> bool:
    return (
        result.memory_worthy
        and result.durability in PASSING_DURABILITY
        and not result.rejection_reasons
    )


def deterministic_rejections(
    claim: str,
    evidence_span: str,
) -> tuple[str, ...]:
    """Named hard reasons a claim cannot become durable memory.

    Each reason here is a rejection, never a flag. The gate reports them in a
    fixed order so a cached run and a fresh run agree.
    """

    reasons: list[str] = []
    if _is_editing_request_span(evidence_span):
        reasons.append(EDITING_REQUEST_AS_PREFERENCE)
    if _is_interrogative_span(evidence_span):
        reasons.append(QUESTION_AS_PERSONAL_FACT)
    if _describes_in_session_activity(claim):
        reasons.append(EPHEMERAL_EVENT_AS_DURABLE_MEMORY)
    return tuple(reasons)


def render_memory_quality_request(
    claim: str,
    evidence_span: str,
    evidence_turns: Sequence[Mapping[str, Any]],
) -> str:
    rendered_source = "\n\n".join(
        f"{str(turn.get('role', '')).title()}: {turn.get('content', '')}"
        for turn in evidence_turns
    )
    return (
        f"Memory claim:\n{claim}\n\n"
        f"Evidence span (the user's own words):\n{evidence_span}\n\n"
        f"Indexed source conversation:\n{rendered_source}"
    )


def memory_quality_cache_key(
    claim: str,
    evidence_span: str,
    evidence_turns: Sequence[Mapping[str, Any]],
    *,
    model: str = MEMORY_QUALITY_MODEL,
    rubric_version: str = MEMORY_QUALITY_RUBRIC,
) -> str:
    return _hash_request(
        {
            "rubric_version": rubric_version,
            "model": model,
            "input": render_memory_quality_request(
                claim, evidence_span, evidence_turns
            ),
        }
    )


def _validate_verdict(verdict: Mapping[str, Any]) -> None:
    category = verdict.get("category")
    if category not in MEMORY_QUALITY_CATEGORIES:
        raise ValueError(f"unknown memory category: {category}")
    durability = verdict.get("durability")
    if durability not in DURABILITY_LEVELS:
        raise ValueError(f"unknown durability level: {durability}")
    for reason in verdict.get("rejection_reasons", ()):
        if reason not in MEMORY_QUALITY_REASONS:
            raise ValueError(f"unknown memory quality reason: {reason}")


def _hash_request(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(request), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _is_editing_request_span(span: str) -> bool:
    """Whether the span is nothing but the user asking for text work.

    A span that also carries a statement about the user is left alone. Only a
    span whose sentences are all rewrite, translation, or polish requests
    counts as the request line itself. A bare greeting carries no fact, so it
    neither saves the span nor condemns it.
    """

    requests = 0
    for sentence in _split_sentences(span):
        if _is_greeting_only(sentence):
            continue
        if not _is_editing_request(sentence):
            return False
        requests += 1
    return requests > 0


def _is_greeting_only(sentence: str) -> bool:
    return not _strip_polite_prefix(_normalise(sentence)).strip(" ,.:;!-?")


def _is_editing_request(sentence: str) -> bool:
    folded = _normalise(sentence)
    if not folded:
        return False
    core = _strip_polite_prefix(folded)
    if _EDITING_IMPERATIVE_PATTERN.search(core):
        return True
    return bool(_EDITING_REQUEST_FRAME_PATTERN.search(folded))


def _strip_polite_prefix(text: str) -> str:
    for _ in range(3):
        stripped = _POLITE_PREFIX_PATTERN.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return text


def _is_interrogative_span(span: str) -> bool:
    sentences = _split_sentences(span)
    if not sentences:
        return False
    return all(_is_interrogative(sentence) for sentence in sentences)


def _is_interrogative(sentence: str) -> bool:
    text = sentence.strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    if text.endswith((".", "!")):
        return False
    match = _FIRST_WORD_PATTERN.match(text.casefold())
    return bool(match) and match.group(0) in _INTERROGATIVE_OPENERS


def _describes_in_session_activity(claim: str) -> bool:
    """Whether the claim's predicate is an activity inside this conversation.

    Deliberately narrow: the claim has to put the user in the middle of
    writing, asking, or chatting and point at the conversation itself. A claim
    about writing a book names an outside referent and survives.
    """

    folded = _normalise(claim)
    if not _IN_SESSION_VERB_PATTERN.search(folded):
        return False
    return any(referent in folded for referent in _IN_SESSION_REFERENTS)


def _split_sentences(text: str) -> list[str]:
    return [
        part
        for part in _SENTENCE_SPLIT_PATTERN.split(text.strip())
        if part.strip()
    ]


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().casefold())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
