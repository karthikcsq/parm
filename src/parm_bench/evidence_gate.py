"""Automated evidence-support gate for candidate benchmark scenarios.

The gate answers one question: does raw, user-authored conversation text
support a proposed personal-memory claim? Deterministic pre-checks run first
and can reject a claim without any model call. Claims that survive them are
graded by a cached LLM rubric.

The judge never sees the deterministic flags. Keeping it blind stops the
pre-checks from steering the grade, so the two signals stay independent.
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


EVIDENCE_GATE_MODEL = "gpt-5-mini"
EVIDENCE_GATE_RUBRIC = "personamem_source_support_v2"
EVIDENCE_GATE_GRADES = ("explicit", "inferable", "unsupported")
PASSING_GRADES = ("explicit", "inferable")

UNJUSTIFIED_FREQUENCY = "unjustified_frequency"
ASSISTANT_ONLY_EVIDENCE = "assistant_only_evidence"
NO_USER_EVIDENCE = "no_user_evidence"
TOPICAL_QUESTION_ONLY = "topical_question_only"
NO_LEXICAL_ANCHOR = "no_lexical_anchor"

HARD_REJECTION_REASONS = (
    UNJUSTIFIED_FREQUENCY,
    ASSISTANT_ONLY_EVIDENCE,
    NO_USER_EVIDENCE,
)
SOFT_FLAG_REASONS = (
    TOPICAL_QUESTION_ONLY,
    NO_LEXICAL_ANCHOR,
)

MINIMUM_HABIT_STATEMENTS = 2

EVIDENCE_GATE_INSTRUCTIONS = """\
Judge whether an indexed conversation supports a proposed personal-memory
claim. Evaluate only the conversation text. Do not assume a hidden persona,
profile, benchmark label, or unstated fact.

Grades:
- explicit: the user directly states or clearly demonstrates the core claim.
- inferable: the core claim follows from the user's words with one small,
  ordinary inference, and no key activity, relation, or identity has to be
  invented.
- unsupported: a key topic, relation, preference, frequency, or identity in
  the claim is absent, or the source could support many incompatible claims.

Apply these rules:
- Assistant suggestions, rewrites, and recommendations are not evidence of the
  user's preference unless the user accepts or independently states them.
- A user question about a topic is never by itself support for a durable
  preference. Curiosity about a subject is not a taste, habit, routine, or
  commitment.
- One instance never supports a frequency word such as "usually", "often",
  "regularly", "always", or "exclusively". A frequency claim needs repeated
  user-authored evidence.
- An inference must not invent a key activity, relation, or identity the user
  never mentioned. If the claim names something absent from the user's own
  words, grade it unsupported.

Keep the rationale short and quote at most two short pieces of source
evidence.
"""

_FREQUENCY_WORDS = frozenset(
    {
        "always",
        "constantly",
        "consistently",
        "continually",
        "customarily",
        "daily",
        "each",
        "every",
        "everyday",
        "frequent",
        "frequently",
        "generally",
        "habitual",
        "habitually",
        "invariably",
        "monthly",
        "never",
        "nightly",
        "normally",
        "often",
        "ordinarily",
        "regular",
        "regularly",
        "repeatedly",
        "routine",
        "routinely",
        "typical",
        "typically",
        "usual",
        "usually",
        "weekly",
        "whenever",
        "yearly",
        "exclusive",
        "exclusively",
    }
)
_FREQUENCY_PHRASES = (
    "all the time",
    "every day",
    "every week",
    "every night",
    "every morning",
    "every time",
    "as a rule",
    "day in and day out",
)
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "more",
        "most",
        "my",
        "of",
        "on",
        "one",
        "or",
        "our",
        "out",
        "over",
        "she",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "user",
        "users",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
_WORD_PATTERN = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class EvidenceGateCachePolicy(str, Enum):
    POPULATE = "populate"
    FROZEN = "frozen"


class EvidenceGateCacheMissError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceGateResult:
    """Final support verdict for one (claim, evidence) pair."""

    grade: str
    rationale: str
    evidence: tuple[str, ...] = ()
    deterministic_rejections: tuple[str, ...] = ()
    rubric_version: str = EVIDENCE_GATE_RUBRIC
    model: str = EVIDENCE_GATE_MODEL
    resolved_model: str | None = None
    response_id: str | None = None

    @property
    def hard_rejections(self) -> tuple[str, ...]:
        return tuple(
            reason
            for reason in self.deterministic_rejections
            if reason in HARD_REJECTION_REASONS
        )

    @property
    def soft_flags(self) -> tuple[str, ...]:
        return tuple(
            reason
            for reason in self.deterministic_rejections
            if reason not in HARD_REJECTION_REASONS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
            "deterministic_rejections": list(self.deterministic_rejections),
            "hard_rejections": list(self.hard_rejections),
            "soft_flags": list(self.soft_flags),
            "rubric_version": self.rubric_version,
            "requested_model": self.model,
            "resolved_model": self.resolved_model,
            "response_id": self.response_id,
        }


class SupportJudge(Protocol):
    model_name: str
    rubric_version: str

    def judge(
        self,
        *,
        claim: str,
        evidence_turns: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]: ...


class CachedOpenAISupportJudge:
    """Cached gpt-5-mini support judge over the versioned rubric."""

    def __init__(
        self,
        cache_dir: str | Path,
        policy: EvidenceGateCachePolicy | str = (
            EvidenceGateCachePolicy.POPULATE
        ),
        *,
        model: str = EVIDENCE_GATE_MODEL,
        rubric_version: str = EVIDENCE_GATE_RUBRIC,
        instructions: str = EVIDENCE_GATE_INSTRUCTIONS,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.policy = EvidenceGateCachePolicy(policy)
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
        evidence_turns: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        input_text = render_evidence_request(claim, evidence_turns)
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
        if self.policy is EvidenceGateCachePolicy.FROZEN:
            raise EvidenceGateCacheMissError(
                f"missing evidence gate cache entry: {request_hash}"
            )

        response = self.client.responses.create(
            model=self.model_name,
            instructions=self.instructions,
            input=input_text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "personamem_source_support",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "grade": {
                                "type": "string",
                                "enum": list(EVIDENCE_GATE_GRADES),
                            },
                            "rationale": {"type": "string"},
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 2,
                            },
                        },
                        "required": ["grade", "rationale", "evidence"],
                        "additionalProperties": False,
                    },
                }
            },
            store=False,
        )
        result = json.loads(response.output_text)
        if result["grade"] not in EVIDENCE_GATE_GRADES:
            raise ValueError(f"unknown support grade: {result['grade']}")
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


def grade_claim(
    claim: str,
    evidence_turns: Sequence[Mapping[str, Any]],
    *,
    model: str = EVIDENCE_GATE_MODEL,
    cache_dir: str | Path | None = None,
    policy: EvidenceGateCachePolicy | str = (
        EvidenceGateCachePolicy.POPULATE
    ),
    client: Any | None = None,
    judge: SupportJudge | None = None,
) -> EvidenceGateResult:
    """Grade whether `evidence_turns` support `claim`.

    Deterministic pre-checks run first. A hard rejection short-circuits to
    "unsupported" without a model call. Soft flags are recorded but never
    shown to the judge.
    """

    flags = deterministic_flags(claim, evidence_turns)
    hard = tuple(
        reason for reason in flags if reason in HARD_REJECTION_REASONS
    )
    if hard:
        return EvidenceGateResult(
            grade="unsupported",
            rationale=(
                "Deterministic pre-check rejected the claim: "
                + ", ".join(hard)
                + "."
            ),
            deterministic_rejections=flags,
            model=model,
        )

    if judge is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required when no judge is supplied")
        judge = CachedOpenAISupportJudge(
            cache_dir,
            policy,
            model=model,
            client=client,
        )
    verdict = judge.judge(claim=claim, evidence_turns=evidence_turns)
    return EvidenceGateResult(
        grade=verdict["grade"],
        rationale=verdict.get("rationale", ""),
        evidence=tuple(verdict.get("evidence", ()))[:2],
        deterministic_rejections=flags,
        rubric_version=judge.rubric_version,
        model=judge.model_name,
        resolved_model=verdict.get("resolved_model"),
        response_id=verdict.get("response_id"),
    )


def passes_gate(result: EvidenceGateResult) -> bool:
    return result.grade in PASSING_GRADES


def deterministic_flags(
    claim: str,
    evidence_turns: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Named deterministic reasons to doubt a claim, hard reasons first."""

    turns = list(evidence_turns)
    user_texts = [
        str(turn.get("content", ""))
        for turn in turns
        if _is_user_turn(turn)
    ]
    user_texts = [text for text in user_texts if text.strip()]

    flags: list[str] = []
    if not user_texts:
        flags.append(
            NO_USER_EVIDENCE if not turns else ASSISTANT_ONLY_EVIDENCE
        )
        return tuple(flags)

    claim_lemmas = _content_lemmas(claim) - _frequency_lemmas()
    if _has_frequency_word(claim):
        supporting = _habit_statements(user_texts, claim_lemmas)
        if len(supporting) < MINIMUM_HABIT_STATEMENTS:
            flags.append(UNJUSTIFIED_FREQUENCY)

    if _is_question_only(user_texts):
        flags.append(TOPICAL_QUESTION_ONLY)

    user_lemmas: set[str] = set()
    for text in user_texts:
        user_lemmas |= _content_lemmas(text)
    if claim_lemmas and not (claim_lemmas & user_lemmas):
        flags.append(NO_LEXICAL_ANCHOR)

    return tuple(flags)


def render_evidence_request(
    claim: str,
    evidence_turns: Sequence[Mapping[str, Any]],
) -> str:
    rendered_source = "\n\n".join(
        f"{str(turn.get('role', '')).title()}: {turn.get('content', '')}"
        for turn in evidence_turns
    )
    return (
        f"Memory claim:\n{claim}\n\n"
        f"Indexed source conversation:\n{rendered_source}"
    )


def evidence_gate_cache_key(
    claim: str,
    evidence_turns: Sequence[Mapping[str, Any]],
    *,
    model: str = EVIDENCE_GATE_MODEL,
    rubric_version: str = EVIDENCE_GATE_RUBRIC,
) -> str:
    return _hash_request(
        {
            "rubric_version": rubric_version,
            "model": model,
            "input": render_evidence_request(claim, evidence_turns),
        }
    )


def _hash_request(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(request), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _is_user_turn(turn: Mapping[str, Any]) -> bool:
    return str(turn.get("role", "")).strip().casefold() == "user"


def _has_frequency_word(text: str) -> bool:
    folded = text.casefold()
    if any(phrase in folded for phrase in _FREQUENCY_PHRASES):
        return True
    return any(
        word in _FREQUENCY_WORDS for word in _WORD_PATTERN.findall(folded)
    )


def _frequency_lemmas() -> set[str]:
    return {_lemma(word) for word in _FREQUENCY_WORDS}


def _habit_statements(
    user_texts: Sequence[str],
    claim_lemmas: set[str],
) -> tuple[str, ...]:
    """Distinct declarative user statements that echo the claim's content."""

    seen: dict[str, str] = {}
    for text in user_texts:
        for sentence in _split_sentences(text):
            if _is_question(sentence):
                continue
            lemmas = _content_lemmas(sentence)
            if claim_lemmas and not (lemmas & claim_lemmas):
                continue
            key = " ".join(sorted(lemmas)) or sentence.strip().casefold()
            seen.setdefault(key, sentence.strip())
    return tuple(seen.values())


def _is_question_only(user_texts: Sequence[str]) -> bool:
    sentences = [
        sentence
        for text in user_texts
        for sentence in _split_sentences(text)
        if sentence.strip()
    ]
    if not sentences:
        return False
    return all(_is_question(sentence) for sentence in sentences)


def _is_question(sentence: str) -> bool:
    return sentence.strip().endswith("?")


def _split_sentences(text: str) -> list[str]:
    return [
        part
        for part in _SENTENCE_SPLIT_PATTERN.split(text.strip())
        if part.strip()
    ]


def _content_lemmas(text: str) -> set[str]:
    return {
        _lemma(word)
        for word in _WORD_PATTERN.findall(text.casefold())
        if word not in _STOPWORDS and len(word) > 2
    }


def _lemma(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 4 and word.endswith("ly"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


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
