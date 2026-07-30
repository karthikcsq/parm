"""Post-construction decision-validity gate for built scenarios.

The evidence gate asks whether the user said it. This gate asks the next
question: does the built task, cue, and choice change follow from the accepted
fact without inventing anything? The v1 relevance calibration audit
(`data/parmbench-v1-supply/relevance-audit-v1.md`) found scenarios that wrap a
supported fact in a strained task, so source support alone cannot decide a
scenario is usable.

The auditor is deliberately given the fixture roles, including which option is
the ordinary winner and which is the memory-conditioned target. It is an
evaluator-only quality check on fixtures. It must never be used to score a
system under test: nothing here may run inside benchmark answer scoring, and
its verdicts belong in construction records, never in a case prompt,
observation, or memory text.

Judgements are cached by the sha256 of `{rubric_version, model, input}`, the
same shape `parm_bench.evidence_gate` uses, so a rebuild replays without new
calls and a frozen policy fails loudly on a miss.
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
    CONTROL_DOES_NOT_REMOVE_ADVANTAGE,
    CUE_ALONE_DETERMINES_CHOICE,
    DECISION_VALIDITY_REASONS,
    INCORRECT_CAPABILITY_LABEL,
    MEDICAL_OR_SENSITIVE_OVERREACH,
)
from .service_tier import service_tier_kwargs


DECISION_VALIDITY_MODEL = "gpt-5-mini"
DECISION_VALIDITY_RUBRIC = "parmbench_decision_validity_v1"

CUE_SPECIFICITY_VALUES = (
    "concrete_affordance",
    "generic_evaluative",
    "wording_echo",
)
ENTAILMENT_VALUES = ("entailed", "plausible_stretch", "invented")
SENSITIVITY_VALUES = ("none", "sensitive_handled", "sensitive_unlabeled")

CAPABILITIES = (
    "direct_lexical_fact",
    "paraphrased_semantic_fact",
    "schedule_commitment",
    "relationship_named_entity",
    "negative_preference_exclusion",
    "one_hop_relational",
)

# Memory-quality categories, taken from item 2 of the audit's relevance
# contract. The pre-construction memory-quality gate assigns one of these to
# every accepted fact; unfamiliar values fall through to the wording split
# rather than forcing a label this module cannot justify.
PREFERENCE = "preference"
CONSTRAINT = "constraint"
EXCLUSION = "exclusion"
ACTIVE_COMMITMENT = "active_commitment"
STABLE_RELATIONSHIP = "stable_relationship"
OWNED_ITEM = "owned_item"
ACCESSIBILITY_NEED = "accessibility_need"
CONCRETE_SCHEDULE = "concrete_schedule"

MEMORY_CATEGORIES = (
    PREFERENCE,
    CONSTRAINT,
    EXCLUSION,
    ACTIVE_COMMITMENT,
    STABLE_RELATIONSHIP,
    OWNED_ITEM,
    ACCESSIBILITY_NEED,
    CONCRETE_SCHEDULE,
)

# The claim drafter's `fact_kind` vocabulary predates the memory-quality gate.
# Old supply rows carry it, so they are translated rather than rejected.
_CATEGORY_ALIASES = {
    "commitment": ACTIVE_COMMITMENT,
    "project": ACTIVE_COMMITMENT,
    "schedule": CONCRETE_SCHEDULE,
    "routine": CONCRETE_SCHEDULE,
    "relationship": STABLE_RELATIONSHIP,
    "named_entity": STABLE_RELATIONSHIP,
    "possession": OWNED_ITEM,
    "owned": OWNED_ITEM,
    "accessibility": ACCESSIBILITY_NEED,
    "taste": PREFERENCE,
    "habit": PREFERENCE,
    "dislike": EXCLUSION,
}

DIRECT_RELATION = "direct"
ONE_HOP_RELATION = "one_hop"
CAUSAL_RELATIONS = (DIRECT_RELATION, ONE_HOP_RELATION)

SHARE_WORDING = "share_wording"
PARAPHRASE_ONLY = "paraphrase_only"

# A stated limitation earns its keep the way a stated dislike does: by ruling
# an option out. Moved here from the builder so the gate and the builder label
# a fact the same way.
EXCLUSION_PATTERN = re.compile(
    r"\b(avoids?|avoiding|skips?|skipping|dislikes?|disliking|never|"
    r"does not|doesn't|do not|don't|will not|won't|no longer|stopped|"
    r"steers? clear|refuses?|declines?|cannot|can't|unable|"
    r"gave up|stays? away|keeps? away|without)\b",
    re.IGNORECASE,
)

# `schedule_commitment` claims to test recall of a time. The audit rejected the
# label on a standing writing project because no time or date appears anywhere
# in the fact, so a commitment needs a temporal marker before it counts.
_TEMPORAL_PATTERN = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|morning|afternoon|evening|night|"
    r"tonight|today|tomorrow|weekday|weekend|daily|nightly|weekly|"
    r"fortnightly|monthly|quarterly|yearly|annual|annually|season|"
    r"seasonal|term|semester|deadline|due|rota|shift|timetable|"
    r"appointment|o'clock|noon|midday|midnight|every|each)\b|"
    r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b|\b\d{1,2}/\d{1,2}\b",
    re.IGNORECASE,
)

_WORD_PATTERN = re.compile(r"[a-z0-9']+")
_NAMED_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]{2,}|[A-Z]{2,})\b")
_NOT_A_NAME = frozenset(
    {
        "The",
        "They",
        "Their",
        "This",
        "That",
        "There",
        "These",
        "Those",
        "User",
        "When",
        "While",
        "Every",
        "Each",
    }
)
_STOPWORDS = frozenset(
    """a about after all also an and any are as at be been but by can did do
    for from had has have her him his how its into is it me my not of on one
    or our out over she should so some than that the their them then there
    these they this to up user users was we were what when where which while
    who why will with would you your""".split()
)


DECISION_VALIDITY_INSTRUCTIONS = """\
You audit one already-built benchmark scenario and decide whether its
personalisation is real. You are shown the fixture's roles on purpose: option A
is meant to be the right answer for a reader who knows nothing about the
person, and option B is meant to become the right answer once the personal fact
is known. Judge whether that design actually holds. You are not answering the
task.

Read the person's own words first. Everything you credit to them has to come
from that text. Assistant turns are not evidence of anything the person wants,
owns, or plans.

Report each field independently.

- ordinary_winner_valid: true only when option A is the better pick on the
  visible ordinary evidence and the stated ordinary mechanism, for a reader
  with no personal information. If option B already reads at least as well on
  ordinary grounds, or if a decisive drawback of A is stated only to make the
  memory bite, report false.
- cue_specificity: concrete_affordance when the cue names a checkable property
  of option B that a reader could verify. generic_evaluative when it is praise,
  an evaluative adjective, or a vague benefit that could attach to any option.
  wording_echo when it repeats the person's own distinctive words closely
  enough that string matching alone would find it.
- memory_affordance_entailment: entailed when the accepted fact plus that
  affordance rationally prefer option B for this person, with no step the
  reader has to supply. plausible_stretch when the preference follows only if
  something likely but unstated is true. invented when it needs a fact the
  person never gave.
- assumptions_required: list every possession, permission, relationship,
  location, medical conclusion, future plan, or unstated preference the
  justification depends on and the person never stated. One short phrase each.
  Leave the list empty only when the justification truly needs nothing added.
- cue_alone_sufficient: true when the cue makes option B the better pick for an
  arbitrary person who has no personal history at all. A cue that is desirable
  to everyone is doing the work the memory should do.
- control_valid: the control arm replaces the cue sentence with the neutral
  replacement and changes nothing else. Read the full bodies of both options as
  they would stand after that swap. Report true only when no personalised
  advantage for option B survives anywhere in either body: not in a clause that
  rules option A out for this person, not in a second affordance kept outside
  the cue sentence.
- sensitivity: none when no health, medical, financial, sexual, religious,
  political, or otherwise sensitive fact is doing work. sensitive_handled when
  such a fact is doing work and the declared sensitive terms cover it.
  sensitive_unlabeled when such a fact is doing work and the declared sensitive
  terms are empty or miss it.
- capability_label_valid and suggested_capability: judge the declared label
  against the accepted fact and the causal relation between the fact and the
  cue, not against the topic either one talks about. schedule_commitment needs
  a time, date, or recurrence in the fact. relationship_named_entity needs a
  named person, organisation, or place in the fact.
  negative_preference_exclusion needs the fact to rule something out; an
  accessibility need is not an exclusion. one_hop_relational needs a real
  inference step from the fact to the affordance. Otherwise choose
  direct_lexical_fact or paraphrased_semantic_fact by whether the cue reuses
  the person's own words. Always name the label you would give in
  suggested_capability, even when the declared one is right.
- rejection_reasons: name every applicable machine-readable reason, and none
  that does not apply.
- rationale: two or three sentences. Quote at most one short phrase from the
  person's words.
- accept: true only when the scenario is a decision you would defend to
  somebody reading the raw source.
"""

DECISION_VALIDITY_SCHEMA = {
    "type": "object",
    "properties": {
        "ordinary_winner_valid": {"type": "boolean"},
        "cue_specificity": {
            "type": "string",
            "enum": list(CUE_SPECIFICITY_VALUES),
        },
        "memory_affordance_entailment": {
            "type": "string",
            "enum": list(ENTAILMENT_VALUES),
        },
        "assumptions_required": {
            "type": "array",
            "items": {"type": "string"},
        },
        "cue_alone_sufficient": {"type": "boolean"},
        "control_valid": {"type": "boolean"},
        "sensitivity": {"type": "string", "enum": list(SENSITIVITY_VALUES)},
        "capability_label_valid": {"type": "boolean"},
        "suggested_capability": {"type": "string"},
        "rejection_reasons": {
            "type": "array",
            "items": {"type": "string", "enum": list(DECISION_VALIDITY_REASONS)},
        },
        "rationale": {"type": "string"},
        "accept": {"type": "boolean"},
    },
    "required": [
        "ordinary_winner_valid",
        "cue_specificity",
        "memory_affordance_entailment",
        "assumptions_required",
        "cue_alone_sufficient",
        "control_valid",
        "sensitivity",
        "capability_label_valid",
        "suggested_capability",
        "rejection_reasons",
        "rationale",
        "accept",
    ],
    "additionalProperties": False,
}


class DecisionValidityCachePolicy(str, Enum):
    POPULATE = "populate"
    FROZEN = "frozen"


class DecisionValidityCacheMissError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionScenario:
    """Everything the auditor reads about one built scenario.

    Option bodies are carried in full. Two of the audit's rejects kept the
    personalised advantage in non-cue option text, so a gate that reads only
    the cue sentence cannot see the control break.
    """

    claim: str
    evidence_span: str
    task_prompt: str
    option_a_label: str
    option_a_body: str
    option_b_label: str
    option_b_body: str
    cue_text: str
    neutral_replacement: str
    ordinary_mechanism: str
    capability_label: str
    evidence_turns: tuple[Mapping[str, Any], ...] = ()
    sensitive_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionValidityResult:
    """One auditor verdict on one built scenario."""

    ordinary_winner_valid: bool
    cue_specificity: str
    memory_affordance_entailment: str
    assumptions_required: tuple[str, ...]
    cue_alone_sufficient: bool
    control_valid: bool
    sensitivity: str
    capability_label_valid: bool
    suggested_capability: str
    rejection_reasons: tuple[str, ...]
    rationale: str
    judge_accept: bool
    rubric_version: str = DECISION_VALIDITY_RUBRIC
    model: str = DECISION_VALIDITY_MODEL
    resolved_model: str | None = None
    response_id: str | None = None

    @property
    def unmet_requirements(self) -> tuple[str, ...]:
        """Named acceptance conditions this verdict fails."""

        unmet: list[str] = []
        if not self.ordinary_winner_valid:
            unmet.append("ordinary_winner_valid")
        if self.cue_specificity != "concrete_affordance":
            unmet.append("cue_specificity")
        if self.memory_affordance_entailment != "entailed":
            unmet.append("memory_affordance_entailment")
        if self.assumptions_required:
            unmet.append("assumptions_required")
        if self.cue_alone_sufficient:
            unmet.append("cue_alone_sufficient")
        if not self.control_valid:
            unmet.append("control_valid")
        if self.sensitivity == "sensitive_unlabeled":
            unmet.append("sensitivity")
        if not self.capability_label_valid:
            unmet.append("capability_label_valid")
        if self.rejection_reasons:
            unmet.append("rejection_reasons")
        if not self.judge_accept:
            unmet.append("judge_accept")
        return tuple(unmet)

    @property
    def accept(self) -> bool:
        """Every acceptance condition holds and the auditor agrees.

        The auditor can refuse a scenario its own field values would let
        through, but it cannot wave one in: each condition is checked here.
        """

        return not self.unmet_requirements

    def to_dict(self) -> dict[str, Any]:
        return {
            "accept": self.accept,
            "judge_accept": self.judge_accept,
            "ordinary_winner_valid": self.ordinary_winner_valid,
            "cue_specificity": self.cue_specificity,
            "memory_affordance_entailment": self.memory_affordance_entailment,
            "assumptions_required": list(self.assumptions_required),
            "cue_alone_sufficient": self.cue_alone_sufficient,
            "control_valid": self.control_valid,
            "sensitivity": self.sensitivity,
            "capability_label_valid": self.capability_label_valid,
            "suggested_capability": self.suggested_capability,
            "rejection_reasons": list(self.rejection_reasons),
            "unmet_requirements": list(self.unmet_requirements),
            "rationale": self.rationale,
            "rubric_version": self.rubric_version,
            "requested_model": self.model,
            "resolved_model": self.resolved_model,
            "response_id": self.response_id,
        }


class DecisionValidityJudge(Protocol):
    model_name: str
    rubric_version: str

    def judge(self, *, scenario: DecisionScenario) -> dict[str, Any]: ...


class CachedOpenAIDecisionValidityJudge:
    """Cached gpt-5-mini auditor over the versioned decision-validity rubric."""

    def __init__(
        self,
        cache_dir: str | Path,
        policy: DecisionValidityCachePolicy | str = (
            DecisionValidityCachePolicy.POPULATE
        ),
        *,
        model: str = DECISION_VALIDITY_MODEL,
        rubric_version: str = DECISION_VALIDITY_RUBRIC,
        instructions: str = DECISION_VALIDITY_INSTRUCTIONS,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.policy = DecisionValidityCachePolicy(policy)
        self.model_name = model
        self.rubric_version = rubric_version
        self.instructions = instructions
        self.client = client
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._used_cache_files: list[Path] = []

    def judge(self, *, scenario: DecisionScenario) -> dict[str, Any]:
        input_text = render_decision_validity_request(scenario)
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
        if self.policy is DecisionValidityCachePolicy.FROZEN:
            raise DecisionValidityCacheMissError(
                f"missing decision validity cache entry: {request_hash}"
            )

        response = self.client.responses.create(
            model=self.model_name,
            instructions=self.instructions,
            input=input_text,
            **service_tier_kwargs(),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parmbench_decision_validity",
                    "strict": True,
                    "schema": DECISION_VALIDITY_SCHEMA,
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


def audit_scenario(
    scenario: DecisionScenario,
    *,
    model: str = DECISION_VALIDITY_MODEL,
    cache_dir: str | Path | None = None,
    policy: DecisionValidityCachePolicy | str = (
        DecisionValidityCachePolicy.POPULATE
    ),
    client: Any | None = None,
    judge: DecisionValidityJudge | None = None,
) -> DecisionValidityResult:
    """Audit one built scenario for decision validity."""

    if judge is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required when no judge is supplied")
        judge = CachedOpenAIDecisionValidityJudge(
            cache_dir,
            policy,
            model=model,
            client=client,
        )
    verdict = judge.judge(scenario=scenario)
    _validate_verdict(verdict)
    reasons = tuple(
        dict.fromkeys(
            tuple(verdict.get("rejection_reasons", ()))
            + derived_rejection_reasons(verdict)
        )
    )
    return DecisionValidityResult(
        ordinary_winner_valid=bool(verdict["ordinary_winner_valid"]),
        cue_specificity=str(verdict["cue_specificity"]),
        memory_affordance_entailment=str(
            verdict["memory_affordance_entailment"]
        ),
        assumptions_required=tuple(
            str(item).strip()
            for item in verdict.get("assumptions_required", ())
            if str(item).strip()
        ),
        cue_alone_sufficient=bool(verdict["cue_alone_sufficient"]),
        control_valid=bool(verdict["control_valid"]),
        sensitivity=str(verdict["sensitivity"]),
        capability_label_valid=bool(verdict["capability_label_valid"]),
        suggested_capability=str(verdict.get("suggested_capability", "")),
        rejection_reasons=reasons,
        rationale=str(verdict.get("rationale", "")),
        judge_accept=bool(verdict["accept"]),
        rubric_version=judge.rubric_version,
        model=judge.model_name,
        resolved_model=verdict.get("resolved_model"),
        response_id=verdict.get("response_id"),
    )


def passes_decision_validity(result: DecisionValidityResult) -> bool:
    return result.accept


def derived_rejection_reasons(
    verdict: Mapping[str, Any],
) -> tuple[str, ...]:
    """Taxonomy reasons a verdict's own field values already establish.

    Every reason here fires only alongside a failing acceptance condition, so
    adding them enriches the record without changing any accept decision.
    """

    reasons: list[str] = []
    if bool(verdict.get("cue_alone_sufficient")):
        reasons.append(CUE_ALONE_DETERMINES_CHOICE)
    if not bool(verdict.get("control_valid", True)):
        reasons.append(CONTROL_DOES_NOT_REMOVE_ADVANTAGE)
    if str(verdict.get("sensitivity", "none")) == "sensitive_unlabeled":
        reasons.append(MEDICAL_OR_SENSITIVE_OVERREACH)
    if not bool(verdict.get("capability_label_valid", True)):
        reasons.append(INCORRECT_CAPABILITY_LABEL)
    return tuple(reasons)


def render_decision_validity_request(scenario: DecisionScenario) -> str:
    turns = "\n\n".join(
        f"{str(turn.get('role', '')).title()}: {turn.get('content', '')}"
        for turn in scenario.evidence_turns
    )
    terms = ", ".join(scenario.sensitive_terms) or "(none declared)"
    ablated_b = _ablate(scenario.option_b_body, scenario)
    return (
        f"Accepted personal fact:\n{scenario.claim}\n\n"
        f"The person's own words:\n{scenario.evidence_span}\n\n"
        f"Raw source turns:\n{turns}\n\n"
        f"Task prompt shown to the reader:\n{scenario.task_prompt}\n\n"
        f"Ordinary evidence mechanism: {scenario.ordinary_mechanism}\n\n"
        f"Option A, the intended ordinary winner - "
        f"{scenario.option_a_label}:\n{scenario.option_a_body}\n\n"
        f"Option B, the intended memory-conditioned target - "
        f"{scenario.option_b_label}:\n{scenario.option_b_body}\n\n"
        f"Cue sentence inside option B:\n{scenario.cue_text}\n\n"
        f"Neutral replacement used in the control arm:\n"
        f"{scenario.neutral_replacement}\n\n"
        f"Option B as it reads in the control arm:\n{ablated_b}\n\n"
        f"Declared capability label: {scenario.capability_label}\n"
        f"Declared sensitive terms: {terms}\n"
    )


def decision_validity_cache_key(
    scenario: DecisionScenario,
    *,
    model: str = DECISION_VALIDITY_MODEL,
    rubric_version: str = DECISION_VALIDITY_RUBRIC,
) -> str:
    return _hash_request(
        {
            "rubric_version": rubric_version,
            "model": model,
            "input": render_decision_validity_request(scenario),
        }
    )


# --------------------------------------------------------------------------
# capability assignment
# --------------------------------------------------------------------------


def capability_for_fact(
    *,
    claim: str,
    memory_category: str,
    causal_relation: str = DIRECT_RELATION,
    evidence_span: str = "",
    cue_text: str = "",
    overlap_mode: str | None = None,
) -> str:
    """Label a scenario by what the memory does, not by what it talks about.

    The audit found twelve of twenty-nine labels wrong, all of them read off
    the surface topic. This reads off the accepted fact's memory-quality
    category and the causal relation between the fact and the cue, and it
    refuses the three labels that carry a claim the fact has to earn: a
    schedule needs a time, a named-entity relation needs a name, and an
    exclusion needs the fact to rule something out.
    """

    category = normalise_memory_category(memory_category)
    relation = str(causal_relation or DIRECT_RELATION).strip().casefold()
    wording = overlap_mode or wording_relation(evidence_span, cue_text)

    if category == EXCLUSION:
        return "negative_preference_exclusion"
    if category in (PREFERENCE, CONSTRAINT, "") and EXCLUSION_PATTERN.search(
        claim
    ):
        return "negative_preference_exclusion"
    if category == CONCRETE_SCHEDULE:
        return "schedule_commitment"
    if category == ACTIVE_COMMITMENT and _TEMPORAL_PATTERN.search(claim):
        return "schedule_commitment"
    if relation == ONE_HOP_RELATION:
        return "one_hop_relational"
    if category == STABLE_RELATIONSHIP and named_entities(claim):
        return "relationship_named_entity"
    return (
        "direct_lexical_fact"
        if wording == SHARE_WORDING
        else "paraphrased_semantic_fact"
    )


def normalise_memory_category(category: str) -> str:
    value = str(category or "").strip().casefold()
    value = _CATEGORY_ALIASES.get(value, value)
    return value if value in MEMORY_CATEGORIES else ""


def named_entities(text: str) -> tuple[str, ...]:
    """Capitalised names in a claim, ignoring the leading sentence word."""

    words = str(text).split()
    if not words:
        return ()
    found = [
        match
        for word in words[1:]
        for match in _NAMED_ENTITY_PATTERN.findall(word)
        if match not in _NOT_A_NAME
    ]
    return tuple(dict.fromkeys(found))


def wording_relation(evidence_span: str, cue_text: str) -> str:
    """Whether the cue reuses the person's own distinctive content words."""

    span_words = _content_words(evidence_span)
    cue_words = _content_words(cue_text)
    if not span_words or not cue_words:
        return PARAPHRASE_ONLY
    return SHARE_WORDING if span_words & cue_words else PARAPHRASE_ONLY


def control_residual_advantage(scenario: DecisionScenario) -> tuple[str, ...]:
    """Distinctive claim words that survive cue ablation in the target.

    The rubric tells the judge to read the full option bodies after the cue
    swap, but the v3 pilot showed it reliably reads only the cue sentence: an
    option literally named after the claim's key word passed `control_valid`
    on every pilot scenario. This is the deterministic backstop for the
    lexical subset of that failure. After ablation, no distinctive content
    word of the claim or evidence span may remain in the target's label or
    body. A word that also appears in the winner's label or body is not
    distinctive: it cannot carry a personalized advantage the winner lacks.
    Semantic residue without shared wording stays the judge's job.
    """

    claim_words = _content_words(scenario.claim) | _content_words(
        scenario.evidence_span
    )
    winner_words = _content_words(scenario.option_a_label) | _content_words(
        scenario.option_a_body
    )
    ablated_target = "{} {}".format(
        scenario.option_b_label, _ablate(scenario.option_b_body, scenario)
    )
    residual = (
        _content_words(ablated_target) & claim_words
    ) - winner_words
    return tuple(sorted(residual))


def capability_conflicts_with_lexical_target(
    capability: str, claim: str, evidence_span: str, target_label: str
) -> bool:
    """A relational capability with the memory's word in the answer's name.

    `one_hop_relational` asserts that a system must traverse a relation the
    cue does not state. When the target's own label contains a distinctive
    content word of the claim, the hop collapses to string matching and the
    label is wrong (pilot example: a "multilingual" claim answered by a
    "Multilingual Data Pack").
    """

    if capability != "one_hop_relational":
        return False
    claim_words = _content_words(claim) | _content_words(evidence_span)
    return bool(_content_words(target_label) & claim_words)


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _ablate(body: str, scenario: DecisionScenario) -> str:
    cue = scenario.cue_text.strip()
    if cue and cue in body:
        return body.replace(cue, scenario.neutral_replacement.strip(), 1)
    return f"{body.rstrip()} {scenario.neutral_replacement.strip()}".strip()


def _validate_verdict(verdict: Mapping[str, Any]) -> None:
    if verdict.get("cue_specificity") not in CUE_SPECIFICITY_VALUES:
        raise ValueError(
            f"unknown cue specificity: {verdict.get('cue_specificity')!r}"
        )
    entailment = verdict.get("memory_affordance_entailment")
    if entailment not in ENTAILMENT_VALUES:
        raise ValueError(f"unknown entailment: {entailment!r}")
    if verdict.get("sensitivity") not in SENSITIVITY_VALUES:
        raise ValueError(f"unknown sensitivity: {verdict.get('sensitivity')!r}")
    unknown = [
        reason
        for reason in verdict.get("rejection_reasons", ())
        if reason not in DECISION_VALIDITY_REASONS
    ]
    if unknown:
        raise ValueError(f"unknown rejection reasons: {unknown}")


def _content_words(text: str) -> set[str]:
    return {
        _lemma(word)
        for word in _WORD_PATTERN.findall(str(text).casefold())
        if word not in _STOPWORDS and len(word) > 3
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
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _hash_request(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(request), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


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


def build_scenario(
    *,
    claim: str,
    evidence_span: str,
    task_prompt: str,
    core: Mapping[str, Any],
    ordinary_mechanism: str,
    capability_label: str,
    evidence_turns: Sequence[Mapping[str, Any]] = (),
    sensitive_terms: Sequence[str] = (),
) -> DecisionScenario:
    """Read a `DecisionScenario` out of a normalised construction core.

    The target body is passed with the cue appended, exactly as the assembled
    document shows it, so the auditor reads the option the reader reads.
    """

    cue = str(core.get("cue_clause", "")).strip()
    target_body = str(core.get("target_body", "")).rstrip()
    return DecisionScenario(
        claim=claim,
        evidence_span=evidence_span,
        task_prompt=task_prompt,
        option_a_label=str(core.get("winner_label", "")),
        option_a_body=str(core.get("winner_body", "")).strip(),
        option_b_label=str(core.get("target_label", "")),
        option_b_body=f"{target_body} {cue}".strip(),
        cue_text=cue,
        neutral_replacement=str(core.get("neutral_clause", "")).strip(),
        ordinary_mechanism=ordinary_mechanism,
        capability_label=capability_label,
        evidence_turns=tuple(evidence_turns),
        sensitive_terms=tuple(sensitive_terms),
    )
