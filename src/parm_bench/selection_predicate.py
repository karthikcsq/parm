"""Selection-predicate stage between an accepted fact and construction.

The v3 pilot showed that one model call cannot invent the task, the visible
affordance, and the causal relationship around a fact and still leave a valid
scenario behind. This stage narrows the job: read an accepted durable fact and
name the decision rule it already implies, then route that rule to a task
family in which the rule naturally ranks ordinary options.

The mapper produces the evaluator-only fields the construction contract names
in `docs/benchmark-construction.md`: `selection_predicate`, `task_family`,
`target_affordance`, `ordinary_mechanism`, `control_affordance`,
`relation_type`, and `material_assumptions`. Those names are the interface the
builder consumes, so they are stable here.

Abstention is a first-class correct outcome. A supported durable fact with no
direct selection implication has no predicate, and forcing one is how the
earlier pipeline manufactured scenarios nobody would defend.

The architecture mirrors `parm_bench.memory_quality`: a deterministic
pre-stage that can abstain without any model call, a cached judge keyed by the
sha256 of `{rubric_version, model, input}`, populate and frozen policies,
atomic cache writes, and a frozen result dataclass. Deterministic post-checks
run on every `maps=true` verdict, because a mapping the registry forbids or a
mapping that leans on an unstated assumption is wrong however confident the
model sounds.
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

from .memory_quality import MEMORY_QUALITY_CATEGORIES, PASSING_DURABILITY
from .relevance_taxonomy import (
    INCOMPATIBLE_TASK_FAMILY_ROUTING,
    MATERIAL_ASSUMPTION_REQUIRED,
    SELECTION_PREDICATE_REASONS,
    UNANCHORED_SELECTION_PREDICATE,
)
from .service_tier import service_tier_kwargs


SELECTION_PREDICATE_MODEL = "gpt-5-mini"
SELECTION_PREDICATE_RUBRIC = "parmbench_selection_predicate_v1"

# How the fact reaches the choice. The contract's list, unchanged.
DIRECT_CONSTRAINT = "direct_constraint"
COMPATIBILITY = "compatibility"
ACTIVE_PROJECT_RELEVANCE = "active_project_relevance"
SCHEDULE_FIT = "schedule_fit"
ACCESSIBILITY_NEED = "accessibility_need"
STABLE_PREFERENCE = "stable_preference"
RELATIONSHIP_OBLIGATION = "relationship_obligation"

RELATION_TYPES = (
    DIRECT_CONSTRAINT,
    COMPATIBILITY,
    ACTIVE_PROJECT_RELEVANCE,
    SCHEDULE_FIT,
    ACCESSIBILITY_NEED,
    STABLE_PREFERENCE,
    RELATIONSHIP_OBLIGATION,
)

# The anchors and the earlier drafting vocabulary name a few of these
# relations differently. Translating is safer than widening the enum: the
# vintage-store anchor records `owned_item_compatibility`, which is the
# compatibility relation with the fact's category spelled into its name.
RELATION_TYPE_ALIASES = {
    "owned_item_compatibility": COMPATIBILITY,
    "owned_item": COMPATIBILITY,
    "possession_compatibility": COMPATIBILITY,
    "device_compatibility": COMPATIBILITY,
    "constraint": DIRECT_CONSTRAINT,
    "exclusion": DIRECT_CONSTRAINT,
    "active_project": ACTIVE_PROJECT_RELEVANCE,
    "active_commitment": ACTIVE_PROJECT_RELEVANCE,
    "schedule": SCHEDULE_FIT,
    "concrete_schedule": SCHEDULE_FIT,
    "accessibility": ACCESSIBILITY_NEED,
    "preference": STABLE_PREFERENCE,
    "stable_relationship": RELATIONSHIP_OBLIGATION,
    "relationship": RELATIONSHIP_OBLIGATION,
}

NO_RELATION = "none"
NO_TASK_FAMILY = "none"


@dataclass(frozen=True)
class TaskFamily:
    """One bounded family of ordinary choices and what may route into it."""

    family_id: str
    description: str
    relation_types: tuple[str, ...]


def _family(
    family_id: str, description: str, *relation_types: str
) -> TaskFamily:
    return TaskFamily(family_id, description, tuple(relation_types))


# The routing list from the construction contract, written out so a reviewer
# can read every legal pairing on one screen. A relation type reaches only the
# families that name it. Breadth belongs to envelopes, names, and ordering
# after the causal structure holds, not here.
TASK_FAMILY_REGISTRY: dict[str, TaskFamily] = {
    family.family_id: family
    for family in (
        _family(
            "menu_or_catering_selection",
            "choose a dish, dessert, or catering package from a menu",
            DIRECT_CONSTRAINT,
            STABLE_PREFERENCE,
        ),
        _family(
            "grocery_selection",
            "choose a grocery product or a delivered food order",
            DIRECT_CONSTRAINT,
            STABLE_PREFERENCE,
        ),
        _family(
            "reading_or_study_material_selection",
            "choose a book, talk, course, or archive to read or study",
            ACTIVE_PROJECT_RELEVANCE,
            STABLE_PREFERENCE,
        ),
        _family(
            "compatible_accessory_or_part_selection",
            "choose an accessory, part, or repair for something owned",
            COMPATIBILITY,
        ),
        _family(
            "vintage_or_secondhand_finds",
            "pick out something worth having from a secondhand listing",
            COMPATIBILITY,
            STABLE_PREFERENCE,
        ),
        _family(
            "event_or_workshop_selection",
            "choose an event, club night, class, or workshop to attend",
            STABLE_PREFERENCE,
            ACTIVE_PROJECT_RELEVANCE,
            SCHEDULE_FIT,
        ),
        _family(
            "supplies_selection",
            "choose supplies, materials, or kit for an activity",
            STABLE_PREFERENCE,
            ACTIVE_PROJECT_RELEVANCE,
            COMPATIBILITY,
        ),
        _family(
            "appointment_or_slot_selection",
            "choose an appointment, delivery, or travel slot",
            SCHEDULE_FIT,
            ACCESSIBILITY_NEED,
        ),
        _family(
            "room_route_or_seating_selection",
            "choose a room, route, seat, or transport option",
            ACCESSIBILITY_NEED,
            DIRECT_CONSTRAINT,
        ),
        _family(
            "gift_or_visit_selection",
            "choose a gift, visit, or trip for a supported relationship",
            RELATIONSHIP_OBLIGATION,
        ),
    )
}

TASK_FAMILIES = tuple(TASK_FAMILY_REGISTRY)

# Categories the memory-quality gate can hand over. "none" never reaches this
# stage, so a fact carrying it abstains before the judge is called.
MAPPABLE_MEMORY_CATEGORIES = tuple(
    category for category in MEMORY_QUALITY_CATEGORIES if category != "none"
)

# Deterministic pre-stage causes. These are not taxonomy rejection reasons:
# each one says the stage had nothing to map, which is an abstention.
UNMAPPABLE_MEMORY_CATEGORY = "unmappable_memory_category"
NON_DURABLE_MEMORY = "non_durable_memory"
MISSING_CLAIM_OR_SPAN = "missing_claim_or_span"

DETERMINISTIC_ABSTENTION_CAUSES = (
    MISSING_CLAIM_OR_SPAN,
    UNMAPPABLE_MEMORY_CATEGORY,
    NON_DURABLE_MEMORY,
)

MAX_MATERIAL_ASSUMPTIONS = 6

SELECTION_PREDICATE_INSTRUCTIONS = """\
Turn one accepted personal fact into the selection rule it already implies, or
abstain. Source support and memory quality are settled elsewhere: assume the
user's own words entail the fact and that the fact is worth retaining.

You are shown the fact, the verbatim span of the user's own words behind it,
the source turns around that span, the memory-quality labels for the fact, and
the task families each relation type may route to. Judge against those. Do not
assume a hidden persona, profile, or benchmark label.

Set maps to true only when the fact on its own decides between ordinary
options. Abstention is a correct answer and a forced mapping is worse than
none. Abstain when:
- the fact records a broad interest, an opinion, or biography that gives no
  lever on any ordinary choice;
- the only rule you can write restates what the fact is about, such as "prefer
  books about history", instead of naming what an option has to offer;
- getting from the fact to the rule needs a possession, permission, location,
  relationship, or plan the user never stated.

Fill these fields when maps is true.

selection_predicate: one sentence naming the condition an option has to meet
for this person. It has to be an action-relevant implication of the fact, not
a topic restatement. "The user owns an NES" implies "prefer vintage items
usable with an NES". It does not imply "the user likes retro games".

relation_type: how the fact reaches the choice. Use direct_constraint when the
fact rules options in or out, compatibility when the option has to fit
something the person already has, active_project_relevance when the option has
to feed live work, schedule_fit when a time has to line up, accessibility_need
when the option has to meet a physical or sensory need, stable_preference when
a standing taste ranks the options, and relationship_obligation when a
supported tie to a person or organisation drives the choice.

task_family: one of the families listed for your relation_type in the request.
A family outside that list is rejected, so abstain rather than reach for a
family that does not fit.

target_affordance: the visible, checkable property that makes one option
satisfy the predicate. A reader has to be able to verify it from that option's
own text.

ordinary_mechanism: the visible reason a different option is the better pick
for somebody who knows nothing about this person. Name evidence anyone can
see, such as rating, availability, price, condition, recency, or prominence.

control_affordance: the neutral property that takes the target affordance's
place when the control arm removes the match. Keep the same register and
roughly the same length. It must not be an antonym of the affordance, must not
announce that the property is missing or unavailable, and must not create a
fresh match for this person anywhere else.

material_assumptions: every possession, permission, relationship, location,
schedule, or preference the predicate leans on and the user never stated, one
short phrase each. Leave it empty when the predicate needs nothing added. A
non-empty list rejects the mapping, so abstain instead of padding it.

rationale: one or two sentences naming the step from the user's own words to
the predicate.

Do not repeat sensitive wording in any field. The sensitivity labels travel
with the mapping on their own.
"""

_WORD_PATTERN = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """a about after all also an and any are as at be been but by can did do
    does for from had has have her him his how into its is it me my not of on
    one or our out over she should so some than that the their them then there
    these they this to up user users was we were what when where which while
    who why will with would you your""".split()
)


class SelectionPredicateCachePolicy(str, Enum):
    POPULATE = "populate"
    FROZEN = "frozen"


class SelectionPredicateCacheMissError(RuntimeError):
    pass


@dataclass(frozen=True)
class SelectionPredicateResult:
    """One mapping verdict for one accepted fact."""

    maps: bool
    selection_predicate: str = ""
    relation_type: str = ""
    task_family: str = ""
    target_affordance: str = ""
    ordinary_mechanism: str = ""
    control_affordance: str = ""
    material_assumptions: tuple[str, ...] = ()
    rationale: str = ""
    rejection_reasons: tuple[str, ...] = ()
    deterministic_abstentions: tuple[str, ...] = ()
    sensitive: bool = False
    sensitive_terms: tuple[str, ...] = ()
    rubric_version: str = SELECTION_PREDICATE_RUBRIC
    model: str = SELECTION_PREDICATE_MODEL
    resolved_model: str | None = None
    response_id: str | None = None

    @property
    def abstained(self) -> bool:
        """The mapper declined to map, which is a correct outcome."""

        return not self.maps

    def to_dict(self) -> dict[str, Any]:
        return {
            "maps": self.maps,
            "selection_predicate": self.selection_predicate,
            "relation_type": self.relation_type,
            "task_family": self.task_family,
            "target_affordance": self.target_affordance,
            "ordinary_mechanism": self.ordinary_mechanism,
            "control_affordance": self.control_affordance,
            "material_assumptions": list(self.material_assumptions),
            "rationale": self.rationale,
            "rejection_reasons": list(self.rejection_reasons),
            "deterministic_abstentions": list(self.deterministic_abstentions),
            "sensitive": self.sensitive,
            "sensitive_terms": list(self.sensitive_terms),
            "rubric_version": self.rubric_version,
            "requested_model": self.model,
            "resolved_model": self.resolved_model,
            "response_id": self.response_id,
        }

    def predicate_fields(self) -> dict[str, Any]:
        """The evaluator-only object the builder consumes.

        The key names come from the construction contract and the builder
        reads them by name, so nothing may be renamed or added here.
        """

        return {
            "selection_predicate": self.selection_predicate,
            "task_family": self.task_family,
            "target_affordance": self.target_affordance,
            "ordinary_mechanism": self.ordinary_mechanism,
            "control_affordance": self.control_affordance,
            "relation_type": self.relation_type,
            "material_assumptions": list(self.material_assumptions),
            "sensitive": self.sensitive,
            "sensitive_terms": list(self.sensitive_terms),
        }


class SelectionPredicateJudge(Protocol):
    model_name: str
    rubric_version: str

    def judge(
        self,
        *,
        claim: str,
        evidence_span: str,
        evidence_turns: Sequence[Mapping[str, Any]],
        memory_category: str,
        durability: str,
        sensitive: bool,
        sensitive_terms: Sequence[str],
    ) -> dict[str, Any]: ...


SELECTION_PREDICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "maps": {"type": "boolean"},
        "selection_predicate": {"type": "string"},
        "relation_type": {
            "type": "string",
            "enum": list(RELATION_TYPES) + [NO_RELATION],
        },
        "task_family": {
            "type": "string",
            "enum": list(TASK_FAMILIES) + [NO_TASK_FAMILY],
        },
        "target_affordance": {"type": "string"},
        "ordinary_mechanism": {"type": "string"},
        "control_affordance": {"type": "string"},
        "material_assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": MAX_MATERIAL_ASSUMPTIONS,
        },
        "rationale": {"type": "string"},
    },
    "required": [
        "maps",
        "selection_predicate",
        "relation_type",
        "task_family",
        "target_affordance",
        "ordinary_mechanism",
        "control_affordance",
        "material_assumptions",
        "rationale",
    ],
    "additionalProperties": False,
}


class CachedOpenAISelectionPredicateJudge:
    """Cached gpt-5-mini mapper over the versioned selection rubric."""

    def __init__(
        self,
        cache_dir: str | Path,
        policy: SelectionPredicateCachePolicy | str = (
            SelectionPredicateCachePolicy.POPULATE
        ),
        *,
        model: str = SELECTION_PREDICATE_MODEL,
        rubric_version: str = SELECTION_PREDICATE_RUBRIC,
        instructions: str = SELECTION_PREDICATE_INSTRUCTIONS,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.policy = SelectionPredicateCachePolicy(policy)
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
        memory_category: str = "",
        durability: str = "",
        sensitive: bool = False,
        sensitive_terms: Sequence[str] = (),
    ) -> dict[str, Any]:
        input_text = render_selection_predicate_request(
            claim,
            evidence_span,
            evidence_turns,
            memory_category=memory_category,
            durability=durability,
            sensitive=sensitive,
            sensitive_terms=sensitive_terms,
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
        if self.policy is SelectionPredicateCachePolicy.FROZEN:
            raise SelectionPredicateCacheMissError(
                f"missing selection predicate cache entry: {request_hash}"
            )

        response = self.client.responses.create(
            model=self.model_name,
            instructions=self.instructions,
            input=input_text,
            **service_tier_kwargs(),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parmbench_selection_predicate",
                    "strict": True,
                    "schema": SELECTION_PREDICATE_SCHEMA,
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


def map_selection_predicate(
    claim: str,
    evidence_span: str,
    evidence_turns: Sequence[Mapping[str, Any]],
    *,
    memory_category: str = "",
    durability: str = "",
    sensitive: bool = False,
    sensitive_terms: Sequence[str] = (),
    model: str = SELECTION_PREDICATE_MODEL,
    cache_dir: str | Path | None = None,
    policy: SelectionPredicateCachePolicy | str = (
        SelectionPredicateCachePolicy.POPULATE
    ),
    client: Any | None = None,
    judge: SelectionPredicateJudge | None = None,
) -> SelectionPredicateResult:
    """Map an accepted fact to the selection rule it implies, or abstain.

    The memory-quality labels travel in and back out untouched. This stage
    never decides that a fact stopped being sensitive, so `sensitive` and
    `sensitive_terms` on the result are the ones the caller supplied.
    """

    forwarded_terms = tuple(
        str(term).strip() for term in sensitive_terms if str(term).strip()
    )
    causes = deterministic_abstentions(
        claim, evidence_span, memory_category, durability
    )
    if causes:
        return SelectionPredicateResult(
            maps=False,
            rationale=(
                "Deterministic pre-stage found nothing to map: "
                + ", ".join(causes)
                + "."
            ),
            deterministic_abstentions=causes,
            sensitive=sensitive,
            sensitive_terms=forwarded_terms if sensitive else (),
            model=model,
        )

    if judge is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required when no judge is supplied")
        judge = CachedOpenAISelectionPredicateJudge(
            cache_dir,
            policy,
            model=model,
            client=client,
        )
    verdict = judge.judge(
        claim=claim,
        evidence_span=evidence_span,
        evidence_turns=evidence_turns,
        memory_category=memory_category,
        durability=durability,
        sensitive=sensitive,
        sensitive_terms=forwarded_terms,
    )
    _validate_verdict(verdict)

    maps = bool(verdict["maps"])
    relation_type = normalise_relation_type(verdict.get("relation_type", ""))
    task_family = normalise_task_family(verdict.get("task_family", ""))
    assumptions = tuple(
        str(item).strip()
        for item in verdict.get("material_assumptions", ())
        if str(item).strip()
    )
    selection = str(verdict.get("selection_predicate", "")).strip()
    target_affordance = str(verdict.get("target_affordance", "")).strip()

    if not maps:
        return SelectionPredicateResult(
            maps=False,
            rationale=str(verdict.get("rationale", "")),
            sensitive=sensitive,
            sensitive_terms=forwarded_terms if sensitive else (),
            rubric_version=judge.rubric_version,
            model=judge.model_name,
            resolved_model=verdict.get("resolved_model"),
            response_id=verdict.get("response_id"),
        )

    reasons = post_check_rejections(
        claim=claim,
        evidence_span=evidence_span,
        selection_predicate=selection,
        relation_type=relation_type,
        task_family=task_family,
        target_affordance=target_affordance,
        material_assumptions=assumptions,
    )
    return SelectionPredicateResult(
        maps=True,
        selection_predicate=selection,
        relation_type=relation_type,
        task_family=task_family,
        target_affordance=target_affordance,
        ordinary_mechanism=str(verdict.get("ordinary_mechanism", "")).strip(),
        control_affordance=str(verdict.get("control_affordance", "")).strip(),
        material_assumptions=assumptions,
        rationale=str(verdict.get("rationale", "")),
        rejection_reasons=reasons,
        sensitive=sensitive,
        sensitive_terms=forwarded_terms if sensitive else (),
        rubric_version=judge.rubric_version,
        model=judge.model_name,
        resolved_model=verdict.get("resolved_model"),
        response_id=verdict.get("response_id"),
    )


def passes_selection_predicate(result: SelectionPredicateResult) -> bool:
    """Whether construction may proceed from this mapping."""

    return result.maps and not result.rejection_reasons


def deterministic_abstentions(
    claim: str,
    evidence_span: str,
    memory_category: str,
    durability: str,
) -> tuple[str, ...]:
    """Causes for abstaining before any model call.

    A fact the memory-quality gate could not categorise, or one it called
    spent, has no selection rule to give. Reporting these as abstention causes
    keeps them out of the rejection taxonomy: the stage found nothing to map
    rather than finding a bad mapping. The empty defaults let a caller that
    has no memory-quality labels reach the judge as before.
    """

    causes: list[str] = []
    if not str(claim).strip() or not str(evidence_span).strip():
        causes.append(MISSING_CLAIM_OR_SPAN)
    category = str(memory_category or "").strip().casefold()
    if category and category not in MAPPABLE_MEMORY_CATEGORIES:
        causes.append(UNMAPPABLE_MEMORY_CATEGORY)
    level = str(durability or "").strip().casefold()
    if level and level not in PASSING_DURABILITY:
        causes.append(NON_DURABLE_MEMORY)
    return tuple(causes)


def post_check_rejections(
    *,
    claim: str,
    evidence_span: str,
    selection_predicate: str,
    relation_type: str,
    task_family: str,
    target_affordance: str,
    material_assumptions: Sequence[str],
) -> tuple[str, ...]:
    """Named taxonomy reasons a `maps=true` verdict cannot stand.

    Each check is conservative. Routing is decided by the registry, which a
    reviewer can read; a material assumption is the failure the decision-
    validity audit traced most rejects to; and a predicate is called
    unanchored only when it shares no content word with the fact or its span
    and names no checkable affordance either, so a paraphrase that keeps the
    affordance survives.
    """

    reasons: list[str] = []
    if not is_legal_routing(relation_type, task_family):
        reasons.append(INCOMPATIBLE_TASK_FAMILY_ROUTING)
    if any(str(item).strip() for item in material_assumptions):
        reasons.append(MATERIAL_ASSUMPTION_REQUIRED)
    if not selection_predicate.strip():
        reasons.append(UNANCHORED_SELECTION_PREDICATE)
    elif not shared_content_words(
        selection_predicate, f"{claim} {evidence_span}"
    ) and not target_affordance.strip():
        reasons.append(UNANCHORED_SELECTION_PREDICATE)
    return tuple(reasons)


def is_legal_routing(relation_type: str, task_family: str) -> bool:
    family = TASK_FAMILY_REGISTRY.get(str(task_family or "").strip())
    if family is None:
        return False
    return str(relation_type or "").strip() in family.relation_types


def allowed_task_families(relation_type: str) -> tuple[str, ...]:
    relation = normalise_relation_type(relation_type)
    return tuple(
        family_id
        for family_id, family in TASK_FAMILY_REGISTRY.items()
        if relation in family.relation_types
    )


def normalise_relation_type(relation_type: str) -> str:
    """Canonical relation name, or "" for the mapper's abstention value."""

    value = str(relation_type or "").strip().casefold().replace("-", "_")
    value = "_".join(value.split())
    if value in ("", NO_RELATION):
        return ""
    value = RELATION_TYPE_ALIASES.get(value, value)
    if value not in RELATION_TYPES:
        raise ValueError(f"unknown relation type: {relation_type!r}")
    return value


def normalise_task_family(task_family: str) -> str:
    value = str(task_family or "").strip().casefold()
    if value in ("", NO_TASK_FAMILY):
        return ""
    if value not in TASK_FAMILY_REGISTRY:
        raise ValueError(f"unknown task family: {task_family!r}")
    return value


def render_routing_table() -> str:
    """The registry as the mapper reads it, one relation type per line."""

    return "\n".join(
        f"- {relation}: " + ", ".join(allowed_task_families(relation))
        for relation in RELATION_TYPES
    )


def render_selection_predicate_request(
    claim: str,
    evidence_span: str,
    evidence_turns: Sequence[Mapping[str, Any]],
    *,
    memory_category: str = "",
    durability: str = "",
    sensitive: bool = False,
    sensitive_terms: Sequence[str] = (),
) -> str:
    rendered_source = "\n\n".join(
        f"{str(turn.get('role', '')).title()}: {turn.get('content', '')}"
        for turn in evidence_turns
    )
    terms = ", ".join(str(term) for term in sensitive_terms) or "(none)"
    return (
        f"Accepted personal fact:\n{claim}\n\n"
        f"Evidence span (the user's own words):\n{evidence_span}\n\n"
        f"Indexed source conversation:\n{rendered_source}\n\n"
        f"Memory-quality category: {memory_category or '(unlabelled)'}\n"
        f"Memory-quality durability: {durability or '(unlabelled)'}\n"
        f"Sensitive: {'yes' if sensitive else 'no'}\n"
        f"Sensitive terms: {terms}\n\n"
        f"Task families each relation type may route to:\n"
        f"{render_routing_table()}\n"
    )


def selection_predicate_cache_key(
    claim: str,
    evidence_span: str,
    evidence_turns: Sequence[Mapping[str, Any]],
    *,
    memory_category: str = "",
    durability: str = "",
    sensitive: bool = False,
    sensitive_terms: Sequence[str] = (),
    model: str = SELECTION_PREDICATE_MODEL,
    rubric_version: str = SELECTION_PREDICATE_RUBRIC,
) -> str:
    return _hash_request(
        {
            "rubric_version": rubric_version,
            "model": model,
            "input": render_selection_predicate_request(
                claim,
                evidence_span,
                evidence_turns,
                memory_category=memory_category,
                durability=durability,
                sensitive=sensitive,
                sensitive_terms=sensitive_terms,
            ),
        }
    )


def shared_content_words(first: str, second: str) -> tuple[str, ...]:
    """Distinctive words two texts have in common, lightly lemmatised."""

    return tuple(sorted(_content_words(first) & _content_words(second)))


def _content_words(text: str) -> set[str]:
    return {
        _lemma(word)
        for word in _WORD_PATTERN.findall(str(text).casefold())
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
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _validate_verdict(verdict: Mapping[str, Any]) -> None:
    if "maps" not in verdict:
        raise ValueError("verdict is missing the maps field")
    normalise_relation_type(verdict.get("relation_type", ""))
    normalise_task_family(verdict.get("task_family", ""))
    for reason in verdict.get("rejection_reasons", ()):
        if reason not in SELECTION_PREDICATE_REASONS:
            raise ValueError(f"unknown selection predicate reason: {reason}")


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
