"""Shared relevance-failure taxonomy for construction gates.

These machine-readable reasons come from the v1 relevance calibration audit
(`data/parmbench-v1-supply/relevance-audit-v1.md`). Source-support rejection
reasons live in `evidence_gate` and are deliberately not mixed in here: a
claim can be perfectly supported by the user's own words and still fail on
memory quality or decision validity.
"""

from __future__ import annotations

# Memory-quality failures: the fact is not worth retaining as durable
# personal memory, however well supported.
EPHEMERAL_EVENT_AS_DURABLE_MEMORY = "ephemeral_event_as_durable_memory"
QUESTION_AS_PERSONAL_FACT = "question_as_personal_fact"
EDITING_REQUEST_AS_PREFERENCE = "editing_request_as_preference"
TOPIC_OVERLAP_WITHOUT_DECISION_RELEVANCE = (
    "topic_overlap_without_decision_relevance"
)
MEDICAL_OR_SENSITIVE_OVERREACH = "medical_or_sensitive_overreach"

MEMORY_QUALITY_REASONS = (
    EPHEMERAL_EVENT_AS_DURABLE_MEMORY,
    QUESTION_AS_PERSONAL_FACT,
    EDITING_REQUEST_AS_PREFERENCE,
    TOPIC_OVERLAP_WITHOUT_DECISION_RELEVANCE,
    MEDICAL_OR_SENSITIVE_OVERREACH,
)

# Decision-validity failures: the built scenario does not follow from the
# accepted fact without invention, or its fixture roles are broken.
INVENTED_RELATIONSHIP_OR_PERMISSION = "invented_relationship_or_permission"
INVENTED_LOCATION_OR_POSSESSION = "invented_location_or_possession"
CUE_ALONE_DETERMINES_CHOICE = "cue_alone_determines_choice"
WEAK_PERSONAL_BENEFIT = (
    "weak_personal_benefit_cannot_override_ordinary_evidence"
)
IMPLAUSIBLE_TASK_OR_AFFORDANCE = "implausible_task_or_affordance"
INCORRECT_CAPABILITY_LABEL = "incorrect_capability_label"
CONTROL_DOES_NOT_REMOVE_ADVANTAGE = (
    "control_does_not_remove_personalized_advantage"
)

DECISION_VALIDITY_REASONS = (
    INVENTED_RELATIONSHIP_OR_PERMISSION,
    INVENTED_LOCATION_OR_POSSESSION,
    MEDICAL_OR_SENSITIVE_OVERREACH,
    CUE_ALONE_DETERMINES_CHOICE,
    WEAK_PERSONAL_BENEFIT,
    IMPLAUSIBLE_TASK_OR_AFFORDANCE,
    INCORRECT_CAPABILITY_LABEL,
    CONTROL_DOES_NOT_REMOVE_ADVANTAGE,
)

# Selection-predicate failures: the accepted fact reached the mapper and no
# usable decision rule came back. `no_selection_predicate` is the abstention
# label the supply pipeline records, and abstaining is a correct outcome, not
# a defect in the fact.
NO_SELECTION_PREDICATE = "no_selection_predicate"
INCOMPATIBLE_TASK_FAMILY_ROUTING = "incompatible_task_family_routing"
MATERIAL_ASSUMPTION_REQUIRED = "material_assumption_required"
UNANCHORED_SELECTION_PREDICATE = "unanchored_selection_predicate"

SELECTION_PREDICATE_REASONS = (
    NO_SELECTION_PREDICATE,
    INCOMPATIBLE_TASK_FAMILY_ROUTING,
    MATERIAL_ASSUMPTION_REQUIRED,
    UNANCHORED_SELECTION_PREDICATE,
    TOPIC_OVERLAP_WITHOUT_DECISION_RELEVANCE,
)

ALL_RELEVANCE_REASONS = tuple(
    dict.fromkeys(
        MEMORY_QUALITY_REASONS
        + DECISION_VALIDITY_REASONS
        + SELECTION_PREDICATE_REASONS
    )
)
