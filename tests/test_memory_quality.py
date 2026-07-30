from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from parm_bench.memory_quality import (
    MEMORY_QUALITY_RUBRIC,
    CachedOpenAIMemoryQualityJudge,
    MemoryQualityCacheMissError,
    MemoryQualityCachePolicy,
    deterministic_rejections,
    grade_memory_quality,
    memory_quality_cache_key,
    passes_memory_quality,
    render_memory_quality_request,
)
from scripts.draft_parmbench_v1_claims import (
    DRAFT_PROMPT_VERSION,
    DRAFT_PROMPT_VERSION_V2,
    memory_quality_enabled,
    memory_quality_rejection_reason,
)
from parm_bench.relevance_taxonomy import (
    EDITING_REQUEST_AS_PREFERENCE,
    EPHEMERAL_EVENT_AS_DURABLE_MEMORY,
    QUESTION_AS_PERSONAL_FACT,
    TOPIC_OVERLAP_WITHOUT_DECISION_RELEVANCE,
)


def _turns(*contents: str) -> list[dict[str, str]]:
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": text}
        for index, text in enumerate(contents)
    ]


class StubJudge:
    """Judge stand-in that records what the gate asked it to grade."""

    model_name = "gpt-5-mini"
    rubric_version = MEMORY_QUALITY_RUBRIC

    def __init__(self, **verdict: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.verdict: dict[str, Any] = {
            "memory_worthy": True,
            "category": "preference",
            "durability": "durable",
            "rejection_reasons": [],
            "rationale": "stub",
            "sensitive": False,
            "sensitive_terms": [],
        }
        self.verdict.update(verdict)

    def judge(
        self,
        *,
        claim: str,
        evidence_span: str,
        evidence_turns: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "claim": claim,
                "evidence_span": evidence_span,
                "evidence_turns": list(evidence_turns),
            }
        )
        return dict(self.verdict)


class RecordingClient:
    """Minimal stand-in for the OpenAI client used by the cached judge."""

    def __init__(self, **verdict: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        payload: dict[str, Any] = {
            "memory_worthy": True,
            "category": "owned_item",
            "durability": "durable",
            "rejection_reasons": [],
            "rationale": "the user owns the console",
            "sensitive": False,
            "sensitive_terms": [],
        }
        payload.update(verdict)
        self._output_text = json.dumps(payload, ensure_ascii=False)
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self._output_text,
            model="resolved-judge",
            id="memory-quality-response",
        )


class ExplodingClient:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        raise AssertionError("the judge must not be called")


class ExplodingJudge:
    model_name = "gpt-5-mini"
    rubric_version = MEMORY_QUALITY_RUBRIC

    def judge(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the judge must not be called")


# Reference spans and claims taken from the v1 relevance audit. The four
# defect rows are the ones that reached construction with a clean support
# grade; the five approved rows are the audit's positive references.
ELEGANT_WORDING_SPAN = (
    "refine the language in this message so that it reads more fluidly "
    "and with a touch more elegance"
)
ELEGANT_WORDING_CLAIM = (
    "The user wants messages to read more fluidly and with a touch more "
    "elegance."
)
SINGING_QUESTION_SPAN = (
    "Why does my singing sometimes feel more relaxed and even better when "
    "I'm performing in a small room with just friends compared to on a "
    "stage?"
)
SINGING_QUESTION_CLAIM = (
    "The user performs in a small room with just friends."
)
POUR_OVER_SPAN = (
    "After an early cycling loop along the river, I stopped for a pour-over "
    "at a small cafe tucked between a bookstore and a weathered little shop "
    "filled with old maps and ceramic mugs."
)
POUR_OVER_CLAIM = (
    "The user stopped for a pour-over at a small cafe tucked between a "
    "bookstore and a weathered little shop."
)
STRANDED_SPAN = (
    "Hi, I'm stuck on the side of I-10 again - car just died on me when I "
    "was halfway through the latest episode of that celebrity gossip show."
)
STRANDED_CLAIM = "The user is stuck on the side of I-10; their car just died."

APPROVED_REFERENCES = (
    (
        "dietary exclusion",
        (
            "I've learned to skip the syrupy desserts at these dinners, and "
            "when the tray comes round I'll choose a small bowl of fresh "
            "fruit instead."
        ),
        (
            "The user skips syrup-heavy desserts and chooses fresh fruit "
            "instead."
        ),
        "exclusion",
    ),
    (
        "federation commitment",
        (
            "I'm partway through a season-long athlete-performance analysis "
            "I owe the national federation before their spring review."
        ),
        (
            "The user owes a season-long athlete-performance analysis to a "
            "partnering federation."
        ),
        "active_commitment",
    ),
    (
        "active writing project",
        (
            "I'm drafting a long piece on theological traditions in "
            "politically diverse societies and I keep hitting gaps in the "
            "sources."
        ),
        (
            "The user is drafting a long piece on theological traditions in "
            "politically diverse societies."
        ),
        "active_commitment",
    ),
    (
        "stand-up practice",
        (
            "I perform stand-up comedy and my whole set is speculative "
            "sci-fi material about first contact."
        ),
        (
            "The user performs stand-up comedy with speculative "
            "science-fiction material."
        ),
        "active_commitment",
    ),
    (
        "NES ownership",
        (
            "I still have my old NES hooked up in the front room and I play "
            "it with my nephew on weekends."
        ),
        "The user owns an NES.",
        "owned_item",
    ),
)


class EditingRequestCheckTests(unittest.TestCase):
    def test_refine_request_line_is_rejected(self) -> None:
        self.assertIn(
            EDITING_REQUEST_AS_PREFERENCE,
            deterministic_rejections(
                ELEGANT_WORDING_CLAIM, ELEGANT_WORDING_SPAN
            ),
        )

    def test_framed_and_imperative_requests_are_rejected(self) -> None:
        for span in (
            "Can you rewrite this paragraph so it sounds warmer?",
            "Please translate the note below into Spanish.",
            "Help me word this apology to my landlord.",
            "Make this sound more urgent.",
            "Hi! Could you please polish the opening line.",
            "Summarise the thread for me.",
        ):
            with self.subTest(span=span):
                self.assertIn(
                    EDITING_REQUEST_AS_PREFERENCE,
                    deterministic_rejections("The user wants X.", span),
                )

    def test_statement_span_is_not_an_editing_request(self) -> None:
        for span in (
            "I edit wedding videos for a living and I work off a laptop.",
            "I write my newsletter every Sunday from the same cafe.",
            (
                "I still have my old NES hooked up in the front room and I "
                "play it with my nephew on weekends."
            ),
        ):
            with self.subTest(span=span):
                self.assertEqual(
                    deterministic_rejections("The user does X.", span), ()
                )

    def test_a_stated_fact_beside_a_request_survives(self) -> None:
        span = (
            "I cannot eat dairy at all. Please rewrite the invitation so it "
            "says so."
        )
        self.assertEqual(
            deterministic_rejections("The user cannot eat dairy.", span), ()
        )


class QuestionCheckTests(unittest.TestCase):
    def test_question_only_span_is_a_hard_rejection(self) -> None:
        self.assertEqual(
            deterministic_rejections(
                SINGING_QUESTION_CLAIM, SINGING_QUESTION_SPAN
            ),
            (QUESTION_AS_PERSONAL_FACT,),
        )

    def test_several_questions_are_still_a_rejection(self) -> None:
        span = (
            "Why did royal courts rely on ceremonial codes? Were the codes "
            "written down anywhere?"
        )
        self.assertIn(
            QUESTION_AS_PERSONAL_FACT,
            deterministic_rejections("The user enjoys court codes.", span),
        )

    def test_unpunctuated_wh_question_is_a_rejection(self) -> None:
        self.assertIn(
            QUESTION_AS_PERSONAL_FACT,
            deterministic_rejections(
                "The user brews pour-over coffee.",
                "how long should a pour-over bloom before the main pour",
            ),
        )

    def test_a_declarative_sentence_clears_the_question_check(self) -> None:
        span = (
            "Why does my singing feel easier in a small room? I sing with "
            "the same four friends every Thursday."
        )
        self.assertEqual(
            deterministic_rejections(
                "The user sings with friends on Thursdays.", span
            ),
            (),
        )


class InSessionActivityCheckTests(unittest.TestCase):
    def test_in_session_predicates_are_rejected(self) -> None:
        for claim in (
            "The user is drafting this message to their landlord.",
            "The user is currently asking how to phrase an apology.",
            "The user is rewriting this paragraph for a colleague.",
            "The user is right now translating this note into Spanish.",
        ):
            with self.subTest(claim=claim):
                self.assertIn(
                    EPHEMERAL_EVENT_AS_DURABLE_MEMORY,
                    deterministic_rejections(claim, "I need a hand with it."),
                )

    def test_outside_session_referents_survive(self) -> None:
        for claim in (
            "The user is writing a book about theological traditions.",
            "The user is drafting a grant application for their federation.",
            "The user is talking to their landlord about a lease renewal.",
            "The user writes a newsletter every Sunday.",
        ):
            with self.subTest(claim=claim):
                self.assertEqual(
                    deterministic_rejections(claim, "I do it every week."), ()
                )


class AuditRegressionTests(unittest.TestCase):
    """The four upstream defects the audit traced to the drafting stage."""

    def test_elegant_wording_request_never_reaches_the_judge(self) -> None:
        result = grade_memory_quality(
            ELEGANT_WORDING_CLAIM,
            ELEGANT_WORDING_SPAN,
            _turns(ELEGANT_WORDING_SPAN, "Here is a smoother version."),
            judge=ExplodingJudge(),
        )
        self.assertFalse(passes_memory_quality(result))
        self.assertEqual(
            result.rejection_reasons, (EDITING_REQUEST_AS_PREFERENCE,)
        )
        self.assertEqual(result.durability, "ephemeral")
        self.assertEqual(result.category, "none")

    def test_question_as_fact_never_reaches_the_judge(self) -> None:
        result = grade_memory_quality(
            SINGING_QUESTION_CLAIM,
            SINGING_QUESTION_SPAN,
            _turns(SINGING_QUESTION_SPAN, "Small rooms are more forgiving."),
            judge=ExplodingJudge(),
        )
        self.assertFalse(passes_memory_quality(result))
        self.assertEqual(
            result.rejection_reasons, (QUESTION_AS_PERSONAL_FACT,)
        )

    def test_single_pour_over_stop_is_rejected_as_ephemeral(self) -> None:
        judge = StubJudge(
            memory_worthy=False,
            category="none",
            durability="ephemeral",
            rejection_reasons=[EPHEMERAL_EVENT_AS_DURABLE_MEMORY],
            rationale="one narrated visit, no standing preference",
        )
        result = grade_memory_quality(
            POUR_OVER_CLAIM,
            POUR_OVER_SPAN,
            _turns(POUR_OVER_SPAN, "Here is the translation."),
            judge=judge,
        )
        self.assertEqual(deterministic_rejections(POUR_OVER_CLAIM, POUR_OVER_SPAN), ())
        self.assertEqual(len(judge.calls), 1)
        self.assertFalse(passes_memory_quality(result))
        self.assertEqual(
            result.rejection_reasons, (EPHEMERAL_EVENT_AS_DURABLE_MEMORY,)
        )

    def test_one_time_stranding_is_rejected_as_ephemeral(self) -> None:
        judge = StubJudge(
            memory_worthy=False,
            category="none",
            durability="ephemeral",
            rejection_reasons=[EPHEMERAL_EVENT_AS_DURABLE_MEMORY],
            rationale="a roadside breakdown resolves within hours",
        )
        result = grade_memory_quality(
            STRANDED_CLAIM,
            STRANDED_SPAN,
            _turns(STRANDED_SPAN, "Here is a more urgent version."),
            judge=judge,
        )
        self.assertEqual(len(judge.calls), 1)
        self.assertFalse(passes_memory_quality(result))
        self.assertEqual(result.durability, "ephemeral")

    def test_broad_interest_is_rejected_on_decision_relevance(self) -> None:
        judge = StubJudge(
            memory_worthy=False,
            category="none",
            durability="durable",
            rejection_reasons=[TOPIC_OVERLAP_WITHOUT_DECISION_RELEVANCE],
            rationale="an interest that ranks no option",
        )
        result = grade_memory_quality(
            "The user is interested in medieval history.",
            "I have always been interested in medieval history.",
            _turns("I have always been interested in medieval history."),
            judge=judge,
        )
        self.assertFalse(passes_memory_quality(result))

    def test_approved_references_reach_and_pass_the_judge(self) -> None:
        for name, span, claim, category in APPROVED_REFERENCES:
            with self.subTest(reference=name):
                judge = StubJudge(
                    category=category,
                    durability=(
                        "currently_operative"
                        if category == "active_commitment"
                        else "durable"
                    ),
                    rationale="a fact that ranks ordinary options",
                )
                result = grade_memory_quality(
                    claim, span, _turns(span, "Understood."), judge=judge
                )
                self.assertEqual(
                    deterministic_rejections(claim, span),
                    (),
                    "no deterministic check may fire on an approved example",
                )
                self.assertEqual(len(judge.calls), 1)
                self.assertTrue(passes_memory_quality(result))
                self.assertEqual(result.category, category)
                self.assertEqual(result.rejection_reasons, ())


class GradeMemoryQualityTests(unittest.TestCase):
    def test_hard_rejection_skips_the_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = grade_memory_quality(
                ELEGANT_WORDING_CLAIM,
                ELEGANT_WORDING_SPAN,
                _turns(ELEGANT_WORDING_SPAN),
                cache_dir=tmp,
                client=ExplodingClient(),
            )
            self.assertEqual(list(Path(tmp).glob("*.json")), [])
        self.assertEqual(
            result.deterministic_rejections, (EDITING_REQUEST_AS_PREFERENCE,)
        )
        self.assertIn(EDITING_REQUEST_AS_PREFERENCE, result.rationale)

    def test_the_judge_sees_claim_span_and_turns(self) -> None:
        span = APPROVED_REFERENCES[4][1]
        claim = APPROVED_REFERENCES[4][2]
        turns = _turns(span, "An NES needs a composite input.")
        judge = StubJudge()
        grade_memory_quality(claim, span, turns, judge=judge)
        call = judge.calls[0]
        self.assertEqual(call["claim"], claim)
        self.assertEqual(call["evidence_span"], span)
        self.assertEqual(call["evidence_turns"], turns)
        rendered = render_memory_quality_request(claim, span, turns)
        self.assertIn(claim, rendered)
        self.assertIn(span, rendered)
        self.assertIn("An NES needs a composite input.", rendered)

    def test_ephemeral_durability_fails_even_when_worthy(self) -> None:
        judge = StubJudge(durability="ephemeral")
        result = grade_memory_quality(
            "The user drank a flat white this morning.",
            "I drank a flat white this morning before the standup.",
            _turns("I drank a flat white this morning before the standup."),
            judge=judge,
        )
        self.assertTrue(result.memory_worthy)
        self.assertFalse(passes_memory_quality(result))

    def test_currently_operative_passes(self) -> None:
        judge = StubJudge(
            category="active_commitment", durability="currently_operative"
        )
        result = grade_memory_quality(
            "The user owes an analysis to the federation.",
            "I owe the federation a season-long analysis by spring.",
            _turns("I owe the federation a season-long analysis by spring."),
            judge=judge,
        )
        self.assertTrue(passes_memory_quality(result))

    def test_sensitive_terms_are_kept_only_when_sensitive(self) -> None:
        sensitive = StubJudge(
            category="accessibility_need",
            sensitive=True,
            sensitive_terms=["knee surgery", " recovery ", ""],
        )
        result = grade_memory_quality(
            "The user is recovering from knee surgery.",
            "I had knee surgery in March and I still avoid stairs.",
            _turns("I had knee surgery in March and I still avoid stairs."),
            judge=sensitive,
        )
        self.assertTrue(result.sensitive)
        self.assertEqual(result.sensitive_terms, ("knee surgery", "recovery"))

        insensitive = StubJudge(sensitive=False, sensitive_terms=["leftover"])
        plain = grade_memory_quality(
            "The user owns an NES.",
            "I still have my old NES hooked up in the front room.",
            _turns("I still have my old NES hooked up in the front room."),
            judge=insensitive,
        )
        self.assertEqual(plain.sensitive_terms, ())

    def test_result_serialises_the_provenance_fields(self) -> None:
        judge = StubJudge(
            category="constraint",
            sensitive=True,
            sensitive_terms=["gluten"],
        )
        payload = grade_memory_quality(
            "The user cannot eat gluten.",
            "I cannot eat gluten, it puts me out for a day.",
            _turns("I cannot eat gluten, it puts me out for a day."),
            judge=judge,
        ).to_dict()
        self.assertEqual(payload["category"], "constraint")
        self.assertEqual(payload["durability"], "durable")
        self.assertEqual(payload["rejection_reasons"], [])
        self.assertEqual(payload["sensitive_terms"], ["gluten"])
        self.assertEqual(payload["rubric_version"], MEMORY_QUALITY_RUBRIC)
        self.assertEqual(payload["requested_model"], "gpt-5-mini")

    def test_unknown_verdict_fields_are_refused(self) -> None:
        for verdict in (
            {"category": "vibe"},
            {"durability": "eternal"},
            {"rejection_reasons": ["cue_alone_determines_choice"]},
        ):
            with self.subTest(verdict=verdict):
                with self.assertRaises(ValueError):
                    grade_memory_quality(
                        "The user owns an NES.",
                        "I still have my old NES in the front room.",
                        _turns("I still have my old NES in the front room."),
                        judge=StubJudge(**verdict),
                    )


class MemoryQualityCacheTests(unittest.TestCase):
    span = "I still have my old NES hooked up in the front room."
    claim = "The user owns an NES."

    def turns(self) -> list[dict[str, str]]:
        return _turns(self.span, "That needs a composite input.")

    def test_cache_key_is_stable_and_input_sensitive(self) -> None:
        turns = self.turns()
        baseline = memory_quality_cache_key(self.claim, self.span, turns)
        self.assertEqual(
            baseline,
            memory_quality_cache_key(self.claim, self.span, list(turns)),
        )
        for other in (
            memory_quality_cache_key(
                "The user owns a Mega Drive.", self.span, turns
            ),
            memory_quality_cache_key(self.claim, "I own an NES.", turns),
            memory_quality_cache_key(self.claim, self.span, turns[:1]),
            memory_quality_cache_key(
                self.claim, self.span, turns, model="other-model"
            ),
            memory_quality_cache_key(
                self.claim,
                self.span,
                turns,
                rubric_version="parmbench_memory_quality_v0",
            ),
        ):
            self.assertNotEqual(baseline, other)

    def test_populate_then_frozen_replay(self) -> None:
        turns = self.turns()
        client = RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            first = grade_memory_quality(
                self.claim, self.span, turns, cache_dir=tmp, client=client
            )
            request_hash = memory_quality_cache_key(
                self.claim, self.span, turns
            )
            self.assertTrue((Path(tmp) / f"{request_hash}.json").exists())

            frozen = CachedOpenAIMemoryQualityJudge(
                tmp,
                MemoryQualityCachePolicy.FROZEN,
                client=ExplodingClient(),
            )
            replay = grade_memory_quality(
                self.claim, self.span, turns, judge=frozen
            )
            self.assertEqual(first, replay)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(first.resolved_model, "resolved-judge")
            self.assertEqual(first.response_id, "memory-quality-response")
            self.assertIsNotNone(frozen.cache_hash)

    def test_frozen_cache_miss_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MemoryQualityCacheMissError):
                grade_memory_quality(
                    self.claim,
                    self.span,
                    self.turns(),
                    cache_dir=tmp,
                    policy="frozen",
                    client=ExplodingClient(),
                )

    def test_cached_judge_is_given_the_versioned_rubric(self) -> None:
        client = RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            grade_memory_quality(
                self.claim,
                self.span,
                self.turns(),
                cache_dir=tmp,
                client=client,
            )
            cached = json.loads(
                next(Path(tmp).glob("*.json")).read_text(encoding="utf-8")
            )
        self.assertEqual(cached["rubric_version"], MEMORY_QUALITY_RUBRIC)
        self.assertEqual(cached["model"], "gpt-5-mini")
        self.assertEqual(client.calls[0]["model"], "gpt-5-mini")


class SupplyStageTests(unittest.TestCase):
    """How the drafting script decides to run the stage and label failures."""

    def test_frozen_rubrics_replay_without_the_gate(self) -> None:
        for rubric in (DRAFT_PROMPT_VERSION, DRAFT_PROMPT_VERSION_V2):
            with self.subTest(rubric=rubric):
                self.assertFalse(memory_quality_enabled(rubric, None))
                self.assertTrue(memory_quality_enabled(rubric, True))

    def test_a_new_rubric_gets_the_gate_by_default(self) -> None:
        self.assertTrue(
            memory_quality_enabled("parmbench_claim_draft_v3", None)
        )
        self.assertFalse(
            memory_quality_enabled("parmbench_claim_draft_v3", False)
        )

    def test_rejection_reason_names_the_taxonomy_reason(self) -> None:
        judge = StubJudge(
            memory_worthy=False,
            category="none",
            durability="ephemeral",
            rejection_reasons=[EPHEMERAL_EVENT_AS_DURABLE_MEMORY],
        )
        result = grade_memory_quality(
            POUR_OVER_CLAIM, POUR_OVER_SPAN, _turns(POUR_OVER_SPAN), judge=judge
        )
        self.assertEqual(
            memory_quality_rejection_reason(result),
            f"memory_quality_{EPHEMERAL_EVENT_AS_DURABLE_MEMORY}",
        )

    def test_rejection_reason_falls_back_to_durability(self) -> None:
        result = grade_memory_quality(
            "The user drank a flat white this morning.",
            "I drank a flat white this morning.",
            _turns("I drank a flat white this morning."),
            judge=StubJudge(durability="ephemeral"),
        )
        self.assertEqual(
            memory_quality_rejection_reason(result), "memory_quality_ephemeral"
        )

    def test_rejection_reason_covers_a_bare_refusal(self) -> None:
        result = grade_memory_quality(
            "The user likes trivia.",
            "I like trivia well enough.",
            _turns("I like trivia well enough."),
            judge=StubJudge(memory_worthy=False, category="none"),
        )
        self.assertEqual(
            memory_quality_rejection_reason(result),
            "memory_quality_not_worth_retaining",
        )


if __name__ == "__main__":
    unittest.main()
