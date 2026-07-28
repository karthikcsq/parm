from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from parm_bench.evidence_gate import (
    ASSISTANT_ONLY_EVIDENCE,
    NO_LEXICAL_ANCHOR,
    NO_USER_EVIDENCE,
    TOPICAL_QUESTION_ONLY,
    UNJUSTIFIED_FREQUENCY,
    CachedOpenAISupportJudge,
    EvidenceGateCacheMissError,
    EvidenceGateCachePolicy,
    EvidenceGateResult,
    deterministic_flags,
    evidence_gate_cache_key,
    grade_claim,
    passes_gate,
)


class RecordingClient:
    """Minimal stand-in for the OpenAI client used by the cached judge."""

    def __init__(
        self,
        grade: str = "explicit",
        rationale: str = "the user says so",
        evidence: tuple[str, ...] = ("I cook salsa on Sundays.",),
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        payload = {
            "grade": grade,
            "rationale": rationale,
            "evidence": list(evidence),
        }
        self._output_text = json.dumps(payload, ensure_ascii=False)
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self._output_text,
            model="resolved-judge",
            id="support-response",
        )


class ExplodingClient:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        raise AssertionError("the judge must not be called")


class DeterministicRejectionTests(unittest.TestCase):
    def test_frequency_word_needs_two_user_statements(self) -> None:
        turns = [
            {"role": "user", "content": "I cooked salsa last Sunday."},
            {"role": "assistant", "content": "That sounds great."},
        ]
        self.assertIn(
            UNJUSTIFIED_FREQUENCY,
            deterministic_flags("The user usually cooks salsa.", turns),
        )

    def test_frequency_word_accepts_repeated_statements(self) -> None:
        turns = [
            {"role": "user", "content": "I cooked salsa last Sunday."},
            {"role": "assistant", "content": "Nice."},
            {
                "role": "user",
                "content": "I made another batch of salsa this weekend.",
            },
        ]
        self.assertNotIn(
            UNJUSTIFIED_FREQUENCY,
            deterministic_flags("The user usually cooks salsa.", turns),
        )

    def test_assistant_only_evidence(self) -> None:
        turns = [
            {"role": "assistant", "content": "You might enjoy The Crown."},
            {"role": "assistant", "content": "Historical dramas suit you."},
        ]
        self.assertEqual(
            deterministic_flags("The user enjoys historical dramas.", turns),
            (ASSISTANT_ONLY_EVIDENCE,),
        )

    def test_empty_evidence(self) -> None:
        self.assertEqual(
            deterministic_flags("The user enjoys historical dramas.", []),
            (NO_USER_EVIDENCE,),
        )

    def test_topical_question_only_is_a_soft_flag(self) -> None:
        turns = [
            {
                "role": "user",
                "content": (
                    "Why did royal courts rely on ceremonial codes? "
                    "Were the codes written down?"
                ),
            },
            {"role": "assistant", "content": "Courts used codes for rank."},
        ]
        flags = deterministic_flags(
            "The user enjoys royal court codes.", turns
        )
        self.assertIn(TOPICAL_QUESTION_ONLY, flags)
        self.assertEqual(
            tuple(
                flag
                for flag in flags
                if flag in (ASSISTANT_ONLY_EVIDENCE, UNJUSTIFIED_FREQUENCY)
            ),
            (),
        )

    def test_declarative_user_turn_clears_question_flag(self) -> None:
        turns = [
            {
                "role": "user",
                "content": (
                    "Why did royal courts use ceremonial codes? "
                    "I watch every costume drama I can find."
                ),
            },
        ]
        self.assertNotIn(
            TOPICAL_QUESTION_ONLY,
            deterministic_flags("The user watches costume drama.", turns),
        )

    def test_no_lexical_anchor_is_a_soft_flag(self) -> None:
        turns = [
            {"role": "user", "content": "I moved my standing desk yesterday."},
        ]
        flags = deterministic_flags("The user practices kite surfing.", turns)
        self.assertEqual(flags, (NO_LEXICAL_ANCHOR,))

    def test_lexical_overlap_survives_inflection(self) -> None:
        turns = [
            {"role": "user", "content": "I have been surfing since April."},
        ]
        self.assertNotIn(
            NO_LEXICAL_ANCHOR,
            deterministic_flags("The user surfs on weekends.", turns),
        )


class GradeClaimTests(unittest.TestCase):
    def test_hard_rejection_skips_the_judge(self) -> None:
        turns = [
            {"role": "assistant", "content": "You might enjoy The Crown."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = grade_claim(
                "The user enjoys historical dramas.",
                turns,
                cache_dir=tmp,
                client=ExplodingClient(),
            )
        self.assertEqual(result.grade, "unsupported")
        self.assertEqual(
            result.deterministic_rejections, (ASSISTANT_ONLY_EVIDENCE,)
        )
        self.assertIn(ASSISTANT_ONLY_EVIDENCE, result.rationale)
        self.assertFalse(passes_gate(result))

    def test_frequency_rejection_skips_the_judge(self) -> None:
        turns = [
            {"role": "user", "content": "I cooked salsa last Sunday."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = grade_claim(
                "The user always cooks salsa.",
                turns,
                cache_dir=tmp,
                client=ExplodingClient(),
            )
        self.assertEqual(result.grade, "unsupported")
        self.assertIn(UNJUSTIFIED_FREQUENCY, result.deterministic_rejections)
        self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_soft_flag_passes_through_to_the_judge(self) -> None:
        turns = [
            {
                "role": "user",
                "content": "Why did royal courts rely on ceremonial codes?",
            },
        ]
        client = RecordingClient(grade="unsupported", rationale="a question")
        with tempfile.TemporaryDirectory() as tmp:
            result = grade_claim(
                "The user enjoys ceremonial court codes.",
                turns,
                cache_dir=tmp,
                client=client,
            )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.grade, "unsupported")
        self.assertEqual(
            result.deterministic_rejections, (TOPICAL_QUESTION_ONLY,)
        )
        self.assertFalse(passes_gate(result))

    def test_judge_never_sees_deterministic_flags(self) -> None:
        turns = [
            {
                "role": "user",
                "content": "Why did royal courts rely on ceremonial codes?",
            },
        ]
        client = RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            grade_claim(
                "The user enjoys ceremonial court codes.",
                turns,
                cache_dir=tmp,
                client=client,
            )
        rendered = "\n".join(
            str(value) for value in client.calls[0].values()
        )
        for flag in (TOPICAL_QUESTION_ONLY, NO_LEXICAL_ANCHOR):
            self.assertNotIn(flag, rendered)

    def test_supported_claim_passes_the_gate(self) -> None:
        turns = [
            {"role": "user", "content": "I cook salsa on Sundays."},
            {"role": "assistant", "content": "Great."},
        ]
        client = RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            result = grade_claim(
                "The user cooks salsa.",
                turns,
                cache_dir=tmp,
                client=client,
            )
        self.assertEqual(result.grade, "explicit")
        self.assertEqual(result.evidence, ("I cook salsa on Sundays.",))
        self.assertEqual(result.resolved_model, "resolved-judge")
        self.assertEqual(result.response_id, "support-response")
        self.assertTrue(passes_gate(result))

    def test_evidence_is_capped_at_two_quotes(self) -> None:
        turns = [{"role": "user", "content": "I cook salsa on Sundays."}]
        client = RecordingClient(evidence=("one", "two", "three"))
        with tempfile.TemporaryDirectory() as tmp:
            result = grade_claim(
                "The user cooks salsa.",
                turns,
                cache_dir=tmp,
                client=client,
            )
        self.assertEqual(result.evidence, ("one", "two"))

    def test_gate_predicate_covers_every_grade(self) -> None:
        for grade, expected in (
            ("explicit", True),
            ("inferable", True),
            ("unsupported", False),
        ):
            result = EvidenceGateResult(grade=grade, rationale="")
            self.assertEqual(passes_gate(result), expected)


class SupportJudgeCacheTests(unittest.TestCase):
    def test_cache_key_is_stable_and_input_sensitive(self) -> None:
        turns = [{"role": "user", "content": "I cook salsa on Sundays."}]
        baseline = evidence_gate_cache_key("The user cooks salsa.", turns)
        self.assertEqual(
            baseline,
            evidence_gate_cache_key("The user cooks salsa.", list(turns)),
        )
        self.assertNotEqual(
            baseline,
            evidence_gate_cache_key("The user cooks curry.", turns),
        )
        self.assertNotEqual(
            baseline,
            evidence_gate_cache_key(
                "The user cooks salsa.",
                [{"role": "assistant", "content": "I cook salsa on Sundays."}],
            ),
        )
        self.assertNotEqual(
            baseline,
            evidence_gate_cache_key(
                "The user cooks salsa.", turns, model="other-model"
            ),
        )
        self.assertNotEqual(
            baseline,
            evidence_gate_cache_key(
                "The user cooks salsa.",
                turns,
                rubric_version="personamem_source_support_v1",
            ),
        )

    def test_populate_then_frozen_replay(self) -> None:
        turns = [{"role": "user", "content": "I cook salsa on Sundays."}]
        claim = "The user cooks salsa."
        client = RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            first = grade_claim(claim, turns, cache_dir=tmp, client=client)
            request_hash = evidence_gate_cache_key(claim, turns)
            self.assertTrue((Path(tmp) / f"{request_hash}.json").exists())

            frozen = CachedOpenAISupportJudge(
                tmp,
                EvidenceGateCachePolicy.FROZEN,
                client=ExplodingClient(),
            )
            replay = grade_claim(claim, turns, judge=frozen)
            self.assertEqual(first, replay)
            self.assertEqual(len(client.calls), 1)
            self.assertIsNotNone(frozen.cache_hash)

    def test_frozen_cache_miss_raises(self) -> None:
        turns = [{"role": "user", "content": "I cook salsa on Sundays."}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(EvidenceGateCacheMissError):
                grade_claim(
                    "The user cooks salsa.",
                    turns,
                    cache_dir=tmp,
                    policy="frozen",
                    client=ExplodingClient(),
                )


if __name__ == "__main__":
    unittest.main()
