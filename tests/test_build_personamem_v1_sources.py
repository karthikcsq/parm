from __future__ import annotations

import unittest

from parm_bench.personamem import PersonaMemSourceRow
from scripts.build_personamem_v1_sources import is_eligible_row


def _row(**overrides: object) -> PersonaMemSourceRow:
    defaults: dict[str, object] = {
        "source_row_id": "train_text:0",
        "persona_id": 1,
        "history_path": "data/chat_history_32k/chat_history_000000_000000_persona1.json",
        "preference": "Enjoys long trail runs on weekends",
        "related_conversation_snippet": (
            {"role": "user", "content": "I usually go for a long trail run on weekends."},
        ),
        "user_query": {"role": "user", "content": "Any trail suggestions nearby?"},
        "correct_answer": "Try the ridge loop, it is a longer trail.",
        "incorrect_answers": ("Try the short paved loop instead.",),
        "updated": False,
        "previous_preference": None,
        "who": "self",
        "sensitive_info": False,
        "preference_type": "explicit",
        "topic_preference": "fitness",
    }
    defaults.update(overrides)
    return PersonaMemSourceRow(**defaults)  # type: ignore[arg-type]


class EligibilityFilterTests(unittest.TestCase):
    def test_ordinary_current_self_row_is_eligible(self) -> None:
        self.assertTrue(is_eligible_row(_row()))

    def test_updated_row_is_not_eligible(self) -> None:
        self.assertFalse(is_eligible_row(_row(updated=True)))

    def test_sensitive_row_is_not_eligible(self) -> None:
        self.assertFalse(is_eligible_row(_row(sensitive_info=True)))

    def test_non_self_row_is_not_eligible(self) -> None:
        self.assertFalse(is_eligible_row(_row(who="assistant")))

    def test_ask_to_forget_preference_type_is_not_eligible(self) -> None:
        self.assertFalse(is_eligible_row(_row(preference_type="ask_to_forget")))

    def test_do_not_remember_preference_text_is_not_eligible(self) -> None:
        self.assertFalse(
            is_eligible_row(_row(preference="Do not remember this trail detail"))
        )

    def test_mixed_case_do_not_remember_preference_is_not_eligible(self) -> None:
        self.assertFalse(
            is_eligible_row(_row(preference="DO NOT REMEMBER the trail name"))
        )


if __name__ == "__main__":
    unittest.main()
