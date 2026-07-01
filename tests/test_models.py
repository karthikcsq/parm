from __future__ import annotations

import unittest
from types import SimpleNamespace

from parm_bench.models import (
    FINAL_ANSWER_INSTRUCTIONS,
    OpenAIResponsesModel,
)


class FakeUsage:
    def model_dump(self, exclude_none: bool = False) -> dict[str, int]:
        return {
            "input_tokens": 20,
            "output_tokens": 3,
            "total_tokens": 23,
        }


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="Chosen Label",
            id="resp_123",
            model="gpt-5-mini-resolved",
            usage=FakeUsage(),
        )


class OpenAIResponsesModelTests(unittest.TestCase):
    def test_generate_uses_non_stored_responses_call(self) -> None:
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        model = OpenAIResponsesModel("gpt-5-mini", client=client)

        response = model.generate(
            prompt="Choose exactly one.",
            observation_kind="tool_result",
            observation_text="Choice A\nChoice B",
        )

        self.assertEqual(
            responses.calls,
            [
                {
                    "model": "gpt-5-mini",
                    "instructions": FINAL_ANSWER_INSTRUCTIONS,
                    "input": (
                        "Task:\nChoose exactly one.\n\n"
                        "Observed tool result:\nChoice A\nChoice B"
                    ),
                    "max_output_tokens": 512,
                    "store": False,
                }
            ],
        )
        self.assertEqual(response.text, "Chosen Label")
        self.assertEqual(response.response_id, "resp_123")
        self.assertEqual(response.resolved_model, "gpt-5-mini-resolved")
        self.assertEqual(response.usage["total_tokens"], 23)

    def test_generate_accepts_baseline_specific_instructions_and_memory(self) -> None:
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        model = OpenAIResponsesModel("gpt-5-mini", client=client)

        model.generate(
            prompt="Choose exactly one.",
            observation_kind="tool_result",
            observation_text="Choice A\nChoice B",
            instructions="Use retrieved memory.",
            memory_context="1. Relevant private fact",
        )

        self.assertEqual(
            responses.calls[0]["instructions"],
            "Use retrieved memory.",
        )
        self.assertEqual(
            responses.calls[0]["input"],
            (
                "Task:\nChoose exactly one.\n\n"
                "Observed tool result:\nChoice A\nChoice B\n\n"
                "Retrieved personal memory:\n1. Relevant private fact"
            ),
        )


if __name__ == "__main__":
    unittest.main()
