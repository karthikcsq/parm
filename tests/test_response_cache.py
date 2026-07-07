from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parm_bench.models import (
    CachingLanguageModel,
    ModelResponse,
    ResponseCacheMissError,
    ResponsePolicy,
)


class FakeModel:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        *,
        prompt: str,
        observation_kind: str,
        observation_text: str,
        instructions: str = "instr",
        memory_context: str | None = None,
    ) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            text=f"answer-{self.calls}",
            response_id=f"resp-{self.calls}",
            resolved_model="fake-model-2025",
            usage={"output_tokens": self.calls},
        )


def _request(**overrides: str) -> dict[str, str]:
    base = {
        "prompt": "pick one",
        "observation_kind": "tool_result",
        "observation_text": "listing text",
        "instructions": "instr",
    }
    base.update(overrides)
    return base


class ResponseCacheTests(unittest.TestCase):
    def test_populate_persists_then_frozen_replays_without_calling_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = FakeModel()
            populate = CachingLanguageModel(base, tmp, ResponsePolicy.POPULATE)
            first = populate.generate(**_request())
            self.assertEqual(base.calls, 1)

            frozen_base = FakeModel()
            frozen = CachingLanguageModel(frozen_base, tmp, ResponsePolicy.FROZEN)
            replayed = frozen.generate(**_request())
            self.assertEqual(frozen_base.calls, 0)
            self.assertEqual(replayed.text, first.text)
            self.assertEqual(replayed.response_id, first.response_id)
            self.assertIsNotNone(frozen.cache_hash)

    def test_frozen_miss_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frozen = CachingLanguageModel(FakeModel(), tmp, ResponsePolicy.FROZEN)
            with self.assertRaises(ResponseCacheMissError):
                frozen.generate(**_request())

    def test_memory_context_changes_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = FakeModel()
            model = CachingLanguageModel(base, tmp, ResponsePolicy.POPULATE)
            model.generate(**_request())
            model.generate(**_request(), memory_context="a retrieved note")
            self.assertEqual(base.calls, 2)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 2)

    def test_tampered_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            CachingLanguageModel(FakeModel(), tmp, ResponsePolicy.POPULATE).generate(
                **_request()
            )
            entry = next(Path(tmp).glob("*.json"))
            entry.write_text(
                entry.read_text(encoding="utf-8").replace("fake-model", "other-model"),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                CachingLanguageModel(FakeModel(), tmp, ResponsePolicy.FROZEN).generate(
                    **_request()
                )


if __name__ == "__main__":
    unittest.main()
