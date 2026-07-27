from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parm_bench.corpus import (
    NormalizedSourceRecord,
    SensitivityMetadata,
    UpdateStatus,
    text_sha256,
)
from parm_bench.corpus_index import write_corpus_retrieval_index
from parm_bench.personamem import PersonaMemSourceRow, PersonaMemV2Adapter
from parm_bench.retrieval import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    IndexRetriever,
    RetrievalIndex,
    RetrievalRequest,
    RetrievalValidationError,
)


REVISION = "b7b42b78917157afed063527a1c959e98f6109f2"


class FakeEmbedder:
    model_name = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def embed(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for index, text in enumerate(texts):
            normalized = text.casefold()
            vectors[index, 0] = normalized.count("alpha")
            vectors[index, 1] = normalized.count("beta")
            if not vectors[index, :2].any():
                vectors[index, 2] = 1.0
        return vectors


def source_mapping(
    *,
    preference: str = "Enjoys historical dramas on TV",
    updated: bool = False,
    sensitive: bool = False,
    pref_type: str = "neutral_preferences",
) -> dict[str, object]:
    snippet = [
        {"role": "user", "content": "I rewatched a historical court drama."},
        {"role": "assistant", "content": "The period detail sounds memorable."},
    ]
    return {
        "persona_id": 887,
        "chat_history_32k_link": (
            "data/chat_history_32k/"
            "chat_history_250913_163134_persona887.json"
        ),
        "preference": preference,
        "related_conversation_snippet": json.dumps(snippet),
        "user_query": repr(
            {"role": "user", "content": "What should I watch this weekend?"}
        ),
        "correct_answer": "Try a historical drama.",
        "incorrect_answers": json.dumps(["Try a sitcom.", "Try a game show."]),
        "updated": updated,
        "prev_pref": "Preferred sitcoms" if updated else None,
        "who": "self",
        "sensitive_info": sensitive,
        "pref_type": pref_type,
        "topic_preference": "Entertainment",
    }


class PersonaMemAdapterTests(unittest.TestCase):
    def test_adapter_excludes_system_and_preserves_gold_snippet(self) -> None:
        row = PersonaMemSourceRow.from_mapping(
            source_mapping(), source_row_id="train_text:7"
        )
        history = {
            "metadata": {"persona_id": 887},
            "chat_history": [
                {
                    "role": "system",
                    "content": "Expanded persona shortcut that must not be indexed.",
                },
                *list(row.related_conversation_snippet),
                {"role": "user", "content": "An unrelated later turn."},
            ],
        }

        records = PersonaMemV2Adapter(REVISION).adapt(
            row,
            history,
            history_sha256="a" * 64,
        )

        self.assertEqual(len(records), 2)
        self.assertFalse(
            any("Expanded persona shortcut" in record.text for record in records)
        )
        gold = next(
            record
            for record in records
            if record.provenance["gold_related_snippet"]
        )
        self.assertEqual(gold.corpus_id, "personamem-v2/train/persona-887")
        self.assertEqual(gold.update_status, UpdateStatus.CURRENT)
        self.assertEqual(gold.annotations["preference"], row.preference)
        self.assertIn("I rewatched a historical court drama.", gold.text)
        self.assertEqual(gold.source_hash, text_sha256(gold.text))

    def test_forget_and_sensitive_rows_are_restrained(self) -> None:
        row = PersonaMemSourceRow.from_mapping(
            source_mapping(
                preference="Do not remember 'Enjoys historical dramas' in memory",
                updated=True,
                sensitive=True,
                pref_type="ask_to_forget",
            ),
            source_row_id="train_text:8",
        )
        history = {
            "metadata": {"persona_id": 887},
            "chat_history": list(row.related_conversation_snippet),
        }

        gold = PersonaMemV2Adapter(REVISION).adapt(
            row,
            history,
            history_sha256="b" * 64,
        )[0]

        self.assertEqual(gold.update_status, UpdateStatus.FORGET_REQUESTED)
        self.assertTrue(gold.sensitivity.flagged)
        self.assertEqual(
            gold.sensitivity.handling, "exclude_from_ordinary_positive"
        )

    def test_updated_preference_keeps_latest_value_current(self) -> None:
        row = PersonaMemSourceRow.from_mapping(
            source_mapping(updated=True),
            source_row_id="train_text:9",
        )
        history = {
            "metadata": {"persona_id": 887},
            "chat_history": list(row.related_conversation_snippet),
        }

        gold = PersonaMemV2Adapter(REVISION).adapt(
            row,
            history,
            history_sha256="d" * 64,
        )[0]

        self.assertEqual(gold.update_status, UpdateStatus.CURRENT)
        self.assertEqual(
            gold.annotations["previous_preference"], "Preferred sitcoms"
        )


class CorpusIndexTests(unittest.TestCase):
    def test_normalized_record_round_trips(self) -> None:
        record = self._record(
            "personamem-v2/train/persona-1", "alpha source", "alpha"
        )
        self.assertEqual(
            NormalizedSourceRecord.from_dict(record.to_dict()),
            record,
        )

    def test_schema_v3_requires_and_enforces_corpus_scope(self) -> None:
        records = (
            self._record("personamem-v2/train/persona-1", "alpha source", "alpha"),
            self._record("personamem-v2/train/persona-2", "beta source", "beta"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "index"
            write_corpus_retrieval_index(
                root,
                records=records,
                embedder=FakeEmbedder(),
                source_manifest_hash="c" * 64,
                dataset_revision=REVISION,
            )
            index = RetrievalIndex.load(root)
            retriever = IndexRetriever(index, "dense", FakeEmbedder())

            with self.assertRaisesRegex(
                RetrievalValidationError, "requires corpus_id"
            ):
                retriever.retrieve(RetrievalRequest("alpha"))
            with self.assertRaisesRegex(
                RetrievalValidationError, "not present"
            ):
                retriever.retrieve(
                    RetrievalRequest("alpha", corpus_id="persona-missing")
                )
            result = retriever.retrieve(
                RetrievalRequest(
                    "alpha",
                    top_k=5,
                    corpus_id="personamem-v2/train/persona-1",
                )
            )

        self.assertEqual(result.trace["corpus_id"], "personamem-v2/train/persona-1")
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(
            result.hits[0].corpus_id, "personamem-v2/train/persona-1"
        )
        self.assertEqual(result.hits[0].slug, "alpha source")

    @staticmethod
    def _record(corpus_id: str, source_id: str, text: str) -> NormalizedSourceRecord:
        return NormalizedSourceRecord(
            corpus_id=corpus_id,
            source_id=source_id,
            timestamp="2025-09-13T16:31:34Z",
            title=source_id,
            text=text,
            provenance={"fixture": True},
            sensitivity=SensitivityMetadata(False, "ordinary"),
            source_hash=text_sha256(text),
        )


if __name__ == "__main__":
    unittest.main()
