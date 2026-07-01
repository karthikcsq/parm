from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parm_bench.retrieval import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    RetrievalIndex,
)
from parm_bench.retrieval_export import write_retrieval_index


class RetrievalExportTests(unittest.TestCase):
    def test_export_writes_loadable_frozen_artifact(self) -> None:
        pages = [
            {"id": 9, "source_id": "amara", "slug": "note/a", "title": "A"}
        ]
        chunks = [
            {
                "id": 20,
                "page_id": 9,
                "chunk_index": 0,
                "chunk_text": "Memory",
                "embedding": [0.0] * EMBEDDING_DIMENSIONS,
                "model": f"llama-server:{EMBEDDING_MODEL}",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "index"
            manifest = write_retrieval_index(
                root,
                pages=pages,
                chunks=chunks,
                links=[],
                corpus_id="amara-life-v1",
                gbrain_version="test",
                chunker_version="test",
                configured_model=f"llama-server:{EMBEDDING_MODEL}",
            )
            loaded = RetrievalIndex.load(root)

        self.assertEqual(manifest["counts"]["vectors"], 1)
        self.assertEqual(loaded.pages[0].page_id, "amara:note/a")
        self.assertEqual(loaded.chunks[0].chunk_id, "amara:note/a:0")

    def test_export_rejects_unexpected_vector_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "vector width"):
                write_retrieval_index(
                    Path(tmp) / "index",
                    pages=[
                        {
                            "id": 1,
                            "source_id": "s",
                            "slug": "p",
                            "title": "P",
                        }
                    ],
                    chunks=[
                        {
                            "id": 1,
                            "page_id": 1,
                            "chunk_index": 0,
                            "chunk_text": "text",
                            "embedding": [0.0],
                            "model": EMBEDDING_MODEL,
                        }
                    ],
                    links=[],
                    corpus_id="fixture",
                    gbrain_version="test",
                    chunker_version="test",
                )


if __name__ == "__main__":
    unittest.main()
