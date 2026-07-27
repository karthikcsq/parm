from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from parm_bench.personamem import (
    PERSONAMEM_ADAPTER_VERSION,
    PERSONAMEM_DATASET_ID,
    PERSONAMEM_SPLIT,
    PersonaMemSourceRow,
    PersonaMemV2Adapter,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "personamem-v2-train-v0"
METADATA_PATH = (
    SOURCE_ROOT
    / "source"
    / "metadata"
    / "train_text_rows_00000_00099.json"
)
HISTORY_ROOT = SOURCE_ROOT / "source" / "histories"
RECORDS_PATH = SOURCE_ROOT / "records.jsonl"
PILOT_SOURCES_PATH = SOURCE_ROOT / "pilot_sources.jsonl"
MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"
DATASET_REVISION = "b7b42b78917157afed063527a1c959e98f6109f2"
RETRIEVED_AT = "2026-07-27"
SELECTED_OFFSETS = (
    7,
    9,
    10,
    14,
    18,
    19,
    21,
    22,
    24,
    29,
    30,
    36,
    38,
    41,
    44,
    45,
    49,
    50,
    51,
    54,
    55,
    60,
    61,
    63,
    66,
    68,
    72,
    73,
    75,
    85,
)


def main() -> None:
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("expected the bounded 100-row datasets-server response")
    adapter = PersonaMemV2Adapter(DATASET_REVISION)
    records = []
    selected_sources: list[dict[str, Any]] = []
    persona_ids: set[int] = set()
    history_files: list[dict[str, Any]] = []
    for offset in SELECTED_OFFSETS:
        source_row_id = f"{PERSONAMEM_SPLIT}:{offset}"
        item = rows[offset]
        if int(item.get("row_idx", -1)) != offset:
            raise ValueError(f"{source_row_id}: datasets-server row_idx mismatch")
        row = PersonaMemSourceRow.from_mapping(
            item["row"],
            source_row_id=source_row_id,
        )
        if row.persona_id in persona_ids:
            raise ValueError(f"{source_row_id}: duplicate pilot persona")
        if row.updated or row.sensitive_info or row.forget_requested or row.who != "self":
            raise ValueError(
                f"{source_row_id}: ordinary pilot rows must be current, "
                "non-sensitive, remembered, and self-authored"
            )
        persona_ids.add(row.persona_id)
        history_path = HISTORY_ROOT / Path(row.history_path).name
        history_sha256 = _sha256(history_path)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        corpus_records = adapter.adapt(
            row,
            history,
            history_sha256=history_sha256,
        )
        gold = [
            record
            for record in corpus_records
            if record.provenance["gold_related_snippet"]
        ]
        if len(gold) != 1:
            raise ValueError(f"{source_row_id}: expected exactly one gold record")
        records.extend(corpus_records)
        selected_sources.append(
            {
                "source_row_id": source_row_id,
                "row_offset": offset,
                "persona_id": row.persona_id,
                "corpus_id": row.corpus_id,
                "history_path": f"histories/{history_path.name}",
                "history_sha256": history_sha256,
                "gold_source_id": gold[0].source_id,
                "gold_source_hash": gold[0].source_hash,
                "related_snippet_sha256": gold[0].provenance[
                    "related_snippet_sha256"
                ],
                "preference": row.preference,
                "updated": row.updated,
                "previous_preference": row.previous_preference,
                "who": row.who,
                "sensitive_info": row.sensitive_info,
                "preference_type": row.preference_type,
                "topic_preference": row.topic_preference,
                "user_query": row.user_query,
                "correct_answer": row.correct_answer,
                "incorrect_answers": list(row.incorrect_answers),
                "related_conversation_snippet": list(
                    row.related_conversation_snippet
                ),
            }
        )
        history_files.append(
            {
                "path": f"source/histories/{history_path.name}",
                "sha256": history_sha256,
                "persona_id": row.persona_id,
                "upstream_path": row.history_path,
                "url": (
                    "https://huggingface.co/datasets/"
                    f"{PERSONAMEM_DATASET_ID}/resolve/{DATASET_REVISION}/"
                    f"{row.history_path}"
                ),
            }
        )

    _write_jsonl(RECORDS_PATH, [record.to_dict() for record in records])
    _write_jsonl(PILOT_SOURCES_PATH, selected_sources)
    counts = Counter(record.corpus_id for record in records)
    manifest = {
        "schema_version": 1,
        "adapter": {
            "name": "personamem_v2",
            "version": PERSONAMEM_ADAPTER_VERSION,
        },
        "dataset": {
            "id": PERSONAMEM_DATASET_ID,
            "revision": DATASET_REVISION,
            "split": PERSONAMEM_SPLIT,
            "retrieved_at": RETRIEVED_AT,
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "dataset_url": (
                "https://huggingface.co/datasets/"
                f"{PERSONAMEM_DATASET_ID}/tree/{DATASET_REVISION}"
            ),
        },
        "selection": {
            "metadata_api_url": (
                "https://datasets-server.huggingface.co/rows"
                f"?dataset={PERSONAMEM_DATASET_ID.replace('/', '%2F')}"
                "&config=benchmark&split=train_text&offset=0&length=100"
            ),
            "metadata_path": str(METADATA_PATH.relative_to(SOURCE_ROOT)).replace(
                "\\", "/"
            ),
            "metadata_sha256": _sha256(METADATA_PATH),
            "bounded_rows": 100,
            "selected_offsets": list(SELECTED_OFFSETS),
            "selected_rows": len(selected_sources),
            "persona_disjoint": True,
            "ordinary_pool_only": True,
        },
        "artifacts": {
            "records": {
                "path": str(RECORDS_PATH.relative_to(SOURCE_ROOT)).replace(
                    "\\", "/"
                ),
                "sha256": _sha256(RECORDS_PATH),
                "count": len(records),
                "corpus_counts": dict(sorted(counts.items())),
            },
            "pilot_sources": {
                "path": str(PILOT_SOURCES_PATH.relative_to(SOURCE_ROOT)).replace(
                    "\\", "/"
                ),
                "sha256": _sha256(PILOT_SOURCES_PATH),
                "count": len(selected_sources),
            },
            "histories": history_files,
            "upstream": [
                _tracked_file("upstream/README.md"),
                _tracked_file("upstream/column_descriptions.md"),
                _tracked_file("upstream/CC-BY-4.0.txt"),
            ],
        },
        "normalization": {
            "system_messages_indexed": False,
            "gold_source": "related_conversation_snippet",
            "gold_source_count_per_corpus": 1,
            "history_page_max_messages": 4,
            "hidden_annotations_indexed": False,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(records)} normalized records across "
        f"{len(counts)} persona corpora"
    )


def _tracked_file(relative_path: str) -> dict[str, str]:
    path = SOURCE_ROOT / relative_path
    return {
        "path": relative_path,
        "sha256": _sha256(path),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
