"""Build the PersonaMem-v2 train source pool v1 (data/personamem-v2-train-v1).

This expands the 30-persona v0 pilot pool (data/personamem-v2-train-v0) to at
least 120 personas, each with a full 32K chat history, for the next stage of
PARMBench construction. It reuses v0's 30 personas by copying their already
verified history files instead of re-downloading them, fetches additional
`train_text` metadata pages from the datasets-server API to discover new
eligible personas beyond the first 100 rows, downloads each new persona's
history from the pinned dataset revision, and normalizes every selected
persona with the same `PersonaMemV2Adapter` used by v0.

v0 is never modified. This script only reads from
data/personamem-v2-train-v0 and writes to data/personamem-v2-train-v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from parm_bench.corpus import NormalizedSourceRecord, text_sha256
from parm_bench.personamem import (
    PERSONAMEM_ADAPTER_VERSION,
    PERSONAMEM_DATASET_ID,
    PERSONAMEM_SPLIT,
    PersonaMemSourceRow,
    PersonaMemV2Adapter,
)


ROOT = Path(__file__).resolve().parents[1]
V0_ROOT = ROOT / "data" / "personamem-v2-train-v0"
SOURCE_ROOT = ROOT / "data" / "personamem-v2-train-v1"
METADATA_DIR = SOURCE_ROOT / "source" / "metadata"
HISTORY_ROOT = SOURCE_ROOT / "source" / "histories"
UPSTREAM_DIR = SOURCE_ROOT / "upstream"
RECORDS_PATH = SOURCE_ROOT / "records.jsonl"
CANDIDATE_SOURCES_PATH = SOURCE_ROOT / "candidate_sources.jsonl"
MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"

DATASET_REVISION = "b7b42b78917157afed063527a1c959e98f6109f2"
RETRIEVED_AT = "2026-07-28"
# Identifies this source-pool construction pass. The underlying adapter code
# (PERSONAMEM_ADAPTER_VERSION) is unchanged from v0; this tag distinguishes
# the expanded v1 selection from the v0 pilot selection in provenance output.
BUILD_TAG = "personamem_v2_train_text_v2"

PAGE_SIZE = 100
TARGET_PERSONAS = 120
RESERVE_BUFFER = 15
MAX_PAGES = 20
REQUEST_DELAY_SECONDS = 0.5
HISTORY_DELAY_SECONDS = 0.3
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"


def metadata_api_url(offset: int, length: int = PAGE_SIZE) -> str:
    dataset = PERSONAMEM_DATASET_ID.replace("/", "%2F")
    return (
        f"{DATASETS_SERVER_URL}?dataset={dataset}&config=benchmark"
        f"&split={PERSONAMEM_SPLIT}&offset={offset}&length={length}"
    )


def history_url(upstream_history_path: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{PERSONAMEM_DATASET_ID}/resolve/{DATASET_REVISION}/{upstream_history_path}"
    )


def is_eligible_row(row: PersonaMemSourceRow) -> bool:
    """Structural eligibility gate, identical to the v0 pilot filter.

    A row is eligible for the ordinary (non-audited) selection pool when its
    preference is self-authored, current, non-sensitive, and not a
    forget-request. This is a pure structural check; it does not judge
    whether the raw conversation actually supports the preference text (that
    is a separate evidence-support gate, out of scope for source selection).
    """
    return not (
        row.updated or row.sensitive_info or row.forget_requested or row.who != "self"
    )


def _snippet_text(messages: Iterable[Mapping[str, str]]) -> str:
    return "\n\n".join(f"{m['role'].title()}: {m['content']}" for m in messages)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _fetch_bytes(url: str) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=90)
            if response.status_code == 404:
                raise FileNotFoundError(url)
            response.raise_for_status()
            return response.content
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-personas", type=int, default=TARGET_PERSONAS)
    parser.add_argument("--reserve-buffer", type=int, default=RESERVE_BUFFER)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = parser.parse_args()

    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)

    v0_manifest = json.loads(
        (V0_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    )
    v0_history_by_persona: dict[int, dict[str, Any]] = {
        int(entry["persona_id"]): entry
        for entry in v0_manifest["artifacts"]["histories"]
    }

    # Personas this pool already holds. Re-running to widen the pool must leave
    # them byte-identical, and re-downloading a file the repository already
    # tracks is a way to lose that guarantee for no gain.
    existing_history_by_persona: dict[int, dict[str, Any]] = {}
    if MANIFEST_PATH.exists():
        existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for entry in existing_manifest.get("artifacts", {}).get("histories", []):
            path = SOURCE_ROOT / entry["path"]
            if path.exists() and _sha256_file(path) == entry["sha256"]:
                existing_history_by_persona[int(entry["persona_id"])] = entry

    # 1. Copy and verify the v0 rows 0-99 metadata page so v1 is
    #    self-contained (does not need to read from v0 at run time later).
    v0_metadata_path = (
        V0_ROOT / "source" / "metadata" / "train_text_rows_00000_00099.json"
    )
    expected_page0_sha = v0_manifest["selection"]["metadata_sha256"]
    page0_bytes = v0_metadata_path.read_bytes()
    if _sha256_bytes(page0_bytes) != expected_page0_sha:
        raise ValueError(
            "v0 rows 0-99 metadata file does not match its recorded manifest hash"
        )
    v1_page0_path = METADATA_DIR / "train_text_rows_00000_00099.json"
    v1_page0_path.write_bytes(page0_bytes)

    # 2. Copy and verify the pinned upstream docs the same way.
    upstream_files: list[dict[str, str]] = []
    for entry in v0_manifest["artifacts"]["upstream"]:
        src = V0_ROOT / entry["path"]
        data = src.read_bytes()
        if _sha256_bytes(data) != entry["sha256"]:
            raise ValueError(f"v0 upstream file hash mismatch: {entry['path']}")
        dst = SOURCE_ROOT / entry["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        upstream_files.append({"path": entry["path"], "sha256": entry["sha256"]})

    metadata_pages_meta: list[dict[str, Any]] = [
        {
            "path": str(v1_page0_path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
            "sha256": expected_page0_sha,
            "offset": 0,
            "length": 100,
            "reused_from_v0": True,
        }
    ]

    combined_rows: dict[int, Mapping[str, Any]] = {}
    payload0 = json.loads(page0_bytes)
    for item in payload0["rows"]:
        combined_rows[int(item["row_idx"])] = item["row"]

    discovered_order: "OrderedDict[int, int]" = OrderedDict()  # persona_id -> primary offset
    eligible_rows_by_persona: dict[int, list[int]] = {}
    parse_skips: list[dict[str, Any]] = []

    def _scan_offset(offset: int, raw_row: Mapping[str, Any]) -> None:
        source_row_id = f"{PERSONAMEM_SPLIT}:{offset}"
        try:
            row = PersonaMemSourceRow.from_mapping(raw_row, source_row_id=source_row_id)
        except Exception as exc:  # noqa: BLE001 - record and skip malformed rows
            parse_skips.append(
                {"source_row_id": source_row_id, "reason": f"parse error: {exc}"}
            )
            return
        if not is_eligible_row(row):
            return
        eligible_rows_by_persona.setdefault(row.persona_id, []).append(offset)
        if row.persona_id not in discovered_order:
            discovered_order[row.persona_id] = offset

    for offset in range(0, 100):
        if offset in combined_rows:
            _scan_offset(offset, combined_rows[offset])

    # 3. Fetch additional metadata pages beyond rows 0-99 until there are
    #    enough eligible personas discovered to hit the target plus a
    #    reserve buffer (spare candidates for skip-and-replace).
    next_offset = 100
    pages_fetched = 1
    max_offset_scanned = 99
    while (
        len(discovered_order) < args.target_personas + args.reserve_buffer
        and pages_fetched < args.max_pages
    ):
        raw = _fetch_bytes(metadata_api_url(next_offset, PAGE_SIZE))
        payload = json.loads(raw)
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            break  # exhausted the split
        last_offset = next_offset + len(rows) - 1
        page_path = METADATA_DIR / (
            f"train_text_rows_{next_offset:05d}_{last_offset:05d}.json"
        )
        page_path.write_bytes(raw)
        metadata_pages_meta.append(
            {
                "path": str(page_path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
                "sha256": _sha256_bytes(raw),
                "offset": next_offset,
                "length": len(rows),
                "reused_from_v0": False,
            }
        )
        for item in rows:
            offset = int(item["row_idx"])
            combined_rows[offset] = item["row"]
            _scan_offset(offset, item["row"])
        max_offset_scanned = max(max_offset_scanned, last_offset)
        pages_fetched += 1
        next_offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY_SECONDS)

    rows_scanned = max_offset_scanned + 1

    # 4. Sanity check: every v0-reused persona must already be part of the
    #    eligible pool discovered within rows 0-99 (it was, by construction,
    #    when v0 was built from the same pinned revision).
    v0_persona_ids = set(v0_history_by_persona)
    missing_v0 = v0_persona_ids - set(discovered_order)
    if missing_v0:
        raise ValueError(f"v0 personas missing from the eligible pool: {sorted(missing_v0)}")
    for persona_id in v0_persona_ids:
        if discovered_order[persona_id] >= 100:
            raise ValueError(
                f"v0 persona {persona_id} was not primary-eligible within rows 0-99"
            )

    discovered_ids = list(discovered_order.keys())
    if len(discovered_ids) < args.target_personas:
        raise ValueError(
            f"only discovered {len(discovered_ids)} eligible personas after "
            f"scanning {rows_scanned} rows across {pages_fetched} page(s); "
            "increase --max-pages"
        )

    selected_ids = discovered_ids[: args.target_personas]
    missing_v0_in_selection = v0_persona_ids - set(selected_ids)
    if missing_v0_in_selection:
        # Not expected given the sanity check above (v0 personas are always
        # among the earliest-discovered eligible rows), but guard defensively
        # so a future upstream change cannot silently drop a reused persona.
        for persona_id in missing_v0_in_selection:
            for idx in range(len(selected_ids) - 1, -1, -1):
                if selected_ids[idx] not in v0_persona_ids:
                    selected_ids[idx] = persona_id
                    break
    reserve_ids = [pid for pid in discovered_ids if pid not in set(selected_ids)]

    adapter = PersonaMemV2Adapter(DATASET_REVISION)

    def _attempt_persona(persona_id: int) -> dict[str, Any]:
        primary_offset = discovered_order[persona_id]
        source_row_id = f"{PERSONAMEM_SPLIT}:{primary_offset}"
        row = PersonaMemSourceRow.from_mapping(
            combined_rows[primary_offset], source_row_id=source_row_id
        )
        if persona_id in v0_history_by_persona:
            entry = v0_history_by_persona[persona_id]
            src = V0_ROOT / entry["path"]
            data = src.read_bytes()
            sha = _sha256_bytes(data)
            if sha != entry["sha256"]:
                raise ValueError("v0 history file hash mismatch on copy")
            filename = Path(entry["path"]).name
            upstream_path = entry["upstream_path"]
            reused = True
        elif persona_id in existing_history_by_persona:
            entry = existing_history_by_persona[persona_id]
            data = (SOURCE_ROOT / entry["path"]).read_bytes()
            sha = entry["sha256"]
            filename = Path(entry["path"]).name
            upstream_path = entry["upstream_path"]
            reused = False
        else:
            upstream_path = row.history_path
            filename = Path(upstream_path).name
            data = _fetch_bytes(history_url(upstream_path))
            sha = _sha256_bytes(data)
            reused = False
            time.sleep(HISTORY_DELAY_SECONDS)
        history_json = json.loads(data.decode("utf-8"))
        corpus_records = adapter.adapt(row, history_json, history_sha256=sha)
        gold = [r for r in corpus_records if r.provenance["gold_related_snippet"]]
        if len(gold) != 1:
            raise ValueError("expected exactly one gold record")
        return {
            "row": row,
            "records": corpus_records,
            "gold": gold[0],
            "history_bytes": data,
            "history_filename": filename,
            "history_upstream_path": upstream_path,
            "history_sha256": sha,
            "reused_from_v0": reused,
            "primary_offset": primary_offset,
        }

    pending = list(selected_ids)
    reserve_queue = list(reserve_ids)
    skips: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    final_ids: list[int] = []
    persona_results: dict[int, dict[str, Any]] = {}

    while pending:
        persona_id = pending.pop(0)
        try:
            result = _attempt_persona(persona_id)
        except Exception as exc:  # noqa: BLE001 - skip and replace on any failure
            skips.append(
                {
                    "persona_id": persona_id,
                    "source_row_id": f"{PERSONAMEM_SPLIT}:{discovered_order[persona_id]}",
                    "reason": str(exc),
                }
            )
            if reserve_queue:
                replacement_id = reserve_queue.pop(0)
                replacements.append(
                    {
                        "skipped_persona_id": persona_id,
                        "replacement_persona_id": replacement_id,
                    }
                )
                pending.append(replacement_id)
            continue
        persona_results[persona_id] = result
        final_ids.append(persona_id)

    # 5. Write history files and records for every persona that made it
    #    through selection.
    all_records: list[NormalizedSourceRecord] = []
    histories_manifest: list[dict[str, Any]] = []
    for persona_id in final_ids:
        result = persona_results[persona_id]
        dest = HISTORY_ROOT / result["history_filename"]
        dest.write_bytes(result["history_bytes"])
        histories_manifest.append(
            {
                "path": str(dest.relative_to(SOURCE_ROOT)).replace("\\", "/"),
                "sha256": result["history_sha256"],
                "persona_id": persona_id,
                "upstream_path": result["history_upstream_path"],
                "url": history_url(result["history_upstream_path"]),
                "reused_from_v0": result["reused_from_v0"],
            }
        )
        all_records.extend(result["records"])

    # 6. Write every eligible source row for every selected persona (not
    #    just the primary row used for records.jsonl) so later scenario
    #    construction can draw on multi-scenario personas.
    candidate_rows: list[dict[str, Any]] = []
    for persona_id in final_ids:
        result = persona_results[persona_id]
        primary_offset = result["primary_offset"]
        gold = result["gold"]
        for offset in sorted(eligible_rows_by_persona[persona_id]):
            source_row_id = f"{PERSONAMEM_SPLIT}:{offset}"
            row = PersonaMemSourceRow.from_mapping(
                combined_rows[offset], source_row_id=source_row_id
            )
            is_primary = offset == primary_offset
            snippet_sha = text_sha256(_snippet_text(row.related_conversation_snippet))
            candidate_rows.append(
                {
                    "source_row_id": source_row_id,
                    "row_offset": offset,
                    "persona_id": persona_id,
                    "corpus_id": row.corpus_id,
                    "is_primary": is_primary,
                    "history_path": f"histories/{result['history_filename']}",
                    "history_sha256": result["history_sha256"],
                    "related_snippet_sha256": snippet_sha,
                    "gold_source_id": gold.source_id if is_primary else None,
                    "gold_source_hash": gold.source_hash if is_primary else None,
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
    candidate_rows.sort(key=lambda r: (r["persona_id"], r["row_offset"]))

    _write_jsonl(RECORDS_PATH, [record.to_dict() for record in all_records])
    _write_jsonl(CANDIDATE_SOURCES_PATH, candidate_rows)

    counts_per_corpus = Counter(record.corpus_id for record in all_records)
    rows_per_persona: Counter[int] = Counter()
    for row_dict in candidate_rows:
        rows_per_persona[row_dict["persona_id"]] += 1
    rows_histogram = Counter(rows_per_persona.values())
    multi_row_personas = sum(1 for count in rows_per_persona.values() if count >= 2)

    manifest = {
        "schema_version": 1,
        "supersedes_note": (
            "Supersedes data/personamem-v2-train-v0 as the source pool for new "
            "PARMBench construction. v0 stays frozen and byte-identical for the "
            "legacy 30-persona slices that already reference it."
        ),
        "adapter": {
            "name": "personamem_v2",
            "version": PERSONAMEM_ADAPTER_VERSION,
            "build_tag": BUILD_TAG,
        },
        "dataset": {
            "id": PERSONAMEM_DATASET_ID,
            "revision": DATASET_REVISION,
            "split": PERSONAMEM_SPLIT,
            "retrieved_at": RETRIEVED_AT,
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "dataset_url": (
                f"https://huggingface.co/datasets/{PERSONAMEM_DATASET_ID}"
                f"/tree/{DATASET_REVISION}"
            ),
        },
        "selection": {
            "metadata_api_url_template": (
                "https://datasets-server.huggingface.co/rows?dataset="
                f"{PERSONAMEM_DATASET_ID.replace('/', '%2F')}&config=benchmark"
                "&split=train_text&offset={offset}&length=100"
            ),
            "metadata_pages": metadata_pages_meta,
            "rows_scanned": rows_scanned,
            "eligible_rows_found": sum(
                len(v) for v in eligible_rows_by_persona.values()
            ),
            "eligible_personas_found": len(discovered_order),
            "eligibility_filter": (
                "who == 'self' and not updated and not sensitive_info and "
                "not forget_requested (preference_type == 'ask_to_forget' or "
                "preference text starting with 'do not remember ')"
            ),
            "target_personas": args.target_personas,
            "reserve_buffer": args.reserve_buffer,
            "selected_personas": len(final_ids),
            "reused_from_v0_personas": sum(
                1 for pid in final_ids if persona_results[pid]["reused_from_v0"]
            ),
            "newly_downloaded_personas": sum(
                1 for pid in final_ids if not persona_results[pid]["reused_from_v0"]
            ),
            "persona_disjoint": True,
            "ordinary_pool_only": True,
            "parse_skips": parse_skips,
            "skips": skips,
            "replacements": replacements,
        },
        "artifacts": {
            "records": {
                "path": str(RECORDS_PATH.relative_to(SOURCE_ROOT)).replace(
                    "\\", "/"
                ),
                "sha256": _sha256_file(RECORDS_PATH),
                "count": len(all_records),
                "corpus_counts": dict(sorted(counts_per_corpus.items())),
            },
            "candidate_sources": {
                "path": str(
                    CANDIDATE_SOURCES_PATH.relative_to(SOURCE_ROOT)
                ).replace("\\", "/"),
                "sha256": _sha256_file(CANDIDATE_SOURCES_PATH),
                "count": len(candidate_rows),
                "rows_per_persona_histogram": {
                    str(k): v for k, v in sorted(rows_histogram.items())
                },
                "personas_with_multiple_rows": multi_row_personas,
            },
            "histories": histories_manifest,
            "upstream": upstream_files,
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
        f"wrote {len(all_records)} normalized records across {len(counts_per_corpus)} "
        f"persona corpora ({manifest['selection']['reused_from_v0_personas']} reused "
        f"from v0, {manifest['selection']['newly_downloaded_personas']} newly "
        "downloaded)"
    )
    print(
        f"wrote {len(candidate_rows)} candidate source rows; "
        f"{multi_row_personas} personas have 2+ eligible rows"
    )
    if skips:
        print(f"skipped {len(skips)} persona(s), replaced from reserve pool: {skips}")


if __name__ == "__main__":
    main()
