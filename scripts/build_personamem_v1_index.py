from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from parm_bench.corpus import NormalizedSourceRecord
from parm_bench.corpus_index import write_corpus_retrieval_index
from parm_bench.retrieval import OpenAIEmbedder


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "personamem-v2-train-v1"
RECORDS_PATH = SOURCE_ROOT / "records.jsonl"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"
OUTPUT = ROOT / "data" / "retrieval-indexes" / "personamem-v2-train-v1"
DATASET_REVISION = "b7b42b78917157afed063527a1c959e98f6109f2"


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    records = [
        NormalizedSourceRecord.from_dict(json.loads(line))
        for line in RECORDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = write_corpus_retrieval_index(
        OUTPUT,
        records=records,
        embedder=OpenAIEmbedder(),
        source_manifest_hash=_sha256(SOURCE_MANIFEST_PATH),
        dataset_revision=DATASET_REVISION,
    )
    print(
        f"wrote schema-v{manifest['schema_version']} index with "
        f"{manifest['counts']['pages']} pages and "
        f"{manifest['counts']['sentences']} sentences"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
