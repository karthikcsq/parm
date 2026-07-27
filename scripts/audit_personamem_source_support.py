from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from personamem_v0_specs import SPECS


MODEL = "gpt-5-mini"
RUBRIC_VERSION = "personamem_source_support_v1"
INSTRUCTIONS = """\
Judge whether an indexed conversation supports a proposed personal-memory
claim. Evaluate only the conversation text. Do not assume a hidden persona,
profile, benchmark label, or unstated fact.

Grades:
- explicit: the user directly states or clearly demonstrates the core claim.
- inferable: the core claim follows from the user's words with one small,
  ordinary inference, and no key relation or activity must be invented.
- unsupported: a key topic, relation, preference, frequency, or identity in
  the claim is absent, or the source could support many incompatible claims.

Assistant suggestions are not evidence of the user's preference unless the
user accepts or independently states them. Keep the rationale short and quote
at most two short pieces of source evidence.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/personamem-v2-train-v0/pilot_sources.jsonl"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "source-support-cache"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "source-support-audit.jsonl"
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()
    sources = {
        row["source_row_id"]: row
        for row in _read_jsonl(args.sources)
    }
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, spec in enumerate(SPECS, start=1):
        source_row_id = f"train_text:{spec['row_offset']}"
        source = sources[source_row_id]
        rendered_source = "\n\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in source["related_conversation_snippet"]
        )
        input_text = (
            f"Memory claim:\n{spec['memory_summary']}\n\n"
            f"Indexed source conversation:\n{rendered_source}"
        )
        request = {
            "rubric_version": RUBRIC_VERSION,
            "model": MODEL,
            "input": input_text,
        }
        request_hash = hashlib.sha256(
            json.dumps(
                request, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        cache_path = args.cache / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            result = cached["result"]
            resolved_model = cached["resolved_model"]
            response_id = cached["response_id"]
        else:
            response = client.responses.create(
                model=MODEL,
                instructions=INSTRUCTIONS,
                input=input_text,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "personamem_source_support",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "grade": {
                                    "type": "string",
                                    "enum": [
                                        "explicit",
                                        "inferable",
                                        "unsupported",
                                    ],
                                },
                                "rationale": {"type": "string"},
                                "evidence": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 2,
                                },
                            },
                            "required": [
                                "grade",
                                "rationale",
                                "evidence",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                store=False,
            )
            result = json.loads(response.output_text)
            resolved_model = response.model
            response_id = response.id
            cache_path.write_text(
                json.dumps(
                    {
                        **request,
                        "request_hash": request_hash,
                        "result": result,
                        "resolved_model": resolved_model,
                        "response_id": response_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        row = {
            "base_case_id": f"parm-personamem-{spec['slug']}",
            "source_row_id": source_row_id,
            "corpus_id": source["corpus_id"],
            "gold_source_id": source["gold_source_id"],
            "memory_claim": spec["memory_summary"],
            "rubric_version": RUBRIC_VERSION,
            "requested_model": MODEL,
            "resolved_model": resolved_model,
            "response_id": response_id,
            **result,
        }
        rows.append(row)
        print(
            f"[{position:02d}/{len(SPECS)}] {spec['slug']}: "
            f"{result['grade']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    counts = {
        grade: sum(row["grade"] == grade for row in rows)
        for grade in ("explicit", "inferable", "unsupported")
    }
    print(json.dumps(counts, sort_keys=True))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
