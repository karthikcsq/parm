"""Build PARMBench scenarios directly from supported facts.

This is the simple construction path, `parmbench_construction_simple_v1`. It
replaces the selection-predicate chain for new generation: one cached model
call per supported fact reads the raw user-authored span, the plain claim, and
the triplet requirements, then writes the scenario the way a person would.
The call may decline a fact it cannot turn into a plausible decision. Task
families, relation types, predicates, and capability labels are not inputs and
cannot reject a scenario here; the old builder keeps them for frozen replay.

Each accepted core is wrapped by the same seeded, model-free envelope assembly
the v1 builder uses, checked by the same deterministic scenario rejections
(minus every predicate-specific guard, which does not run without a predicate
object), and written as a positive / cue-ablated / memory-included triplet
under the `parmbench_v1` validation profile.

Usage:

    $env:PYTHONPATH = 'src'
    & 'C:\\Users\\karth\\anaconda3\\python.exe' \\
      scripts\\build_parmbench_simple_v1.py \\
      --select data\\parmbench-v1-supply\\simple_v1_spec_examples.json
    & 'C:\\Users\\karth\\anaconda3\\python.exe' \\
      scripts\\build_parmbench_simple_v1.py \\
      --supply data\\parmbench-v1-supply\\gated_claims_pilot2.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parmbench_v1_envelopes import ENVELOPES  # noqa: E402

from build_parmbench_v1_benchmark import (  # noqa: E402
    ANSWER_CONTRACTS,
    CONSTRUCTION_MODEL,
    RECORDS_PATH,
    ROOT,
    SOURCE_ROOT_RELATIVE,
    TOKENIZER,
    assemble_observation,
    atomic_write_json,
    build_case_rows,
    case_sensitive_terms,
    choose_distractors,
    hash_request,
    load_env,
    load_jsonl,
    normalise_core,
    scenario_rejection,
    sha256_text,
    write_jsonl,
)
from parm_bench.service_tier import service_tier_kwargs  # noqa: E402


SIMPLE_PROMPT_VERSION = "parmbench_construction_simple_v1"
SIMPLE_BUILDER_VERSION = "parmbench_simple_builder_v1"
SIMPLE_CACHE = ROOT / "data" / "construction-caches" / "parmbench-simple-v1"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark_parmbench_pilot_simple_v1"
DEFAULT_SUPPLY = ROOT / "data" / "parmbench-v1-supply" / "gated_claims_pilot2.jsonl"
BASE_ID_PREFIX = "parmbench-s1"
OBSERVATION_KINDS = ("tool_result", "assistant_output")
DECOY_RANGE = (2, 4)
DISTRACTOR_COUNT = 4

SIMPLE_INSTRUCTIONS = """\
You write one scenario for a personal-memory benchmark, starting from one
supported personal fact.

You are given the fact and the exact words the person wrote. Work the way a
person would set an exam question:

1. Name one plausible request in which this fact could change which option the
   person should pick. Plausible means a normal thing somebody might ask an
   assistant, even if it only comes up now and then. Finding a souvenir on a
   trip qualifies; the situation does not need to be an everyday one.
2. Write an ordinary set of options for that request.
3. Give one option, the winner, a default advantage that has nothing to do
   with the fact: popularity, a strong recommendation, price, convenience,
   condition, or availability.
4. Give a different option, the target, one concrete property that matches
   the fact. That property arrives as a late detail, the cue.
5. Without the fact, the winner is the sensible answer.
6. With the fact and the cue, the target is the sensible answer.
7. In the control, the cue is replaced by a neutral detail, and the winner is
   the sensible answer again, even for somebody who knows the fact.

If, after a serious attempt, no plausible decision turns on this fact, set
decline to true, explain why in decline_reason, and leave every other field
an empty string or empty list. Declining is better than forcing a scenario.

When you do build, return these fields.

- task: one or two sentences stating the ordinary request. It must read as a
  routine ask, must not mention the person's history, remembering, or
  personalisation, and must not name any option.
- item_noun: the singular noun for what the options are, such as dessert,
  book, listing, souvenir.
- winner_label, target_label, and one label per decoy: short distinct names of
  two to four words. Every label reads like a real name or title the document
  would print: a dish name from a menu, a plausible book title, a shop, event,
  or listing name. A category description such as "Accessible Overview" is not
  a label. No shared numbering scheme, no shared prefix, and no label may be a
  substring of another. Do not use the words option, choice, pick,
  recommended, or best inside a label. The target's label and its sentences
  before the cue must leave a reader unable to guess that the cue's property
  holds; when the kind of item is itself what matches the fact, keep even the
  era or format of the item unstated until the cue.
- winner_body: two or three sentences that make the winner the sensible pick
  through its default advantage alone. Never reference the personal fact.
- target_body: two or three sentences that read as unremarkable on their own.
  They must not reveal the property that matches the fact and must not claim
  the target shares the winner's advantage.
- cue_clause: exactly one sentence, written to sit at the end of target_body.
  It states the concrete property of the target that matches the fact. State
  the property and stop: a phrase that explains who the property suits, such
  as "ideal for collectors", tells every reader the answer. A reader who does
  not know the fact takes it as an ordinary detail.
- neutral_clause: exactly one sentence in the same register that replaces
  cue_clause in the control, between about three-quarters and one-and-a-half
  times the cue sentence's length. An unrelated ordinary detail of the same
  option. It must not deny or invert the cue's property, must not create a new
  match with the fact anywhere, and must leave the target neither more nor
  less attractive.
- decoys: two to four further ordinary options. Each falls short of the
  winner's default advantage in a way its text states plainly, and none may
  mention the fact's subject matter, its absence, or a substitute for it.
- memory_text: one short sentence stating the durable personal fact plainly,
  starting with "The user". State the standing fact in general terms; do not
  retell the specific episode the person described, and add nothing they did
  not say.
- why_ordinary_wins: one sentence saying why a reader without the fact picks
  the winner.
- why_memory_plus_cue_prefers_b: one sentence saying why a reader who knows
  the fact switches to the target once the cue is visible.
- why_control_removes_advantage: one sentence saying why the neutral detail
  sends even a fact-aware reader back to the winner.

Rules:
- Do not invent possessions, relationships, medical needs, locations, or plans
  the person never stated. The scenario must need nothing beyond the fact.
- The fact must not be stated or paraphrased anywhere in the document text you
  return; only memory_text states it.
- Do not reuse distinctive multi-word phrases from the person's own words
  anywhere in the document. Carry the matching property in your own wording,
  as a stranger describing the item would put it.
- Do not use the words memory, preference, persona, or profile.
- Never write that an option is the strongest, the consensus, the default, the
  obvious one, or narrower than another, and never write that nobody
  challenged it. Do not mention ranking or how many options exist.
- Plain prose, no scores or star ratings unless a price or count is the
  winner's stated advantage.
"""

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "decline": {"type": "boolean"},
        "decline_reason": {"type": "string"},
        "task": {"type": "string"},
        "item_noun": {"type": "string"},
        "winner_label": {"type": "string"},
        "winner_body": {"type": "string"},
        "target_label": {"type": "string"},
        "target_body": {"type": "string"},
        "cue_clause": {"type": "string"},
        "neutral_clause": {"type": "string"},
        "memory_text": {"type": "string"},
        "why_ordinary_wins": {"type": "string"},
        "why_memory_plus_cue_prefers_b": {"type": "string"},
        "why_control_removes_advantage": {"type": "string"},
        "decoys": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["label", "body"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "decline",
        "decline_reason",
        "task",
        "item_noun",
        "winner_label",
        "winner_body",
        "target_label",
        "target_body",
        "cue_clause",
        "neutral_clause",
        "memory_text",
        "why_ordinary_wins",
        "why_memory_plus_cue_prefers_b",
        "why_control_removes_advantage",
        "decoys",
    ],
    "additionalProperties": False,
}


def simple_base_case_id(row: Mapping[str, Any]) -> str:
    return (
        f"{BASE_ID_PREFIX}-p{row['persona_id']}-"
        f"{str(row['source_row_id']).split(':')[-1]}"
    )


def render_simple_request(
    row: Mapping[str, Any],
    *,
    request_hint: str = "",
    variation: int = 0,
) -> str:
    """The generation input: the fact, the raw span, and nothing structural.

    The hint carries an agreed request shape for a specification example; the
    variation number distinguishes parallel candidates for one fact. Both are
    part of the cached request input, so they key the cache naturally.
    """

    request = (
        f"Personal fact: {row['draft']['claim']}\n"
        f"The person's own words: {row['draft']['evidence_span']}\n"
    )
    if request_hint.strip():
        request += f"Suggested request: {request_hint.strip()}\n"
    if variation:
        request += f"Candidate variation: {variation}\n"
    return request


def strip_predicate(row: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the supply row with every mapper artifact removed.

    The deterministic rejection set is shared with the v1 builder, and its
    predicate-specific guards key off `row['predicate']`. The simple path
    treats those fields as analysis metadata, never as construction inputs.
    """

    stripped = dict(row)
    stripped.pop("predicate", None)
    stripped.pop("selection_predicate", None)
    return stripped


def supply_skip_reason(row: Mapping[str, Any]) -> str | None:
    """Reasons a supply row is not usable for any construction path."""

    if row.get("outcome") != "passed":
        return f"supply_outcome_{row.get('outcome', 'missing')}"
    rejections = row.get("grade", {}).get("deterministic_rejections", [])
    if rejections:
        return "evidence_deterministic_rejection: " + ", ".join(
            str(item) for item in rejections
        )
    return None


_SCRUB_ARTICLE_ARTIFACT = re.compile(r"\b([Aa]n?) (this|another)\b")
_SENTENCE_OPENING = re.compile(r"(^|[.!?]\s+)([a-z])")


def tidy_scrub_artifacts(core: dict[str, Any]) -> dict[str, Any]:
    """Repair the article breakage the shared label scrubber leaves behind.

    `normalise_core` rewrites a body's mention of its own label to "this
    <item>", so a body that opened "A Hand-Painted Tin Truck in..." becomes
    "A this collectible in...". The determiner pair is never intentional
    English, so collapsing it to the scrubbed phrase is safe.
    """

    def tidy(text: str) -> str:
        repaired = _SCRUB_ARTICLE_ARTIFACT.sub(r"\2", str(text))
        if repaired == str(text):
            return repaired
        return _SENTENCE_OPENING.sub(
            lambda m: m.group(1) + m.group(2).upper(), repaired
        )

    for key in ("winner_body", "target_body", "cue_clause", "neutral_clause"):
        core[key] = tidy(core.get(key, ""))
    for decoy in core.get("decoys", []):
        decoy["body"] = tidy(decoy.get("body", ""))
    return core


def core_declines(core: Mapping[str, Any]) -> bool:
    return bool(core.get("decline"))


def core_shape_rejection(core: Mapping[str, Any]) -> str | None:
    """Cheap shape assertions on a freshly generated core."""

    for field in (
        "task",
        "item_noun",
        "winner_label",
        "winner_body",
        "target_label",
        "target_body",
        "cue_clause",
        "neutral_clause",
        "memory_text",
    ):
        if not str(core.get(field, "")).strip():
            return f"empty_core_field: {field}"
    decoys = core.get("decoys", [])
    if not DECOY_RANGE[0] <= len(decoys) <= DECOY_RANGE[1]:
        return f"decoy_count_out_of_range: {len(decoys)}"
    if not str(core["memory_text"]).strip().startswith("The user"):
        return "memory_text_missing_user_subject"
    return None


class SimpleCachedGenerator:
    """Cached one-call scenario generator for the simple path."""

    def __init__(self, cache_dir: Path, *, offline: bool = False) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.live_calls = 0
        self._lock = threading.Lock()
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _create(self, **kwargs: Any) -> Any:
        import time

        for attempt in range(6):
            try:
                return self.client.responses.create(
                    **kwargs, **service_tier_kwargs()
                )
            except Exception:  # noqa: BLE001 - retried, then re-raised
                if attempt == 5:
                    raise
                time.sleep(
                    min(60.0, 2.0 * 2**attempt) * (0.5 + random.random())
                )
        raise AssertionError("unreachable generation retry state")

    def build(
        self,
        row: Mapping[str, Any],
        *,
        request_hint: str = "",
        variation: int = 0,
    ) -> tuple[dict[str, Any], str]:
        input_text = render_simple_request(
            row, request_hint=request_hint, variation=variation
        )
        request = {
            "prompt_version": SIMPLE_PROMPT_VERSION,
            "model": CONSTRUCTION_MODEL,
            "input": input_text,
        }
        request_hash = hash_request(request)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["result"], request_hash
        if self.offline:
            raise RuntimeError(f"missing generation cache entry: {request_hash}")
        response = self._create(
            model=CONSTRUCTION_MODEL,
            instructions=SIMPLE_INSTRUCTIONS,
            input=input_text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parmbench_simple_scenario",
                    "strict": True,
                    "schema": SIMPLE_SCHEMA,
                }
            },
            store=False,
        )
        result = json.loads(response.output_text)
        with self._lock:
            self.live_calls += 1
        atomic_write_json(
            cache_path,
            {
                **request,
                "request_hash": request_hash,
                "result": result,
                "resolved_model": response.model,
                "response_id": response.id,
            },
        )
        return result, request_hash


def scenario_seed(base_case_id: str, variation: int) -> int:
    digest = hashlib.sha256(
        f"{SIMPLE_PROMPT_VERSION}:{base_case_id}:{variation}".encode()
    ).hexdigest()
    return int(digest[:16], 16)


def load_selection(path: Path) -> list[dict[str, Any]]:
    """Read a selection file naming rows, hints, and variations to build."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("examples", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty list of examples")
    selection = []
    for entry in entries:
        selection.append(
            {
                "supply": str(entry["supply"]),
                "persona_id": int(entry["persona_id"]),
                "source_row_id": str(entry["source_row_id"]),
                "request_hint": str(entry.get("request_hint", "")),
                "variation": int(entry.get("variation", 0)),
            }
        )
    return selection


def resolve_selection(
    selection: Sequence[Mapping[str, Any]], supply_root: Path
) -> list[dict[str, Any]]:
    """Turn selection entries into build jobs with their supply rows."""

    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    jobs: list[dict[str, Any]] = []
    for entry in selection:
        supply_path = supply_root / entry["supply"]
        key = str(supply_path)
        if key not in rows_by_file:
            rows_by_file[key] = load_jsonl(supply_path)
        matches = [
            row
            for row in rows_by_file[key]
            if row.get("persona_id") == entry["persona_id"]
            and str(row.get("source_row_id")) == entry["source_row_id"]
        ]
        if not matches:
            raise ValueError(
                f"no supply row for persona {entry['persona_id']} "
                f"{entry['source_row_id']} in {supply_path}"
            )
        jobs.append(
            {
                "row": matches[0],
                "request_hint": entry["request_hint"],
                "variation": entry["variation"],
            }
        )
    return jobs


def jobs_from_supply(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """One build job per distinct supported fact across the supply files."""

    jobs: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()
    for path in paths:
        for row in load_jsonl(path):
            key = (row.get("persona_id"), str(row.get("source_row_id")))
            if key in seen:
                continue
            seen.add(key)
            jobs.append({"row": row, "request_hint": "", "variation": 0})
    return jobs


def build_prompt(core: Mapping[str, Any], rng: random.Random) -> str:
    contract = rng.choice(ANSWER_CONTRACTS)
    return (
        f"{str(core['task']).strip()} "
        + contract.format(item=str(core["item_noun"]).strip())
    )


def review_table(records: Sequence[Mapping[str, Any]]) -> str:
    """A compact manual-inspection table over the built scenarios."""

    lines = [
        "# Simple-path scenarios for manual review",
        "",
        "One row per built scenario. Verdicts belong in the adjudication",
        "file, not here.",
        "",
        "| base_case_id | fact | request | A (winner) | B (target) | cue | control |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def cell(text: Any) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ").strip()

    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(record["base_case_id"]),
                    cell(record["claim"]),
                    cell(record["task"]),
                    cell(record["choices"]["output_only"]),
                    cell(record["choices"]["memory_conditioned"]),
                    cell(record["control_replacement"]["old"]),
                    cell(record["control_replacement"]["new"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supply",
        action="append",
        default=None,
        help="gated-claims JSONL to build from; repeatable",
    )
    parser.add_argument(
        "--select",
        default=None,
        help=(
            "JSON file naming specific rows to build, with optional "
            "request hints and candidate variations"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    load_env(ROOT / ".env")
    encoding = tiktoken.get_encoding(TOKENIZER)
    output_root = Path(args.output_dir)
    context_root = output_root / "contexts"

    if args.select:
        selection = load_selection(Path(args.select))
        jobs = resolve_selection(
            selection, ROOT / "data" / "parmbench-v1-supply"
        )
    else:
        supply_paths = [
            Path(item) for item in (args.supply or [str(DEFAULT_SUPPLY)])
        ]
        jobs = jobs_from_supply(supply_paths)
    if args.limit:
        jobs = jobs[: args.limit]

    records_by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_jsonl(RECORDS_PATH):
        records_by_corpus[record["corpus_id"]].append(record)

    generator = SimpleCachedGenerator(SIMPLE_CACHE, offline=args.offline)

    def warm(job: Mapping[str, Any]) -> None:
        if supply_skip_reason(job["row"]):
            return
        try:
            generator.build(
                job["row"],
                request_hint=job["request_hint"],
                variation=job["variation"],
            )
        except Exception as exc:  # noqa: BLE001 - reported during assembly
            print(
                f"generation failed for {simple_base_case_id(job['row'])}: {exc}"
            )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(warm, jobs))

    context_root.mkdir(parents=True, exist_ok=True)
    for stale in context_root.glob("*.md"):
        stale.unlink()

    cases: list[dict[str, Any]] = []
    construction_records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    corpora: dict[str, dict[str, Any]] = {}

    for job in jobs:
        row = job["row"]
        base_case_id = simple_base_case_id(row)
        skip = supply_skip_reason(row)
        if skip:
            dropped.append({"base_case_id": base_case_id, "reason": skip})
            continue
        try:
            raw_core, request_hash = generator.build(
                row,
                request_hint=job["request_hint"],
                variation=job["variation"],
            )
        except Exception as exc:  # noqa: BLE001
            dropped.append(
                {
                    "base_case_id": base_case_id,
                    "reason": f"generation_error: {exc}",
                }
            )
            continue
        if core_declines(raw_core):
            dropped.append(
                {
                    "base_case_id": base_case_id,
                    "reason": "declined",
                    "decline_reason": str(raw_core.get("decline_reason", "")),
                }
            )
            continue
        shape_reason = core_shape_rejection(raw_core)
        if shape_reason:
            dropped.append(
                {"base_case_id": base_case_id, "reason": shape_reason}
            )
            continue
        core = tidy_scrub_artifacts(normalise_core(raw_core))

        seed = scenario_seed(base_case_id, job["variation"])
        prompt_rng = random.Random(seed + 11)
        prompt = build_prompt(core, prompt_rng)
        stripped_row = strip_predicate(row)

        text = ""
        metrics: dict[str, Any] = {}
        reason: str | None = "not_attempted"
        envelope_name = ""
        attempt = 0
        for attempt in range(6):
            rng = random.Random(seed + attempt * 7919)
            envelope = rng.choice(ENVELOPES)
            envelope_name = envelope.name
            cue_fraction = rng.uniform(0.55, 0.92)
            winner_fraction = rng.uniform(0.05, 0.45)
            token_target = rng.randrange(6_800, 11_500)
            text, metrics = assemble_observation(
                envelope,
                core,
                rng=rng,
                encoding=encoding,
                cue_fraction=cue_fraction,
                winner_fraction=winner_fraction,
                token_target=token_target,
            )
            reason = scenario_rejection(text, core, prompt, stripped_row)
            if reason is None:
                break
        if reason is not None:
            dropped.append({"base_case_id": base_case_id, "reason": reason})
            continue

        corpus_id = row["corpus_id"]
        persona_id = row["persona_id"]
        distractors = choose_distractors(
            records_by_corpus.get(corpus_id, []),
            row["gold_source_id"],
            str(row["draft"]["claim"]),
            random.Random(seed + 104_729),
            abstention_pressure=True,
            count=DISTRACTOR_COUNT,
        )
        if len(distractors) < 3:
            dropped.append(
                {"base_case_id": base_case_id, "reason": "too_few_distractors"}
            )
            continue

        content_path = f"contexts/{base_case_id}.md"
        (output_root / content_path).write_text(
            text, encoding="utf-8", newline="\n"
        )
        corpora[corpus_id] = {
            "corpus_id": corpus_id,
            "source_root": SOURCE_ROOT_RELATIVE,
            "source_id_prefix": f"notes/persona-{persona_id}/",
        }
        gold_source = {
            "source_id": row["gold_source_id"],
            "path": row["history_path"],
            "perturbations": [],
            "sha256": row["history_sha256"],
            "evidence_span": {"text": row["draft"]["evidence_span"]},
        }
        kind_rng = random.Random(seed + 13)
        observation_kind = kind_rng.choice(OBSERVATION_KINDS)
        provenance = {
            "approved": True,
            "builder_version": SIMPLE_BUILDER_VERSION,
            "construction_model": CONSTRUCTION_MODEL,
            "construction_prompt_version": SIMPLE_PROMPT_VERSION,
            "construction_request_hash": request_hash,
            "envelope_style": envelope_name,
            "evaluation_split": "calibration",
            "evidence_grade": row["grade"]["grade"],
            "evidence_gate_rubric": row["grade"]["rubric_version"],
            "fact_kind": row["draft"]["fact_kind"],
            "persona_id": persona_id,
            "request_hint": job["request_hint"],
            "seeds": {
                "scenario_seed": seed,
                "assembly_attempt": attempt,
                "variation": job["variation"],
            },
            "source_dataset": "bowen-upenn/PersonaMem-v2",
            "source_revision": "b7b42b78917157afed063527a1c959e98f6109f2",
            "source_row_id": row["source_row_id"],
            "source_split": "train_text",
            "hashes": {
                "claim_draft_hash": row["draft_hash"],
                "context_sha256": sha256_text(text),
                "gold_source_hash": row["gold_source_hash"],
                "history_sha256": row["history_sha256"],
            },
        }
        spec = {"base_case_id": base_case_id, "observation_kind": observation_kind}
        cases.extend(
            build_case_rows(
                spec=spec,
                row=row,
                core=core,
                prompt=prompt,
                corpus_id=corpus_id,
                content_path=content_path,
                gold_source=gold_source,
                distractors=distractors,
                provenance=provenance,
            )
        )
        construction_records.append(
            {
                "base_case_id": base_case_id,
                "persona_id": persona_id,
                "corpus_id": corpus_id,
                "source_row_id": row["source_row_id"],
                "claim": row["draft"]["claim"],
                "evidence_span": row["draft"]["evidence_span"],
                "gold_source_id": row["gold_source_id"],
                "task": str(core["task"]).strip(),
                "prompt": prompt,
                "request_hint": job["request_hint"],
                "variation": job["variation"],
                "choices": {
                    "output_only": str(core["winner_label"]),
                    "memory_conditioned": str(core["target_label"]),
                },
                "memory_text": str(core["memory_text"]).strip(),
                "control_replacement": {
                    "old": str(core["cue_clause"]).strip(),
                    "new": str(core["neutral_clause"]).strip(),
                },
                "why": {
                    "ordinary_wins": str(
                        core.get("why_ordinary_wins", "")
                    ).strip(),
                    "memory_plus_cue_prefers_b": str(
                        core.get("why_memory_plus_cue_prefers_b", "")
                    ).strip(),
                    "control_removes_advantage": str(
                        core.get("why_control_removes_advantage", "")
                    ).strip(),
                },
                "sensitive_terms": case_sensitive_terms(row),
                "envelope_style": envelope_name,
                "observation_kind": observation_kind,
                "metrics": metrics,
                "seeds": {
                    "scenario_seed": seed,
                    "assembly_attempt": attempt,
                },
                "model_calls": {
                    "claim_draft_hash": row["draft_hash"],
                    "construction_request_hash": request_hash,
                    "construction_prompt_version": SIMPLE_PROMPT_VERSION,
                    "construction_model": CONSTRUCTION_MODEL,
                },
            }
        )

    write_jsonl(output_root / "cases.jsonl", cases)
    write_jsonl(
        output_root / "construction_records.jsonl", construction_records
    )
    manifest = {
        "schema_version": 1,
        "validation_profile": "parmbench_v1",
        "builder_version": SIMPLE_BUILDER_VERSION,
        "construction_records": {"path": "construction_records.jsonl"},
        "corpora": [corpora[key] for key in sorted(corpora)],
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if dropped:
        write_jsonl(output_root / "dropped_scenarios.jsonl", dropped)
    elif (output_root / "dropped_scenarios.jsonl").exists():
        (output_root / "dropped_scenarios.jsonl").unlink()
    if construction_records:
        (output_root / "review_table.md").write_text(
            review_table(construction_records), encoding="utf-8", newline="\n"
        )

    summary = {
        "scenarios": len(construction_records),
        "cases": len(cases),
        "personas": len(
            {record["persona_id"] for record in construction_records}
        ),
        "dropped": len(dropped),
        "dropped_reasons": dict(
            sorted(
                {
                    reason: sum(
                        1
                        for item in dropped
                        if item["reason"].split(":")[0] == reason
                    )
                    for reason in {
                        item["reason"].split(":")[0] for item in dropped
                    }
                }.items()
            )
        ),
        "declined": sum(
            1 for item in dropped if item["reason"] == "declined"
        ),
        "live_generation_calls": generator.live_calls,
        "construction_prompt_version": SIMPLE_PROMPT_VERSION,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
