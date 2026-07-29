"""Draft and gate personal-fact claims for PARMBench v1 construction.

Phase A of the construction contract. For every candidate source row in
`data/personamem-v2-train-v1/candidate_sources.jsonl` this script:

1. asks a cached `gpt-5-mini` drafter to write one conservative personal-fact
   claim from the raw user-authored text of the row's conversation snippet,
   returning the exact verbatim user span that entails it;
2. verifies the span deterministically against the persona's normalized
   records and the tracked history file; and
3. runs `parm_bench.evidence_gate.grade_claim` on the claim and the snippet
   turns, keeping only explicit or inferable grades.

The upstream `preference` label is passed to the drafter only as a hint of
where to look. The drafter is told it is unverified, must not copy its
wording, and may return "no supportable claim".

Two drafting rubrics are versioned here.

`parmbench_claim_draft_v1` asks only for a conservative, auditable fact. It
built the first 65-scenario batch and is kept so that batch replays.

`parmbench_claim_draft_v2` keeps every soundness rule of v1 and adds a
decision-relevance requirement: the fact has to be a durable preference,
constraint, exclusion, requirement, schedule or commitment, or relationship
that could steer a choice among ordinary task options. The drafter also names
the `decision_lever`, the kind of choice the fact could steer. Biographical
trivia with no lever is declined. v1's supply lost roughly half its scenarios
to ceilings that no rewrite could fix, because an auditable fact such as an old
fracture decides nothing.

The `decision_lever` is construction metadata. It may inform the construction
call, and the builder rejects any scenario that reuses its wording verbatim in
the observation, the prompt, or the memory sentence.

Outputs land in `data/parmbench-v1-supply/`, suffixed by rubric version for
anything past v1:

- `claim_drafts.jsonl` - one row per candidate, drafter verdict and checks;
- `gated_claims.jsonl` - the rows that passed both the span checks and the
  evidence gate, with the gold record resolved; and
- `supply_summary.json` - counts by outcome and rejection reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from parm_bench.evidence_gate import (
    CachedOpenAISupportJudge,
    EvidenceGateCacheMissError,
    EvidenceGateCachePolicy,
    grade_claim,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "personamem-v2-train-v1"
CANDIDATES_PATH = SOURCE_ROOT / "candidate_sources.jsonl"
RECORDS_PATH = SOURCE_ROOT / "records.jsonl"
HISTORY_ROOT = SOURCE_ROOT / "source"
SUPPLY_ROOT = ROOT / "data" / "parmbench-v1-supply"
DRAFT_CACHE = ROOT / "data" / "construction-caches" / "parmbench-v1" / "claim-drafts"
GATE_CACHE = ROOT / "data" / "evidence-gate-caches" / "parmbench-v1"

DRAFT_MODEL = "gpt-5-mini"
DRAFT_PROMPT_VERSION = "parmbench_claim_draft_v1"
DRAFT_PROMPT_VERSION_V2 = "parmbench_claim_draft_v2"

MIN_SPAN_CHARS = 30
MAX_SPAN_CHARS = 320
MAX_LEVER_CHARS = 120

# A v2 claim this close to the v1 claim that already failed the fairness sweep
# twice is the same fact under a new name, so the row is skipped rather than
# re-rolled.
MAX_REDRAFT_CLAIM_JACCARD = 0.7

FACT_KINDS = (
    "taste",
    "habit",
    "routine",
    "relationship",
    "named_entity",
    "commitment",
    "schedule",
    "constraint",
    "identity",
    "exclusion",
    "none",
)

DRAFT_INSTRUCTIONS = """\
You draft one conservative personal-fact claim about a user from a raw
conversation excerpt.

Only the user's own turns are evidence. Assistant turns are context and never
support a claim on their own.

Write a claim only when the user's own words state or clearly demonstrate a
durable personal fact: a taste, a habit, a routine, a relationship, a named
person or place in the user's life, a commitment, a constraint, an identity, or
an explicit dislike or exclusion. Write it as one short sentence that starts
with "The user".

Rules:
- The claim must be entailed by a span of the user's own words that you quote
  back verbatim.
- Name only the activities, places, people, and identities the user actually
  mentioned. Never introduce an activity or relation the user did not name.
- Do not use a frequency word such as usually, often, regularly, always,
  every, daily, weekly, or exclusively unless two or more separate user
  statements support that frequency.
- A user asking about a topic is not a preference. Curiosity about a subject is
  not a taste, a habit, or a commitment.
- Do not restate an assistant suggestion, rewrite, or recommendation.
- The hint field is an unverified upstream label. Treat it only as a pointer to
  where to look. Never copy its wording, and ignore it when the user's own
  words do not support it.
- When nothing durable is supportable, set supportable to false and leave the
  claim and the span empty.

The evidence span must be an exact contiguous substring of a single user turn,
between 30 and 300 characters, with no line break and no double-quote
character. Copy it character for character.

Set relational_hop to true only when the fact connects the user to a named
third party, place, or organisation, so that recognising the fact from a
mention of that named thing needs one hop through the relation.
"""

DRAFT_INSTRUCTIONS_V2 = """\
You draft one conservative personal-fact claim about a user from a raw
conversation excerpt, and the claim has to be one that could change an ordinary
decision.

Only the user's own turns are evidence. Assistant turns are context and never
support a claim on their own.

A claim qualifies only when both tests pass.

Support test. The user's own words state or clearly demonstrate the fact, and
you can quote back verbatim the span that entails it.

Decision-relevance test. The fact is durable and could steer a choice among
ordinary options: a preference, a constraint, an exclusion, a requirement, a
schedule or a commitment, or a relationship to a person, place, or
organisation that would make one option better or worse for this person. Name
the everyday choice that would come out differently for someone with this fact.
If you cannot name one, the fact does not qualify.

Decline these even when the user plainly stated them:
- something finished that bears on no present choice, such as a healed injury,
  a trip already taken, or a job already left;
- a one-off incident or a passing mood;
- an opinion about an abstract subject, a public figure, or an idea, which
  changes no ordinary decision;
- a fact so general that every option would suit it equally.

Write the claim as one short sentence that starts with "The user".

Write decision_lever as a short phrase naming the kind of choice the fact could
steer, phrased as what this person would do. Examples of the shape: "would pick
a screen-free option over a connected one", "cannot attend on Monday mornings",
"avoids dairy". Keep it under 90 characters. It is a note about the claim, not
part of it.

Rules:
- The claim must be entailed by a span of the user's own words that you quote
  back verbatim.
- Name only the activities, places, people, and identities the user actually
  mentioned. Never introduce an activity or relation the user did not name.
- Do not use a frequency word such as usually, often, regularly, always,
  every, daily, weekly, or exclusively unless two or more separate user
  statements support that frequency.
- A user asking about a topic is not a preference. Curiosity about a subject is
  not a taste, a habit, or a commitment.
- Do not restate an assistant suggestion, rewrite, or recommendation.
- The hint field is an unverified upstream label. Treat it only as a pointer to
  where to look. Never copy its wording, and ignore it when the user's own
  words do not support it.
- When nothing durable and decision-relevant is supportable, set supportable to
  false and leave the claim, the span, and the lever empty.

The evidence span must be an exact contiguous substring of a single user turn,
between 30 and 300 characters, with no line break and no double-quote
character. Copy it character for character.

Set relational_hop to true only when the fact connects the user to a named
third party, place, or organisation, so that recognising the fact from a
mention of that named thing needs one hop through the relation.
"""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "supportable": {"type": "boolean"},
        "claim": {"type": "string"},
        "evidence_span": {"type": "string"},
        "fact_kind": {"type": "string", "enum": list(FACT_KINDS)},
        "relational_hop": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": [
        "supportable",
        "claim",
        "evidence_span",
        "fact_kind",
        "relational_hop",
        "rationale",
    ],
    "additionalProperties": False,
}

DRAFT_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        **{
            key: dict(value)
            for key, value in DRAFT_SCHEMA["properties"].items()
        },
        "decision_lever": {"type": "string"},
    },
    "required": list(DRAFT_SCHEMA["required"]) + ["decision_lever"],
    "additionalProperties": False,
}

RUBRICS = {
    DRAFT_PROMPT_VERSION: {
        "instructions": DRAFT_INSTRUCTIONS,
        "schema": DRAFT_SCHEMA,
        "schema_name": "parmbench_claim_draft",
        "requires_lever": False,
    },
    DRAFT_PROMPT_VERSION_V2: {
        "instructions": DRAFT_INSTRUCTIONS_V2,
        "schema": DRAFT_SCHEMA_V2,
        "schema_name": "parmbench_claim_draft_v2",
        "requires_lever": True,
    },
}

_WHITESPACE = re.compile(r"\s+")


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hash_request(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(request), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def render_draft_request(
    snippet: Sequence[Mapping[str, str]],
    hint: str,
) -> str:
    rendered = "\n\n".join(
        f"{str(turn.get('role', '')).title()}: {turn.get('content', '')}"
        for turn in snippet
    )
    return (
        f"Hint (unverified upstream label, not evidence):\n{hint}\n\n"
        f"Conversation excerpt:\n{rendered}"
    )


class DraftCacheMissError(RuntimeError):
    """A cache-only run reached a candidate the drafter has never seen."""


class CachedDrafter:
    """Cached `gpt-5-mini` claim drafter keyed by prompt version, model, input."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        model: str = DRAFT_MODEL,
        prompt_version: str = DRAFT_PROMPT_VERSION,
        offline: bool = False,
    ) -> None:
        from openai import OpenAI

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.prompt_version = prompt_version
        self.rubric = RUBRICS[prompt_version]
        self.offline = offline
        self.client = OpenAI()
        self.live_calls = 0
        self._lock = threading.Lock()

    def _create(self, **kwargs: Any) -> Any:
        """Call the API, waiting out rate limits and transient network faults."""

        import random
        import time

        for attempt in range(6):
            try:
                return self.client.responses.create(**kwargs)
            except Exception:  # noqa: BLE001 - retried, then re-raised
                if attempt == 5:
                    raise
                time.sleep(min(60.0, 2.0 * 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable draft retry state")

    def draft(
        self,
        snippet: Sequence[Mapping[str, str]],
        hint: str,
    ) -> tuple[dict[str, Any], str]:
        input_text = render_draft_request(snippet, hint)
        request = {
            "prompt_version": self.prompt_version,
            "model": self.model,
            "input": input_text,
        }
        request_hash = hash_request(request)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["result"], request_hash
        if self.offline:
            raise DraftCacheMissError(f"missing claim draft cache entry: {request_hash}")
        response = self._create(
            model=self.model,
            instructions=self.rubric["instructions"],
            input=input_text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": self.rubric["schema_name"],
                    "strict": True,
                    "schema": self.rubric["schema"],
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def base_case_id_for(row: Mapping[str, Any]) -> str:
    """The scenario id the builder derives from a candidate row."""

    return (
        f"parmbench-v1-p{row['persona_id']}-"
        f"{str(row['source_row_id']).split(':')[-1]}"
    )


def load_records_by_corpus(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_corpus: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            by_corpus.setdefault(record["corpus_id"], []).append(record)
    return by_corpus


def user_turns(snippet: Sequence[Mapping[str, str]]) -> list[str]:
    return [
        str(turn.get("content", ""))
        for turn in snippet
        if str(turn.get("role", "")).strip().casefold() == "user"
    ]


def span_rejection(span: str, snippet: Sequence[Mapping[str, str]]) -> str | None:
    """Deterministic reasons the drafted span cannot anchor a benchmark case."""

    if not span.strip():
        return "empty_span"
    if "\n" in span or "\r" in span:
        return "span_has_line_break"
    if '"' in span or "\\" in span:
        return "span_has_unquotable_character"
    if not MIN_SPAN_CHARS <= len(span) <= MAX_SPAN_CHARS:
        return "span_length_out_of_range"
    if not any(span in text for text in user_turns(snippet)):
        return "span_not_verbatim_in_user_turn"
    return None


def resolve_gold_record(
    span: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    matches = [record for record in records if span in str(record.get("text", ""))]
    if not matches:
        return None
    # Prefer the gold snippet record when several pages quote the same span.
    for record in matches:
        if record.get("provenance", {}).get("gold_related_snippet"):
            return dict(record)
    return dict(matches[0])


_CLAIM_WORD = re.compile(r"[a-z0-9']+")
_CLAIM_STOPWORDS = frozenset(
    """a an and are as at be been but by can did do does for from had has have
    her his in into is it its of on or our that the their them they this to
    user was were will with would their's""".split()
)


def claim_tokens(claim: str) -> frozenset[str]:
    return frozenset(
        word
        for word in _CLAIM_WORD.findall(claim.casefold())
        if word not in _CLAIM_STOPWORDS and len(word) > 2
    )


def same_fact_as(first: str, second: str) -> bool:
    """Whether two claims describe the same fact closely enough to skip.

    A row whose scenario already failed the fairness sweep twice does not get a
    second life under a new claim that says the same thing. Overlap of the
    claims' content words is a blunt test, but it is the one that can run over
    every row without another model call.
    """

    left, right = claim_tokens(first), claim_tokens(second)
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= MAX_REDRAFT_CLAIM_JACCARD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offline", action="store_true", help="cache-only run")
    parser.add_argument(
        "--rubric",
        default=DRAFT_PROMPT_VERSION,
        choices=sorted(RUBRICS),
        help="drafting rubric version; anything past v1 writes suffixed files",
    )
    parser.add_argument(
        "--skip-base-case-ids",
        default="",
        help=(
            "path to a newline-separated list of base case ids whose rows are "
            "already spent on a frozen scenario and must not be redrafted"
        ),
    )
    parser.add_argument(
        "--previous-drafts",
        default="",
        help=(
            "path to an earlier claim_drafts.jsonl; a row whose new claim says "
            "the same thing as its previous one is skipped"
        ),
    )
    args = parser.parse_args()

    load_env(ROOT / ".env")
    SUPPLY_ROOT.mkdir(parents=True, exist_ok=True)
    GATE_CACHE.mkdir(parents=True, exist_ok=True)

    suffix = "" if args.rubric == DRAFT_PROMPT_VERSION else "_v2"
    requires_lever = bool(RUBRICS[args.rubric]["requires_lever"])

    candidates = [
        json.loads(line)
        for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    spent: set[str] = set()
    if args.skip_base_case_ids:
        spent = {
            line.strip()
            for line in Path(args.skip_base_case_ids)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        candidates = [
            row for row in candidates if base_case_id_for(row) not in spent
        ]
    previous_claims: dict[str, str] = {}
    if args.previous_drafts:
        for row in load_jsonl(Path(args.previous_drafts)):
            claim = str(row.get("draft", {}).get("claim", "")).strip()
            if claim:
                previous_claims[str(row["source_row_id"])] = claim
    if args.limit:
        candidates = candidates[: args.limit]
    records_by_corpus = load_records_by_corpus(RECORDS_PATH)
    history_cache: dict[str, str] = {}

    drafter = CachedDrafter(
        DRAFT_CACHE, prompt_version=args.rubric, offline=args.offline
    )
    judge = CachedOpenAISupportJudge(
        GATE_CACHE,
        EvidenceGateCachePolicy.FROZEN
        if args.offline
        else EvidenceGateCachePolicy.POPULATE,
        client=drafter.client,
    )

    def history_text(relative_path: str) -> str:
        if relative_path not in history_cache:
            history_cache[relative_path] = (HISTORY_ROOT / relative_path).read_text(
                encoding="utf-8"
            )
        return history_cache[relative_path]

    lock = threading.Lock()

    def process(row: Mapping[str, Any]) -> dict[str, Any]:
        snippet = row["related_conversation_snippet"]
        out: dict[str, Any] = {
            "source_row_id": row["source_row_id"],
            "persona_id": row["persona_id"],
            "corpus_id": row["corpus_id"],
            "preference_type": row["preference_type"],
            "is_primary": row["is_primary"],
            "history_path": row["history_path"],
            "history_sha256": row["history_sha256"],
            "base_case_id": base_case_id_for(row),
            "draft_prompt_version": args.rubric,
        }
        try:
            draft, draft_hash = drafter.draft(snippet, str(row.get("preference", "")))
        except DraftCacheMissError:
            # An offline resume reports the rows a later run still owes rather
            # than pretending the drafter declined them.
            out["outcome"] = "pending"
            out["reason"] = "draft_not_cached"
            return out
        out["draft_hash"] = draft_hash
        out["draft"] = draft
        if not draft.get("supportable"):
            out["outcome"] = "rejected"
            out["reason"] = "drafter_declined"
            return out
        claim = str(draft.get("claim", "")).strip()
        span = str(draft.get("evidence_span", ""))
        if not claim:
            out["outcome"] = "rejected"
            out["reason"] = "empty_claim"
            return out
        if requires_lever:
            lever = str(draft.get("decision_lever", "")).strip()
            if not lever:
                out["outcome"] = "rejected"
                out["reason"] = "empty_decision_lever"
                return out
            if len(lever) > MAX_LEVER_CHARS:
                out["outcome"] = "rejected"
                out["reason"] = "decision_lever_too_long"
                return out
        previous = previous_claims.get(str(row["source_row_id"]))
        if previous and same_fact_as(claim, previous):
            out["outcome"] = "rejected"
            out["reason"] = "restates_the_previous_claim"
            out["previous_claim"] = previous
            return out
        reason = span_rejection(span, snippet)
        if reason is not None:
            out["outcome"] = "rejected"
            out["reason"] = reason
            return out
        with lock:
            raw_history = history_text(row["history_path"])
        if span not in raw_history:
            out["outcome"] = "rejected"
            out["reason"] = "span_not_verbatim_in_history_file"
            return out
        record = resolve_gold_record(span, records_by_corpus.get(row["corpus_id"], []))
        if record is None:
            out["outcome"] = "rejected"
            out["reason"] = "span_not_in_any_normalized_record"
            return out
        out["gold_source_id"] = record["source_id"]
        out["gold_source_hash"] = record["source_hash"]
        out["gold_record_text"] = record["text"]

        try:
            result = grade_claim(claim, snippet, judge=judge)
        except EvidenceGateCacheMissError:
            out["outcome"] = "pending"
            out["reason"] = "gate_not_cached"
            return out
        out["grade"] = result.to_dict()
        if result.grade not in ("explicit", "inferable"):
            out["outcome"] = "rejected"
            out["reason"] = f"gate_{result.grade}"
            if result.hard_rejections:
                out["reason"] = f"gate_{result.hard_rejections[0]}"
            return out
        out["outcome"] = "passed"
        out["reason"] = f"gate_{result.grade}"
        return out

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        drafted = list(pool.map(process, candidates))

    drafted.sort(key=lambda row: (row["persona_id"], row["source_row_id"]))
    passed = [row for row in drafted if row["outcome"] == "passed"]

    write_jsonl(SUPPLY_ROOT / f"claim_drafts{suffix}.jsonl", drafted)
    write_jsonl(SUPPLY_ROOT / f"gated_claims{suffix}.jsonl", passed)

    reasons = Counter(row["reason"] for row in drafted)
    grades = Counter(
        row.get("grade", {}).get("grade", "not_graded") for row in drafted
    )
    pending = [row for row in drafted if row["outcome"] == "pending"]
    summary = {
        "candidates_graded": len(drafted) - len(pending),
        "passed": len(passed),
        "failed": len(drafted) - len(passed) - len(pending),
        "pending": len(pending),
        "personas_with_a_passing_row": len({row["persona_id"] for row in passed}),
        "by_reason": dict(sorted(reasons.items())),
        "by_grade": dict(sorted(grades.items())),
        "by_fact_kind": dict(
            sorted(Counter(row["draft"]["fact_kind"] for row in passed).items())
        ),
        "relational_hop_rows": sum(
            1 for row in passed if row["draft"].get("relational_hop")
        ),
        "rows_skipped_as_already_built": len(spent),
        "draft_prompt_version": args.rubric,
        "draft_model": DRAFT_MODEL,
        "live_draft_calls": drafter.live_calls,
    }
    atomic_write_json(SUPPLY_ROOT / f"supply_summary{suffix}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    sys.exit(main())
