"""Build `data/benchmark_parmbench_v1` from gated PersonaMem claims.

Phase B of the construction contract. Every scenario starts from one row of
`data/parmbench-v1-supply/gated_claims.jsonl`: a personal-fact claim drafted
from raw user-authored text, its verbatim evidence span, and the evidence-gate
verdict that accepted it.

Each scenario is built in two steps.

1. A cached `gpt-5-mini` construction call turns the claim plus a sampled set
   of construction axes into the semantic core of a scenario: an ordinary task,
   an output-only winner defensible on visible evidence alone, a target whose
   decisive affordance connects to the claim, near-miss decoys, and one neutral
   replacement for the cue-ablation control. The cache is keyed by the sha256 of
   `{prompt_version, model, input}`, matching `parm_bench.evidence_gate`, so a
   rebuild replays without new calls.
2. A seeded, model-free assembly step wraps that core in one of fourteen
   observation envelopes, places the cue anywhere across the document, and pads
   the document with combinatorial filler until it lands inside the
   `parmbench_v1` token band.

Axes are assigned from balanced pools under one fixed seed, so domain,
envelope, mechanism, and wording relationship stay evenly spread without a
positional formula.

Usage:

    $env:PYTHONPATH = 'src'
    & 'C:\\Users\\karth\\anaconda3\\python.exe' \\
      scripts\\build_parmbench_v1_benchmark.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parmbench_v1_envelopes import (  # noqa: E402
    ENVELOPES,
    ENVELOPE_NAMES,
    Envelope,
    balanced_assignment,
    detail_line,
    filler_body,
    proper_name,
    render_entry,
    render_noise,
    render_opening,
    render_section,
    sentences,
)

from parm_bench.decision_validity import (  # noqa: E402
    CAPABILITIES,
    CachedOpenAIDecisionValidityJudge,
    DecisionValidityCachePolicy,
    DecisionValidityResult,
    ONE_HOP_RELATION,
    DIRECT_RELATION,
    audit_scenario,
    build_scenario,
    capability_for_fact,
)
from parm_bench.service_tier import service_tier_kwargs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SUPPLY_PATH = ROOT / "data" / "parmbench-v1-supply" / "gated_claims.jsonl"
FAIRNESS_REPAIRS_PATH = (
    ROOT / "data" / "parmbench-v1-supply" / "fairness_repairs.json"
)
RECORDS_PATH = ROOT / "data" / "personamem-v2-train-v1" / "records.jsonl"
DATASET_ROOT = ROOT / "data" / "benchmark_parmbench_v1"
CONTEXT_ROOT = DATASET_ROOT / "contexts"
CONSTRUCTION_CACHE = ROOT / "data" / "construction-caches" / "parmbench-v1"
DECISION_VALIDITY_CACHE = (
    ROOT / "data" / "decision-validity-caches" / "parmbench-v1"
)
SOURCE_ROOT_RELATIVE = "../personamem-v2-train-v1/source"

BUILDER_VERSION = "parmbench_v1_builder_v1"
# v3 added the evaluator-only causal chain to the construction response. v4
# restricts assumptions_required to MATERIAL assumptions: the first pilot
# showed a model told to list every obvious assumption lists one for all 17
# fresh cores, so the hard reject on a non-empty list starved the batch. Old
# caches are keyed by the previous versions and stay untouched.
CONSTRUCTION_PROMPT_VERSION = "parmbench_construction_v4"
CONSTRUCTION_MODEL = "gpt-5-mini"
AXIS_SEED = 20260728

TOKENIZER = "cl100k_base"
MIN_TOKENS = 6_400
MAX_TOKENS = 15_400

DOMAINS = (
    "choosing a class or workshop to attend",
    "picking a place to hold a small gathering",
    "booking a repeat appointment",
    "shortlisting a flat or room to view",
    "choosing a route or a way to travel",
    "picking a shop or supplier to order from",
    "choosing a volunteering slot",
    "selecting a course reading or study group",
    "picking a café or eating place for a meeting",
    "choosing a community group to join",
    "booking a short break or day trip",
    "choosing a service provider for a household job",
    "picking a venue for a rehearsal or practice session",
    "selecting a delivery or collection option",
    "choosing a seat, desk, or working space",
    "picking an evening event to go to",
)

MECHANISMS = (
    "schedule fit",
    "feasibility",
    "stated requirements",
    "recency",
    "quality of the write-up",
    "cost",
    "an explicit constraint in the task",
)

OVERLAP_MODES = ("share_wording", "paraphrase_only")

ANSWER_CONTRACTS = (
    "Name exactly one {item} from the material below and write nothing else.",
    "Reply with exactly one {item} name taken from the material below.",
    "Pick exactly one {item} from what follows and give only its name.",
    "Answer with exactly one {item} name from the document below, nothing more.",
    "Return exactly one {item} from the material below, named as it is written there.",
    "Choose exactly one {item} below and reply with its name alone.",
    "Give exactly one {item} name from the material that follows.",
    "State exactly one {item} from the list below and no other text.",
)

CONSTRUCTION_INSTRUCTIONS = """\
You build the semantic core of one scenario for a personal-memory benchmark.

You are given a personal fact about a person, the person's own words that
support it, a task domain, and a few construction settings. Produce a realistic
set of options for an ordinary task. One option is the sensible pick on the
visible evidence alone. A different option becomes the better pick for a reader
who already knows the personal fact.

Return these fields.

- task: one or two sentences describing the ordinary task somebody is doing. It
  must read as routine work in the domain. It must not mention the personal
  fact, the person's history, remembering, or personalisation, and it must not
  name any option.
- item_noun: the singular noun for what the options are, for example workshop,
  flat, slot, supplier.
- winner_label, target_label, and one label per decoy: short distinct names of
  two to four words. No shared numbering scheme, no shared prefix, and no label
  may be a substring of another. Do not use the words option, choice, pick,
  recommended, or best inside a label.
- winner_body: two or three sentences that make the winner the sensible pick
  through the stated ordinary mechanism alone. Never reference the personal
  fact.
- target_body: two or three sentences about the target that read as
  unremarkable on their own. They must not claim the target is strong on the
  ordinary mechanism.
- cue_clause: exactly one sentence, written to sit at the end of target_body. It
  states a concrete, checkable affordance of the target. A reader who knows the
  personal fact sees at once why this makes the target the right pick. A reader
  who does not know the fact reads it as an ordinary detail. Describe a property
  of the target. Never describe the person, and never state the personal fact.
- neutral_clause: exactly one sentence of similar length and register that
  replaces cue_clause when the affordance is removed. Neutral logistics or an
  unrelated detail. It must not make the target more or less attractive, and it
  must not be the opposite of the affordance.
- decoys: near misses. Each one nearly satisfies either the ordinary mechanism
  or the affordance and falls short in a way the text states plainly.
- memory_text: one short sentence stating the personal fact plainly, starting
  with "The user". Keep it faithful to the person's own words and add nothing.

Then state the causal chain behind the scenario. These five fields are read by
the builder alone. They are never shown to any system under test, so write them
plainly and do not soften them.

- why_ordinary_wins: why a reader with no personal information picks the winner
  on the stated ordinary mechanism.
- why_cue_neutral_without_memory: why that same reader treats cue_clause as an
  ordinary detail worth no extra weight.
- why_memory_plus_cue_prefers_b: why the personal fact together with the
  affordance makes the target the better pick for this person, and why the
  winner becomes unsuitable or clearly worse for them.
- assumptions_required: every MATERIAL assumption your justification needs
  that the person never stated: a possession, permission, relationship,
  location, medical conclusion, future plan, or unstated preference that is
  not already part of the personal fact. One short phrase each. Do not list
  restatements of the fact itself (if the fact is that they own a console,
  "they still own the console" is not an assumption), background facts true
  of nearly anyone, or ordinary implications of the stated fact. Design the
  scenario so this list is genuinely empty: a scenario that needs even one
  material assumption will be rejected, so choose an affordance that follows
  from the fact alone.
- why_control_removes_advantage: why swapping cue_clause for neutral_clause
  leaves the target with no remaining advantage for this person anywhere in its
  entry, including sentences you wrote outside the cue.

Wording relationship setting:
- share_wording: cue_clause may reuse some of the same content words the person
  used.
- paraphrase_only: cue_clause must not reuse the person's distinctive content
  words. Carry the affordance with different vocabulary.

Rules:
- No ratings, scores, star counts, or decimal numbers anywhere unless numeric
  ratings are allowed.
- Never write that an option is the strongest, the consensus, the default, the
  obvious one, or narrower than another, and never write that nobody challenged
  it.
- Do not mention ranking, ordering, or how many options exist.
- Do not use the words memory, preference, persona, or profile.
- The personal fact must not be stated or paraphrased anywhere in the document
  text you return. Only memory_text may state it.
- Write every body in the given document register.
"""

NEUTRAL_REPAIR_PROMPT_VERSION = "parmbench_neutral_repair_v1"
NEUTRAL_REPAIR_INSTRUCTIONS = """\
You write one replacement sentence for the control arm of a benchmark.

You are given a sentence that states a decisive affordance of an option in a
document, and the length window the replacement has to sit inside. Write a
replacement sentence that:

- describes the same option;
- carries neutral logistics or an ordinary unrelated detail;
- leaves the option neither more nor less attractive than the rest of the
  document already makes it;
- is not the opposite, the denial, or the withdrawal of the affordance, and
  does not mention the affordance at all; and
- matches the register of the sentence it replaces.

Return only the sentence, inside the given character window.
"""
NEUTRAL_REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"neutral_clause": {"type": "string"}},
    "required": ["neutral_clause"],
    "additionalProperties": False,
}

CONSTRUCTION_SCHEMA = {
    "type": "object",
    "properties": {
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
        "why_cue_neutral_without_memory": {"type": "string"},
        "why_memory_plus_cue_prefers_b": {"type": "string"},
        "assumptions_required": {
            "type": "array",
            "items": {"type": "string"},
        },
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
        "why_cue_neutral_without_memory",
        "why_memory_plus_cue_prefers_b",
        "assumptions_required",
        "why_control_removes_advantage",
        "decoys",
    ],
    "additionalProperties": False,
}

# The causal chain is construction provenance. It explains the scenario to the
# builder and to the decision-validity gate, and it must not reach a model
# under test through the prompt, the observation, or the injected memory.
CAUSAL_CHAIN_FIELDS = (
    "why_ordinary_wins",
    "why_cue_neutral_without_memory",
    "why_memory_plus_cue_prefers_b",
    "why_control_removes_advantage",
)

_STOPWORDS = frozenset(
    """a about after all also an and any are as at be been but by can did do
    for from had has have her him his how i if in into is it its me my not of
    on one or our out over she should so some than that the their them then
    there these they this to up was we were what when where which while who
    why will with would you your""".split()
)
_WORD = re.compile(r"[a-z0-9']+")


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def content_words(text: str) -> set[str]:
    return {
        word
        for word in _WORD.findall(text.casefold())
        if word not in _STOPWORDS and len(word) > 3
    }


# A prompt sharing more content words than this with the personal fact starts
# to hand a memory query to a system that only ever reads the prompt, which is
# what acceptance criterion 3 forbids.
MAX_PROMPT_CLAIM_OVERLAP = 2
_SHORT_WORD = re.compile(r"[a-z0-9']+")


def prompt_claim_overlap(prompt: str, claim: str) -> tuple[str, ...]:
    """Content words the ordinary prompt shares with the personal fact.

    Deliberately looser than `content_words`: three-letter tokens such as
    "gpu" or "bus" are exactly the ones that would make a prompt-only query
    work, so they are counted here even though they are noise elsewhere.
    """

    def words(text: str) -> set[str]:
        return {
            word
            for word in _SHORT_WORD.findall(text.casefold())
            if word not in _STOPWORDS and len(word) > 2
        }

    return tuple(sorted(words(prompt) & words(claim)))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


CEILING_MEMORY_HEADING = "Known personal memory:"


def ceiling_prompt(memory_text: str, prompt: str) -> str:
    """Render the memory-included prompt with the fact under its own heading.

    The first fairness sweep concatenated the memory sentence onto the front of
    the task. The answer model is instructed to follow the task using only the
    supplied observation, and an unlabelled leading sentence reads as part of
    that task, so the fact was discounted and the ceiling collapsed on
    scenarios whose cue plainly matched it. Naming the fact as personal memory,
    the way the PersonaMem pilot did, separates it from the task text.
    """

    return f"{CEILING_MEMORY_HEADING}\n{memory_text.strip()}\n\n{prompt.strip()}"


# --------------------------------------------------------------------------
# construction call
# --------------------------------------------------------------------------


REPAIR_PREAMBLE = """\
An earlier version of this scenario was tested on a reader who was given the
document with no personal information, and on a reader who was given the
personal fact as well. It did not behave as it must. Write a completely
different set of options for the same personal fact and the same task domain.

Three readings have to come out right at once:

1. A reader with no personal information picks the winner.
2. That reader still picks the winner when the cue sentence is swapped for the
   neutral sentence.
3. A reader who is told the personal fact picks the target instead.
"""

REPAIR_NOTES = {
    "positive": """\
What went wrong: a reader who knew nothing about the person already picked the
target. The affordance was doing work on its own. Make the cue sentence read as
a plain factual detail that no uninformed reader would weigh, and make the
winner's case on the ordinary mechanism unmistakably stronger.
""",
    "cue-ablated": """\
What went wrong: a reader who knew nothing about the person did not pick the
winner even after the affordance was removed. Either a decoy read better or the
target read better on ordinary grounds. State the winner's advantage on the
ordinary mechanism in concrete, checkable terms, give the target no ordinary
advantage at all, and make every decoy fall visibly short.
""",
    "memory-included": """\
What went wrong: a reader who was told the personal fact still picked the
winner, so the affordance was not decisive for this task. Choose an affordance
that settles the task for someone who knows the fact: knowing it must make the
winner unsuitable or unusable for this person, not merely make the target a
pleasant extra. The affordance must bear on the task being done, not on an
unrelated interest of theirs.
""",
}


def render_construction_request(spec: Mapping[str, Any]) -> str:
    request = (
        f"Personal fact: {spec['claim']}\n"
        f"The person's own words: {spec['evidence_span']}\n"
        f"Task domain: {spec['domain']}\n"
        f"Ordinary evidence mechanism: {spec['mechanism']}\n"
        f"Numeric ratings allowed: {'yes' if spec['ratings_allowed'] else 'no'}\n"
        f"Wording relationship: {spec['overlap_mode']}\n"
        f"Number of near-miss decoys: {spec['decoy_count']}\n"
        f"Document register: {spec['register']}\n"
    )
    # A repaired scenario keeps its claim, evidence span, and axes and asks for
    # a fresh core. The attempt number and the variant that broke are part of
    # the request, so the rebuild cannot replay the core that failed and the
    # rewrite is aimed at the reading that came out wrong.
    if spec.get("repair_attempt"):
        request += f"Regeneration attempt: {spec['repair_attempt']}\n"
        request += REPAIR_PREAMBLE
        for variant in spec.get("repair_failed_variants", ()):
            request += REPAIR_NOTES[variant]
    return request


class CachedConstructor:
    """Cached `gpt-5-mini` scenario-core generator."""

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
        """Call the API, waiting out rate limits and transient network faults.

        A batch of repairs is large enough that one refused connection would
        otherwise drop a scenario that has nothing wrong with it.
        """

        import random
        import time

        for attempt in range(6):
            try:
                return self.client.responses.create(
                    **kwargs, **service_tier_kwargs()
                )
            except Exception:  # noqa: BLE001 - retried, then re-raised
                if attempt == 5:
                    raise
                time.sleep(min(60.0, 2.0 * 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable construction retry state")

    def build(self, spec: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        input_text = render_construction_request(spec)
        request = {
            "prompt_version": CONSTRUCTION_PROMPT_VERSION,
            "model": CONSTRUCTION_MODEL,
            "input": input_text,
        }
        request_hash = hash_request(request)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["result"], request_hash
        if self.offline:
            raise RuntimeError(f"missing construction cache entry: {request_hash}")
        response = self._create(
            model=CONSTRUCTION_MODEL,
            instructions=CONSTRUCTION_INSTRUCTIONS,
            input=input_text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parmbench_scenario_core",
                    "strict": True,
                    "schema": CONSTRUCTION_SCHEMA,
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

    def repair_neutral(
        self,
        *,
        cue: str,
        target_body: str,
        register: str,
        low: int,
        high: int,
    ) -> tuple[str, str]:
        """Rewrite a control replacement that missed the length window."""

        input_text = (
            f"Sentence to replace: {cue}\n"
            f"Surrounding entry: {target_body}\n"
            f"Document register: {register}\n"
            f"Replacement length window: {low} to {high} characters\n"
        )
        request = {
            "prompt_version": NEUTRAL_REPAIR_PROMPT_VERSION,
            "model": CONSTRUCTION_MODEL,
            "input": input_text,
        }
        request_hash = hash_request(request)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return str(cached["result"]["neutral_clause"]), request_hash
        if self.offline:
            raise RuntimeError(f"missing repair cache entry: {request_hash}")
        response = self._create(
            model=CONSTRUCTION_MODEL,
            instructions=NEUTRAL_REPAIR_INSTRUCTIONS,
            input=input_text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parmbench_neutral_repair",
                    "strict": True,
                    "schema": NEUTRAL_REPAIR_SCHEMA,
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
        return str(result["neutral_clause"]), request_hash


NEUTRAL_LENGTH_BAND = (0.55, 1.8)


def neutral_length_ratio(core: Mapping[str, Any]) -> float:
    cue = str(core.get("cue_clause", "")).strip()
    neutral = str(core.get("neutral_clause", "")).strip()
    return len(neutral) / max(1, len(cue))


# --------------------------------------------------------------------------
# observation assembly
# --------------------------------------------------------------------------


_SENTENCE_START = re.compile(r"(\A|[.!?:;]\s+|\n\s*)([a-z])")


def _recapitalise(text: str) -> str:
    return _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def _scrub_label_mentions(core: dict[str, Any]) -> None:
    """Leave each option name in exactly one place: its own entry heading.

    The construction model routinely opens a body by repeating the option's
    own name and sometimes points at a rival by name. Either habit puts a
    choice string in the document twice, which the exactly-once answer check
    rejects. Self-mentions become "this <item>" and cross-mentions become
    "another <item>", so the prose still reads while each name stays unique.
    """

    item = (str(core.get("item_noun") or "option")).strip().strip(".").lower()
    labels = [str(core["winner_label"]), str(core["target_label"])] + [
        str(decoy["label"]) for decoy in core.get("decoys", [])
    ]
    ordered = sorted((label for label in labels if label.strip()), key=len, reverse=True)

    def scrub(text: str, own: str) -> str:
        for label in ordered:
            replacement = f"this {item}" if label == own else f"another {item}"
            text = re.sub(re.escape(label), replacement, text, flags=re.IGNORECASE)
        return _recapitalise(text.strip())

    core["winner_body"] = scrub(str(core.get("winner_body", "")), labels[0])
    core["target_body"] = scrub(str(core.get("target_body", "")), labels[1])
    core["cue_clause"] = scrub(str(core.get("cue_clause", "")), labels[1])
    core["neutral_clause"] = scrub(str(core.get("neutral_clause", "")), labels[1])
    for decoy in core.get("decoys", []):
        decoy["body"] = scrub(str(decoy.get("body", "")), str(decoy["label"]))


def normalise_core(core: Mapping[str, Any]) -> dict[str, Any]:
    """Make the cue appear once, at the end of the target body.

    The construction model is told the cue sits at the end of the target body
    and often writes it there itself. Appending it again would put the decisive
    sentence in the document twice, so it is lifted out of every body first.
    Option names are then reduced to one appearance each.
    """

    normalised = json.loads(json.dumps(dict(core)))
    cue = str(normalised.get("cue_clause", "")).strip()
    normalised["cue_clause"] = cue
    if cue:
        for key in ("target_body", "winner_body"):
            body = str(normalised.get(key, ""))
            if cue in body:
                normalised[key] = body.replace(cue, " ").strip()
        for decoy in normalised.get("decoys", []):
            body = str(decoy.get("body", ""))
            if cue in body:
                decoy["body"] = body.replace(cue, " ").strip()
    normalised["neutral_clause"] = str(normalised.get("neutral_clause", "")).strip()
    _scrub_label_mentions(normalised)
    return normalised


def assemble_observation(
    envelope: Envelope,
    core: Mapping[str, Any],
    *,
    rng: random.Random,
    encoding: Any,
    cue_fraction: float,
    winner_fraction: float,
    token_target: int,
) -> tuple[str, dict[str, Any]]:
    """Wrap the scenario core in a padded, seeded document."""

    used_labels = {
        str(core["winner_label"]).casefold(),
        str(core["target_label"]).casefold(),
    }
    used_labels.update(str(decoy["label"]).casefold() for decoy in core["decoys"])

    def block_tokens(block: str) -> int:
        # Two for the blank-line separator that joins blocks.
        return len(encoding.encode(block)) + 2

    blocks: list[str] = [render_opening(envelope, rng)]
    blocks.append(render_noise(envelope, rng))
    running = sum(block_tokens(block) for block in blocks)
    ordinal = 1
    while running < token_target:
        draw = rng.random()
        if draw < 0.10:
            block = render_section(envelope, rng)
        elif draw < 0.66:
            label = proper_name(rng)
            guard = 0
            while label.casefold() in used_labels and guard < 12:
                label = proper_name(rng)
                guard += 1
            used_labels.add(label.casefold())
            block = render_entry(
                envelope, rng, label, filler_body(rng), ordinal, filler=True
            )
            ordinal += 1
        elif draw < 0.90:
            block = render_noise(envelope, rng)
        else:
            block = detail_line(rng)
        blocks.append(block)
        running += block_tokens(block)

    body_slots = len(blocks) - 2
    winner_index = 2 + min(body_slots, max(0, int(winner_fraction * body_slots)))
    cue_index = 2 + min(body_slots, max(0, int(cue_fraction * body_slots)))
    decoy_indices = sorted(
        rng.randrange(2, len(blocks) + 1) for _ in range(len(core["decoys"]))
    )

    placements: list[tuple[int, str]] = []
    winner_entry = render_entry(
        envelope, rng, str(core["winner_label"]), str(core["winner_body"]), ordinal
    )
    ordinal += 1
    target_entry = render_entry(
        envelope,
        rng,
        str(core["target_label"]),
        f"{str(core['target_body']).rstrip()} {str(core['cue_clause']).strip()}",
        ordinal,
    )
    ordinal += 1
    placements.append((winner_index, winner_entry))
    placements.append((cue_index, target_entry))
    for index, decoy in zip(decoy_indices, core["decoys"]):
        placements.append(
            (
                index,
                render_entry(
                    envelope, rng, str(decoy["label"]), str(decoy["body"]), ordinal
                ),
            )
        )
        ordinal += 1

    # Insert from the back so earlier indices stay meaningful.
    for index, block in sorted(placements, key=lambda item: -item[0]):
        blocks.insert(min(index, len(blocks)), block)

    protected = {entry for _, entry in placements}
    text = "\n\n".join(blocks) + "\n"
    token_count = len(encoding.encode(text))
    while token_count > MAX_TOKENS and len(blocks) > 12:
        removable = [
            position
            for position, block in enumerate(blocks)
            if position > 1 and block not in protected
        ]
        if not removable:
            break
        token_count -= block_tokens(blocks.pop(removable[-1]))
    while token_count < MIN_TOKENS:
        block = render_noise(envelope, rng)
        blocks.append(block)
        token_count += block_tokens(block)
    text = "\n\n".join(blocks) + "\n"
    token_count = len(encoding.encode(text))

    metrics = {
        "token_count": token_count,
        "block_count": len(blocks),
        "cue_char_fraction": round(
            text.find(str(core["cue_clause"]).strip())
            / max(1, len(text) - len(str(core["cue_clause"]).strip())),
            4,
        ),
        "winner_char_fraction": round(
            text.casefold().find(str(core["winner_label"]).casefold())
            / max(1, len(text)),
            4,
        ),
    }
    return text, metrics


_WHITESPACE = re.compile(r"\s+")


def flatten(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text)).strip().casefold()


def causal_chain_texts(core: Mapping[str, Any]) -> tuple[str, ...]:
    """Every sentence of the causal chain, including declared assumptions."""

    values = [str(core.get(field, "")).strip() for field in CAUSAL_CHAIN_FIELDS]
    values.extend(
        str(item).strip() for item in core.get("assumptions_required", ())
    )
    return tuple(value for value in values if value)


def causal_chain_record(core: Mapping[str, Any]) -> dict[str, Any]:
    """The causal chain as it is persisted, in construction records alone."""

    record: dict[str, Any] = {
        field: str(core.get(field, "")).strip() for field in CAUSAL_CHAIN_FIELDS
    }
    record["assumptions_required"] = [
        str(item).strip()
        for item in core.get("assumptions_required", ())
        if str(item).strip()
    ]
    return record


def causal_chain_rejection(core: Mapping[str, Any]) -> str | None:
    """Reasons a scenario core is dropped before the gate is called.

    A construction model that has to write down what its justification assumes
    usually knows when it is inventing something. Taking it at its word costs
    one scenario and saves a judge call.
    """

    if "assumptions_required" not in core:
        return "missing_causal_chain"
    for field in CAUSAL_CHAIN_FIELDS:
        if not str(core.get(field, "")).strip():
            return "missing_causal_chain"
    if any(str(item).strip() for item in core["assumptions_required"]):
        return "construction_declares_required_assumptions"
    return None


# Short fragments recur in ordinary prose, so only a substantial run of the
# chain counts as a leak.
MIN_CHAIN_LEAK_LENGTH = 24


def causal_chain_leak(
    core: Mapping[str, Any],
    text: str,
    prompt: str,
    memory_text: str,
) -> str | None:
    """Reject a scenario whose causal chain reached model-visible text."""

    flat_text = flatten(text)
    flat_prompt = flatten(prompt)
    flat_memory = flatten(memory_text)
    for value in causal_chain_texts(core):
        needle = flatten(value)
        if len(needle) < MIN_CHAIN_LEAK_LENGTH:
            continue
        if needle in flat_text:
            return "causal_chain_leaks_into_observation"
        if needle in flat_memory:
            return "causal_chain_leaks_into_memory_text"
        if needle in flat_prompt:
            return "causal_chain_leaks_into_prompt"
    return None


def scenario_rejection(
    text: str,
    core: Mapping[str, Any],
    prompt: str,
    claim_row: Mapping[str, Any],
) -> str | None:
    """Deterministic reasons an assembled scenario cannot be shipped."""

    folded = text.casefold()
    cue = str(core["cue_clause"]).strip()
    neutral = str(core["neutral_clause"]).strip()
    labels = [str(core["winner_label"]), str(core["target_label"])] + [
        str(decoy["label"]) for decoy in core["decoys"]
    ]
    if len({label.casefold() for label in labels}) != len(labels):
        return "duplicate_labels"
    for label in labels:
        if not label.strip():
            return "empty_label"
        if folded.count(label.casefold()) != 1:
            return "label_not_unique_in_observation"
    for first in labels:
        for second in labels:
            if first is not second and first.casefold() in second.casefold():
                return "label_is_substring_of_another"
    if not cue or text.count(cue) != 1:
        return "cue_not_unique_in_observation"
    if not neutral or neutral in text:
        return "neutral_replacement_already_present"
    ratio = len(neutral) / max(1, len(cue))
    if not NEUTRAL_LENGTH_BAND[0] <= ratio <= NEUTRAL_LENGTH_BAND[1]:
        return "neutral_replacement_length_mismatch"
    for label in labels:
        if label.casefold() in cue.casefold():
            return "cue_names_an_option"
    memory_text = str(core["memory_text"]).strip()
    if not memory_text:
        return "empty_memory_text"
    if memory_text.casefold() in folded:
        return "memory_text_leaks_into_observation"
    span = str(claim_row["draft"]["evidence_span"])
    if span.casefold() in folded:
        return "evidence_span_leaks_into_observation"
    # The ceiling prompt carries the memory on purpose. Nothing else may ride
    # along with it: not an option name, not the cue, not the raw user span.
    if span.casefold() in memory_text.casefold():
        return "evidence_span_leaks_into_memory_text"
    prompt_folded = prompt.casefold()
    ceiling_folded = ceiling_prompt(memory_text, prompt).casefold()
    if "exactly one" not in prompt_folded:
        return "prompt_missing_answer_contract"
    for label in labels:
        if label.casefold() in ceiling_folded:
            return "label_leaks_into_prompt"
    if cue.casefold() in ceiling_folded or memory_text.casefold() in prompt_folded:
        return "cue_or_memory_leaks_into_prompt"
    claim = str(claim_row["draft"]["claim"])
    if len(prompt_claim_overlap(prompt, claim)) > MAX_PROMPT_CLAIM_OVERLAP:
        return "prompt_shares_too_much_wording_with_the_claim"
    ablated = text.replace(cue, neutral, 1)
    if cue in ablated:
        return "ablation_leaves_cue"
    if ablated.casefold().count(str(core["winner_label"]).casefold()) != 1:
        return "ablation_breaks_winner_uniqueness"
    return causal_chain_leak(
        core, text, ceiling_prompt(memory_text, prompt), memory_text
    )


# --------------------------------------------------------------------------
# capability tagging and distractors
# --------------------------------------------------------------------------


def capability_for(
    claim: str,
    fact_kind: str,
    relational_hop: bool,
    overlap_mode: str,
    *,
    memory_category: str = "",
    cue_text: str = "",
    evidence_span: str = "",
) -> str:
    """Label a scenario from its accepted fact and its causal relation.

    The supply row's `fact_kind` is the older vocabulary; the memory-quality
    gate's category wins when the row carries one. Labelling itself lives in
    `parm_bench.decision_validity`, so the gate that checks the label and the
    builder that assigns it cannot drift apart.
    """

    return capability_for_fact(
        claim=claim,
        memory_category=memory_category or fact_kind,
        causal_relation=(
            ONE_HOP_RELATION if relational_hop else DIRECT_RELATION
        ),
        evidence_span=evidence_span,
        cue_text=cue_text,
        overlap_mode=None if cue_text and evidence_span else overlap_mode,
    )


def choose_distractors(
    records: Sequence[Mapping[str, Any]],
    gold_source_id: str,
    claim: str,
    rng: random.Random,
    *,
    abstention_pressure: bool,
    count: int,
) -> list[dict[str, Any]]:
    pool = [
        record for record in records if record["source_id"] != gold_source_id
    ]
    if not pool:
        return []
    if abstention_pressure:
        claim_words = content_words(claim)
        ranked = sorted(
            pool,
            key=lambda record: (
                -len(claim_words & content_words(str(record["text"])[:2000])),
                record["source_id"],
            ),
        )
        chosen = ranked[:count]
    else:
        chosen = rng.sample(pool, min(count, len(pool)))
    return [
        {
            "source_id": record["source_id"],
            "perturbations": [],
            "text": str(record["text"])[:240],
        }
        for record in chosen
    ]


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_fairness_repairs(
    path: Path | None = FAIRNESS_REPAIRS_PATH,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Read the tracked repair and drop decisions from the fairness run.

    `repairs` maps a scenario to its attempt number and the variants that
    failed, which reseeds the scenario and asks the construction model for a
    fresh core aimed at the reading that broke. `drops` maps a scenario to the
    reason it was abandoned. Both are declared in one tracked file so a rebuild
    reproduces the repaired batch exactly.
    """

    # None means repairs are disabled (a pilot with no fairness history);
    # the default carries the tracked v1 file for frozen-batch replays.
    if path is None or not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    repairs = {
        str(key): {
            "attempt": int(value["attempt"]),
            "failed_variants": tuple(
                variant
                for variant in ("positive", "cue-ablated", "memory-included")
                if variant in set(value.get("failed_variants", ()))
            ),
        }
        for key, value in dict(payload.get("repairs", {})).items()
    }
    drops = {
        str(item["base_case_id"]): str(item["reason"])
        for item in payload.get("drops", [])
    }
    overlap = sorted(set(repairs) & set(drops))
    if overlap:
        raise ValueError(f"scenarios both repaired and dropped: {overlap}")
    return repairs, drops


def build_specs(
    claims: Sequence[Mapping[str, Any]],
    repairs_path: Path | None = FAIRNESS_REPAIRS_PATH,
) -> list[dict[str, Any]]:
    """Assign every construction axis from balanced, seeded pools."""

    repairs, _ = load_fairness_repairs(repairs_path)
    axis_rng = random.Random(AXIS_SEED)
    count = len(claims)
    domains = balanced_assignment(DOMAINS, count, axis_rng)
    envelopes = balanced_assignment(ENVELOPE_NAMES, count, axis_rng)
    mechanisms = balanced_assignment(MECHANISMS, count, axis_rng)
    overlaps = balanced_assignment(OVERLAP_MODES, count, axis_rng)
    contracts = balanced_assignment(ANSWER_CONTRACTS, count, axis_rng)
    ratings_pool = balanced_assignment(
        ("none",) * 5 + ("numeric",), count, axis_rng
    )
    abstention_pool = balanced_assignment(
        ("plain",) * 3 + ("pressure",), count, axis_rng
    )

    specs: list[dict[str, Any]] = []
    for index, row in enumerate(claims):
        base_case_id = (
            f"parmbench-v1-p{row['persona_id']}-"
            f"{str(row['source_row_id']).split(':')[-1]}"
        )
        repair = repairs.get(base_case_id)
        repair_attempt = repair["attempt"] if repair else 0
        repair_failed_variants = repair["failed_variants"] if repair else ()
        seed_key = (
            f"{base_case_id}#repair{repair_attempt}"
            if repair_attempt
            else base_case_id
        )
        seed = int(sha256_text(seed_key)[:16], 16)
        rng = random.Random(seed)
        envelope = next(e for e in ENVELOPES if e.name == envelopes[index])
        specs.append(
            {
                "base_case_id": base_case_id,
                "repair_attempt": repair_attempt,
                "repair_failed_variants": repair_failed_variants,
                "claim_row": row,
                "claim": row["draft"]["claim"],
                "evidence_span": row["draft"]["evidence_span"],
                "domain": domains[index],
                "envelope": envelope.name,
                "register": envelope.register,
                "mechanism": mechanisms[index],
                "overlap_mode": overlaps[index],
                "answer_contract": contracts[index],
                "ratings_allowed": ratings_pool[index] == "numeric",
                "abstention_pressure": abstention_pool[index] == "pressure",
                "decoy_count": rng.randrange(3, 6),
                "observation_kind": rng.choice(envelope.kinds),
                "cue_fraction": round(rng.uniform(0.05, 0.94), 4),
                "winner_fraction": round(rng.uniform(0.04, 0.93), 4),
                "token_target": rng.randrange(6_800, 13_000),
                "distractor_count": rng.randrange(3, 6),
                "seed": seed,
            }
        )
    return specs


_TURN_SPLIT = re.compile(r"^(User|Assistant):\s*", re.MULTILINE)


def source_turns(record_text: str) -> tuple[dict[str, str], ...]:
    """Split a stored source record back into role-tagged turns.

    The decision-validity auditor has to tell the person's own words from the
    assistant's, so it is given the record as turns rather than as one blob.
    """

    parts = _TURN_SPLIT.split(str(record_text).strip())
    if len(parts) < 3:
        return ({"role": "source record", "content": str(record_text).strip()},)
    turns: list[dict[str, str]] = []
    for role, content in zip(parts[1::2], parts[2::2]):
        body = content.strip()
        if body:
            turns.append({"role": role.casefold(), "content": body})
    return tuple(turns)


class OfflineJudgeClient:
    """Stand-in client that refuses a live call during an offline build."""

    class _Responses:
        @staticmethod
        def create(**kwargs: Any) -> Any:
            raise RuntimeError(
                "offline build cannot make decision validity calls"
            )

    responses = _Responses()


def add_decision_validity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--decision-validity-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="audit every constructed scenario before it is kept",
    )
    parser.add_argument(
        "--decision-validity-cache",
        default=str(DECISION_VALIDITY_CACHE),
        help="cache directory for decision-validity verdicts",
    )
    parser.add_argument(
        "--decision-validity-cache-policy",
        choices=[policy.value for policy in DecisionValidityCachePolicy],
        default=DecisionValidityCachePolicy.POPULATE.value,
        help="populate the verdict cache, or replay it and fail on a miss",
    )


def make_decision_validity_judge(
    args: argparse.Namespace,
) -> CachedOpenAIDecisionValidityJudge | None:
    if not args.decision_validity_gate:
        return None
    # An offline build replays, so a gap in the cache should name the missing
    # entry rather than fail at the first live call.
    policy = (
        DecisionValidityCachePolicy.FROZEN
        if args.offline
        else args.decision_validity_cache_policy
    )
    return CachedOpenAIDecisionValidityJudge(
        Path(args.decision_validity_cache),
        policy,
        client=OfflineJudgeClient() if args.offline else None,
    )


def gate_scenario(
    judge: CachedOpenAIDecisionValidityJudge | None,
    *,
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
    core: Mapping[str, Any],
    prompt: str,
    capability: str,
    sensitive_terms: Sequence[str] = (),
) -> DecisionValidityResult | None:
    if judge is None:
        return None
    scenario = build_scenario(
        claim=str(spec["claim"]),
        evidence_span=str(row["draft"]["evidence_span"]),
        task_prompt=prompt,
        core=core,
        ordinary_mechanism=str(spec["mechanism"]),
        capability_label=capability,
        evidence_turns=source_turns(row.get("gold_record_text", "")),
        sensitive_terms=sensitive_terms,
    )
    return audit_scenario(scenario, judge=judge)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(DATASET_ROOT),
        help="dataset directory to write; point a pilot away from the frozen batch",
    )
    parser.add_argument(
        "--fairness-repairs",
        default=str(FAIRNESS_REPAIRS_PATH),
        help=(
            "repair and drop decisions file from a fairness run; pass "
            "'none' for a pilot with no prior fairness history"
        ),
    )
    parser.add_argument(
        "--supply",
        default=str(SUPPLY_PATH),
        help=(
            "gated-claims JSONL to build from; point a pilot at its own "
            "supply file instead of the frozen gated_claims.jsonl"
        ),
    )
    add_decision_validity_arguments(parser)
    args = parser.parse_args()

    load_env(ROOT / ".env")
    encoding = tiktoken.get_encoding(TOKENIZER)
    output_root = Path(args.output_dir)
    context_root = output_root / "contexts"

    claims = load_jsonl(Path(args.supply))
    if args.limit:
        claims = claims[: args.limit]
    records_by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_jsonl(RECORDS_PATH):
        records_by_corpus[record["corpus_id"]].append(record)

    repairs_path = (
        None
        if args.fairness_repairs.strip().lower() == "none"
        else Path(args.fairness_repairs)
    )
    specs = build_specs(claims, repairs_path)
    _, fairness_drops = load_fairness_repairs(repairs_path)
    constructor = CachedConstructor(CONSTRUCTION_CACHE, offline=args.offline)
    validity_judge = make_decision_validity_judge(args)

    def warm(spec: Mapping[str, Any]) -> None:
        try:
            constructor.build(spec)
        except Exception as exc:  # noqa: BLE001 - reported during assembly
            print(f"construction failed for {spec['base_case_id']}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(
            pool.map(
                warm,
                [
                    spec
                    for spec in specs
                    if spec["base_case_id"] not in fairness_drops
                ],
            )
        )

    context_root.mkdir(parents=True, exist_ok=True)
    for stale in context_root.glob("*.md"):
        stale.unlink()

    cases: list[dict[str, Any]] = []
    construction_records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    corpora: dict[str, dict[str, Any]] = {}

    for spec in specs:
        if args.max_scenarios and len(construction_records) >= args.max_scenarios:
            break
        if spec["base_case_id"] in fairness_drops:
            dropped.append(
                {
                    "base_case_id": spec["base_case_id"],
                    "reason": fairness_drops[spec["base_case_id"]],
                }
            )
            continue
        row = spec["claim_row"]
        try:
            raw_core, request_hash = constructor.build(spec)
        except Exception as exc:  # noqa: BLE001
            dropped.append(
                {
                    "base_case_id": spec["base_case_id"],
                    "reason": f"construction_error: {exc}",
                }
            )
            continue
        core = normalise_core(raw_core)
        chain_reason = causal_chain_rejection(core)
        if chain_reason is not None:
            dropped.append(
                {
                    "base_case_id": spec["base_case_id"],
                    "reason": chain_reason,
                    "assumptions_required": [
                        str(item)
                        for item in core.get("assumptions_required", ())
                        if str(item).strip()
                    ],
                }
            )
            continue
        neutral_repaired = False
        if not (
            NEUTRAL_LENGTH_BAND[0]
            <= neutral_length_ratio(core)
            <= NEUTRAL_LENGTH_BAND[1]
        ):
            cue_length = len(str(core["cue_clause"]).strip())
            try:
                repaired, _ = constructor.repair_neutral(
                    cue=str(core["cue_clause"]).strip(),
                    target_body=str(core["target_body"]).strip(),
                    register=spec["register"],
                    low=int(cue_length * 0.7),
                    high=int(cue_length * 1.5),
                )
            except Exception:  # noqa: BLE001 - keep the original and let it drop
                repaired = ""
            if repaired.strip():
                raw_core["neutral_clause"] = repaired.strip()
                core = normalise_core(raw_core)
                neutral_repaired = True

        prompt = (
            f"{str(core['task']).strip()} "
            + spec["answer_contract"].format(item=str(core["item_noun"]).strip())
        )
        text = ""
        metrics: dict[str, Any] = {}
        reason: str | None = "not_attempted"
        attempt = 0
        for attempt in range(6):
            rng = random.Random(spec["seed"] + attempt * 7919)
            envelope = next(e for e in ENVELOPES if e.name == spec["envelope"])
            text, metrics = assemble_observation(
                envelope,
                core,
                rng=rng,
                encoding=encoding,
                cue_fraction=spec["cue_fraction"],
                winner_fraction=spec["winner_fraction"],
                token_target=spec["token_target"],
            )
            reason = scenario_rejection(text, core, prompt, row)
            if reason is None:
                break
        if reason is not None:
            dropped.append({"base_case_id": spec["base_case_id"], "reason": reason})
            continue

        cue = str(core["cue_clause"]).strip()
        neutral = str(core["neutral_clause"]).strip()
        capability = capability_for(
            spec["claim"],
            row["draft"]["fact_kind"],
            bool(row["draft"].get("relational_hop")),
            spec["overlap_mode"],
            memory_category=str(row["draft"].get("memory_category", "")),
            cue_text=cue,
            evidence_span=str(row["draft"]["evidence_span"]),
        )
        # The gate runs once the scenario is otherwise shippable, so a judge
        # call is never spent on a core the deterministic checks would drop.
        validity = gate_scenario(
            validity_judge,
            spec=spec,
            row=row,
            core=core,
            prompt=prompt,
            capability=capability,
        )
        if validity is not None and not validity.accept:
            dropped.append(
                {
                    "base_case_id": spec["base_case_id"],
                    "reason": "decision_validity_rejected",
                    "decision_validity": validity.to_dict(),
                }
            )
            continue

        content_path = f"contexts/{spec['base_case_id']}.md"
        (output_root / content_path).write_text(text, encoding="utf-8", newline="\n")

        corpus_id = row["corpus_id"]
        persona_id = row["persona_id"]
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
        distractors = choose_distractors(
            records_by_corpus.get(corpus_id, []),
            row["gold_source_id"],
            spec["claim"],
            random.Random(spec["seed"] + 104_729),
            abstention_pressure=spec["abstention_pressure"],
            count=spec["distractor_count"],
        )
        if len(distractors) < 3:
            dropped.append(
                {"base_case_id": spec["base_case_id"], "reason": "too_few_distractors"}
            )
            (output_root / content_path).unlink(missing_ok=True)
            continue

        memory_text = str(core["memory_text"]).strip()
        winner = str(core["winner_label"])
        target = str(core["target_label"])
        provenance = {
            "approved": True,
            "abstention_pressure": spec["abstention_pressure"],
            "builder_version": BUILDER_VERSION,
            "capability": capability,
            "construction_model": CONSTRUCTION_MODEL,
            "construction_prompt_version": CONSTRUCTION_PROMPT_VERSION,
            "construction_request_hash": request_hash,
            "domain": spec["domain"],
            "envelope_style": spec["envelope"],
            "evaluation_split": "calibration",
            "evidence_grade": row["grade"]["grade"],
            "evidence_gate_rubric": row["grade"]["rubric_version"],
            "fact_kind": row["draft"]["fact_kind"],
            "lexical_overlap_mode": spec["overlap_mode"],
            "ordinary_evidence_mechanism": spec["mechanism"],
            "persona_id": persona_id,
            "ratings_allowed": spec["ratings_allowed"],
            "neutral_clause_repaired": neutral_repaired,
            "fairness_repair_attempt": spec["repair_attempt"],
            "seeds": {
                "axis_seed": AXIS_SEED,
                "scenario_seed": spec["seed"],
                "assembly_attempt": attempt,
            },
            "source_dataset": "bowen-upenn/PersonaMem-v2",
            "source_revision": "b7b42b78917157afed063527a1c959e98f6109f2",
            "source_row_id": row["source_row_id"],
            "source_split": "train_text",
            "hashes": {
                "context_sha256": sha256_text(text),
                "gold_source_hash": row["gold_source_hash"],
                "history_sha256": row["history_sha256"],
                "claim_draft_hash": row["draft_hash"],
            },
        }

        for variant in ("positive", "cue-ablated", "memory-included"):
            cue_present = variant != "cue-ablated"
            case_prompt = (
                ceiling_prompt(memory_text, prompt)
                if variant == "memory-included"
                else prompt
            )
            cases.append(
                {
                    "case_id": f"{spec['base_case_id']}-{variant}",
                    "base_case_id": spec["base_case_id"],
                    "corpus_id": corpus_id,
                    "variant": variant,
                    "prompt": case_prompt,
                    "observation": {
                        "kind": spec["observation_kind"],
                        "content_path": content_path,
                        "replacements": (
                            []
                            if cue_present
                            else [{"old": cue, "new": neutral}]
                        ),
                    },
                    "cue": {
                        "present": cue_present,
                        "type": "affordance_clause",
                        "text": cue,
                    },
                    "memory": {
                        "corpus_id": corpus_id,
                        "text": memory_text,
                        "gold_source_ids": [row["gold_source_id"]],
                        "sensitive_terms": [],
                        "sources": [json.loads(json.dumps(gold_source))],
                    },
                    "decisions": {
                        "answer_type": "natural_language_choice",
                        "output_only": {"choice": winner},
                        "memory_conditioned": {
                            "choice": winner if variant == "cue-ablated" else target
                        },
                    },
                    "distractors": {"sources": json.loads(json.dumps(distractors))},
                    "provenance": dict(provenance),
                }
            )

        construction_records.append(
            {
                "base_case_id": spec["base_case_id"],
                "persona_id": persona_id,
                "corpus_id": corpus_id,
                "source_row_id": row["source_row_id"],
                "claim": spec["claim"],
                "evidence_span": row["draft"]["evidence_span"],
                "evidence_grade": row["grade"],
                "gold_source_id": row["gold_source_id"],
                "capability": capability,
                "abstention_pressure": spec["abstention_pressure"],
                "neutral_clause_repaired": neutral_repaired,
                "fairness_repair_attempt": spec["repair_attempt"],
                "axes": {
                    "domain": spec["domain"],
                    "envelope_style": spec["envelope"],
                    "observation_kind": spec["observation_kind"],
                    "ordinary_evidence_mechanism": spec["mechanism"],
                    "lexical_overlap_mode": spec["overlap_mode"],
                    "ratings_allowed": spec["ratings_allowed"],
                    "requested_cue_fraction": spec["cue_fraction"],
                    "requested_winner_fraction": spec["winner_fraction"],
                    "decoy_count": len(core["decoys"]),
                    "distractor_count": len(distractors),
                },
                "metrics": metrics,
                "seeds": {
                    "axis_seed": AXIS_SEED,
                    "scenario_seed": spec["seed"],
                    "assembly_attempt": attempt,
                },
                "model_calls": {
                    "claim_draft_hash": row["draft_hash"],
                    "construction_request_hash": request_hash,
                    "construction_prompt_version": CONSTRUCTION_PROMPT_VERSION,
                    "construction_model": CONSTRUCTION_MODEL,
                },
                "choices": {"output_only": winner, "memory_conditioned": target},
                "memory_text": memory_text,
                "control_replacement": {"old": cue, "new": neutral},
                "causal_chain": causal_chain_record(core),
                "decision_validity": (
                    validity.to_dict() if validity is not None else None
                ),
            }
        )
        write_jsonl(output_root / "construction_records.jsonl", construction_records)

    write_jsonl(output_root / "cases.jsonl", cases)
    write_jsonl(output_root / "construction_records.jsonl", construction_records)
    manifest = {
        "schema_version": 1,
        "validation_profile": "parmbench_v1",
        "builder_version": BUILDER_VERSION,
        "construction_records": {"path": "construction_records.jsonl"},
        "corpora": [corpora[key] for key in sorted(corpora)],
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if dropped:
        write_jsonl(output_root / "dropped_scenarios.jsonl", dropped)
    elif (output_root / "dropped_scenarios.jsonl").exists():
        (output_root / "dropped_scenarios.jsonl").unlink()

    summary = {
        "scenarios": len(construction_records),
        "cases": len(cases),
        "personas": len({record["persona_id"] for record in construction_records}),
        "repaired": sum(
            1
            for record in construction_records
            if record["fairness_repair_attempt"]
        ),
        "dropped": len(dropped),
        "dropped_reasons": dict(
            sorted(Counter(item["reason"].split(":")[0] for item in dropped).items())
        ),
        "live_construction_calls": constructor.live_calls,
        "capability": dict(
            sorted(Counter(r["capability"] for r in construction_records).items())
        ),
        "decision_validity_gate": args.decision_validity_gate,
        "decision_validity_rejections": dict(
            sorted(
                Counter(
                    requirement
                    for item in dropped
                    for requirement in item.get("decision_validity", {}).get(
                        "unmet_requirements", ()
                    )
                ).items()
            )
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
