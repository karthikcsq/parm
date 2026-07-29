"""Add a supplement round of scenarios to `data/benchmark_parmbench_v1`.

The first construction round left the batch at 65 fairness-passing scenarios,
below the contract's 100-scenario calibration floor. The diagnosis was a supply
problem: `parmbench_claim_draft_v1` asked for auditable facts and got plenty of
them, but roughly half described something no ordinary task could turn on, so
their ceilings could not be repaired.

This script builds a second round from `gated_claims_v2.jsonl`, drafted under
`parmbench_claim_draft_v2`, which additionally requires a decision lever. It
reuses every piece of the round-one builder: the same envelopes, axis pools,
assembly, rejection checks, distractor selection, capability tagging, and
caches. Three things differ.

1. Supplement scenarios carry an `-s2` suffix, so no id collides with a frozen
   scenario and the round is legible in the record.
2. Axes are assigned under their own seed over the supplement's own claim
   count. Round one's assignment is a function of its claim count and must not
   be disturbed.
3. The construction request carries the claim's decision lever, and a scenario
   is rejected when any field reuses the lever's wording. The lever is
   construction metadata; it may steer the choice of affordance but must never
   reach a model under test.

The 65 frozen scenarios are never rebuilt. Their case rows, construction
records, and observation files are copied through byte for byte.

Usage:

    $env:PYTHONPATH = 'src'
    & 'C:\\Users\\karth\\anaconda3\\python.exe' \\
      scripts\\build_parmbench_v1_supplement.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_parmbench_v1_benchmark as builder  # noqa: E402
from parmbench_v1_envelopes import (  # noqa: E402
    ENVELOPES,
    ENVELOPE_NAMES,
    balanced_assignment,
)


ROOT = builder.ROOT
SUPPLY_PATH = ROOT / "data" / "parmbench-v1-supply" / "gated_claims_v2.jsonl"
FAIRNESS_REPAIRS_PATH = (
    ROOT / "data" / "parmbench-v1-supply" / "fairness_repairs_v2.json"
)
DATASET_ROOT = builder.DATASET_ROOT
CONTEXT_ROOT = builder.CONTEXT_ROOT

SUPPLEMENT_SUFFIX = "-s2"
SUPPLEMENT_ROUND = 2
BUILDER_VERSION = "parmbench_v1_builder_v1_supplement_v2"
CONSTRUCTION_PROMPT_VERSION = "parmbench_construction_v2"
AXIS_SEED = 20260729

CONSTRUCTION_INSTRUCTIONS = (
    builder.CONSTRUCTION_INSTRUCTIONS
    + """
The request carries a line naming what the personal fact could decide. Use it
to choose an affordance that settles this task for somebody with the fact,
rather than one that merely suits them. Never reuse its wording: that line is a
construction note, and it must not appear in any field you return.
"""
)

_WHITESPACE = re.compile(r"\s+")

# A decision lever names what this person would choose, need, or rule out. The
# openers below are the ones that say so. Anything else turned out to mark a
# claim that fails the decision-relevance test rather than a lever that is
# merely worded oddly: levers such as "eligible for Columbia-affiliated
# research collaborations" or "invite to speak on justice panels" describe what
# somebody else would do about the person, and "might consider seeking clinical
# assessment" hedges instead of deciding. The list is deliberately a closed
# allowlist, so an unfamiliar shape costs supply rather than admitting a
# scenario whose memory decides nothing.
LEVER_OPENERS = frozenset(
    """attends avoid avoids books brings buys can cannot choose chooses declines
    drinks eats favors favours keeps limits must need needs orders picks plays
    prefer prefers reads refuses requires reserves rides schedule schedules
    seeks shops skips sticks takes travels uses want wants watches will works
    would""".split()
)

# A claim that reports what somebody asked about is not a durable fact, and the
# evidence gate cannot see that: a conversation always supports a faithful
# report of its own question, so the gate grades it "explicit". Both drafting
# rubrics forbid treating curiosity as a preference and both were talked past,
# 24 times out of 247 under v2.
_QUESTION_SHAPED_CLAIM = re.compile(
    r"^the user (is asking|asked|asks|is curious|wants to know|is wondering|"
    r"wondered|has noticed|noticed|is considering|is interested in "
    r"(?:learning|knowing)|inquired|is seeking (?:information|advice|"
    r"clarification))",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text)).strip().casefold()


def supply_rejection(claim: str, lever: str) -> str | None:
    """Reasons a gated claim must not reach the construction call at all.

    Both checks run before any model call, so a claim that cannot carry a
    scenario costs nothing to reject.
    """

    if _QUESTION_SHAPED_CLAIM.match(str(claim).strip()):
        return "claim_restates_a_question_rather_than_a_fact"
    tokens = str(lever).strip().split()
    if not tokens:
        return "empty_decision_lever"
    if tokens[0].lower().strip(",.;:") not in LEVER_OPENERS:
        return "decision_lever_is_not_a_person_choice"
    return None


class SupplementConstructor(builder.CachedConstructor):
    """Round-one constructor under a request that names the decision lever."""

    def build(self, spec: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        input_text = render_construction_request(spec)
        request = {
            "prompt_version": CONSTRUCTION_PROMPT_VERSION,
            "model": builder.CONSTRUCTION_MODEL,
            "input": input_text,
        }
        request_hash = builder.hash_request(request)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["result"], request_hash
        if self.offline:
            raise RuntimeError(f"missing construction cache entry: {request_hash}")
        response = self._create(
            model=builder.CONSTRUCTION_MODEL,
            instructions=CONSTRUCTION_INSTRUCTIONS,
            input=input_text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parmbench_scenario_core",
                    "strict": True,
                    "schema": builder.CONSTRUCTION_SCHEMA,
                }
            },
            store=False,
        )
        result = json.loads(response.output_text)
        with self._lock:
            self.live_calls += 1
        builder.atomic_write_json(
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


def render_construction_request(spec: Mapping[str, Any]) -> str:
    request = (
        f"Personal fact: {spec['claim']}\n"
        f"The person's own words: {spec['evidence_span']}\n"
        f"What the fact could decide: {spec['decision_lever']}\n"
        f"Task domain: {spec['domain']}\n"
        f"Ordinary evidence mechanism: {spec['mechanism']}\n"
        f"Numeric ratings allowed: {'yes' if spec['ratings_allowed'] else 'no'}\n"
        f"Wording relationship: {spec['overlap_mode']}\n"
        f"Number of near-miss decoys: {spec['decoy_count']}\n"
        f"Document register: {spec['register']}\n"
    )
    if spec.get("repair_attempt"):
        request += f"Regeneration attempt: {spec['repair_attempt']}\n"
        request += builder.REPAIR_PREAMBLE
        for variant in spec.get("repair_failed_variants", ()):
            request += builder.REPAIR_NOTES[variant]
    return request


def lever_rejection(
    text: str,
    core: Mapping[str, Any],
    prompt: str,
    lever: str,
) -> str | None:
    """Reject a scenario that reuses the construction lever's own wording.

    The lever names the choice the personal fact could steer. It exists so the
    construction call can pick a decisive affordance. Repeating it in the
    document, the task, or the injected memory would hand a system under test
    the construction note itself.
    """

    needle = normalise(lever).rstrip(".")
    if not needle:
        return "empty_decision_lever"
    memory_text = str(core.get("memory_text", ""))
    if needle in normalise(text):
        return "decision_lever_leaks_into_observation"
    if needle in normalise(memory_text):
        return "decision_lever_leaks_into_memory_text"
    if needle in normalise(builder.ceiling_prompt(memory_text, prompt)):
        return "decision_lever_leaks_into_prompt"
    return None


def load_fairness_repairs() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if not FAIRNESS_REPAIRS_PATH.exists():
        return {}, {}
    payload = json.loads(FAIRNESS_REPAIRS_PATH.read_text(encoding="utf-8"))
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


def build_specs(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    repairs, _ = load_fairness_repairs()
    axis_rng = random.Random(AXIS_SEED)
    count = len(claims)
    domains = balanced_assignment(builder.DOMAINS, count, axis_rng)
    envelopes = balanced_assignment(ENVELOPE_NAMES, count, axis_rng)
    mechanisms = balanced_assignment(builder.MECHANISMS, count, axis_rng)
    overlaps = balanced_assignment(builder.OVERLAP_MODES, count, axis_rng)
    contracts = balanced_assignment(builder.ANSWER_CONTRACTS, count, axis_rng)
    ratings_pool = balanced_assignment(("none",) * 5 + ("numeric",), count, axis_rng)
    abstention_pool = balanced_assignment(
        ("plain",) * 3 + ("pressure",), count, axis_rng
    )

    specs: list[dict[str, Any]] = []
    for index, row in enumerate(claims):
        base_case_id = (
            f"parmbench-v1-p{row['persona_id']}-"
            f"{str(row['source_row_id']).split(':')[-1]}{SUPPLEMENT_SUFFIX}"
        )
        repair = repairs.get(base_case_id)
        repair_attempt = repair["attempt"] if repair else 0
        repair_failed_variants = repair["failed_variants"] if repair else ()
        seed_key = (
            f"{base_case_id}#repair{repair_attempt}"
            if repair_attempt
            else base_case_id
        )
        seed = int(builder.sha256_text(seed_key)[:16], 16)
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
                "decision_lever": row["draft"]["decision_lever"],
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


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_supplement(base_case_id: str) -> bool:
    return str(base_case_id).endswith(SUPPLEMENT_SUFFIX)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    builder.load_env(ROOT / ".env")
    encoding = tiktoken.get_encoding(builder.TOKENIZER)

    all_claims = builder.load_jsonl(SUPPLY_PATH)
    claims: list[dict[str, Any]] = []
    supply_rejected: list[dict[str, str]] = []
    for row in all_claims:
        reason = supply_rejection(
            row["draft"]["claim"], row["draft"]["decision_lever"]
        )
        if reason is None:
            claims.append(row)
        else:
            supply_rejected.append(
                {"source_row_id": row["source_row_id"], "reason": reason}
            )
    if args.limit:
        claims = claims[: args.limit]
    records_by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in builder.load_jsonl(builder.RECORDS_PATH):
        records_by_corpus[record["corpus_id"]].append(record)

    # Round one's rows, verbatim. Nothing below rewrites or re-validates them.
    kept_case_lines = [
        line
        for line in read_lines(DATASET_ROOT / "cases.jsonl")
        if not is_supplement(json.loads(line)["base_case_id"])
    ]
    kept_record_lines = [
        line
        for line in read_lines(DATASET_ROOT / "construction_records.jsonl")
        if not is_supplement(json.loads(line)["base_case_id"])
    ]
    kept_drop_lines = [
        line
        for line in read_lines(DATASET_ROOT / "dropped_scenarios.jsonl")
        if not is_supplement(json.loads(line)["base_case_id"])
    ]
    frozen_ids = {json.loads(line)["base_case_id"] for line in kept_record_lines}
    manifest = json.loads(
        (DATASET_ROOT / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    corpora = {entry["corpus_id"]: dict(entry) for entry in manifest["corpora"]}

    specs = build_specs(claims)
    collisions = sorted(
        spec["base_case_id"] for spec in specs if spec["base_case_id"] in frozen_ids
    )
    if collisions:
        raise ValueError(f"supplement ids collide with frozen scenarios: {collisions}")
    _, fairness_drops = load_fairness_repairs()
    constructor = SupplementConstructor(
        builder.CONSTRUCTION_CACHE, offline=args.offline
    )

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

    CONTEXT_ROOT.mkdir(parents=True, exist_ok=True)
    for stale in CONTEXT_ROOT.glob(f"*{SUPPLEMENT_SUFFIX}.md"):
        stale.unlink()

    cases: list[dict[str, Any]] = []
    construction_records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for spec in specs:
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
        core = builder.normalise_core(raw_core)
        neutral_repaired = False
        if not (
            builder.NEUTRAL_LENGTH_BAND[0]
            <= builder.neutral_length_ratio(core)
            <= builder.NEUTRAL_LENGTH_BAND[1]
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
                core = builder.normalise_core(raw_core)
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
            text, metrics = builder.assemble_observation(
                envelope,
                core,
                rng=rng,
                encoding=encoding,
                cue_fraction=spec["cue_fraction"],
                winner_fraction=spec["winner_fraction"],
                token_target=spec["token_target"],
            )
            reason = builder.scenario_rejection(text, core, prompt, row)
            if reason is None:
                reason = lever_rejection(text, core, prompt, spec["decision_lever"])
            if reason is None:
                break
        if reason is not None:
            dropped.append({"base_case_id": spec["base_case_id"], "reason": reason})
            continue

        cue = str(core["cue_clause"]).strip()
        neutral = str(core["neutral_clause"]).strip()
        content_path = f"contexts/{spec['base_case_id']}.md"
        (DATASET_ROOT / content_path).write_text(text, encoding="utf-8", newline="\n")

        corpus_id = row["corpus_id"]
        persona_id = row["persona_id"]
        corpora[corpus_id] = {
            "corpus_id": corpus_id,
            "source_root": builder.SOURCE_ROOT_RELATIVE,
            "source_id_prefix": f"notes/persona-{persona_id}/",
        }
        gold_source = {
            "source_id": row["gold_source_id"],
            "path": row["history_path"],
            "perturbations": [],
            "sha256": row["history_sha256"],
            "evidence_span": {"text": row["draft"]["evidence_span"]},
        }
        distractors = builder.choose_distractors(
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
            (DATASET_ROOT / content_path).unlink(missing_ok=True)
            continue

        capability = builder.capability_for(
            spec["claim"],
            row["draft"]["fact_kind"],
            bool(row["draft"].get("relational_hop")),
            spec["overlap_mode"],
        )
        memory_text = str(core["memory_text"]).strip()
        winner = str(core["winner_label"])
        target = str(core["target_label"])
        provenance = {
            "approved": True,
            "abstention_pressure": spec["abstention_pressure"],
            "builder_version": BUILDER_VERSION,
            "capability": capability,
            "construction_model": builder.CONSTRUCTION_MODEL,
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
            "claim_draft_prompt_version": row.get(
                "draft_prompt_version", "parmbench_claim_draft_v2"
            ),
            "construction_round": SUPPLEMENT_ROUND,
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
                "context_sha256": builder.sha256_text(text),
                "gold_source_hash": row["gold_source_hash"],
                "history_sha256": row["history_sha256"],
                "claim_draft_hash": row["draft_hash"],
            },
        }

        for variant in ("positive", "cue-ablated", "memory-included"):
            cue_present = variant != "cue-ablated"
            case_prompt = (
                builder.ceiling_prompt(memory_text, prompt)
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
                            [] if cue_present else [{"old": cue, "new": neutral}]
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
                "decision_lever": spec["decision_lever"],
                "evidence_span": row["draft"]["evidence_span"],
                "evidence_grade": row["grade"],
                "gold_source_id": row["gold_source_id"],
                "capability": capability,
                "abstention_pressure": spec["abstention_pressure"],
                "neutral_clause_repaired": neutral_repaired,
                "fairness_repair_attempt": spec["repair_attempt"],
                "construction_round": SUPPLEMENT_ROUND,
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
                    "construction_model": builder.CONSTRUCTION_MODEL,
                },
                "choices": {"output_only": winner, "memory_conditioned": target},
                "memory_text": memory_text,
                "control_replacement": {"old": cue, "new": neutral},
            }
        )

    def dump(rows: Sequence[Mapping[str, Any]]) -> list[str]:
        return [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]

    write_lines(DATASET_ROOT / "cases.jsonl", kept_case_lines + dump(cases))
    write_lines(
        DATASET_ROOT / "construction_records.jsonl",
        kept_record_lines + dump(construction_records),
    )
    write_lines(
        DATASET_ROOT / "dropped_scenarios.jsonl", kept_drop_lines + dump(dropped)
    )

    builder.atomic_write_json(
        SUPPLY_PATH.parent / "supplement_supply_rejections.json",
        {
            "gated_claims": len(all_claims),
            "rejected": sorted(
                supply_rejected, key=lambda item: item["source_row_id"]
            ),
        },
    )
    manifest["corpora"] = [corpora[key] for key in sorted(corpora)]
    manifest["supplement_builder_version"] = BUILDER_VERSION
    # The gate block describes a sweep over a different case list until the
    # fairness evaluator runs again over the merged set.
    manifest.pop("fairness_gate", None)
    (DATASET_ROOT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    summary = {
        "gated_claims": len(all_claims),
        "claims_rejected_before_construction": dict(
            sorted(Counter(item["reason"] for item in supply_rejected).items())
        ),
        "claims_built_from": len(claims),
        "frozen_scenarios": len(kept_record_lines),
        "supplement_scenarios": len(construction_records),
        "total_scenarios": len(kept_record_lines) + len(construction_records),
        "total_cases": len(kept_case_lines) + len(cases),
        "supplement_repaired": sum(
            1 for record in construction_records if record["fairness_repair_attempt"]
        ),
        "supplement_dropped": len(dropped),
        "dropped_reasons": dict(
            sorted(Counter(item["reason"].split(":")[0] for item in dropped).items())
        ),
        "live_construction_calls": constructor.live_calls,
        "supplement_capability": dict(
            sorted(Counter(r["capability"] for r in construction_records).items())
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.write_text(
        "".join(line + "\n" for line in lines), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    sys.exit(main())
