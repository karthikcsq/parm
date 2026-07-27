from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .corpus import (
    NormalizedSourceRecord,
    SensitivityMetadata,
    UpdateStatus,
    text_sha256,
    validate_normalized_records,
)


PERSONAMEM_DATASET_ID = "bowen-upenn/PersonaMem-v2"
PERSONAMEM_SPLIT = "train_text"
PERSONAMEM_ADAPTER_VERSION = "personamem_v2_train_text_v1"
_HISTORY_TIMESTAMP = re.compile(
    r"chat_history_(?P<date>\d{6})_(?P<time>\d{6})_persona\d+\.json$"
)


@dataclass(frozen=True)
class PersonaMemSourceRow:
    source_row_id: str
    persona_id: int
    history_path: str
    preference: str
    related_conversation_snippet: tuple[dict[str, str], ...]
    user_query: dict[str, str]
    correct_answer: str
    incorrect_answers: tuple[str, ...]
    updated: bool
    previous_preference: str | None
    who: str
    sensitive_info: bool
    preference_type: str
    topic_preference: str

    @property
    def corpus_id(self) -> str:
        return f"personamem-v2/train/persona-{self.persona_id}"

    @property
    def forget_requested(self) -> bool:
        return (
            self.preference_type == "ask_to_forget"
            or self.preference.casefold().startswith("do not remember ")
        )

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        source_row_id: str,
    ) -> "PersonaMemSourceRow":
        required = {
            "persona_id",
            "chat_history_32k_link",
            "preference",
            "related_conversation_snippet",
            "user_query",
            "correct_answer",
            "incorrect_answers",
            "updated",
            "prev_pref",
            "who",
            "sensitive_info",
            "pref_type",
            "topic_preference",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"{source_row_id}: missing PersonaMem fields: {', '.join(missing)}"
            )
        snippet = _parse_messages(row["related_conversation_snippet"])
        if not snippet:
            raise ValueError(f"{source_row_id}: related snippet is empty")
        user_query = _parse_message(row["user_query"])
        incorrect_answers = _parse_string_list(row["incorrect_answers"])
        return cls(
            source_row_id=source_row_id,
            persona_id=int(row["persona_id"]),
            history_path=_non_empty(row["chat_history_32k_link"], "history path"),
            preference=_non_empty(row["preference"], "preference"),
            related_conversation_snippet=snippet,
            user_query=user_query,
            correct_answer=_non_empty(row["correct_answer"], "correct answer"),
            incorrect_answers=incorrect_answers,
            updated=bool(row["updated"]),
            previous_preference=(
                str(row["prev_pref"]).strip()
                if row["prev_pref"] is not None
                else None
            ),
            who=_non_empty(row["who"], "who"),
            sensitive_info=bool(row["sensitive_info"]),
            preference_type=_non_empty(row["pref_type"], "preference type"),
            topic_preference=_non_empty(
                row["topic_preference"], "preference topic"
            ),
        )


class PersonaMemV2Adapter:
    adapter_name = "personamem_v2"
    adapter_version = PERSONAMEM_ADAPTER_VERSION

    def __init__(self, dataset_revision: str):
        if not dataset_revision.strip():
            raise ValueError("dataset_revision must be non-empty")
        self.dataset_revision = dataset_revision

    def adapt(
        self,
        row: PersonaMemSourceRow,
        history: Mapping[str, Any],
        *,
        history_sha256: str,
    ) -> tuple[NormalizedSourceRecord, ...]:
        metadata = history.get("metadata")
        messages = history.get("chat_history")
        if not isinstance(metadata, Mapping) or not isinstance(messages, list):
            raise ValueError(f"{row.source_row_id}: invalid chat history payload")
        if int(metadata.get("persona_id", -1)) != row.persona_id:
            raise ValueError(f"{row.source_row_id}: chat history persona mismatch")
        normalized_messages = tuple(_normalize_history_message(item) for item in messages)
        gold_start = _find_unique_subsequence(
            normalized_messages, row.related_conversation_snippet
        )
        gold_end = gold_start + len(row.related_conversation_snippet) - 1
        timestamp = _history_timestamp(row.history_path)
        records: list[NormalizedSourceRecord] = []
        position = 0
        while position < len(normalized_messages):
            message = normalized_messages[position]
            if message["role"] == "system":
                position += 1
                continue
            if position == gold_start:
                segment = normalized_messages[gold_start : gold_end + 1]
                records.append(
                    self._record(
                        row,
                        segment,
                        start=gold_start,
                        end=gold_end,
                        timestamp=timestamp,
                        history_sha256=history_sha256,
                        is_gold=True,
                    )
                )
                position = gold_end + 1
                continue
            segment_start = position
            segment: list[dict[str, str]] = []
            while (
                position < len(normalized_messages)
                and position != gold_start
                and normalized_messages[position]["role"] != "system"
                and len(segment) < 4
            ):
                segment.append(normalized_messages[position])
                position += 1
            if not segment:
                position += 1
                continue
            records.append(
                self._record(
                    row,
                    tuple(segment),
                    start=segment_start,
                    end=position - 1,
                    timestamp=timestamp,
                    history_sha256=history_sha256,
                    is_gold=False,
                )
            )
        validate_normalized_records(records)
        return tuple(records)

    def _record(
        self,
        row: PersonaMemSourceRow,
        messages: Sequence[dict[str, str]],
        *,
        start: int,
        end: int,
        timestamp: str,
        history_sha256: str,
        is_gold: bool,
    ) -> NormalizedSourceRecord:
        text = "\n\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in messages
        )
        suffix = (
            f"turns-{start:04d}-{end:04d}"
            if start != end
            else f"turn-{start:04d}-{messages[0]['role']}"
        )
        source_id = f"notes/persona-{row.persona_id}/{suffix}"
        update_status = UpdateStatus.UNSPECIFIED
        sensitivity = SensitivityMetadata(False, "ordinary")
        who = "unspecified"
        annotations: dict[str, Any] = {}
        if is_gold:
            update_status = (
                UpdateStatus.FORGET_REQUESTED
                if row.forget_requested
                else UpdateStatus.CURRENT
            )
            handling = (
                "exclude_from_ordinary_positive"
                if row.sensitive_info or row.forget_requested
                else "ordinary"
            )
            sensitivity = SensitivityMetadata(
                row.sensitive_info,
                handling,
                (row.preference_type,) if row.sensitive_info else (),
            )
            who = row.who
            annotations = {
                "source_row_id": row.source_row_id,
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
        return NormalizedSourceRecord(
            corpus_id=row.corpus_id,
            source_id=source_id,
            timestamp=timestamp,
            title=(
                f"Persona {row.persona_id} source dialogue"
                if is_gold
                else f"Persona {row.persona_id} history turn {start}"
            ),
            text=text,
            provenance={
                "dataset_id": PERSONAMEM_DATASET_ID,
                "dataset_revision": self.dataset_revision,
                "split": PERSONAMEM_SPLIT,
                "persona_id": row.persona_id,
                "history_path": row.history_path,
                "history_sha256": history_sha256,
                "turn_start": start,
                "turn_end": end,
                "gold_related_snippet": is_gold,
                "related_snippet_sha256": (
                    text_sha256(text) if is_gold else None
                ),
            },
            update_status=update_status,
            sensitivity=sensitivity,
            who=who,
            source_hash=text_sha256(text),
            annotations=annotations,
        )


def _parse_messages(value: Any) -> tuple[dict[str, str], ...]:
    parsed = _parse_structured(value)
    if not isinstance(parsed, list):
        raise ValueError("related conversation snippet must be a list")
    return tuple(_normalize_history_message(item) for item in parsed)


def _parse_message(value: Any) -> dict[str, str]:
    parsed = _parse_structured(value)
    return _normalize_history_message(parsed)


def _parse_string_list(value: Any) -> tuple[str, ...]:
    parsed = _parse_structured(value)
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) or not item.strip() for item in parsed
    ):
        raise ValueError("incorrect_answers must be a string list")
    return tuple(item.strip() for item in parsed)


def _parse_structured(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("PersonaMem structured field is invalid") from exc


def _normalize_history_message(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("chat message must be an object")
    role = _non_empty(value.get("role"), "message role").casefold()
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"unsupported chat message role: {role!r}")
    return {
        "role": role,
        "content": _non_empty(value.get("content"), "message content"),
    }


def _find_unique_subsequence(
    messages: Sequence[dict[str, str]],
    target: Sequence[dict[str, str]],
) -> int:
    matches = [
        start
        for start in range(0, len(messages) - len(target) + 1)
        if tuple(messages[start : start + len(target)]) == tuple(target)
    ]
    if len(matches) != 1:
        raise ValueError(
            "related conversation snippet must occur exactly once in chat history"
        )
    return matches[0]


def _history_timestamp(history_path: str) -> str:
    match = _HISTORY_TIMESTAMP.search(Path(history_path).name)
    if match is None:
        raise ValueError(f"cannot parse history timestamp from {history_path!r}")
    parsed = datetime.strptime(
        match.group("date") + match.group("time"), "%y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def personamem_pilot_validation_issues(
    cases: Sequence[dict[str, Any]],
) -> list[tuple[str, str]]:
    if not cases:
        return [("<dataset>", "PersonaMem pilot is empty")]
    root = Path(cases[0].get("_dataset_root", "."))
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[tuple[str, str]] = []
    for key in ("source_manifest", "source_records", "pilot_sources"):
        artifact = manifest.get(key, {})
        path = (root / str(artifact.get("path", ""))).resolve()
        if not path.is_file():
            issues.append(("<dataset>", f"missing {key} artifact"))
        elif _sha256_file(path) != artifact.get("sha256"):
            issues.append(("<dataset>", f"{key} hash mismatch"))
    construction = manifest.get("construction_records", {})
    construction_path = (
        root / str(construction.get("path", ""))
    ).resolve()
    if not construction_path.is_file():
        issues.append(("<dataset>", "missing construction records"))
    elif _sha256_file(construction_path) != construction.get("sha256"):
        issues.append(("<dataset>", "construction records hash mismatch"))

    source_rows = {
        row["source_row_id"]: row
        for row in _read_jsonl(
            (root / manifest["pilot_sources"]["path"]).resolve()
        )
    }
    records = {
        (row["corpus_id"], row["source_id"]): row
        for row in _read_jsonl(
            (root / manifest["source_records"]["path"]).resolve()
        )
    }
    construction_rows = {
        row["base_case_id"]: row
        for row in _read_jsonl(construction_path)
    }
    fairness = manifest.get("fairness_gate", {})
    fairness_path = (root / str(fairness.get("path", ""))).resolve()
    predictions_path = (
        root / str(fairness.get("predictions_path", ""))
    ).resolve()
    if (
        fairness.get("evaluator_version") != "fixture_fairness_v1"
        or fairness.get("passed_cases") != 90
        or fairness.get("passed_scenarios") != 30
    ):
        issues.append(("<dataset>", "fairness gate is absent or incomplete"))
    for label, path, expected_hash in (
        ("fairness report", fairness_path, fairness.get("sha256")),
        (
            "fairness predictions",
            predictions_path,
            fairness.get("predictions_sha256"),
        ),
    ):
        if not path.is_file():
            issues.append(("<dataset>", f"missing {label}"))
        elif _sha256_file(path) != expected_hash:
            issues.append(("<dataset>", f"{label} hash mismatch"))
    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for case in cases:
        by_base.setdefault(case["base_case_id"], {})[case["variant"]] = case
    if len(by_base) != 30:
        issues.append(("<dataset>", "PersonaMem v0 must contain 30 scenarios"))
    if len({case["corpus_id"] for case in cases}) != 30:
        issues.append(("<dataset>", "PersonaMem v0 must use 30 persona corpora"))
    for base_case_id, variants in by_base.items():
        positive = variants.get("positive")
        control = variants.get("cue-ablated")
        ceiling = variants.get("memory-included")
        if positive is None or control is None or ceiling is None:
            continue
        source_row_id = positive.get("provenance", {}).get("source_row_id")
        source = source_rows.get(source_row_id)
        if source is None:
            issues.append((base_case_id, "source row is absent from pilot sources"))
            continue
        if (
            source["updated"]
            or source["sensitive_info"]
            or source["who"] != "self"
            or source["preference_type"] == "ask_to_forget"
        ):
            issues.append(
                (base_case_id, "ordinary pilot includes a restrained source row")
            )
        if positive["corpus_id"] != source["corpus_id"]:
            issues.append((base_case_id, "case corpus does not match source persona"))
        gold_key = (source["corpus_id"], source["gold_source_id"])
        gold_record = records.get(gold_key)
        if gold_record is None:
            issues.append((base_case_id, "gold normalized source is missing"))
        else:
            if gold_record["update_status"] != "current":
                issues.append((base_case_id, "gold source is not current"))
            if gold_record["sensitivity"]["handling"] != "ordinary":
                issues.append((base_case_id, "gold source is not ordinary-use"))
            if gold_record["annotations"].get("source_row_id") != source_row_id:
                issues.append((base_case_id, "gold source annotations mismatch"))
            if (
                gold_record["source_hash"] != source["gold_source_hash"]
                or gold_record["provenance"].get("history_sha256")
                != source["history_sha256"]
            ):
                issues.append((base_case_id, "gold source provenance mismatch"))
        prompt = positive["prompt"].casefold()
        preference = str(source["preference"]).casefold()
        query = str(source["user_query"].get("content", "")).casefold()
        if preference in prompt or query in prompt:
            issues.append((base_case_id, "source preference or query leaks into prompt"))
        if _content_word_overlap(positive["memory"]["text"], source["preference"]) < 2:
            issues.append((base_case_id, "ceiling memory is not source-anchored"))
        positive_lines = positive["observation_text"].splitlines()
        control_lines = control["observation_text"].splitlines()
        if len(positive_lines) != len(control_lines):
            issues.append((base_case_id, "control changes observation line count"))
        else:
            changed = [
                index
                for index, (left, right) in enumerate(
                    zip(positive_lines, control_lines)
                )
                if left != right
            ]
            if len(changed) != 1:
                issues.append(
                    (base_case_id, "control must change exactly one observation line")
                )
        cue_words = len(positive["cue"]["text"].split())
        replacement = control["observation"]["replacements"][0]["new"]
        replacement_words = len(replacement.split())
        ratio = replacement_words / max(cue_words, 1)
        if not 0.65 <= ratio <= 1.35:
            issues.append((base_case_id, "cue and control detail are not symmetric"))
        if positive["observation_text"] != ceiling["observation_text"]:
            issues.append((base_case_id, "ceiling observation differs from positive"))
        construction_record = construction_rows.get(base_case_id)
        if construction_record is None:
            issues.append((base_case_id, "construction record is missing"))
        elif (
            construction_record["acceptance"]["status"] != "accepted"
            or construction_record["acceptance"]["fairness_status"] != "passed"
            or construction_record["acceptance"][
                "fairness_evaluator_version"
            ]
            != "fixture_fairness_v1"
            or len(
                construction_record["acceptance"]["fairness_case_ids"]
            )
            != 3
            or construction_record["transformation"]["control_edits"] != 1
            or construction_record["raw_model_output_path"] is not None
            or construction_record["generation_model"] is not None
        ):
            issues.append((base_case_id, "construction record is incomplete"))
    return issues


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_word_overlap(left: str, right: str) -> int:
    ignored = {
        "about",
        "after",
        "before",
        "during",
        "enjoys",
        "often",
        "prefers",
        "regularly",
        "their",
        "there",
        "these",
        "those",
        "through",
        "user",
        "uses",
        "with",
    }
    left_words = {
        word
        for word in re.findall(r"[a-z0-9]+", left.casefold())
        if len(word) >= 4 and word not in ignored
    }
    right_words = {
        word
        for word in re.findall(r"[a-z0-9]+", right.casefold())
        if len(word) >= 4 and word not in ignored
    }
    return len(left_words & right_words)
