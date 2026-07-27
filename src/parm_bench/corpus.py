from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence


class UpdateStatus(str, Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    FORGET_REQUESTED = "forget_requested"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class SensitivityMetadata:
    flagged: bool
    handling: str
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.handling.strip():
            raise ValueError("sensitivity handling must be non-empty")
        if any(not label.strip() for label in self.labels):
            raise ValueError("sensitivity labels must be non-empty")


@dataclass(frozen=True)
class NormalizedSourceRecord:
    corpus_id: str
    source_id: str
    timestamp: str
    title: str
    text: str
    provenance: dict[str, Any]
    update_status: UpdateStatus = UpdateStatus.UNSPECIFIED
    sensitivity: SensitivityMetadata = field(
        default_factory=lambda: SensitivityMetadata(False, "ordinary")
    )
    who: str = "unspecified"
    source_hash: str = ""
    perturbations: tuple[str, ...] = ()
    annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("corpus_id", self.corpus_id),
            ("source_id", self.source_id),
            ("timestamp", self.timestamp),
            ("title", self.title),
            ("text", self.text),
            ("who", self.who),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be non-empty")
        if not self.provenance:
            raise ValueError("provenance must be non-empty")
        if self.source_hash and self.source_hash != text_sha256(self.text):
            raise ValueError("source_hash does not match normalized text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "title": self.title,
            "text": self.text,
            "provenance": self.provenance,
            "update_status": self.update_status.value,
            "sensitivity": {
                "flagged": self.sensitivity.flagged,
                "handling": self.sensitivity.handling,
                "labels": list(self.sensitivity.labels),
            },
            "who": self.who,
            "source_hash": self.source_hash or text_sha256(self.text),
            "perturbations": list(self.perturbations),
            "annotations": self.annotations,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalizedSourceRecord":
        sensitivity = value.get("sensitivity", {})
        return cls(
            corpus_id=str(value["corpus_id"]),
            source_id=str(value["source_id"]),
            timestamp=str(value["timestamp"]),
            title=str(value["title"]),
            text=str(value["text"]),
            provenance=dict(value["provenance"]),
            update_status=UpdateStatus(value.get("update_status", "unspecified")),
            sensitivity=SensitivityMetadata(
                bool(sensitivity.get("flagged")),
                str(sensitivity.get("handling", "")),
                tuple(str(item) for item in sensitivity.get("labels", [])),
            ),
            who=str(value.get("who", "unspecified")),
            source_hash=str(value.get("source_hash", "")),
            perturbations=tuple(
                str(item) for item in value.get("perturbations", [])
            ),
            annotations=dict(value.get("annotations", {})),
        )


class SourceAdapter(Protocol):
    adapter_name: str
    adapter_version: str

    def adapt(self, *args: Any, **kwargs: Any) -> Sequence[NormalizedSourceRecord]:
        ...


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_normalized_records(records: Sequence[NormalizedSourceRecord]) -> None:
    if not records:
        raise ValueError("normalized corpus must contain at least one record")
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.corpus_id, record.source_id)
        if key in seen:
            raise ValueError(
                f"duplicate normalized source_id {record.source_id!r} "
                f"in corpus {record.corpus_id!r}"
            )
        seen.add(key)
        if record.source_hash and record.source_hash != text_sha256(record.text):
            raise ValueError(
                f"normalized source hash mismatch for {record.source_id!r}"
            )
