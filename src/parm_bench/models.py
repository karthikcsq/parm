from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI


RESPONSE_CACHE_SCHEMA = 1


FINAL_ANSWER_INSTRUCTIONS = (
    "Follow the selection task using only the supplied observation. "
    "Return exactly one requested label or name and no explanation."
)

MAX_OUTPUT_TOKENS = 4096


class ModelTruncationError(RuntimeError):
    """Raised when the Responses API stops before emitting a full answer.

    Reasoning models spend ``max_output_tokens`` on hidden reasoning plus the
    visible answer. When the budget runs out mid-response the API returns
    ``status="incomplete"`` and an empty ``output_text``. Surfacing that as an
    error keeps a truncated run from being silently scored as a wrong answer.
    """

    def __init__(self, reason: str, response_id: str) -> None:
        self.reason = reason
        self.response_id = response_id
        super().__init__(
            f"model response truncated before completion "
            f"(reason={reason!r}, response_id={response_id!r})"
        )


@dataclass(frozen=True)
class ModelResponse:
    text: str
    response_id: str
    resolved_model: str
    usage: dict[str, Any]


class LanguageModel(Protocol):
    model_name: str

    def generate(
        self,
        *,
        prompt: str,
        observation_kind: str,
        observation_text: str,
        instructions: str = FINAL_ANSWER_INSTRUCTIONS,
        memory_context: str | None = None,
    ) -> ModelResponse: ...


class OpenAIResponsesModel:
    def __init__(self, model_name: str, client: OpenAI | None = None):
        self.model_name = model_name
        self.client = client or OpenAI()

    def generate(
        self,
        *,
        prompt: str,
        observation_kind: str,
        observation_text: str,
        instructions: str = FINAL_ANSWER_INSTRUCTIONS,
        memory_context: str | None = None,
    ) -> ModelResponse:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=instructions,
            input=_render_input(
                prompt,
                observation_kind,
                observation_text,
                memory_context,
            ),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
        if getattr(response, "status", None) == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) or "unknown"
            raise ModelTruncationError(reason, response.id)
        return ModelResponse(
            text=response.output_text,
            response_id=response.id,
            resolved_model=response.model,
            usage=_usage_dict(response.usage),
        )


def _render_input(
    prompt: str,
    observation_kind: str,
    observation_text: str,
    memory_context: str | None = None,
) -> str:
    rendered = f"Task:\n{prompt}"
    if observation_text:
        label = observation_kind.replace("_", " ")
        rendered += f"\n\nObserved {label}:\n{observation_text}"
    if memory_context:
        rendered += f"\n\nRetrieved personal memory:\n{memory_context}"
    return rendered


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {
        key: value
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if (value := getattr(usage, key, None)) is not None
    }


class ResponsePolicy(str, Enum):
    POPULATE = "populate"
    FROZEN = "frozen"


class ResponseCacheMissError(RuntimeError):
    def __init__(self, request_hash: str) -> None:
        self.request_hash = request_hash
        super().__init__(f"frozen response cache miss for {request_hash}")


def _response_request_hash(
    model_name: str,
    prompt: str,
    observation_kind: str,
    observation_text: str,
    instructions: str,
    memory_context: str | None,
) -> str:
    payload = {
        "schema": RESPONSE_CACHE_SCHEMA,
        "model": model_name,
        "prompt": prompt,
        "observation_kind": observation_kind,
        "observation_text": observation_text,
        "instructions": instructions,
        "memory_context": memory_context or "",
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class CachingLanguageModel:
    """Freeze answer-model outputs so output-derived retrieval is reproducible.

    Enhanced retrieval expands a query cached by its text. When the retrieval
    query is the model's own output (``model_output_only`` and
    ``tool_then_model_output``) it is nondeterministic, so a frozen expansion
    cache can never hit and a populate run rewrites a fresh entry every time.
    Freezing the model response makes the derived query deterministic, which
    lets the expansion cache -- and the whole run -- replay exactly.

    Keyed by a hash of the full request (model, prompt, observation, instructions,
    and injected memory), so the tool-phase retrieval that feeds an intermediate
    response is captured too and the ``tool_then_model_output`` chain resolves
    deterministically.
    """

    def __init__(
        self,
        base: LanguageModel,
        cache_dir: str | Path,
        policy: ResponsePolicy | str = ResponsePolicy.FROZEN,
    ) -> None:
        self.base = base
        self.cache_dir = Path(cache_dir)
        self.policy = ResponsePolicy(policy)
        self._used_cache_files: list[Path] = []

    @property
    def model_name(self) -> str:
        return self.base.model_name

    def generate(
        self,
        *,
        prompt: str,
        observation_kind: str,
        observation_text: str,
        instructions: str = FINAL_ANSWER_INSTRUCTIONS,
        memory_context: str | None = None,
    ) -> ModelResponse:
        request_hash = _response_request_hash(
            self.base.model_name,
            prompt,
            observation_kind,
            observation_text,
            instructions,
            memory_context,
        )
        path = self.cache_dir / f"{request_hash}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("request_hash") != request_hash
                or payload.get("model") != self.base.model_name
            ):
                raise RuntimeError(f"invalid response cache entry: {path}")
            self._used_cache_files.append(path)
            return ModelResponse(
                text=payload["text"],
                response_id=payload["response_id"],
                resolved_model=payload["resolved_model"],
                usage=payload.get("usage", {}),
            )
        if self.policy is ResponsePolicy.FROZEN:
            raise ResponseCacheMissError(request_hash)
        response = self.base.generate(
            prompt=prompt,
            observation_kind=observation_kind,
            observation_text=observation_text,
            instructions=instructions,
            memory_context=memory_context,
        )
        _atomic_write_json(
            path,
            {
                "schema": RESPONSE_CACHE_SCHEMA,
                "request_hash": request_hash,
                "model": self.base.model_name,
                "text": response.text,
                "response_id": response.response_id,
                "resolved_model": response.resolved_model,
                "usage": response.usage,
            },
        )
        self._used_cache_files.append(path)
        return response

    @property
    def cache_hash(self) -> str | None:
        if not self._used_cache_files:
            return None
        digest = hashlib.sha256()
        for path in sorted(set(self._used_cache_files)):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()
