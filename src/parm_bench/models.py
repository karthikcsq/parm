from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI


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
