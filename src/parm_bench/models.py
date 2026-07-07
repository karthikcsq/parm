from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI


FINAL_ANSWER_INSTRUCTIONS = (
    "Follow the selection task using only the supplied observation. "
    "Return exactly one requested label or name and no explanation."
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
            max_output_tokens=512,
            store=False,
        )
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
