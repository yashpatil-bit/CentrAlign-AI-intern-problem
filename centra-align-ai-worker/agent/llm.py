"""LLM client abstraction and the OpenAI Responses API implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.errors import LLMError

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str  # raw JSON string exactly as produced by the model


@dataclass
class LLMTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    label: str

    def next_turn(self, instructions: str, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        ...


class OpenAIResponsesClient:
    """Calls ``client.responses.create`` with function tools and parses tool calls."""

    def __init__(self, api_key: str | None, model: str, timeout_s: float = 60.0) -> None:
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env, add your key, and restart the app "
                "(or choose the clearly labelled offline non-LLM demo mode)."
            )
        from openai import OpenAI  # imported lazily so tests without the SDK configured still run

        self._client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model = model
        self.label = f"OpenAI Responses API ({model})"

    def next_turn(self, instructions: str, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        import openai

        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_items,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                store=False,
            )
        except openai.AuthenticationError as exc:
            raise LLMError("The OpenAI API rejected the API key. Check OPENAI_API_KEY in .env.", str(exc)) from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"Model '{self.model}' was not found or is not available to your account. "
                           "Set OPENAI_MODEL in .env to a model that supports the Responses API.", str(exc)) from exc
        except openai.RateLimitError as exc:
            raise LLMError("OpenAI rate limit or quota exceeded. Check your plan/billing and retry.", str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not reach the OpenAI API. Check your internet connection or proxy.", str(exc)) from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"OpenAI API error (HTTP {exc.status_code}).", str(exc)) from exc

        turn = LLMTurn()
        texts: list[str] = []
        for item in response.output:
            if item.type == "function_call":
                turn.tool_calls.append(ToolCall(call_id=item.call_id, name=item.name, arguments=item.arguments))
            elif item.type == "message":
                for part in item.content:
                    if getattr(part, "type", "") == "output_text":
                        texts.append(part.text)
        turn.text = "\n".join(texts).strip()
        return turn
