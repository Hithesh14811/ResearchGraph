"""A deterministic, offline chat model that speaks LangChain's tool-calling protocol.

``MockChatModel`` implements ``bind_tools`` and returns ``AIMessage.tool_calls`` — so the
*real* ``BaseChatModel.with_structured_output`` → ``PydanticToolsParser`` path runs, including
Pydantic validation failures. Nothing in the graph knows it is talking to a mock.

What the model "says" is delegated to a ``Responder`` (see ``researchgraph.demo.responder``),
which reads the structured context blocks in the prompt and produces grounded output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict


@dataclass
class MockRequest:
    messages: list[BaseMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None

    @property
    def structured_schema(self) -> str | None:
        """Name of the schema when this is a forced structured-output call."""
        if self.tool_choice in ("any", "required") and len(self.tools) == 1:
            return str(self.tools[0]["function"]["name"])
        if isinstance(self.tool_choice, str) and self.tool_choice not in ("auto", "none"):
            return self.tool_choice
        return None

    @property
    def prompt_text(self) -> str:
        """Concatenated human-authored content (where the structured context blocks live)."""
        return "\n".join(str(m.content) for m in self.messages if isinstance(m, HumanMessage))

    @property
    def tool_messages(self) -> list[ToolMessage]:
        return [m for m in self.messages if isinstance(m, ToolMessage)]

    @property
    def is_repair(self) -> bool:
        """True when the caller is asking for a corrected structured response."""
        last = self.messages[-1] if self.messages else None
        return isinstance(last, HumanMessage) and "could not be used" in str(last.content)


@dataclass
class MockResponse:
    content: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


Responder = Callable[[MockRequest], MockResponse]


def _estimate_tokens(chars: int) -> int:
    """Rough token estimate (~4 characters per token) so usage tracking works offline."""
    return max(1, chars // 4)


class MockChatModel(BaseChatModel):
    """Deterministic chat model for demo mode, evaluation smoke tests and the test suite."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Any
    model_name: str = "mock-research-model"
    latency_ms: int = 0
    before_call: Any = None  # optional hook(request) that may raise to simulate outages

    @property
    def _llm_type(self) -> str:
        return "researchgraph-mock"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)

    # -- generation ---------------------------------------------------------------------
    def _respond(self, messages: list[BaseMessage], kwargs: dict[str, Any]) -> ChatResult:
        request = MockRequest(
            messages=messages,
            tools=list(kwargs.get("tools") or []),
            tool_choice=kwargs.get("tool_choice"),
        )
        if self.before_call is not None:
            self.before_call(request)
        response: MockResponse = self.responder(request)

        tool_calls = []
        for index, (name, args) in enumerate(response.tool_calls):
            digest = hashlib.sha256(
                f"{name}:{index}:{json.dumps(args, sort_keys=True, default=str)}".encode()
            ).hexdigest()
            tool_calls.append(
                {"name": name, "args": args, "id": f"call_{digest[:16]}", "type": "tool_call"}
            )

        prompt_chars = sum(len(str(m.content)) for m in messages)
        completion_chars = len(response.content) + sum(
            len(json.dumps(a, default=str)) for _, a in response.tool_calls
        )
        input_tokens, output_tokens = (
            _estimate_tokens(prompt_chars),
            _estimate_tokens(completion_chars),
        )
        message = AIMessage(
            content=response.content,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._respond(messages, kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        return self._respond(messages, kwargs)
