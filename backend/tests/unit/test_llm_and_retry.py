import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from researchgraph.core.errors import LLMCallError, StructuredOutputError
from researchgraph.core.retry import BackoffPolicy, is_transient_error, retry_async
from researchgraph.llm.mock import MockChatModel, MockRequest, MockResponse
from researchgraph.llm.structured import LLMCallPolicy, invoke_structured
from researchgraph.llm.usage import UsageTracker

FAST = BackoffPolicy(max_attempts=3, initial_delay=0.001, max_delay=0.002, jitter=False)
POLICY = LLMCallPolicy(timeout_seconds=5, backoff=FAST, max_repairs=2)


class Answer(BaseModel):
    value: int = Field(ge=0)
    label: str


class Scripted:
    """Responder that replays a script of outcomes (exceptions or argument dicts)."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[MockRequest] = []

    def __call__(self, request: MockRequest) -> MockResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return MockResponse(tool_calls=[("Answer", outcome)])  # type: ignore[list-item]


async def test_structured_output_uses_real_tool_calling_parser() -> None:
    responder = Scripted({"value": 3, "label": "ok"})
    result = await invoke_structured(
        MockChatModel(responder=responder),
        Answer,
        [HumanMessage("q")],
        operation="t",
        policy=POLICY,
    )
    assert (
        result.value == Answer(value=3, label="ok") and result.attempts == 1 and not result.repaired
    )
    assert responder.requests[0].structured_schema == "Answer"


async def test_invalid_output_is_repaired_with_validation_feedback() -> None:
    responder = Scripted({"value": -1, "label": "bad"}, {"value": 2, "label": "fixed"})
    repairs: list[str] = []
    result = await invoke_structured(
        MockChatModel(responder=responder),
        Answer,
        [HumanMessage("q")],
        operation="t",
        policy=POLICY,
        on_repair=repairs.append,
    )
    assert result.value.value == 2 and result.repaired and len(repairs) == 1
    assert "could not be used" in str(responder.requests[1].messages[-1].content)


async def test_persistently_invalid_output_raises_typed_error() -> None:
    responder = Scripted(*[{"value": -5, "label": "x"}] * 3)
    with pytest.raises(StructuredOutputError) as info:
        await invoke_structured(
            MockChatModel(responder=responder),
            Answer,
            [HumanMessage("q")],
            operation="t",
            policy=POLICY,
        )
    assert info.value.attempts == 3


async def test_transient_failures_are_retried_and_counted() -> None:
    responder = Scripted(TimeoutError("slow"), {"value": 1, "label": "ok"})
    tracker = UsageTracker()
    model = MockChatModel(responder=responder, callbacks=[tracker])
    result = await invoke_structured(
        model, Answer, [HumanMessage("q")], operation="t", policy=POLICY
    )
    assert result.attempts == 2
    snapshot = tracker.snapshot()
    assert snapshot.llm_calls == 1 and snapshot.llm_errors == 1 and snapshot.total_tokens > 0


async def test_non_transient_errors_fail_fast() -> None:
    responder = Scripted(KeyError("bug"), {"value": 1, "label": "never"})
    with pytest.raises(LLMCallError):
        await invoke_structured(
            MockChatModel(responder=responder),
            Answer,
            [HumanMessage("q")],
            operation="t",
            policy=POLICY,
        )
    assert len(responder.requests) == 1


async def test_retry_async_is_bounded_and_reports_attempts() -> None:
    calls: list[int] = []
    seen: list[int] = []

    async def always_times_out() -> None:
        calls.append(1)
        raise TimeoutError

    with pytest.raises(TimeoutError):
        await retry_async(
            always_times_out, policy=FAST, on_retry=lambda attempt, exc, delay: seen.append(attempt)
        )
    assert len(calls) == 3 and seen == [1, 2]


def test_transient_error_classification_and_backoff() -> None:
    import httpx

    assert is_transient_error(TimeoutError())
    assert is_transient_error(httpx.ConnectError("x"))

    class RateLimitError(Exception):
        status_code = 429

    assert is_transient_error(RateLimitError())
    assert not is_transient_error(ValueError("bad input"))
    policy = BackoffPolicy(initial_delay=1, multiplier=2, max_delay=5, jitter=False)
    assert [policy.delay_for(n) for n in (1, 2, 3, 4)] == [1, 2, 4, 5]


def test_structured_output_method_resolution() -> None:
    from researchgraph.config.settings import Settings
    from researchgraph.llm.factory import structured_output_method

    def method(**overrides: object) -> str | None:
        return structured_output_method(Settings(_env_file=None, **overrides))  # type: ignore[arg-type]

    assert method() is None  # mock model: BaseChatModel default
    assert (
        method(llm_provider="anthropic") == "json_schema"
    )  # newer Claude models reject forced tool calls
    assert method(llm_provider="openai") == "json_schema"
    assert method(llm_provider="ollama") is None  # integration default
    assert (
        method(llm_provider="anthropic", llm_structured_output_method="function_calling")
        == "function_calling"
    )


def test_openai_compatible_endpoints_use_base_url_and_json_mode() -> None:
    from researchgraph.config.settings import Settings
    from researchgraph.llm.factory import _provider_kwargs, structured_output_method

    compatible = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="openai",
        llm_base_url="https://api.deepseek.com",
        openai_api_key="sk-test",
    )
    assert _provider_kwargs(compatible)["base_url"] == "https://api.deepseek.com"
    assert structured_output_method(compatible) == "json_mode"
    pinned = compatible.model_copy(update={"llm_structured_output_method": "function_calling"})
    assert structured_output_method(pinned) == "function_calling"
    assert "base_url" not in _provider_kwargs(Settings(_env_file=None, llm_provider="openai"))  # type: ignore[call-arg]


async def test_json_mode_states_the_schema_and_parses_the_reply() -> None:
    """Drives the real ChatOpenAI client against a fake OpenAI-compatible endpoint."""
    import json

    import httpx
    from langchain_openai import ChatOpenAI

    requests: list[dict[str, object]] = []

    def endpoint(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = '{"value": 7, "label": "seven"}' if len(requests) > 1 else '{"value": -1}'
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "compatible-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    llm = ChatOpenAI(
        model="compatible-model",
        api_key="sk-test",  # type: ignore[arg-type]
        base_url="https://llm.example/v1",
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint)),
    )
    policy = LLMCallPolicy(
        timeout_seconds=5, backoff=FAST, max_repairs=2, structured_method="json_mode"
    )
    result = await invoke_structured(
        llm, Answer, [HumanMessage(content="q")], operation="t", policy=policy
    )

    assert result.value == Answer(value=7, label="seven") and result.repaired
    first = requests[0]
    assert first["response_format"] == {"type": "json_object"}
    system = first["messages"][0]  # type: ignore[index]
    assert system["role"] == "system" and '"label"' in system["content"]  # schema is stated
    assert "Respond again with a JSON object" in str(requests[1]["messages"])
