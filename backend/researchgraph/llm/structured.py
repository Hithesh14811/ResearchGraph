"""Reliable structured-output calls.

``invoke_structured`` wraps ``llm.with_structured_output(schema, include_raw=True)`` with:

* per-attempt timeouts and exponential-backoff retries for transient provider failures;
* *validation repair*: when the model returns arguments that fail Pydantic validation, the
  validation error is fed back and the model is asked to try again (bounded);
* typed failures (``LLMCallError`` / ``StructuredOutputError``) so callers can degrade
  gracefully instead of crashing the graph.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from researchgraph.core.errors import LLMCallError, StructuredOutputError
from researchgraph.core.retry import BackoffPolicy, retry_async
from researchgraph.core.text import truncate


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    attempts: int
    repaired: bool


@dataclass(frozen=True)
class LLMCallPolicy:
    timeout_seconds: float = 120.0
    backoff: BackoffPolicy = field(default_factory=BackoffPolicy)
    max_repairs: int = 2
    structured_method: str | None = None  # None = the integration's default


def json_mode_instruction(schema: type[BaseModel]) -> SystemMessage:
    """JSON mode guarantees syntactically valid JSON, not its shape: state the schema."""
    return SystemMessage(
        content=(
            f"Respond with a single JSON object for `{schema.__name__}` and nothing else: no prose, "
            "no Markdown fences. It must validate against this JSON Schema:\n"
            + json.dumps(schema.model_json_schema(), separators=(",", ":"))
        )
    )


def _describe_parse_failure(raw: Any, error: Any) -> str:
    if error is not None:
        return truncate(str(error), 1200)
    tool_calls = getattr(raw, "tool_calls", None)
    if not tool_calls:
        return "the response did not contain a structured tool call"
    return "the structured arguments were empty or invalid"


async def invoke_structured[T: BaseModel](
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    operation: str,
    policy: LLMCallPolicy,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    on_repair: Callable[[str], None] | None = None,
) -> StructuredResult[T]:
    options: dict[str, Any] = {"include_raw": True}
    if policy.structured_method:
        options["method"] = policy.structured_method
    runnable = llm.with_structured_output(schema, **options).with_config(
        run_name=operation, tags=["structured", operation]
    )
    json_mode = policy.structured_method == "json_mode"
    if json_mode:
        messages = [json_mode_instruction(schema), *messages]
    conversation = list(messages)
    attempts = 0
    last_problem = ""

    for repair_round in range(policy.max_repairs + 1):

        async def _call(convo: list[BaseMessage] = conversation) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            result = await asyncio.wait_for(runnable.ainvoke(convo), timeout=policy.timeout_seconds)
            return cast(dict[str, Any], result)

        try:
            output = await retry_async(_call, policy=policy.backoff, on_retry=on_retry)
        except Exception as exc:  # provider/network failure after retries
            raise LLMCallError(
                f"{operation} failed after {attempts} attempt(s): {type(exc).__name__}: {exc}"
            ) from exc

        parsed, error = output.get("parsed"), output.get("parsing_error")
        if isinstance(parsed, schema) and error is None:
            return StructuredResult(value=parsed, attempts=attempts, repaired=repair_round > 0)

        last_problem = _describe_parse_failure(output.get("raw"), error)
        if on_repair is not None and repair_round < policy.max_repairs:
            on_repair(last_problem)
        conversation = [
            *messages,
            HumanMessage(
                content=(
                    f"Your previous response could not be used: {last_problem}\n"
                    + (
                        f"Respond again with a JSON object for `{schema.__name__}`"
                        if json_mode
                        else f"Respond again by calling `{schema.__name__}` with arguments"
                    )
                    + " that satisfy the schema exactly (correct field names, types and allowed "
                    "enum values)."
                )
            ),
        ]

    raise StructuredOutputError(
        f"{operation}: no valid {schema.__name__} after {policy.max_repairs + 1} rounds ({last_problem})",
        schema=schema.__name__,
        attempts=attempts,
    )
