"""Provider-agnostic chat model construction.

The provider is selected purely by configuration (``LLM_PROVIDER`` / ``MODEL_NAME``) through
LangChain's ``init_chat_model``, so swapping OpenAI for Anthropic, Gemini, Ollama, etc. is an
environment change, not a code change. Two tiers let cheap, high-volume steps use a smaller
model than the reasoning-heavy steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.language_models import BaseChatModel

from researchgraph.config.settings import Settings
from researchgraph.core.errors import ConfigurationError
from researchgraph.core.faults import FaultInjector, NoFaults
from researchgraph.core.retry import BackoffPolicy
from researchgraph.llm.mock import MockChatModel, MockRequest
from researchgraph.llm.structured import LLMCallPolicy

_PROVIDER_PACKAGES = {
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "google_genai": "langchain-google-genai",
    "ollama": "langchain-ollama",
    "groq": "langchain-groq",
    "mistralai": "langchain-mistralai",
}


class ModelTier(StrEnum):
    STRONG = "strong"  # planning, synthesis, critique, report writing
    FAST = "fast"  # query writing, tool selection, evidence extraction


@dataclass(frozen=True)
class ModelRegistry:
    strong: BaseChatModel
    fast: BaseChatModel
    policy: LLMCallPolicy

    def get(self, tier: ModelTier) -> BaseChatModel:
        return self.strong if tier is ModelTier.STRONG else self.fast


def _provider_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "timeout": settings.llm_timeout_seconds,
        # Provider SDK retries handle Retry-After on 429s; our wrapper adds bounded
        # retries for timeouts and invalid output, so keep the SDK's count low.
        "max_retries": 1,
        "max_tokens": settings.llm_max_output_tokens,
    }
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    # pydantic-settings reads .env into Settings, not os.environ, so pass keys explicitly.
    if settings.llm_provider == "openai" and settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key.get_secret_value()
    if settings.llm_provider == "openai" and settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
    return kwargs


def _init_provider_model(settings: Settings, model_name: str) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    try:
        model = init_chat_model(
            model_name, model_provider=settings.llm_provider, **_provider_kwargs(settings)
        )
    except ImportError as exc:
        package = _PROVIDER_PACKAGES.get(
            settings.llm_provider, f"langchain-{settings.llm_provider}"
        )
        raise ConfigurationError(
            f"LLM_PROVIDER={settings.llm_provider!r} requires the '{package}' package "
            f"(pip install {package})"
        ) from exc
    except Exception as exc:
        raise ConfigurationError(f"Could not initialise model {model_name!r}: {exc}") from exc
    if not isinstance(model, BaseChatModel):
        raise ConfigurationError(f"init_chat_model returned unexpected type {type(model).__name__}")
    return model


# Native JSON-schema structured output where the integration supports it well (newer
# Claude models do not support forced tool calling); elsewhere keep the integration default.
# OpenAI-compatible endpoints often support neither JSON-schema output nor forced tool calls
# (e.g. reasoning models), but nearly all support JSON mode.
_NATIVE_STRUCTURED_OUTPUT = {"anthropic": "json_schema", "openai": "json_schema"}


def structured_output_method(settings: Settings) -> str | None:
    if settings.is_mock_llm:
        return None
    if settings.llm_structured_output_method != "auto":
        return settings.llm_structured_output_method
    if settings.llm_provider == "openai" and settings.llm_base_url:
        return "json_mode"
    return _NATIVE_STRUCTURED_OUTPUT.get(settings.llm_provider)


def call_policy(settings: Settings) -> LLMCallPolicy:
    return LLMCallPolicy(
        structured_method=structured_output_method(settings),
        timeout_seconds=settings.llm_timeout_seconds,
        backoff=BackoffPolicy(
            max_attempts=settings.llm_max_retries,
            initial_delay=0.05 if settings.is_mock_llm else 1.0,
            max_delay=0.2 if settings.is_mock_llm else 20.0,
        ),
        max_repairs=2,
    )


def build_model_registry(
    settings: Settings, *, faults: FaultInjector | None = None
) -> ModelRegistry:
    """Create the strong/fast model pair for the configured provider.

    In mock mode the models are bound to the run's fault injector, so demo failure
    scenarios (timeouts, malformed output, critic rejection) apply per run.
    """
    policy = call_policy(settings)
    if settings.is_mock_llm:
        from researchgraph.demo.responder import DemoResponder

        injector = faults or NoFaults()
        responder = DemoResponder(faults=injector)

        def before_call(request: MockRequest) -> None:
            schema = request.structured_schema or "tool_call"
            if injector.should_fail("llm_timeout", schema):
                raise TimeoutError(f"Simulated provider timeout while generating {schema}")

        def make(name: str) -> MockChatModel:
            return MockChatModel(
                responder=responder,
                model_name=name,
                latency_ms=settings.mock_llm_latency_ms,
                before_call=before_call,
            )

        mock_strong = make(settings.model_name)
        mock_fast = make(settings.fast_model_name) if settings.fast_model_name else mock_strong
        return ModelRegistry(strong=mock_strong, fast=mock_fast, policy=policy)

    strong = _init_provider_model(settings, settings.model_name)
    fast = (
        _init_provider_model(settings, settings.fast_model_name)
        if settings.fast_model_name and settings.fast_model_name != settings.model_name
        else strong
    )
    return ModelRegistry(strong=strong, fast=fast, policy=policy)
