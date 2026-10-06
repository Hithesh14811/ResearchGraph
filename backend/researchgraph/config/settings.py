"""Application settings, loaded from environment variables (and an optional ``.env`` file).

Every tunable limit in the system lives here so that nothing is a magic number buried in
a node. Secrets are typed as ``SecretStr`` so they never appear in logs or reprs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SearchProviderName = Literal["mock", "auto", "tavily", "arxiv", "semantic_scholar"]
VectorStoreName = Literal["auto", "memory", "pgvector"]
ResearchAgentMode = Literal["llm", "deterministic"]


class Settings(BaseSettings):
    """Typed, validated configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # `KEY=` in .env means "unset" (an empty API_TOKEN must not enable auth with "").
        env_ignore_empty=True,
    )

    # --- Application -------------------------------------------------------------------
    app_name: str = "ResearchGraph"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # --- LLM provider (any provider supported by langchain's init_chat_model) -------------
    llm_provider: str = Field(
        default="mock",
        description="'mock' for the deterministic offline model, or any init_chat_model "
        "provider: openai, anthropic, google_genai, ollama, groq, mistralai, ...",
    )
    model_name: str = "mock-research-model"
    fast_model_name: str | None = Field(
        default=None,
        description="Optional cheaper model for high-volume steps (query writing, extraction).",
    )
    llm_temperature: float | None = 0.1
    llm_timeout_seconds: float = 180.0
    llm_max_retries: int = Field(default=3, ge=1, le=6)
    llm_structured_output_method: Literal[
        "auto", "function_calling", "json_schema", "json_mode"
    ] = Field(
        default="auto",
        description="auto = native JSON-schema output for anthropic/openai, JSON mode for "
        "OpenAI-compatible endpoints (LLM_BASE_URL), the integration's default elsewhere.",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible endpoint (DeepSeek, vLLM, Together, ...) used with "
        "LLM_PROVIDER=openai.",
    )
    llm_max_output_tokens: int = 8192
    llm_input_cost_per_mtok: float | None = Field(
        default=None, description="USD per 1M input tokens; enables cost estimates."
    )
    llm_output_cost_per_mtok: float | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Embeddings -----------------------------------------------------------------------
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=768, ge=64, le=4096)
    embedding_batch_size: int = Field(default=64, ge=1, le=2048)

    # --- Search / retrieval ---------------------------------------------------------------
    search_provider: SearchProviderName = "mock"
    tavily_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    search_results_per_query: int = Field(default=5, ge=1, le=10)
    research_agent_mode: ResearchAgentMode = Field(
        default="llm",
        description="'llm' lets the research worker choose tools; 'deterministic' executes "
        "the planned queries directly (cheaper, fully reproducible).",
    )
    max_search_rounds_per_task: int = Field(default=2, ge=1, le=4)
    max_sources_per_task: int = Field(default=4, ge=1, le=12)
    chunk_size: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    passages_per_task: int = Field(default=8, ge=2, le=30)
    max_chunks_per_source: int = Field(default=3, ge=1, le=10)
    max_extraction_context_chars: int = Field(default=14_000, ge=2_000, le=100_000)

    # --- Persistence ----------------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./data/researchgraph.db"
    checkpoint_url: str | None = Field(
        default=None,
        description="LangGraph checkpointer location. Defaults to a sibling of DATABASE_URL.",
    )
    vector_store: VectorStoreName = "auto"

    # --- Safety & budget limits -----------------------------------------------------------
    max_research_iterations: int = Field(default=3, ge=1, le=6)
    max_tool_calls: int = Field(default=60, ge=5, le=500)
    max_parallel_workers: int = Field(default=5, ge=1, le=16)
    max_active_runs: int = Field(
        default=4, ge=1, le=100, description="Concurrently executing runs per process."
    )
    max_subquestions: int = Field(default=6, ge=2, le=10)
    max_queries_per_subquestion: int = Field(default=3, ge=1, le=6)
    max_citation_repair_attempts: int = Field(default=2, ge=0, le=5)
    max_question_chars: int = Field(default=2000, ge=50, le=10_000)
    max_document_bytes: int = Field(default=5_000_000, ge=10_000)
    max_document_chars: int = Field(default=200_000, ge=1_000)
    max_pdf_pages: int = Field(default=40, ge=1, le=500)
    max_report_chars: int = Field(default=150_000, ge=5_000)
    fetch_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    max_redirects: int = Field(default=3, ge=0, le=10)
    graph_recursion_limit: int = Field(default=200, ge=50, le=2000)

    # --- Quality thresholds ----------------------------------------------------------------
    min_evidence_per_subquestion: int = Field(default=2, ge=1, le=10)
    credible_source_threshold: float = Field(default=0.45, ge=0, le=1)
    high_quality_source_threshold: float = Field(default=0.7, ge=0, le=1)
    quality_gate_threshold: float = Field(default=0.65, ge=0, le=1)
    citation_relevance_threshold: float = Field(default=0.18, ge=0, le=1)
    recency_half_life_years: float = Field(default=4.0, gt=0)

    # --- Observability (LangSmith) ---------------------------------------------------------
    langchain_tracing_v2: bool = False
    langchain_api_key: SecretStr | None = None
    langchain_project: str = "researchgraph"
    langchain_endpoint: str | None = None

    # --- API ---------------------------------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080"
    api_token: SecretStr | None = Field(
        default=None, description="If set, every API request must send 'Authorization: Bearer'."
    )
    resume_runs_on_startup: bool = True
    shutdown_drain_timeout_seconds: float = 20.0

    # --- Live mode gate (public deployments) ---------------------------------------------------
    public_demo: bool = Field(
        default=False,
        description="Public deployment: runs use the offline demo lane; real-provider runs need "
        "LIVE_MODE_PASSWORD (and are impossible without it).",
    )
    live_mode_password: SecretStr | None = Field(
        default=None,
        description="Unlocks live (real-provider) runs alongside the demo lane.",
    )
    live_token_ttl_hours: float = Field(default=12.0, gt=0, le=168)

    # --- Demo / failure simulation -------------------------------------------------------------
    demo_failures: str = Field(
        default="",
        description="Comma-separated failure scenarios for demo mode: failed_source, "
        "llm_timeout, invalid_output, insufficient_evidence, critic_rejection (or 'all').",
    )
    mock_llm_latency_ms: int = Field(default=0, ge=0, le=5000)

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: str) -> str:
        return str(value).strip().lower()

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_mock_llm(self) -> bool:
        return self.llm_provider == "mock"

    @property
    def live_gate_enabled(self) -> bool:
        """Demo and live lanes side by side: demo by default, live only when unlocked."""
        return not self.is_mock_llm and (self.public_demo or self.live_mode_password is not None)

    @property
    def live_mode_available(self) -> bool:
        return self.live_gate_enabled and self.live_mode_password is not None

    def demo_variant(self) -> Settings:
        """The offline demo lane: mock model over the synthetic corpus."""
        return self.model_copy(
            update={
                "llm_provider": "mock",
                "model_name": "mock-research-model",
                "fast_model_name": None,
                "search_provider": "mock",
            }
        )

    @property
    def uses_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def resolved_vector_store(self) -> Literal["memory", "pgvector"]:
        if self.vector_store == "auto":
            return "pgvector" if self.uses_postgres else "memory"
        return self.vector_store

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langchain_tracing_v2 and self.langchain_api_key)


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (read-only after construction)."""
    return Settings()
