import pytest
from pydantic import ValidationError

from researchgraph.graph.serde import state_serializer, state_types
from researchgraph.graph.state import append_unique, merge_by_id, merge_sources
from researchgraph.schemas.api import CreateResearchRequest
from researchgraph.schemas.evidence import Contradiction
from researchgraph.schemas.research import PlanDecision, ResearchSubquestion, SearchQuery
from researchgraph.schemas.sources import Source


def _source(source_id: str, status: str = "processed", sq: str = "SQ1") -> Source:
    return Source(
        id=source_id,
        url=f"https://example.org/{source_id}",
        title="T",
        provider="p",
        status=status,
        subquestion_ids=[sq],
    )  # type: ignore[arg-type]


def test_domain_models_are_frozen() -> None:
    source = _source("S-1")
    with pytest.raises(ValidationError):
        source.title = "changed"  # type: ignore[misc]


def test_subquestion_id_format_and_query_normalisation() -> None:
    sq = ResearchSubquestion(
        id="SQ3",
        question="What is measured here?",
        search_queries=[SearchQuery(query="  rag   accuracy ")],
    )
    assert sq.search_queries[0].query == "rag accuracy"
    with pytest.raises(ValidationError):
        ResearchSubquestion(id="Q3", question="What is measured here?")


def test_contradiction_requires_exactly_two_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        Contradiction(id="C1", subquestion_id="SQ1", evidence_ids=["E-1"], description="x")


def test_plan_decision_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        PlanDecision.model_validate({"action": "delete"})


def test_create_request_sanitises_and_validates_question() -> None:
    request = CreateResearchRequest(question="  Compare\x00 RAG and fine-tuning for QA  ")
    assert request.question == "Compare RAG and fine-tuning for QA"
    with pytest.raises(ValidationError):
        CreateResearchRequest(question="short")
    with pytest.raises(ValidationError):
        CreateResearchRequest.model_validate(
            {"question": "A long enough question here", "unexpected": 1}
        )


def test_merge_sources_prefers_processed_and_unions_links() -> None:
    left = {"S-1": _source("S-1", status="failed", sq="SQ1")}
    right = {"S-1": _source("S-1", status="processed", sq="SQ2")}
    merged = merge_sources(left, right)
    assert merged["S-1"].status == "processed"
    assert merged["S-1"].subquestion_ids == ["SQ1", "SQ2"]
    # Commutative on status choice and idempotent.
    assert merge_sources(right, left)["S-1"].status == "processed"
    assert merge_sources(merged, merged) == merged


def test_merge_by_id_and_append_unique() -> None:
    assert merge_by_id({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_by_id(None, {"a": 1}) == {"a": 1}
    assert append_unique(["q1", "q2"], ["q2", "q3"]) == ["q1", "q2", "q3"]


def test_serde_allowlist_covers_schema_types_and_roundtrips() -> None:
    names = {t.__name__ for t in state_types()}
    assert {"Source", "Evidence", "ResearchPlan", "FinalReport", "RunStatus", "SourceType"} <= names
    serde = state_serializer()
    source = _source("S-9")
    restored = serde.loads_typed(serde.dumps_typed({"sources": {"S-9": source}}))
    assert restored["sources"]["S-9"] == source


def test_env_example_loads_and_blank_values_mean_unset() -> None:
    from pathlib import Path

    from researchgraph.config.settings import Settings

    example = Path(__file__).resolve().parents[3] / ".env.example"
    settings = Settings(_env_file=example)  # type: ignore[call-arg]
    assert settings.llm_provider == "mock"
    assert settings.api_token is None  # `API_TOKEN=` must not enable auth with an empty token
    assert settings.llm_input_cost_per_mtok is None and settings.checkpoint_url is None
