from datetime import date

from researchgraph.schemas.evidence import Evidence, EvidenceType
from researchgraph.schemas.report import ResearchFinding, Synthesis
from researchgraph.schemas.research import ResearchPlan, ResearchSubquestion, RunLimits, SearchQuery
from researchgraph.schemas.review import Critique, CritiqueIssue, IssueType, Severity
from researchgraph.schemas.sources import (
    ContentSignals,
    QualityTier,
    Source,
    SourceQuality,
    SourceType,
)
from researchgraph.services.coverage import assess_coverage
from researchgraph.services.quality_gate import assess_quality, budget_remaining
from researchgraph.services.source_quality import classify_source, recency_score, score_source

TODAY = date(2026, 10, 1)
LIMITS = RunLimits(
    max_iterations=3,
    max_tool_calls=60,
    max_citation_repairs=2,
    min_evidence_per_subquestion=2,
    quality_threshold=0.65,
    credible_source_threshold=0.45,
)


def _source(
    sid: str,
    url: str,
    *,
    hint: SourceType | None = None,
    published: date | None = None,
    signals: ContentSignals | None = None,
) -> Source:
    return Source(
        id=sid,
        url=url,
        title="Retrieval accuracy study",
        provider="test",
        type_hint=hint,
        published_date=published,
        signals=signals,
        status="processed",
        subquestion_ids=["SQ1"],
    )


def _score(source: Source, evidence: list[Evidence] | None = None) -> SourceQuality:
    return score_source(
        source,
        subquestion_texts=["retrieval accuracy"],
        evidence=evidence or [],
        today=TODAY,
        half_life_years=4,
        high_threshold=0.7,
        credible_threshold=0.45,
    )


def test_classification_uses_provider_hint_then_domain() -> None:
    assert (
        classify_source(_source("S1", "https://anything.example/x", hint=SourceType.FORUM))[0]
        is SourceType.FORUM
    )
    assert (
        classify_source(_source("S2", "https://arxiv.org/abs/1"))[0] is SourceType.PRIMARY_RESEARCH
    )
    assert (
        classify_source(_source("S3", "https://docs.python.org/3/"))[0]
        is SourceType.OFFICIAL_DOCUMENTATION
    )
    assert (
        classify_source(_source("S4", "https://cs.stanford.edu/paper"))[0]
        is SourceType.INSTITUTIONAL
    )
    assert classify_source(_source("S5", "https://www.reddit.com/r/ml"))[0] is SourceType.FORUM
    assert classify_source(_source("S6", "https://someone.medium.com/post"))[0] is SourceType.BLOG
    assert classify_source(_source("S7", "https://unknown-site.io/page"))[0] is SourceType.UNKNOWN


def test_recency_decays_with_age() -> None:
    fresh, _ = recency_score(date(2026, 6, 1), today=TODAY, half_life_years=4)
    old, _ = recency_score(date(2014, 6, 1), today=TODAY, half_life_years=4)
    unknown, reason = recency_score(None, today=TODAY, half_life_years=4)
    assert fresh > 0.9 > old and unknown == 0.4 and "unknown" in reason


def test_primary_research_outscores_forum_and_failed_sources_are_penalised() -> None:
    signals = ContentSignals(
        word_count=3000,
        reference_count=40,
        methodology_terms=30,
        numeric_density=3,
        has_abstract=True,
    )
    paper = _score(
        _source("S1", "https://arxiv.org/abs/1", published=date(2025, 1, 1), signals=signals)
    )
    forum = _score(_source("S2", "https://reddit.com/r/x", published=date(2025, 1, 1)))
    assert paper.overall > forum.overall
    assert paper.tier in (QualityTier.HIGH, QualityTier.MEDIUM) and forum.tier is QualityTier.LOW
    failed = _score(
        _source(
            "S1", "https://arxiv.org/abs/1", published=date(2025, 1, 1), signals=signals
        ).model_copy(update={"status": "failed"})
    )
    assert failed.overall < paper.overall
    assert any("Classified as primary research" in r for r in paper.rationale)


def _evidence(
    eid: str,
    source_id: str,
    sq: str = "SQ1",
    claim: str = "Retrieval improved accuracy by 9 points.",
) -> Evidence:
    return Evidence(
        id=eid,
        subquestion_id=sq,
        claim=claim,
        supporting_text=claim,
        source_id=source_id,
        source_url=f"https://e.org/{source_id}",
        source_title="t",
        passage_id=f"{source_id}:0",
        chunk_index=0,
        confidence=0.9,
        relevance=0.9,
        evidence_type=EvidenceType.EMPIRICAL_RESULT,
    )


def _quality(sid: str, overall: float) -> SourceQuality:
    return SourceQuality(
        source_id=sid,
        source_type=SourceType.PRIMARY_RESEARCH,
        authority=overall,
        recency=overall,
        relevance=overall,
        primary_source=overall,
        methodology=overall,
        citation_signal=overall,
        overall=overall,
        tier=QualityTier.HIGH if overall >= 0.7 else QualityTier.LOW,
    )


PLAN = ResearchPlan(
    objective="o",
    subquestions=[
        ResearchSubquestion(
            id="SQ1",
            question="How accurate is retrieval?",
            information_requirements=["accuracy"],
            search_queries=[SearchQuery(query="retrieval accuracy")],
        ),
        ResearchSubquestion(
            id="SQ2",
            question="What does retrieval cost?",
            search_queries=[SearchQuery(query="retrieval cost")],
        ),
    ],
)


def test_coverage_requires_two_credible_sources() -> None:
    evidence = {
        "E1": _evidence("E1", "S1"),
        "E2": _evidence("E2", "S1", claim="Another claim from the same source."),
        "E3": _evidence("E3", "S2", sq="SQ2"),
        "E4": _evidence("E4", "S3", sq="SQ2"),
    }
    quality = {"S1": _quality("S1", 0.8), "S2": _quality("S2", 0.8), "S3": _quality("S3", 0.8)}
    assessment = assess_coverage(PLAN, evidence, quality, iteration=1, limits=LIMITS)
    by_sq = {c.subquestion_id: c for c in assessment.coverage}
    assert not by_sq["SQ1"].sufficient  # two items but one source
    assert by_sq["SQ2"].sufficient
    assert [g.subquestion_id for g in assessment.gaps] == ["SQ1"]
    assert all(
        "sourc " not in q for q in assessment.gaps[0].suggested_queries
    )  # unstemmed keywords


def _synthesis(*findings: ResearchFinding) -> Synthesis:
    return Synthesis(iteration=1, findings=list(findings), overall_assessment="x")


def _finding(fid: str, sq: str, evidence_ids: list[str]) -> ResearchFinding:
    return ResearchFinding(
        id=fid, subquestion_id=sq, statement="s", evidence_ids=evidence_ids, confidence="moderate"
    )


EVIDENCE = {"E1": _evidence("E1", "S1"), "E2": _evidence("E2", "S2", sq="SQ2")}
QUALITY = {"S1": _quality("S1", 0.8), "S2": _quality("S2", 0.8)}


def _assess(
    critique: Critique, *, iteration: int = 1, findings: tuple[ResearchFinding, ...] | None = None
):  # type: ignore[no-untyped-def]
    findings = findings or (_finding("F1", "SQ1", ["E1"]), _finding("F2", "SQ2", ["E2"]))
    return assess_quality(
        plan=PLAN,
        synthesis=_synthesis(*findings),
        critique=critique,
        evidence=EVIDENCE,
        quality=QUALITY,
        contradictions=[],
        iteration=iteration,
        tool_calls_used=10,
        limits=LIMITS,
    )


def test_quality_gate_passes_clean_research() -> None:
    result = _assess(Critique(iteration=1, issues=[], verdict="acceptable"))
    assert result.passed and result.decision == "proceed" and result.score >= 0.65


def test_critical_issue_blocks_and_triggers_targeted_research() -> None:
    issue = CritiqueIssue(
        id="I1",
        issue_type=IssueType.MISSING_COVERAGE,
        severity=Severity.CRITICAL,
        description="gap",
        subquestion_id="SQ2",
        suggested_queries=["retrieval cost study"],
    )
    result = _assess(Critique(iteration=1, issues=[issue], verdict="major_problems"))
    assert not result.passed and result.decision == "research_more"
    assert [g.subquestion_id for g in result.gaps] == ["SQ2"]
    assert result.gaps[0].suggested_queries == ["retrieval cost study"]


def test_uncovered_subquestion_blocks_and_budget_exhaustion_proceeds_with_limitations() -> None:
    critique = Critique(iteration=3, issues=[], verdict="acceptable")
    result = _assess(critique, iteration=3, findings=(_finding("F1", "SQ1", ["E1"]),))
    assert not result.passed
    assert result.decision == "proceed_with_limitations"  # iteration == max_iterations
    assert any("SQ2" in b for b in result.blocking_issues)


def test_budget_remaining() -> None:
    assert budget_remaining(iteration=1, tool_calls_used=10, limits=LIMITS)
    assert not budget_remaining(iteration=3, tool_calls_used=10, limits=LIMITS)
    assert not budget_remaining(iteration=1, tool_calls_used=59, limits=LIMITS)


def test_literature_surveys_score_as_secondary_sources() -> None:
    """Regression (found in the first live run): arXiv surveys scored as primary research."""
    study = _source("S1", "https://arxiv.org/abs/1", published=date(2025, 1, 1))
    survey = study.model_copy(
        update={"title": "Mitigating Hallucination in LLMs: A Survey on RAG and Agentic Systems"}
    )
    study_q, survey_q = _score(study), _score(survey)
    assert survey_q.primary_source < study_q.primary_source == 1.0
    assert survey_q.overall < study_q.overall
    assert any("secondary source" in r for r in survey_q.rationale)
    for title in ("Peer review of code changes at scale", "Reviewing retrieval accuracy"):
        assert _score(study.model_copy(update={"title": title})).primary_source == 1.0
