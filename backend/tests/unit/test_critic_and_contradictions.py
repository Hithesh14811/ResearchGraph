from researchgraph.agents.critic import _merge, rule_based_issues
from researchgraph.schemas.evidence import Contradiction, Evidence, EvidenceType
from researchgraph.schemas.report import ResearchFinding, Synthesis
from researchgraph.schemas.research import ResearchPlan, ResearchSubquestion
from researchgraph.schemas.review import CritiqueIssue, IssueType, Severity
from researchgraph.schemas.sources import QualityTier, SourceQuality, SourceType
from researchgraph.services.contradictions import find_candidate_pairs


def _ev(eid: str, source_id: str, claim: str, sq: str = "SQ1") -> Evidence:
    return Evidence(
        id=eid,
        subquestion_id=sq,
        claim=claim,
        supporting_text=claim,
        source_id=source_id,
        source_url="https://e.org",
        source_title="t",
        passage_id="p",
        chunk_index=0,
        confidence=0.8,
        relevance=0.8,
        evidence_type=EvidenceType.EMPIRICAL_RESULT,
    )


def _q(sid: str, overall: float) -> SourceQuality:
    return SourceQuality(
        source_id=sid,
        source_type=SourceType.BLOG,
        authority=0,
        recency=0,
        relevance=0,
        primary_source=0,
        methodology=0,
        citation_signal=0,
        overall=overall,
        tier=QualityTier.LOW,
    )


PLAN = ResearchPlan(
    objective="o",
    subquestions=[
        ResearchSubquestion(id="SQ1", question="Does retrieval help accuracy?"),
        ResearchSubquestion(id="SQ2", question="What does retrieval cost?"),
    ],
)


def test_rule_based_critic_flags_weak_overconfident_uncovered_and_ignored_contradictions() -> None:
    evidence = {
        "E1": _ev("E1", "S1", "Retrieval always wins."),
        "E2": _ev("E2", "S2", "Retrieval improved accuracy by 9 points."),
    }
    synthesis = Synthesis(
        iteration=1,
        overall_assessment="x",
        findings=[
            ResearchFinding(
                id="F1",
                subquestion_id="SQ1",
                statement="Retrieval always wins.",
                evidence_ids=["E1"],
                confidence="high",
            ),
            ResearchFinding(
                id="F2",
                subquestion_id="SQ1",
                statement="Ghost finding",
                evidence_ids=["E404"],
                confidence="low",
            ),
        ],
    )
    contradiction = Contradiction(
        id="C1",
        subquestion_id="SQ1",
        evidence_ids=["E1", "E2"],
        description="conflict",
        severity="major",
    )
    issues = rule_based_issues(
        plan=PLAN,
        synthesis=synthesis,
        evidence=evidence,
        quality={"S1": _q("S1", 0.2), "S2": _q("S2", 0.8)},
        contradictions=[contradiction],
        credible_threshold=0.45,
    )
    kinds = {(i.issue_type, tuple(i.finding_ids) or i.subquestion_id) for i in issues}
    assert (IssueType.WEAK_SOURCE, ("F1",)) in kinds
    assert (IssueType.OVERCONFIDENCE, ("F1",)) in kinds
    assert (IssueType.UNSUPPORTED_CLAIM, ("F2",)) in kinds
    assert (IssueType.MISSING_COVERAGE, "SQ2") in kinds
    assert any(
        i.issue_type is IssueType.UNREPRESENTED_CONTRADICTION and i.severity is Severity.HIGH
        for i in issues
    )


def test_merge_keeps_most_severe_duplicate() -> None:
    """Regression: a duplicate rule issue must never downgrade an LLM-raised critical issue."""
    rule = CritiqueIssue(
        id="R1",
        issue_type=IssueType.MISSING_COVERAGE,
        severity=Severity.HIGH,
        description="rule",
        subquestion_id="SQ2",
        suggested_queries=["a"],
        origin="rule",
    )
    llm = CritiqueIssue(
        id="x",
        issue_type=IssueType.MISSING_COVERAGE,
        severity=Severity.CRITICAL,
        description="llm",
        subquestion_id="SQ2",
        suggested_queries=["b"],
        origin="llm",
    )
    merged = _merge([rule], [llm])
    assert len(merged) == 1
    assert merged[0].severity is Severity.CRITICAL and merged[0].id == "I1"
    assert merged[0].suggested_queries == ["b", "a"]


def test_candidate_pairs_require_different_sources_and_a_conflict_signal() -> None:
    evidence = [
        _ev("E1", "S1", "Fine-tuning always beats retrieval for domain question answering."),
        _ev(
            "E2",
            "S2",
            "Retrieval reached 71.4% accuracy on domain question answering versus 64.2% for fine-tuning.",
        ),
        _ev(
            "E3", "S2", "Fine-tuning always beats retrieval for domain question answering."
        ),  # same source as E2
        _ev("E4", "S3", "Unrelated statement about GPU memory bandwidth."),
        _ev(
            "E5", "S4", "Retrieval reached 71.4% accuracy on domain question answering.", sq="SQ2"
        ),  # other subquestion
    ]
    pairs = find_candidate_pairs(evidence, max_pairs=10)
    ids = {frozenset((p.first.id, p.second.id)) for p in pairs}
    assert frozenset(("E1", "E2")) in ids
    assert frozenset(("E2", "E3")) not in ids  # same source
    assert not any("E4" in pair for pair in ids)
    assert not any("E5" in pair for pair in ids)
    assert all(p.pair_id.startswith("PAIR") for p in pairs)


def test_candidate_pairs_are_deduplicated_across_subquestions() -> None:
    """Regression: one conflicting quote pair stored under three subquestions was judged
    (and reported) three times."""
    absolute = "Small models now match large models on all tasks for production inference."
    measured = (
        "Large models scored 11.2 points higher than small models on multi-step reasoning tasks."
    )
    evidence = [
        item
        for sq in ("SQ1", "SQ2", "SQ5")
        for item in (_ev(f"A-{sq}", "S1", absolute, sq=sq), _ev(f"B-{sq}", "S2", measured, sq=sq))
    ]
    pairs = find_candidate_pairs(evidence, max_pairs=10)
    assert len(pairs) == 1
    assert {pairs[0].first.source_id, pairs[0].second.source_id} == {"S1", "S2"}
