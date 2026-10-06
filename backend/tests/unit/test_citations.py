from researchgraph.schemas.evidence import Evidence, EvidenceType
from researchgraph.schemas.report import ReportClaim, ReportDraft, ReportSection
from researchgraph.schemas.sources import Source
from researchgraph.services.citations import (
    build_bibliography,
    check_claim,
    citation_marker,
    dedupe_claims,
    repair_citations,
    verify_citations,
)
from researchgraph.services.report_renderer import render_markdown

THRESHOLD = 0.18


def _ev(eid: str, source_id: str, claim: str) -> Evidence:
    return Evidence(
        id=eid,
        subquestion_id="SQ1",
        claim=claim,
        supporting_text=claim,
        source_id=source_id,
        source_url=f"https://e.org/{source_id}",
        source_title=f"Title {source_id}",
        passage_id="p",
        chunk_index=0,
        confidence=0.9,
        relevance=0.9,
        evidence_type=EvidenceType.EMPIRICAL_RESULT,
    )


EVIDENCE = {
    "E1": _ev(
        "E1",
        "S1",
        "Retrieval-augmented generation reached 71.4% exact-match accuracy on domain QA.",
    ),
    "E2": _ev("E2", "S2", "Fine-tuning improved terminology accuracy from 61% to 90%."),
    "E3": _ev("E3", "S1", "Long-context prompts cost 12 times more per query than retrieval."),
}
SOURCES = {
    sid: Source(
        id=sid, url=f"https://e.org/{sid}", title=f"Title {sid}", provider="t", status="processed"
    )
    for sid in ("S1", "S2")
}


def _claim(cid: str, text: str, ids: list[str]) -> ReportClaim:
    return ReportClaim(id=cid, text=text, evidence_ids=ids)


def test_check_claim_detects_each_problem_type() -> None:
    ok = _claim("C1", "Retrieval-augmented generation reached 71.4% exact-match accuracy.", ["E1"])
    assert check_claim(ok, EVIDENCE, relevance_threshold=THRESHOLD) is None
    assert (
        check_claim(
            _claim("C2", "An uncited claim.", []), EVIDENCE, relevance_threshold=THRESHOLD
        ).problem
        == "missing_citation"
    )  # type: ignore[union-attr]
    assert (
        check_claim(_claim("C3", "x", ["E404"]), EVIDENCE, relevance_threshold=THRESHOLD).problem
        == "unknown_evidence"
    )  # type: ignore[union-attr]
    unrelated = _claim("C4", "Bananas are rich in potassium and vitamins.", ["E1"])
    assert (
        check_claim(unrelated, EVIDENCE, relevance_threshold=THRESHOLD).problem == "low_relevance"
    )  # type: ignore[union-attr]
    wrong_number = _claim(
        "C5",
        "Retrieval-augmented generation reached 85.0% exact-match accuracy on domain QA.",
        ["E1"],
    )
    assert (
        check_claim(wrong_number, EVIDENCE, relevance_threshold=THRESHOLD).problem
        == "numeric_mismatch"
    )  # type: ignore[union-attr]


def _draft() -> ReportDraft:
    return ReportDraft(
        title="Report",
        executive_summary=[
            _claim(
                "C1",
                "Retrieval-augmented generation reached 71.4% exact-match accuracy on domain QA.",
                ["E1"],
            )
        ],
        sections=[
            ReportSection(
                heading="Findings",
                paragraphs=[
                    [
                        _claim(
                            "C2",
                            "Fine-tuning improved terminology accuracy from 61% to 90%.",
                            ["E2"],
                        ),
                        _claim(
                            "C3",
                            "Long-context prompts cost 12 times more per query than retrieval.",
                            [],
                        ),  # repairable
                        _claim(
                            "C4", "Quantum computers will replace all GPUs by next year.", []
                        ),  # unsupported
                    ]
                ],
            )
        ],
        recommendations=[],
    )


def test_verify_then_repair_reattributes_or_removes() -> None:
    draft = _draft()
    check = verify_citations(draft, EVIDENCE, attempt=0, relevance_threshold=THRESHOLD)
    assert check.total_claims == 4 and check.cited_claims == 2 and len(check.issues) == 2
    outcome = repair_citations(draft, check, EVIDENCE, relevance_threshold=THRESHOLD)
    assert outcome.reattributed == 1 and outcome.removed == 1
    repaired = {c.id: c for c in outcome.draft.all_claims()}
    assert repaired["C3"].evidence_ids == ["E3"]
    assert "C4" not in repaired and outcome.draft.removed_claims[0].id == "C4"
    assert verify_citations(
        outcome.draft, EVIDENCE, attempt=1, relevance_threshold=THRESHOLD
    ).passed


def test_bibliography_numbers_sources_by_first_appearance_and_renders_markers() -> None:
    draft = _draft()
    numbers, bibliography = build_bibliography(draft, EVIDENCE, SOURCES, {})
    assert numbers == {"S1": 1, "S2": 2}
    assert [c.source_id for c in bibliography] == ["S1", "S2"] and bibliography[0].evidence_ids == [
        "E1"
    ]
    marker = citation_marker(_claim("C9", "x", ["E2", "E1", "E3"]), EVIDENCE, numbers)
    assert marker == "[1][2]"


def test_rendered_markdown_contains_citations_and_only_known_urls() -> None:
    from researchgraph.schemas.report import ReportMetrics

    draft = repair_citations(
        _draft(),
        verify_citations(_draft(), EVIDENCE, attempt=0, relevance_threshold=THRESHOLD),
        EVIDENCE,
        relevance_threshold=THRESHOLD,
    ).draft
    numbers, bibliography = build_bibliography(draft, EVIDENCE, SOURCES, {})
    metrics = ReportMetrics(
        total_claims=3,
        cited_claims=3,
        citation_coverage=1,
        unsupported_claim_rate=0.25,
        removed_claims=1,
        sources_cited=2,
        mean_cited_source_quality=0,
        evidence_items=3,
        research_iterations=1,
    )
    md = render_markdown(
        question="Q?",
        draft=draft,
        evidence=EVIDENCE,
        numbers=numbers,
        bibliography=bibliography,
        contradictions=[],
        quality=None,
        metrics=metrics,
    )
    assert "domain QA [1]." in md and "## References" in md and "~~Quantum computers" in md
    urls = {
        line.split("<")[1].split(">")[0]
        for line in md.splitlines()
        if line.startswith("[") and "<http" in line
    }
    assert urls == {s.url for s in SOURCES.values()}


def test_dedupe_claims_removes_repeated_sentences() -> None:
    claim = _claim(
        "C1",
        "Retrieval-augmented generation reached 71.4% exact-match accuracy on domain QA.",
        ["E1"],
    )
    draft = ReportDraft(
        title="t",
        executive_summary=[claim],
        recommendations=[],
        sections=[
            ReportSection(heading="A", paragraphs=[[claim.model_copy(update={"id": "C2"})]]),
            ReportSection(heading="B", paragraphs=[[claim.model_copy(update={"id": "C3"})]]),
        ],
    )
    deduped, removed = dedupe_claims(draft)
    assert removed == 1  # summary repetition is allowed; the second body copy is not
    assert [s.heading for s in deduped.sections] == ["A"]
