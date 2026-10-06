"""The citation system: claim → evidence → source provenance, verification and repair.

The report writer only ever attaches *evidence IDs* to sentences. Citation numbers, URLs
and the bibliography are produced here, deterministically, from stored provenance — so a
citation can never point at a source the system did not actually retrieve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from researchgraph.core.text import coverage, extract_quantities, normalize_for_match
from researchgraph.schemas.evidence import Evidence
from researchgraph.schemas.report import Citation, ReportClaim, ReportDraft, ReportSection
from researchgraph.schemas.review import CitationCheck, CitationIssue
from researchgraph.schemas.sources import Source, SourceQuality, SourceType


def _evidence_text(items: list[Evidence]) -> str:
    return " ".join(f"{e.claim} {e.supporting_text}" for e in items)


def check_claim(
    claim: ReportClaim, evidence: Mapping[str, Evidence], *, relevance_threshold: float
) -> CitationIssue | None:
    """Return the first problem with a claim's citations, or None if it is supported."""
    if not claim.evidence_ids:
        return CitationIssue(
            claim_id=claim.id,
            claim_text=claim.text,
            problem="missing_citation",
            detail="Sentence has no supporting evidence",
        )
    unknown = [eid for eid in claim.evidence_ids if eid not in evidence]
    if unknown:
        return CitationIssue(
            claim_id=claim.id,
            claim_text=claim.text,
            problem="unknown_evidence",
            detail=f"Cites evidence that does not exist: {', '.join(unknown)}",
        )
    cited = [evidence[eid] for eid in claim.evidence_ids]
    support_text = _evidence_text(cited)
    relevance = coverage(claim.text, support_text)
    if relevance < relevance_threshold:
        return CitationIssue(
            claim_id=claim.id,
            claim_text=claim.text,
            problem="low_relevance",
            detail=f"Cited evidence covers only {relevance:.0%} of the sentence's key terms",
        )
    missing_numbers = extract_quantities(claim.text) - extract_quantities(support_text)
    if missing_numbers:
        return CitationIssue(
            claim_id=claim.id,
            claim_text=claim.text,
            problem="numeric_mismatch",
            detail=f"Numbers not found in cited evidence: {', '.join(sorted(missing_numbers))}",
        )
    return None


def verify_citations(
    draft: ReportDraft,
    evidence: Mapping[str, Evidence],
    *,
    attempt: int,
    relevance_threshold: float,
) -> CitationCheck:
    claims = draft.all_claims()
    issues = [
        issue
        for claim in claims
        if (issue := check_claim(claim, evidence, relevance_threshold=relevance_threshold))
        is not None
    ]
    cited = sum(1 for c in claims if any(eid in evidence for eid in c.evidence_ids))
    return CitationCheck(
        attempt=attempt,
        total_claims=len(claims),
        cited_claims=cited,
        verified_claims=len(claims) - len(issues),
        coverage=round(cited / len(claims), 4) if claims else 1.0,  # vacuous: nothing uncited
        issues=issues,
    )


@dataclass(frozen=True)
class RepairOutcome:
    draft: ReportDraft
    reattributed: int
    removed: int


def _best_support(
    claim: ReportClaim, pool: list[Evidence], *, relevance_threshold: float
) -> list[str]:
    numbers = extract_quantities(claim.text)
    ranked = sorted(
        ((coverage(claim.text, _evidence_text([e])), e) for e in pool),
        key=lambda pair: (pair[0], pair[1].confidence),
        reverse=True,
    )
    chosen: list[Evidence] = []
    for score, item in ranked[:5]:
        if score < relevance_threshold + 0.1:
            break
        chosen.append(item)
        if (
            numbers <= extract_quantities(_evidence_text(chosen))
            and coverage(claim.text, _evidence_text(chosen)) >= relevance_threshold + 0.1
        ):
            return [e.id for e in chosen]
        if len(chosen) == 2:
            break
    return []


def repair_citations(
    draft: ReportDraft,
    check: CitationCheck,
    evidence: Mapping[str, Evidence],
    *,
    relevance_threshold: float,
) -> RepairOutcome:
    """Re-attribute each failing sentence to the best-matching evidence, or remove it.

    Removal is preferred over keeping an unsupported sentence; removed sentences are kept
    in ``removed_claims`` and listed in the report for transparency.
    """
    failing = {issue.claim_id for issue in check.issues}
    pool = list(evidence.values())
    removed: list[ReportClaim] = list(draft.removed_claims)
    reattributed = 0

    def fix(claim: ReportClaim) -> ReportClaim | None:
        nonlocal reattributed
        if claim.id not in failing:
            return claim
        known = [eid for eid in claim.evidence_ids if eid in evidence]
        candidate = claim.model_copy(update={"evidence_ids": known})
        if (
            known
            and check_claim(candidate, evidence, relevance_threshold=relevance_threshold) is None
        ):
            reattributed += 1
            return candidate
        support = _best_support(claim, pool, relevance_threshold=relevance_threshold)
        if support:
            repaired = claim.model_copy(update={"evidence_ids": support})
            if check_claim(repaired, evidence, relevance_threshold=relevance_threshold) is None:
                reattributed += 1
                return repaired
        removed.append(claim)
        return None

    def fix_all(claims: list[ReportClaim]) -> list[ReportClaim]:
        return [fixed for claim in claims if (fixed := fix(claim)) is not None]

    sections = []
    for section in draft.sections:
        paragraphs = [p for p in (fix_all(paragraph) for paragraph in section.paragraphs) if p]
        if paragraphs:
            sections.append(ReportSection(heading=section.heading, paragraphs=paragraphs))

    repaired_draft = draft.model_copy(
        update={
            "executive_summary": fix_all(draft.executive_summary),
            "sections": sections,
            "recommendations": fix_all(draft.recommendations),
            "removed_claims": removed,
        }
    )
    return RepairOutcome(
        draft=repaired_draft,
        reattributed=reattributed,
        removed=len(removed) - len(draft.removed_claims),
    )


def dedupe_claims(draft: ReportDraft) -> tuple[ReportDraft, int]:
    """Drop verbatim-repeated sentences within the summary, the body and the
    recommendations (repeating a finding in the executive summary is fine)."""
    removed = 0

    def unique(claims: list[ReportClaim], seen: set[str]) -> list[ReportClaim]:
        nonlocal removed
        kept = []
        for claim in claims:
            key = normalize_for_match(claim.text)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept.append(claim)
        return kept

    summary = unique(draft.executive_summary, set())
    body_seen: set[str] = set()
    sections = []
    for section in draft.sections:
        paragraphs = [
            p for p in (unique(paragraph, body_seen) for paragraph in section.paragraphs) if p
        ]
        if paragraphs:
            sections.append(ReportSection(heading=section.heading, paragraphs=paragraphs))
    recommendations = unique(draft.recommendations, set())
    if not removed:
        return draft, 0
    return draft.model_copy(
        update={
            "executive_summary": summary,
            "sections": sections,
            "recommendations": recommendations,
        }
    ), removed


def build_bibliography(
    draft: ReportDraft,
    evidence: Mapping[str, Evidence],
    sources: Mapping[str, Source],
    quality: Mapping[str, SourceQuality],
) -> tuple[dict[str, int], list[Citation]]:
    """Number sources by first appearance; returns (source_id → number, bibliography)."""
    numbers: dict[str, int] = {}
    evidence_by_source: dict[str, list[str]] = {}
    for claim in draft.all_claims():
        for eid in claim.evidence_ids:
            item = evidence.get(eid)
            if item is None or item.source_id not in sources:
                continue
            numbers.setdefault(item.source_id, len(numbers) + 1)
            bucket = evidence_by_source.setdefault(item.source_id, [])
            if eid not in bucket:
                bucket.append(eid)

    bibliography = []
    for source_id, number in sorted(numbers.items(), key=lambda kv: kv[1]):
        source = sources[source_id]
        q = quality.get(source_id)
        bibliography.append(
            Citation(
                number=number,
                source_id=source_id,
                title=source.title,
                url=source.url,
                authors=source.authors,
                published_date=source.published_date,
                venue=source.venue,
                source_type=q.source_type if q else (source.type_hint or SourceType.UNKNOWN),
                quality=q.overall if q else None,
                evidence_ids=evidence_by_source[source_id],
            )
        )
    return numbers, bibliography


def citation_marker(
    claim: ReportClaim, evidence: Mapping[str, Evidence], numbers: Mapping[str, int]
) -> str:
    cited = sorted(
        {
            numbers[evidence[eid].source_id]
            for eid in claim.evidence_ids
            if eid in evidence and evidence[eid].source_id in numbers
        }
    )
    return "".join(f"[{n}]" for n in cited)
