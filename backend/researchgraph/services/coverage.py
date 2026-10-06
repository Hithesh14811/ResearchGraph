"""Evidence-coverage assessment: the deterministic input to the post-research gate."""

from __future__ import annotations

from collections.abc import Mapping

from researchgraph.core.text import keywords
from researchgraph.schemas.evidence import Evidence
from researchgraph.schemas.research import ResearchGap, ResearchPlan, RunLimits
from researchgraph.schemas.review import EvidenceAssessment, SubquestionCoverage
from researchgraph.schemas.sources import SourceQuality


def assess_coverage(
    plan: ResearchPlan,
    evidence: Mapping[str, Evidence],
    quality: Mapping[str, SourceQuality],
    *,
    iteration: int,
    limits: RunLimits,
) -> EvidenceAssessment:
    """A subquestion is sufficiently covered when it has at least
    ``min_evidence_per_subquestion`` credible evidence items drawn from at least two
    distinct credible sources (one source is never enough to corroborate a finding)."""
    coverage: list[SubquestionCoverage] = []
    gaps: list[ResearchGap] = []
    required_sources = min(2, limits.min_evidence_per_subquestion)

    for sq in plan.subquestions:
        items = [e for e in evidence.values() if e.subquestion_id == sq.id]
        credible = [
            e
            for e in items
            if (q := quality.get(e.source_id)) is not None
            and q.overall >= limits.credible_source_threshold
            and e.relevance >= limits.min_evidence_relevance
        ]
        credible_sources = {e.source_id for e in credible}
        scores = [quality[s].overall for s in {e.source_id for e in items} if s in quality]
        sufficient = (
            len(credible) >= limits.min_evidence_per_subquestion
            and len(credible_sources) >= required_sources
        )
        coverage.append(
            SubquestionCoverage(
                subquestion_id=sq.id,
                evidence_count=len(items),
                credible_evidence_count=len(credible),
                source_count=len({e.source_id for e in items}),
                mean_source_quality=round(sum(scores) / len(scores), 3) if scores else 0.0,
                sufficient=sufficient,
            )
        )
        if not sufficient:
            topic = keywords(sq.question, 6)
            suggestions = [f"{topic} {keywords(req, 5)}" for req in sq.information_requirements[:2]]
            suggestions.append(f"{topic} empirical study")
            gaps.append(
                ResearchGap(
                    subquestion_id=sq.id,
                    description=(
                        f"{len(credible)} credible evidence item(s) from {len(credible_sources)} source(s); "
                        f"need {limits.min_evidence_per_subquestion} from {required_sources}+ sources"
                    ),
                    reason="insufficient_evidence",
                    suggested_queries=suggestions,
                )
            )
    return EvidenceAssessment(
        iteration=iteration,
        coverage=coverage,
        sufficient=not gaps,
        gaps=gaps,
    )
