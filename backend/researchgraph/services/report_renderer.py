"""Deterministic Markdown rendering of the verified report."""

from __future__ import annotations

from collections.abc import Mapping

from researchgraph.schemas.evidence import Contradiction, Evidence
from researchgraph.schemas.report import Citation, ReportClaim, ReportDraft, ReportMetrics
from researchgraph.schemas.review import QualityAssessment
from researchgraph.services.citations import citation_marker

_TYPE_LABELS = {
    "primary_research": "Primary research",
    "official_documentation": "Official documentation",
    "institutional": "Institutional",
    "reputable_technical": "Technical publication",
    "news": "News",
    "blog": "Blog",
    "forum": "Forum",
    "unknown": "Unclassified",
}


def _sentence(
    claim: ReportClaim, evidence: Mapping[str, Evidence], numbers: Mapping[str, int]
) -> str:
    text = claim.text.rstrip()
    marker = citation_marker(claim, evidence, numbers)
    if not marker:
        return text
    if text and text[-1] in ".!?":
        return f"{text[:-1]} {marker}{text[-1]}"
    return f"{text} {marker}"


def _paragraph(
    claims: list[ReportClaim], evidence: Mapping[str, Evidence], numbers: Mapping[str, int]
) -> str:
    return " ".join(_sentence(c, evidence, numbers) for c in claims)


def format_reference(citation: Citation) -> str:
    authors = ", ".join(citation.authors[:3]) + (" et al." if len(citation.authors) > 3 else "")
    year = f" ({citation.published_date.year})" if citation.published_date else " (n.d.)"
    venue = f" *{citation.venue}*." if citation.venue else ""
    label = _TYPE_LABELS.get(citation.source_type.value, citation.source_type.value)
    quality = f", quality {citation.quality:.2f}" if citation.quality is not None else ""
    lead = f"{authors}{year}. " if authors else f"{year.strip()}. "
    return f"[{citation.number}] {lead}{citation.title}.{venue} <{citation.url}> — {label}{quality}"


def render_executive_summary(
    draft: ReportDraft, evidence: Mapping[str, Evidence], numbers: Mapping[str, int]
) -> str:
    if not draft.executive_summary:
        return "_No statement could be verified against the retrieved evidence; see Limitations._"
    lines = [_paragraph(draft.executive_summary, evidence, numbers)]
    if draft.recommendations:
        lines.append("")
        lines.append("**Key recommendations**")
        lines.extend(f"- {_sentence(r, evidence, numbers)}" for r in draft.recommendations[:5])
    return "\n".join(lines).strip()


def render_markdown(
    *,
    question: str,
    draft: ReportDraft,
    evidence: Mapping[str, Evidence],
    numbers: Mapping[str, int],
    bibliography: list[Citation],
    contradictions: list[Contradiction],
    quality: QualityAssessment | None,
    metrics: ReportMetrics,
) -> str:
    out: list[str] = [f"# {draft.title}", "", f"> **Research question:** {question}", ""]
    out += ["## Executive Summary", "", render_executive_summary(draft, evidence, numbers), ""]

    for section in draft.sections:
        out += [f"## {section.heading}", ""]
        for paragraph in section.paragraphs:
            out += [_paragraph(paragraph, evidence, numbers), ""]

    if draft.recommendations:
        out += ["## Recommendations", ""]
        out += [f"- {_sentence(r, evidence, numbers)}" for r in draft.recommendations]
        out.append("")

    cited_contradictions = [
        c
        for c in contradictions
        if all(e in evidence and evidence[e].source_id in numbers for e in c.evidence_ids)
    ]
    if cited_contradictions:
        out += ["## Conflicting Evidence", ""]
        for c in cited_contradictions:
            refs = "".join(f"[{numbers[evidence[e].source_id]}]" for e in c.evidence_ids)
            out.append(f"- **{c.severity.capitalize()}:** {c.description} {refs}")
        out.append("")

    if draft.limitations:
        out += ["## Limitations", ""]
        out += [f"- {item}" for item in draft.limitations]
        out.append("")

    out += [
        "## Methodology",
        "",
        f"This report was produced by an automated, multi-step research workflow over "
        f"{metrics.research_iterations} research iteration(s). {metrics.evidence_items} evidence items were "
        f"extracted from retrieved sources; every sentence above is tied to stored evidence and verified before "
        f"publication ({metrics.citation_coverage:.0%} citation coverage across {metrics.total_claims} claims).",
    ]
    if quality is not None:
        out.append(
            f"The final quality-gate score was {quality.score:.2f} (threshold {quality.threshold:.2f}; "
            f"decision: {quality.decision.replace('_', ' ')})."
        )
    out.append("")

    if draft.removed_claims:
        out += ["## Claims Removed During Verification", ""]
        out += [f"- ~~{c.text}~~ (insufficient supporting evidence)" for c in draft.removed_claims]
        out.append("")

    out += ["## References", ""]
    out += [format_reference(c) for c in bibliography] or ["_No sources were cited._"]
    out.append("")
    return "\n".join(out)
