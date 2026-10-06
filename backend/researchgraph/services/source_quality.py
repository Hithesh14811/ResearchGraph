"""Deterministic, explainable source-quality scoring.

Search rank is not trusted as a quality signal. Each source gets six sub-scores with a
human-readable rationale, and an overall weighted score that downstream steps (evidence
gate, synthesis, critic, quality gate) use to prefer strong evidence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from researchgraph.core.text import coverage
from researchgraph.schemas.evidence import Evidence
from researchgraph.schemas.sources import QualityTier, Source, SourceQuality, SourceType
from researchgraph.tools.url_safety import domain_of

PRIMARY_RESEARCH_DOMAINS = frozenset(
    {
        "arxiv.org",
        "aclanthology.org",
        "openreview.net",
        "proceedings.neurips.cc",
        "papers.nips.cc",
        "proceedings.mlr.press",
        "jmlr.org",
        "dl.acm.org",
        "ieeexplore.ieee.org",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "link.springer.com",
        "springer.com",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "biorxiv.org",
        "medrxiv.org",
        "semanticscholar.org",
        "ojs.aaai.org",
        "ijcai.org",
        "openaccess.thecvf.com",
        "pnas.org",
        "cell.com",
        "thelancet.com",
        "nejm.org",
        "journals.plos.org",
        "frontiersin.org",
        "mdpi.com",
        "aclweb.org",
        "usenix.org",
    }
)
OFFICIAL_DOC_DOMAINS = frozenset(
    {
        "python.langchain.com",
        "langchain-ai.github.io",
        "platform.openai.com",
        "docs.anthropic.com",
        "pytorch.org",
        "tensorflow.org",
        "kubernetes.io",
        "developer.mozilla.org",
        "learn.microsoft.com",
        "cloud.google.com",
        "ai.google.dev",
        "docs.python.org",
        "postgresql.org",
        "readthedocs.io",
    }
)
INSTITUTIONAL_DOMAINS = frozenset(
    {
        "who.int",
        "oecd.org",
        "europa.eu",
        "un.org",
        "worldbank.org",
        "imf.org",
        "mlcommons.org",
        "ietf.org",
        "w3.org",
    }
)
INSTITUTIONAL_SUFFIXES = (
    ".gov",
    ".edu",
    ".mil",
    ".ac.uk",
    ".ac.in",
    ".ac.jp",
    ".edu.au",
    ".gov.uk",
)
REPUTABLE_TECHNICAL_DOMAINS = frozenset(
    {
        "research.google",
        "blog.research.google",
        "deepmind.google",
        "openai.com",
        "anthropic.com",
        "ai.meta.com",
        "huggingface.co",
        "microsoft.com",
        "developer.nvidia.com",
        "blogs.nvidia.com",
        "engineering.fb.com",
        "distill.pub",
        "thegradient.pub",
        "lilianweng.github.io",
        "eugeneyan.com",
        "paperswithcode.com",
        "databricks.com",
        "pinecone.io",
        "netflixtechblog.com",
        "aws.amazon.com",
        "cohere.com",
        "mistral.ai",
        "stanford.edu",
        "crfm.stanford.edu",
    }
)
NEWS_DOMAINS = frozenset(
    {
        "nytimes.com",
        "reuters.com",
        "bloomberg.com",
        "theverge.com",
        "techcrunch.com",
        "wired.com",
        "arstechnica.com",
        "venturebeat.com",
        "bbc.co.uk",
        "bbc.com",
        "theguardian.com",
        "wsj.com",
        "ft.com",
        "cnbc.com",
        "zdnet.com",
        "forbes.com",
        "axios.com",
        "technologyreview.com",
    }
)
BLOG_DOMAINS = frozenset(
    {
        "medium.com",
        "substack.com",
        "dev.to",
        "hashnode.dev",
        "towardsdatascience.com",
        "blogspot.com",
        "wordpress.com",
        "hackernoon.com",
        "analyticsvidhya.com",
        "linkedin.com",
        "quora.com",
    }
)
FORUM_DOMAINS = frozenset(
    {
        "reddit.com",
        "stackoverflow.com",
        "stackexchange.com",
        "news.ycombinator.com",
        "discord.com",
        "x.com",
        "twitter.com",
        "facebook.com",
        "lemmy.world",
    }
)

AUTHORITY_PRIOR = {
    SourceType.PRIMARY_RESEARCH: 0.9,
    SourceType.OFFICIAL_DOCUMENTATION: 0.85,
    SourceType.INSTITUTIONAL: 0.85,
    SourceType.REPUTABLE_TECHNICAL: 0.72,
    SourceType.NEWS: 0.5,
    SourceType.BLOG: 0.35,
    SourceType.FORUM: 0.2,
    SourceType.UNKNOWN: 0.3,
}
PRIMARY_PRIOR = {
    SourceType.PRIMARY_RESEARCH: 1.0,
    SourceType.OFFICIAL_DOCUMENTATION: 0.8,
    SourceType.INSTITUTIONAL: 0.7,
    SourceType.REPUTABLE_TECHNICAL: 0.5,
    SourceType.NEWS: 0.3,
    SourceType.BLOG: 0.2,
    SourceType.FORUM: 0.1,
    SourceType.UNKNOWN: 0.2,
}


# A literature survey published by a research venue is still a secondary source: it reports
# other people's results. Matched on the title, where authors reliably say so.
_SECONDARY_LITERATURE = re.compile(
    r"\bsurvey\b|\b(systematic|scoping|literature|narrative)\s+review\b|^\s*(a\s+)?review\s+of\b"
    r"|:\s*an?\s+(comprehensive\s+|critical\s+)?(review|overview)\b",
    re.IGNORECASE,
)
SECONDARY_LITERATURE_PRIMARY = 0.5


@dataclass(frozen=True)
class ScoringWeights:
    authority: float = 0.25
    recency: float = 0.12
    relevance: float = 0.2
    primary_source: float = 0.15
    methodology: float = 0.18
    citation_signal: float = 0.10


def _matches(domain: str, known: Iterable[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in known)


def classify_source(source: Source) -> tuple[SourceType, str]:
    """Return the source type and the reason it was chosen."""
    if source.type_hint is not None:
        return source.type_hint, f"type reported by search provider '{source.provider}'"
    domain = domain_of(source.url)
    path = source.url.lower()
    if _matches(domain, PRIMARY_RESEARCH_DOMAINS):
        return SourceType.PRIMARY_RESEARCH, f"{domain} publishes primary research"
    if _matches(domain, OFFICIAL_DOC_DOMAINS) or domain.startswith("docs.") or "/docs/" in path:
        return SourceType.OFFICIAL_DOCUMENTATION, f"{domain} is official documentation"
    if _matches(domain, INSTITUTIONAL_DOMAINS) or domain.endswith(INSTITUTIONAL_SUFFIXES):
        return SourceType.INSTITUTIONAL, f"{domain} is an institutional domain"
    if _matches(domain, FORUM_DOMAINS):
        return SourceType.FORUM, f"{domain} is a discussion forum"
    if _matches(domain, BLOG_DOMAINS):
        return SourceType.BLOG, f"{domain} is a blogging platform"
    if _matches(domain, NEWS_DOMAINS):
        return SourceType.NEWS, f"{domain} is a news outlet"
    if _matches(domain, REPUTABLE_TECHNICAL_DOMAINS):
        return SourceType.REPUTABLE_TECHNICAL, f"{domain} is a recognised technical publisher"
    if "/blog" in path:
        return SourceType.BLOG, "URL path indicates a blog post"
    return SourceType.UNKNOWN, f"{domain} is not a recognised publisher"


def recency_score(
    published: date | None, *, today: date, half_life_years: float
) -> tuple[float, str]:
    if published is None:
        return 0.4, "publication date unknown"
    age_years = max(0.0, (today - published).days / 365.25)
    score = max(0.05, min(1.0, 0.5 ** (age_years / half_life_years)))
    return score, f"published {published.isoformat()} ({age_years:.1f} years ago)"


def score_source(
    source: Source,
    *,
    subquestion_texts: list[str],
    evidence: list[Evidence],
    today: date,
    half_life_years: float,
    high_threshold: float,
    credible_threshold: float,
    weights: ScoringWeights = ScoringWeights(),
) -> SourceQuality:
    source_type, type_reason = classify_source(source)
    rationale = [f"Classified as {source_type.value.replace('_', ' ')}: {type_reason}"]

    authority = AUTHORITY_PRIOR[source_type]
    if source_type is SourceType.PRIMARY_RESEARCH and source.venue:
        if "preprint" in source.venue.lower() or "arxiv" in source.venue.lower():
            authority -= 0.08
            rationale.append("Preprint (not yet peer reviewed)")
        else:
            authority += 0.05
            rationale.append(f"Published venue: {source.venue}")
    authority = max(0.0, min(1.0, authority))

    recency, recency_reason = recency_score(
        source.published_date, today=today, half_life_years=half_life_years
    )
    rationale.append(recency_reason.capitalize())

    profile = f"{source.title} {source.snippet}"
    lexical = max((coverage(text, profile) for text in subquestion_texts), default=0.0)
    if evidence:
        relevance = 0.5 * lexical + 0.5 * (sum(e.relevance for e in evidence) / len(evidence))
    else:
        relevance = lexical * 0.8
    relevance = max(0.0, min(1.0, relevance))

    primary = PRIMARY_PRIOR[source_type]
    if source_type is SourceType.PRIMARY_RESEARCH and _SECONDARY_LITERATURE.search(source.title):
        primary = SECONDARY_LITERATURE_PRIMARY
        rationale.append("Literature survey or review: a secondary source")

    signals = source.signals
    if signals is not None and signals.word_count > 0:
        per_thousand = signals.methodology_terms / max(1.0, signals.word_count / 1000)
        methodology = (
            0.4 * min(1.0, per_thousand / 8)
            + 0.2 * (1.0 if signals.has_abstract else 0.0)
            + 0.2 * min(1.0, signals.numeric_density / 3)
            + 0.2 * primary
        )
        if per_thousand >= 4:
            rationale.append("Describes experimental methodology")
    else:
        methodology = 0.15 + 0.35 * primary
        rationale.append("Only a snippet/abstract was available")
    methodology = max(0.0, min(1.0, methodology))

    if source.citation_count is not None:
        citation_signal = min(1.0, math.log10(1 + source.citation_count) / 3)
        rationale.append(f"Cited {source.citation_count} times")
    elif signals is not None and signals.reference_count:
        citation_signal = min(1.0, signals.reference_count / 30)
        rationale.append(f"Contains ~{signals.reference_count} references")
    else:
        citation_signal = 0.3

    overall = (
        weights.authority * authority
        + weights.recency * recency
        + weights.relevance * relevance
        + weights.primary_source * primary
        + weights.methodology * methodology
        + weights.citation_signal * citation_signal
    ) / (
        weights.authority
        + weights.recency
        + weights.relevance
        + weights.primary_source
        + weights.methodology
        + weights.citation_signal
    )
    if source.status == "failed":
        overall *= 0.5
        rationale.append("Content could not be retrieved")

    tier = (
        QualityTier.HIGH
        if overall >= high_threshold
        else QualityTier.MEDIUM
        if overall >= credible_threshold
        else QualityTier.LOW
    )
    return SourceQuality(
        source_id=source.id,
        source_type=source_type,
        authority=round(authority, 3),
        recency=round(recency, 3),
        relevance=round(relevance, 3),
        primary_source=round(primary, 3),
        methodology=round(methodology, 3),
        citation_signal=round(citation_signal, 3),
        overall=round(overall, 3),
        tier=tier,
        rationale=rationale,
    )
