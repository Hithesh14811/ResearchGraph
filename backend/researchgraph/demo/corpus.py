"""The synthetic demo corpus, plus a search provider and fetcher that serve it offline.

Documents live in ``demo/corpus/*.md`` with front matter. Depending on their ``format``
they are served as Markdown, as HTML (with citation meta tags) or as generated PDF bytes,
so demo runs exercise every document parser.
"""

from __future__ import annotations

import html
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from importlib import resources

from researchgraph.core.errors import FetchError
from researchgraph.core.text import content_terms
from researchgraph.demo.pdf import build_pdf
from researchgraph.schemas.sources import SearchResult, SourceType
from researchgraph.tools.documents import markdown_to_text, parse_frontmatter
from researchgraph.tools.fetcher import FetchedContent
from researchgraph.tools.metadata import parse_date
from researchgraph.tools.search.base import SearchChannel
from researchgraph.tools.url_safety import canonicalize_url

DISCLAIMER_MARKER = "Synthetic demo document"


@dataclass(frozen=True)
class CorpusDocument:
    slug: str
    title: str
    url: str
    authors: list[str]
    published: date | None
    venue: str | None
    kind: SourceType
    channel: SearchChannel
    format: str
    tags: list[str]
    markdown: str
    terms: Counter[str] = field(compare=False, hash=False)

    @property
    def abstract(self) -> str:
        """First substantive paragraph (used as the search snippet)."""
        text = markdown_to_text(self.markdown)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        body = [p for p in paragraphs if DISCLAIMER_MARKER not in p and len(p) > 120]
        return " ".join(body[:1])[:600]


@cache
def load_corpus() -> tuple[CorpusDocument, ...]:
    docs = []
    folder = resources.files("researchgraph.demo").joinpath("corpus")
    for entry in sorted(folder.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".md"):
            continue
        raw = entry.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
        terms = Counter(
            content_terms(
                f"{meta['title']} {meta['title']} {' '.join(tags)} {' '.join(tags)} {markdown_to_text(body)}"
            )
        )
        docs.append(
            CorpusDocument(
                slug=entry.name.removesuffix(".md"),
                title=meta["title"],
                url=meta["url"],
                authors=[a.strip() for a in meta.get("authors", "").split(";") if a.strip()],
                published=parse_date(meta.get("date")),
                venue=meta.get("venue"),
                kind=SourceType(meta.get("kind", "unknown")),
                channel=SearchChannel(meta.get("channel", "web")),
                format=meta.get("format", "markdown"),
                tags=tags,
                markdown=raw,
                terms=terms,
            )
        )
    return tuple(docs)


class CorpusSearchProvider:
    """BM25-style keyword search over the synthetic corpus for one channel."""

    def __init__(
        self, channel: SearchChannel, documents: tuple[CorpusDocument, ...] | None = None
    ) -> None:
        self.name = f"demo-{channel.value}"
        self.channel = channel
        self._docs = [d for d in (documents or load_corpus()) if d.channel is channel]
        n = len(self._docs)
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(set(doc.terms))
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._avg_len = sum(sum(d.terms.values()) for d in self._docs) / max(1, n)

    def _score(self, doc: CorpusDocument, query_terms: list[str]) -> float:
        k1, b = 1.4, 0.75
        length = sum(doc.terms.values())
        score = 0.0
        for term in set(query_terms):
            tf = doc.terms.get(term, 0)
            if tf:
                score += (
                    self._idf.get(term, 0.0)
                    * tf
                    * (k1 + 1)
                    / (tf + k1 * (1 - b + b * length / self._avg_len))
                )
        return score

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        terms = content_terms(query)
        scored = sorted(
            ((self._score(d, terms), d) for d in self._docs),
            key=lambda pair: (-pair[0], pair[1].slug),
        )
        top = [(s, d) for s, d in scored if s > 0][:max_results]
        best = top[0][0] if top else 1.0
        return [
            SearchResult(
                url=doc.url,
                title=doc.title,
                snippet=doc.abstract,
                provider=self.name,
                published_date=doc.published,
                authors=doc.authors,
                venue=doc.venue,
                type_hint=doc.kind,
                score=round(score / best, 4),
            )
            for score, doc in top
        ]


def _markdown_to_html(doc: CorpusDocument) -> str:
    _, body = parse_frontmatter(doc.markdown)
    blocks = []
    for block in re.split(r"\n\s*\n", body.strip()):
        block = block.strip()
        if block.startswith("# "):
            blocks.append(f"<h1>{html.escape(block[2:])}</h1>")
        elif block.startswith("## "):
            blocks.append(f"<h2>{html.escape(block[3:])}</h2>")
        else:
            blocks.append(f"<p>{html.escape(markdown_to_text(block).strip())}</p>")
    authors = "".join(
        f'<meta name="citation_author" content="{html.escape(a)}">' for a in doc.authors
    )
    published = doc.published.strftime("%Y/%m/%d") if doc.published else ""
    return (
        f"<!doctype html><html><head><title>{html.escape(doc.title)}</title>"
        f'<meta name="citation_title" content="{html.escape(doc.title)}">{authors}'
        f'<meta name="citation_publication_date" content="{published}">'
        f'<meta property="og:site_name" content="{html.escape(doc.venue or "")}">'
        f"</head><body><nav>Home | Topics | Subscribe to our newsletter</nav>"
        f"<article>{''.join(blocks)}</article>"
        f"<footer>Accept all cookies to continue. All rights reserved.</footer></body></html>"
    )


class CorpusFetcher:
    """Serves corpus documents by URL in their declared format (no network access)."""

    def __init__(self, documents: tuple[CorpusDocument, ...] | None = None) -> None:
        self._by_url = {canonicalize_url(d.url): d for d in (documents or load_corpus())}

    async def fetch(self, url: str) -> FetchedContent:
        doc = self._by_url.get(canonicalize_url(url))
        if doc is None:
            raise FetchError(f"HTTP 404 for {url}", url=url, status_code=404)
        if doc.format == "pdf":
            _, body = parse_frontmatter(doc.markdown)
            payload = build_pdf(
                markdown_to_text(body), title=doc.title, author="; ".join(doc.authors)
            )
            return FetchedContent(
                url=url, final_url=url, content_type="application/pdf", body=payload
            )
        if doc.format == "html":
            return FetchedContent(
                url=url,
                final_url=url,
                content_type="text/html",
                body=_markdown_to_html(doc).encode("utf-8"),
            )
        return FetchedContent(
            url=url, final_url=url, content_type="text/markdown", body=doc.markdown.encode("utf-8")
        )
