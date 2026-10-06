"""Document parsing for PDF, HTML, Markdown and plain text, with size limits."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from researchgraph.core.errors import DocumentExtractionError
from researchgraph.schemas.sources import DocumentFormat
from researchgraph.tools.metadata import extract_html_metadata, parse_date

logger = logging.getLogger(__name__)

_CONTENT_TYPE_FORMATS = {
    "text/html": DocumentFormat.HTML,
    "application/xhtml+xml": DocumentFormat.HTML,
    "application/pdf": DocumentFormat.PDF,
    "text/markdown": DocumentFormat.MARKDOWN,
    "text/x-markdown": DocumentFormat.MARKDOWN,
    "text/plain": DocumentFormat.TEXT,
}
_SUFFIX_FORMATS = {
    ".pdf": DocumentFormat.PDF,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".txt": DocumentFormat.TEXT,
    ".html": DocumentFormat.HTML,
    ".htm": DocumentFormat.HTML,
}
_STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "svg",
    "iframe",
    "button",
)
_BLOCK_TAGS = (
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "dt",
    "dd",
    "blockquote",
    "pre",
    "tr",
    "caption",
    "figcaption",
)
_BLOCK_BREAK = "\u2029"  # Unicode paragraph separator; a stray one in page text only adds a break


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    format: DocumentFormat
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    published_date: date | None = None
    venue: str | None = None
    doi: str | None = None
    truncated: bool = False


def detect_format(content_type: str | None, url: str = "") -> DocumentFormat | None:
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in _CONTENT_TYPE_FORMATS:
        return _CONTENT_TYPE_FORMATS[mime]
    path = url.lower().split("?", 1)[0]
    for suffix, fmt in _SUFFIX_FORMATS.items():
        if path.endswith(suffix):
            return fmt
    return None


def _decode(body: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def _limit(text: str, max_chars: int) -> tuple[str, bool]:
    return (text, False) if len(text) <= max_chars else (text[:max_chars], True)


def extract_pdf(body: bytes, *, max_chars: int, max_pages: int) -> ExtractedDocument:
    try:
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentExtractionError("PDF is encrypted")
        pages = []
        for index, page in enumerate(reader.pages):
            if index >= max_pages:
                break
            pages.append(page.extract_text() or "")
        meta: Any = reader.metadata or {}
    except DocumentExtractionError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise DocumentExtractionError(f"Malformed PDF: {exc}") from exc
    text, truncated = _limit("\n\n".join(pages), max_chars)
    if not text.strip():
        raise DocumentExtractionError("PDF contains no extractable text (scanned image?)")
    title = getattr(meta, "title", None) or None
    author = getattr(meta, "author", None)
    created = meta.get("/CreationDate") if hasattr(meta, "get") else None
    return ExtractedDocument(
        text=text,
        format=DocumentFormat.PDF,
        title=str(title) if title else None,
        authors=[a.strip() for a in re.split(r"[;,]| and ", str(author)) if a.strip()]
        if author
        else [],
        published_date=parse_date(created),
        truncated=truncated or len(reader.pages) > max_pages,
    )


def extract_html(body: bytes, *, max_chars: int) -> ExtractedDocument:
    soup = BeautifulSoup(_decode(body), "html.parser")
    metadata: dict[str, Any] = extract_html_metadata(soup)
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    root = (
        soup.find("article")
        or soup.find("main")
        or soup.select_one('[role="main"]')
        or soup.body
        or soup
    )
    # Block elements become paragraphs separated by a blank line, so a heading can never be
    # reflowed into the sentence after it (the markup already says where blocks end).
    for tag in root.find_all(_BLOCK_TAGS):
        tag.insert_before(_BLOCK_BREAK)
        tag.append(_BLOCK_BREAK)
    blocks = []
    for block in root.get_text("\n").split(_BLOCK_BREAK):
        lines = [line.strip() for line in block.splitlines()]
        if joined := "\n".join(line for line in lines if line):
            blocks.append(joined)
    text = "\n\n".join(blocks)
    if not text.strip():
        raise DocumentExtractionError("HTML page has no readable text")
    text, truncated = _limit(text, max_chars)
    return ExtractedDocument(
        text=text,
        format=DocumentFormat.HTML,
        title=metadata["title"],
        authors=metadata["authors"],
        published_date=metadata["published_date"],
        venue=metadata["venue"],
        doi=metadata["doi"],
        truncated=truncated,
    )


_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal ``key: value`` front-matter parser (no YAML dependency needed)."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip().strip("\"'")
    return fields, text[match.end() :]


def markdown_to_text(markdown: str) -> str:
    text = re.sub(r"```[a-zA-Z0-9]*\n", "", markdown).replace("```", "")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> anchor text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # heading markers
    text = re.sub(r"(\*\*|__|\*|_|`)(.+?)\1", r"\2", text)  # emphasis / inline code
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)  # block quotes
    return text


def extract_markdown(body: bytes, *, max_chars: int) -> ExtractedDocument:
    raw = _decode(body)
    front, content = parse_frontmatter(raw)
    heading = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    text, truncated = _limit(markdown_to_text(content).strip(), max_chars)
    if not text:
        raise DocumentExtractionError("Markdown document is empty")
    authors = [
        a.strip() for a in front.get("authors", front.get("author", "")).split(";") if a.strip()
    ]
    return ExtractedDocument(
        text=text,
        format=DocumentFormat.MARKDOWN,
        title=front.get("title") or (heading.group(1).strip() if heading else None),
        authors=authors,
        published_date=parse_date(front.get("date")),
        venue=front.get("venue"),
        doi=front.get("doi"),
        truncated=truncated,
    )


def extract_text(body: bytes, *, max_chars: int) -> ExtractedDocument:
    text, truncated = _limit(_decode(body).strip(), max_chars)
    if not text:
        raise DocumentExtractionError("Text document is empty")
    first_line = text.splitlines()[0][:200] if text else None
    return ExtractedDocument(
        text=text, format=DocumentFormat.TEXT, title=first_line, truncated=truncated
    )


def extract_document(
    body: bytes,
    *,
    content_type: str | None,
    url: str,
    max_chars: int,
    max_pdf_pages: int,
) -> ExtractedDocument:
    """Dispatch on detected format. Raises ``DocumentExtractionError`` for malformed input."""
    fmt = detect_format(content_type, url)
    if fmt is None:
        raise DocumentExtractionError(f"Unsupported document type {content_type!r}")
    if fmt is DocumentFormat.PDF:
        return extract_pdf(body, max_chars=max_chars, max_pages=max_pdf_pages)
    if fmt is DocumentFormat.HTML:
        return extract_html(body, max_chars=max_chars)
    if fmt is DocumentFormat.MARKDOWN:
        return extract_markdown(body, max_chars=max_chars)
    return extract_text(body, max_chars=max_chars)
