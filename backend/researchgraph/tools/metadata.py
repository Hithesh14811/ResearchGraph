"""Source metadata extraction: titles, authors, publication dates and identifiers."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from bs4 import BeautifulSoup

_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+[^\s\"'<>.,;])", re.IGNORECASE)
_MONTHS = {
    m: i + 1
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ]
    )
    for m in names
}


def parse_date(value: Any) -> date | None:
    """Best-effort parsing of the many date formats found in web/PDF metadata."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("D:"):  # PDF dates: D:YYYYMMDDHHmmSS
        text = text[2:]
        if len(text) >= 8 and text[:8].isdigit():
            return _safe_date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    iso = re.match(r"^(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", text)
    if iso:
        return _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3) or 1))
    if re.fullmatch(r"\d{4}", text):
        return _safe_date(int(text), 1, 1)
    words = re.findall(r"[A-Za-z]+|\d+", text)
    year = next((int(w) for w in words if w.isdigit() and len(w) == 4), None)
    month = next((_MONTHS[w.lower()] for w in words if w.lower() in _MONTHS), None)
    day = next((int(w) for w in words if w.isdigit() and len(w) <= 2), 1)
    if year and month:
        return _safe_date(year, month, day)
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        parsed = date(year, month, day)
    except ValueError:
        try:
            parsed = date(year, month, 1)
        except ValueError:
            return None
    return parsed if 1900 <= parsed.year <= 2100 else None


def extract_doi(text: str) -> str | None:
    match = _DOI.search(text)
    return match.group(1) if match else None


def _meta(soup: BeautifulSoup, *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        for tag in soup.find_all("meta", attrs={"name": name}) + soup.find_all(
            "meta", attrs={"property": name}
        ):
            content = tag.get("content")
            if isinstance(content, str) and content.strip():
                values.append(content.strip())
    return values


def extract_html_metadata(soup: BeautifulSoup) -> dict[str, Any]:
    """Pull bibliographic metadata from Highwire/Dublin Core/OpenGraph tags."""
    title = next(
        iter(_meta(soup, "citation_title", "og:title", "dc.title", "twitter:title")),
        None,
    )
    if title is None and soup.title and soup.title.string:
        title = soup.title.string.strip()
    if title is None and (h1 := soup.find("h1")):
        title = h1.get_text(" ", strip=True)

    authors = _meta(soup, "citation_author", "dc.creator", "author", "article:author")
    date_values = _meta(
        soup,
        "citation_publication_date",
        "citation_date",
        "article:published_time",
        "dc.date",
        "date",
        "pubdate",
        "publish_date",
    )
    if not date_values and (time_tag := soup.find("time")):
        dt = time_tag.get("datetime")
        if isinstance(dt, str):
            date_values.append(dt)
    published = next((d for d in (parse_date(v) for v in date_values) if d), None)
    doi = next(iter(_meta(soup, "citation_doi", "dc.identifier")), None)
    venue = next(
        iter(_meta(soup, "citation_journal_title", "citation_conference_title", "og:site_name")),
        None,
    )
    return {
        "title": title,
        "authors": list(dict.fromkeys(authors))[:20],
        "published_date": published,
        "doi": extract_doi(doi) if doi else None,
        "venue": venue,
    }
