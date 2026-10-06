"""Text cleaning and structural signal extraction for retrieved documents."""

from __future__ import annotations

import re
import unicodedata

from researchgraph.schemas.sources import ContentSignals

_BOILERPLATE = re.compile(
    r"(accept (all )?cookies|cookie (policy|settings)|subscribe to|sign up for|all rights reserved|"
    r"privacy policy|terms of (service|use)|share this (article|post)|advertisement|skip to (main )?content)",
    re.IGNORECASE,
)
_SECTION_HEADING = re.compile(
    r"^(\d+(\.\d+)*\.?\s+)?(abstract|introduction|background|related work|methods?|methodology|approach|"
    r"experiments?|evaluation|results|findings|analysis|discussion|limitations|conclusions?|summary|"
    r"recommendations?|operational notes|references|bibliography|appendix)$",
    re.IGNORECASE,
)
_REFERENCES_HEADING = re.compile(
    r"^\s*(references|bibliography|works cited)\s*$", re.IGNORECASE | re.MULTILINE
)
_REFERENCE_ENTRY = re.compile(
    r"(^\s*\[\d+\])|(\bet al\.)|(\b(19|20)\d{2}[a-z]?\b\.)|(doi\.org/|arxiv:)",
    re.IGNORECASE | re.MULTILINE,
)
_METHOD_TERMS = re.compile(
    r"\b(experiment\w*|evaluat\w+|benchmark\w*|dataset\w*|baseline\w*|ablation\w*|methodolog\w+|"
    r"statistically|significan\w+|sample size|participants|accuracy|f1|exact match|we (propose|evaluate|measure|compare))\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?")


_LINE_BREAK_HYPHEN = re.compile(r"(\w+)-\n(\w+)")


def dehyphenate(text: str) -> str:
    """Rejoin words split across lines by a hyphen (PDF layout).

    A line-break hyphen is ambiguous: "accu-/racy" is a syllable break, "long-/context" is a
    real compound. Keep the hyphen when the hyphenated form occurs elsewhere in the same
    document; otherwise join the parts.
    """
    lowered = text.lower()

    def join(match: re.Match[str]) -> str:
        first, second = match.group(1), match.group(2)
        separator = "-" if f"{first}-{second}".lower() in lowered else ""
        return f"{first}{separator}{second}"

    return _LINE_BREAK_HYPHEN.sub(join, text)


def clean_text(text: str) -> str:
    """Normalise extracted text: unicode, de-hyphenation, boilerplate removal, reflow."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = dehyphenate(text)
    lines = []
    for raw in text.split("\n"):
        line = " ".join(raw.split())
        if line and len(line) < 160 and _BOILERPLATE.search(line):
            continue
        lines.append(line)
    # Reflow: join lines broken mid-sentence (PDF layout), keep headings and paragraphs.
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for index, line in enumerate(lines):
        if not line:
            flush()
            continue
        if _SECTION_HEADING.match(line):
            # PDFs often lose blank lines; known section names always start a new block,
            # so a title or heading never fuses with the first sentence that follows it.
            flush()
            paragraphs.append(line)
            continue
        following = next((nxt for nxt in lines[index + 1 :] if nxt), "")
        if (
            not current
            and len(line) < 70
            and len(line.split()) <= 8
            and not line.endswith((".", ",", ";", ":", "!", "?", "-"))
            and not following[:1].islower()
        ):
            paragraphs.append(line)  # heading: short, unpunctuated, not followed by a continuation
            continue
        current.append(line)
        if line.endswith((".", "!", "?")):
            flush()
    flush()
    return "\n\n".join(p for p in paragraphs if p.strip())


def split_references(text: str) -> tuple[str, str]:
    """Separate a trailing references section (only if it appears in the last 45%)."""
    matches = list(_REFERENCES_HEADING.finditer(text))
    if not matches:
        return text, ""
    last = matches[-1]
    if last.start() < len(text) * 0.55:
        return text, ""
    return text[: last.start()].rstrip(), text[last.end() :].strip()


def compute_signals(body: str, references: str) -> ContentSignals:
    words = body.split()
    word_count = len(words)
    head = body[:3000].lower()
    return ContentSignals(
        word_count=word_count,
        reference_count=len(_REFERENCE_ENTRY.findall(references)) if references else 0,
        methodology_terms=len(_METHOD_TERMS.findall(body)),
        numeric_density=round(len(_NUMBER.findall(body)) / max(1, word_count) * 100, 3),
        has_abstract="abstract" in head,
    )
