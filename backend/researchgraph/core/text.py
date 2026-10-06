"""Small, dependency-free text utilities used across retrieval, scoring and verification."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "vs",
        "versus",
        "via",
        "use",
        "used",
        "using",
        "across",
        "within",
        "without",
        "among",
        "per",
        "may",
        "might",
        "must",
    ]
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%|x|×|percent|pp)?", re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "vs.", "fig.", "approx.", "etc.", "cf.", "no.")
_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " "})


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def sanitize_user_text(text: str, *, max_chars: int) -> str:
    """Normalise untrusted user input: NFKC, strip control characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    return normalize_whitespace(text)[:max_chars]


def normalize_for_match(text: str) -> str:
    """Canonical form for quote matching: lowercase, unified quotes/dashes, single spaces."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP).lower()
    return normalize_whitespace(text).strip(" .,;:\"'")


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return _WORD.findall(normalize_for_match(text))


def content_terms(text: str) -> list[str]:
    """Stopword-filtered, lightly stemmed tokens (order preserved, duplicates kept)."""
    return [_stem(tok) for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1]


def term_set(text: str) -> set[str]:
    return set(content_terms(text))


def keywords(text: str, limit: int = 8) -> str:
    """Unstemmed, stopword-free keywords in original order — suitable for search queries
    (stemmed tokens like 'sourc' must never reach a search engine)."""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        if token not in STOPWORDS and len(token) > 1:
            seen.setdefault(token, None)
    return " ".join(list(seen)[:limit])


def coverage(query: str, text: str) -> float:
    """Fraction of the query's content terms that occur in ``text`` (asymmetric)."""
    q = term_set(query)
    if not q:
        return 0.0
    return len(q & term_set(text)) / len(q)


def jaccard(a: str, b: str) -> float:
    sa, sb = term_set(a), term_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that protects common abbreviations and decimals.

    Line breaks are treated as hard boundaries first, so headings never fuse with the
    sentence that follows them."""
    sentences = []
    for block in re.split(r"\s*\n\s*", text):
        protected = block
        for i, abbr in enumerate(_ABBREVIATIONS):
            protected = re.sub(re.escape(abbr), f"\u0000{i}\u0000", protected, flags=re.IGNORECASE)
        for part in _SENTENCE_BOUNDARY.split(normalize_whitespace(protected)):
            for i, abbr in enumerate(_ABBREVIATIONS):
                part = part.replace(f"\u0000{i}\u0000", abbr)
            if part.strip():
                sentences.append(part.strip())
    return sentences


def extract_quantities(text: str) -> set[str]:
    """Numbers that state a quantity (percentages, decimals, multipliers, counts >= 10).

    Calendar years are excluded: they appear legitimately in prose without being claims.
    Used to catch numeric hallucination — a number in a report sentence must appear in the
    evidence that sentence cites.
    """
    found: set[str] = set()
    for match in _NUMBER.finditer(text):
        raw, unit = match.group(1).replace(",", ""), match.group(2)
        try:
            value = float(raw)
        except ValueError:
            continue
        if unit is None and "." not in raw and (value < 10 or 1900 <= value <= 2100):
            continue
        found.add(raw.rstrip("0").rstrip(".") if "." in raw else raw)
    return found


def quote_in_text(quote: str, text: str, *, min_ratio: float = 0.9) -> bool:
    """True if ``quote`` appears in ``text`` verbatim (modulo case/whitespace/quote style),
    or as a near-exact contiguous match (tolerates a dropped character or ellipsis)."""
    q, t = normalize_for_match(quote), normalize_for_match(text)
    if not q:
        return False
    if q in t:
        return True
    if len(q) < 20:
        return False
    matcher = SequenceMatcher(None, t, q, autojunk=False)
    block = matcher.find_longest_match(0, len(t), 0, len(q))
    if block.size / len(q) >= min_ratio:
        return True
    # Allow "..." elisions: every fragment must appear in order.
    fragments = [f.strip() for f in re.split(r"\.\.\.|…", q) if len(f.strip()) > 10]
    if len(fragments) > 1:
        position = 0
        for fragment in fragments:
            position = t.find(fragment, position)
            if position < 0:
                return False
        return True
    return False


_BRACKET_CITATION = re.compile(r"\s*\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)])")


def strip_citation_markers(text: str) -> str:
    """Remove numeric bracket citations such as "[14]", "[ 3, 5]" or "[12–15]".

    Sentences copied from papers carry the *source's* reference numbers; left in claim or
    report text they would be indistinguishable from ResearchGraph's own citation markers.
    """
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _BRACKET_CITATION.sub("", text)).strip()


def truncate(text: str, max_chars: int, *, suffix: str = "…") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix
