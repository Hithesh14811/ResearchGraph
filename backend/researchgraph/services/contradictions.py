"""Candidate generation for contradiction detection.

Comparing every evidence pair with an LLM is O(n²) in cost. Instead we pre-select pairs
that are about the same thing (shared terms, same subquestion, different sources) *and*
show a conflict signal (opposite polarity or diverging quantities), then ask the model to
adjudicate only those.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from researchgraph.core.text import extract_quantities, jaccard, tokenize
from researchgraph.schemas.evidence import Evidence

# Prefix markers; deliberately excludes direction-ambiguous words ("lower latency" is good,
# "lower accuracy" is bad). This only *selects* candidates — the LLM makes the judgement.
_POSITIVE = (
    "improv",
    "outperform",
    "better",
    "gain",
    "boost",
    "effective",
    "superior",
    "benefit",
    "robust",
)
_NEGATIVE = (
    "degrad",
    "worse",
    "underperform",
    "fail",
    "hurt",
    "harm",
    "ineffective",
    "inferior",
    "negligible",
    "marginal",
    "unreliable",
    "no",
    "not",
    "never",
)


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    first: Evidence
    second: Evidence
    similarity: float
    signal: str


_ABSOLUTE = re.compile(
    r"\b(always|never|every (time|domain|case|task)|on all tasks|obsolete|eliminat\w*|completely|guarantee\w*)\b",
    re.IGNORECASE,
)


def _polarity(text: str) -> int:
    tokens = tokenize(text)
    positive = any(t.startswith(_POSITIVE) for t in tokens)
    negative = any(
        t in ("no", "not", "never") or (len(t) > 3 and t.startswith(_NEGATIVE)) for t in tokens
    )
    return int(positive) - int(negative)


def find_candidate_pairs(evidence: list[Evidence], *, max_pairs: int = 12) -> list[CandidatePair]:
    by_subquestion: dict[str, list[Evidence]] = {}
    for item in evidence:
        by_subquestion.setdefault(item.subquestion_id, []).append(item)

    scored: list[tuple[float, Evidence, Evidence, str]] = []
    for items in by_subquestion.values():
        for a, b in itertools.combinations(sorted(items, key=lambda e: e.id), 2):
            if a.source_id == b.source_id:
                continue
            similarity = jaccard(a.claim, b.claim)
            absolute = bool(_ABSOLUTE.search(a.claim)) != bool(_ABSOLUTE.search(b.claim))
            if similarity < (0.06 if absolute else 0.15):
                continue
            signals = ["absolute claim vs. qualified evidence"] if absolute else []
            pa, pb = _polarity(a.claim), _polarity(b.claim)
            if pa * pb < 0:
                signals.append("opposite polarity")
            qa, qb = extract_quantities(a.claim), extract_quantities(b.claim)
            if qa and qb and not (qa & qb) and similarity >= 0.25:
                signals.append("diverging quantities")
            if not signals and similarity < 0.5:
                continue
            priority = similarity + 0.3 * len(signals)
            scored.append((priority, a, b, ", ".join(signals) or "high overlap"))

    scored.sort(key=lambda item: item[0], reverse=True)
    # The same quote is stored once per subquestion it answers, so one conflicting pair of
    # quotes can surface under several subquestions. Adjudicate (and report) it only once.
    unique: list[tuple[float, Evidence, Evidence, str]] = []
    seen: set[frozenset[tuple[str, str]]] = set()
    for entry in scored:
        key = frozenset((e.source_id, " ".join(e.claim.lower().split())) for e in entry[1:3])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return [
        CandidatePair(
            pair_id=f"PAIR{i + 1}", first=a, second=b, similarity=round(priority, 3), signal=signal
        )
        for i, (priority, a, b, signal) in enumerate(unique[:max_pairs])
    ]
