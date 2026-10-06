"""Contradiction adjudication over pre-selected candidate pairs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from researchgraph.agents.base import (
    UNTRUSTED_CONTENT_POLICY,
    call_structured,
    make_prompt,
    render_block,
)
from researchgraph.core.text import truncate
from researchgraph.llm.factory import ModelTier
from researchgraph.schemas.evidence import Contradiction
from researchgraph.schemas.llm import ContradictionOutput
from researchgraph.schemas.sources import Source
from researchgraph.services.contradictions import CandidatePair

if TYPE_CHECKING:
    from researchgraph.graph.context import ResearchContext

SYSTEM = """You check whether pairs of evidence claims genuinely contradict each other.

A pair is a contradiction only if both claims cannot be true under the same conditions.
Claims about different datasets, model sizes, domains or settings are usually NOT contradictions — say so in the explanation.
severity = 'major' if the disagreement would change a practical conclusion, otherwise 'minor'.
{policy}"""

HUMAN = "{pairs}"

PROMPT = make_prompt(SYSTEM, HUMAN)


async def judge_contradictions(
    ctx: ResearchContext, pairs: list[CandidatePair], sources: Mapping[str, Source]
) -> list[Contradiction]:
    if not pairs:
        return []

    def describe(e_id: str, claim: str, quote: str, source_id: str) -> dict[str, str]:
        source = sources.get(source_id)
        return {
            "evidence_id": e_id,
            "claim": claim,
            "quote": truncate(quote, 300),
            "source": truncate(source.title, 120) if source else source_id,
        }

    block = render_block(
        "pairs",
        [
            {
                "pair_id": p.pair_id,
                "subquestion_id": p.first.subquestion_id,
                "signal": p.signal,
                "a": describe(
                    p.first.id, p.first.claim, p.first.supporting_text, p.first.source_id
                ),
                "b": describe(
                    p.second.id, p.second.claim, p.second.supporting_text, p.second.source_id
                ),
            }
            for p in pairs
        ],
    )
    output = await call_structured(
        ctx,
        tier=ModelTier.FAST,
        prompt=PROMPT,
        variables={"pairs": block, "policy": UNTRUSTED_CONTENT_POLICY},
        schema=ContradictionOutput,
        operation="contradiction_detection",
    )
    by_id = {p.pair_id: p for p in pairs}
    contradictions: list[Contradiction] = []
    for judgement in output.judgements:
        pair = by_id.get(judgement.pair_id)
        if pair is None or not judgement.is_contradiction:
            continue
        contradictions.append(
            Contradiction(
                id=f"C{len(contradictions) + 1}",
                subquestion_id=pair.first.subquestion_id,
                evidence_ids=[pair.first.id, pair.second.id],
                description=truncate(judgement.explanation, 500),
                severity=judgement.severity,
            )
        )
    return contradictions
