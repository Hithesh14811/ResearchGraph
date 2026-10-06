"""Deterministic responses for the offline mock model.

The responder reads the same structured context blocks a real model receives and produces
*grounded* output with simple, transparent heuristics: extractive evidence (quotes are real
sentences from the passages), findings built from the strongest evidence, a rule-plus-
heuristic critique, and a report assembled from findings. It is a stand-in that exercises
every code path of the workflow — it is not meant to imitate the prose quality of an LLM.
"""

from __future__ import annotations

import re
from typing import Any

from researchgraph.agents.base import extract_block
from researchgraph.agents.planner import heuristic_plan
from researchgraph.core.faults import FaultInjector, NoFaults
from researchgraph.core.text import (
    coverage,
    extract_quantities,
    jaccard,
    keywords,
    normalize_for_match,
    split_sentences,
    term_set,
    truncate,
)
from researchgraph.demo.corpus import DISCLAIMER_MARKER
from researchgraph.demo.scenarios import match_sample
from researchgraph.llm.mock import MockRequest, MockResponse
from researchgraph.schemas.llm import (
    ContradictionJudgement,
    ContradictionOutput,
    CriticOutput,
    CritiqueIssueOutput,
    DraftParagraph,
    DraftSection,
    DraftSentence,
    EvidenceExtractionOutput,
    ExtractedEvidence,
    PlannedSubquestion,
    PlannerOutput,
    QueryGenerationOutput,
    QueryPlan,
    ReportDraftOutput,
    SynthesisOutput,
    SynthesizedFinding,
)

ABSOLUTE = re.compile(
    r"\b(always|every (time|domain|case|task)|on all tasks|never|obsolete|eliminates?|completely|no hallucinations)\b",
    re.IGNORECASE,
)
RESULT_VERBS = re.compile(
    r"\b(improv\w*|reduc\w*|achiev\w*|outperform\w*|scor\w*|retain\w*|detect\w*|increas\w*|lost|fell|reached|cost\w*|correlat\w*)\b",
    re.IGNORECASE,
)
OPINION = re.compile(
    r"\b(in my experience|i think|we never|reply:|original post:|according to one|said one)\b",
    re.IGNORECASE,
)
LIMITATION = re.compile(
    r"\b(limitation|however|struggl\w*|fail\w*|degrad\w*|misclassif\w*|only|underestimat\w*|worse)\b",
    re.IGNORECASE,
)
CREDIBLE = 0.45
GENERIC_TERMS = term_set(
    "effect effects approach approaches compare comparison evaluate evaluation analyze analysis current "
    "use using identify evidence limitation limitations benchmark benchmarks practical recommendation "
    "recommendations source sources credible recent academic paper papers technical tradeoff tradeoffs "
    "long-term impact study studies research question questions best practice practices role method "
    "methods overview approach state knowledge implication implications follow regarding reported "
    "uncertainty uncertainties risk risks measurement measurements empirical recommend exist"
)

RECOMMENDATIONS = {
    "rag-vs-finetuning": [
        "Use retrieval-augmented generation when answers depend on a large or frequently changing knowledge base.",
        "Use fine-tuning to teach consistent output format, tone and domain terminology rather than to add new facts.",
        "Reserve long-context prompting for small document sets, keeping the most relevant material at the beginning of the context.",
        "Monitor retrieval quality continuously, because retrieval failures dominate answer errors.",
    ],
    "small-vs-large-models": [
        "Use fine-tuned or distilled small models for narrow, stable tasks such as classification and extraction.",
        "Route complex multi-step reasoning requests to larger models, for example with a confidence-based cascade.",
        "Apply 4-bit quantization to large models to reduce memory and GPU count at a small accuracy cost.",
        "Monitor small models for input drift, since out-of-distribution performance drops sharply.",
    ],
    "rag-hallucinations": [
        "Improve retrieval first with reranking and hybrid lexical and dense retrieval, since retrieval failures cause most hallucinated answers.",
        "Require citations for every sentence and verify them after generation.",
        "Allow the system to abstain when the retrieved evidence is insufficient.",
        "Track faithfulness metrics continuously and evaluate retrieval separately from generation.",
    ],
}
SYNTHESIS_SENTENCE = "Overall, no single approach dominates across all conditions; the tradeoffs depend on the task, on how often knowledge changes and on cost constraints."
_INTERROGATIVE = re.compile(
    r"^(how (accurate|effective|well|much|do|does|should)|what (are|is|role does|practical guidance do|limitations and failure modes)|which)\s+(the\s+)?",
    re.IGNORECASE,
)


def _heading(question: str) -> str:
    text = _INTERROGATIVE.sub("", question.strip()).rstrip("?")
    text = re.sub(r"^(\w+ )?(sources give for|affect|is|are)\s+", "", text)
    text = text[:1].upper() + text[1:]
    if len(text) <= 72:
        return text
    return text[:72].rsplit(" ", 1)[0] + "…"


def _attribute(sentence: str) -> str:
    """Report claims in the third person ("We compare" -> "The authors compare")."""
    sentence = re.sub(r"^We\b", "The authors", sentence)
    return re.sub(r"^Our\b", "The authors'", sentence)


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text and not text[:2].isupper() else text


def _section_heading(sample: Any, subquestion: dict[str, Any]) -> str:
    """Curated heading for an unedited sample-plan subquestion, else a derived one."""
    if sample is not None:
        match = re.fullmatch(r"SQ(\d+)", str(subquestion.get("id", "")))
        index = int(match.group(1)) - 1 if match else -1
        planned = sample.plan.subquestions
        if (
            0 <= index < min(len(planned), len(sample.headings))
            and planned[index].question == subquestion["question"]
        ):
            return str(sample.headings[index])
    return _heading(str(subquestion["question"]))


class DemoResponder:
    def __init__(self, faults: FaultInjector | None = None) -> None:
        self.faults = faults or NoFaults()

    def __call__(self, request: MockRequest) -> MockResponse:
        schema = request.structured_schema
        if schema is None:
            return (
                self._research_agent(request)
                if request.tools
                else MockResponse(content="Acknowledged.")
            )
        handler = getattr(self, f"_{re.sub(r'(?<!^)(?=[A-Z])', '_', schema).lower()}", None)
        if handler is None:
            raise ValueError(f"DemoResponder cannot produce {schema}")
        return MockResponse(tool_calls=[(schema, handler(request))])

    # -- planner -----------------------------------------------------------------------
    def _planner_output(self, request: MockRequest) -> dict[str, Any]:
        req = extract_block(request.prompt_text, "planning_request") or {}
        question = str(req.get("question", ""))
        max_sq = int(req.get("max_subquestions", 6))
        max_q = int(req.get("max_queries_per_subquestion", 3))
        sample = match_sample(question)
        if sample is not None:
            plan = sample.plan.model_copy(deep=True)
        else:
            fallback = heuristic_plan(question, max_subquestions=max_sq, max_queries=max_q)
            plan = PlannerOutput(
                objective=fallback.objective,
                scope=fallback.scope,
                subquestions=[
                    PlannedSubquestion(
                        question=sq.question,
                        rationale="Needed to answer the research question.",
                        information_requirements=sq.information_requirements,
                        search_queries=[q.query for q in sq.search_queries],
                        preferred_sources=sq.preferred_sources,
                    )
                    for sq in fallback.subquestions
                ],
                source_strategy=fallback.source_strategy,
                stopping_criteria=fallback.stopping_criteria,
            )
        feedback = [str(f) for f in req.get("reviewer_feedback") or [] if str(f).strip()]
        if feedback:
            focus = feedback[-1].strip().rstrip(".?")
            terms = keywords(focus, 6) or focus
            plan.subquestions.append(
                PlannedSubquestion(
                    question=f"What does the evidence show regarding the reviewer's request: {truncate(focus, 200)}?",
                    rationale="Added in response to reviewer feedback.",
                    information_requirements=[truncate(focus, 120)],
                    search_queries=[terms, f"{terms} evaluation"],
                    preferred_sources="mixed",
                )
            )
        plan.subquestions = plan.subquestions[:max_sq]
        return plan.model_dump(mode="json")

    # -- query generation ------------------------------------------------------------------
    def _query_generation_output(self, request: MockRequest) -> dict[str, Any]:
        req = extract_block(request.prompt_text, "query_request") or {}
        history = {normalize_for_match(h) for h in req.get("search_history", [])}
        limit = int(req.get("max_queries", 3))
        plans = []
        for target in req.get("targets", []):
            terms = keywords(target["question"], 5)
            candidates = list(target.get("draft_queries") or [])
            if target.get("gap"):
                gap_terms = [
                    f"{terms} {keywords(req_, 4)}"
                    for req_ in target.get("information_requirements", [])[:2]
                ]
                candidates = [*gap_terms, f"{terms} evaluation study", *candidates]
            queries = [
                q for q in dict.fromkeys(candidates) if normalize_for_match(q) not in history
            ][:limit]
            plans.append(
                QueryPlan(
                    subquestion_id=target["subquestion_id"],
                    queries=queries or [f"{terms} study"],
                    preferred_sources="mixed"
                    if target.get("gap")
                    else target.get("preferred_sources", "mixed"),
                )
            )
        return QueryGenerationOutput(plans=plans).model_dump(mode="json")

    # -- research agent (tool calling) --------------------------------------------------------
    def _research_agent(self, request: MockRequest) -> MockResponse:
        task = extract_block(request.prompt_text, "task") or {}
        queries = [str(q) for q in task.get("suggested_queries", [])]
        pref = task.get("preferred_sources", "mixed")
        tool_turns = sum(1 for m in request.messages if getattr(m, "tool_calls", None))
        if not request.tool_messages:
            calls = []
            for i, query in enumerate(queries):
                tool = {"academic": "academic_search", "web": "web_search"}.get(
                    pref, "academic_search" if i % 2 == 0 else "web_search"
                )
                calls.append((tool, {"query": query, "max_results": 5}))
            if pref in ("academic", "web") and queries:
                other = "web_search" if pref == "academic" else "academic_search"
                calls.append((other, {"query": queries[0], "max_results": 3}))
            return MockResponse(tool_calls=calls)
        found = sum(len(m.artifact or []) for m in request.tool_messages)
        if found < 3 and tool_turns < 2:
            broad = keywords(str(task.get("subquestion", "")), 6)
            return MockResponse(
                tool_calls=[
                    ("academic_search", {"query": broad, "max_results": 5}),
                    ("web_search", {"query": broad, "max_results": 5}),
                ]
            )
        return MockResponse(
            content=f"Collected {found} search results across {len(request.tool_messages)} searches; enough candidates to proceed."
        )

    # -- evidence extraction -----------------------------------------------------------------
    @staticmethod
    def _evidence_type(sentence: str) -> str:
        if OPINION.search(sentence):
            return "expert_opinion"
        quantities = extract_quantities(sentence)
        if quantities and re.search(
            r"\b(accuracy|exact[- ]match|benchmark|score|points?|faithfulness|precision)\b",
            sentence,
            re.I,
        ):
            return "benchmark_result"
        if re.search(
            r"\b(our (internal|first)|we now|in production|deployments?)\b", sentence, re.I
        ):
            return "case_study"
        if quantities:
            return "empirical_result"
        if LIMITATION.search(sentence):
            return "limitation"
        return "background"

    def _evidence_extraction_output(self, request: MockRequest) -> dict[str, Any]:
        req = extract_block(request.prompt_text, "extraction_request") or {}
        passages = extract_block(request.prompt_text, "passages") or []
        target = " ".join(
            [
                str(req.get("subquestion", "")),
                *map(str, req.get("search_intent", [])),
                str(req.get("focus") or ""),
            ]
        )
        # Topical anchor: a sentence must share a distinctive term with the research question,
        # otherwise generic words ("cost", "performance") would let off-topic text through.
        # The anchor comes only from what was asked (question + subquestion): planned queries can
        # carry generic padding ("... cost tradeoffs") that would make off-topic text look relevant.
        topic_terms = (
            term_set(f"{req.get('research_question', '')} {req.get('subquestion', '')}")
            - GENERIC_TERMS
        )
        facets = [
            str(part)
            for part in [req.get("subquestion"), *req.get("search_intent", []), req.get("focus")]
            if part
        ] or [target]
        scored: list[tuple[float, ExtractedEvidence]] = []
        for passage in passages:
            candidates = []
            for sentence in split_sentences(passage["text"]):
                clean = re.sub(r"^(reply|original post):\s*", "", sentence, flags=re.I).strip()
                if (
                    DISCLAIMER_MARKER.lower() in clean.lower()
                    or not 40 <= len(clean) <= 400
                    or clean.startswith("#")
                    or not clean.endswith((".", "!", "?"))  # headings and titles are not claims
                    or len(clean.split()) < 6
                    or (topic_terms and not topic_terms & term_set(clean))
                ):
                    continue
                # Score against the best-matching facet (the subquestion or any single planned
                # query): concatenating them would dilute coverage for focused sentences.
                relevance = max(coverage(facet, clean) for facet in facets)
                if relevance < 0.15:
                    continue
                score = (
                    relevance
                    + (0.25 if extract_quantities(clean) else 0.0)
                    + (0.1 if RESULT_VERBS.search(clean) else 0.0)
                )
                kind = self._evidence_type(sentence)
                candidates.append(
                    (
                        score,
                        ExtractedEvidence(
                            passage_id=passage["passage_id"],
                            claim=_attribute(clean),
                            quote=clean,
                            evidence_type=kind,
                            confidence=0.5
                            if kind == "expert_opinion"
                            else (0.85 if extract_quantities(clean) else 0.7),
                            relevance=round(min(1.0, 0.35 + relevance), 2),
                        ),
                    )
                )
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            scored.extend(candidates[:2])
        scored.sort(key=lambda pair: pair[0], reverse=True)
        items = [item for _, item in scored[: int(req.get("max_items", 12))]]
        return EvidenceExtractionOutput(evidence=items).model_dump(mode="json")

    # -- contradictions ----------------------------------------------------------------------
    def _contradiction_output(self, request: MockRequest) -> dict[str, Any]:
        pairs = extract_block(request.prompt_text, "pairs") or []
        judgements = []
        flagged: set[str] = set()  # one contradiction per sweeping claim is enough
        for pair in pairs:
            a, b = pair["a"], pair["b"]
            # A sweeping claim is an unmeasured generalisation ("always beats"); a claim that
            # reports a measurement ("40% relative to always answering") is not one.
            sweeping = next(
                (
                    x
                    for x in (a, b)
                    if ABSOLUTE.search(x["claim"]) and not extract_quantities(x["claim"])
                ),
                None,
            )
            other = b if sweeping is a else a
            qualified = bool(
                extract_quantities(other["claim"]) or LIMITATION.search(other["claim"])
            )
            absolute = sweeping is not None and qualified and sweeping["evidence_id"] not in flagged
            if absolute and sweeping is not None:
                flagged.add(sweeping["evidence_id"])
            conflict = absolute or (
                "opposite polarity" in pair["signal"] and jaccard(a["claim"], b["claim"]) >= 0.3
            )
            explanation = (
                f"'{truncate(a['claim'], 110)}' ({truncate(a['source'], 60)}) is inconsistent with "
                f"'{truncate(b['claim'], 110)}' ({truncate(b['source'], 60)})."
                if conflict
                else "The claims describe different conditions and are compatible."
            )
            judgements.append(
                ContradictionJudgement(
                    pair_id=pair["pair_id"],
                    is_contradiction=conflict,
                    explanation=explanation,
                    severity="major" if absolute else "minor",
                )
            )
        return ContradictionOutput(judgements=judgements).model_dump(mode="json")

    # -- synthesis -------------------------------------------------------------------------------
    def _synthesis_output(self, request: MockRequest) -> dict[str, Any]:
        req = extract_block(request.prompt_text, "synthesis_request") or {}
        evidence = extract_block(request.prompt_text, "evidence") or []
        contradictions = extract_block(request.prompt_text, "contradictions") or []
        findings: list[SynthesizedFinding] = []
        used_claims: set[str] = set()
        for sq in req.get("subquestions", []):
            items = [e for e in evidence if e["subquestion_id"] == sq["id"]]
            credible = [e for e in items if (e.get("source_quality") or 0) >= CREDIBLE]
            pool = sorted(
                credible or items,
                key=lambda e: (
                    (e.get("source_quality") or 0),
                    bool(extract_quantities(e["claim"])),
                ),
                reverse=True,
            )
            used_sources: set[str] = set()
            for item in pool:
                if len([f for f in findings if f.subquestion_id == sq["id"]]) >= 3:
                    break
                claim_key = normalize_for_match(item["claim"])
                if (
                    item["source_title"] in used_sources
                    or claim_key in used_claims
                    or ABSOLUTE.search(item["claim"])
                ):
                    continue
                used_claims.add(claim_key)
                corroborating = [
                    o
                    for o in pool
                    if o["source_title"] != item["source_title"]
                    and jaccard(o["claim"], item["claim"]) >= 0.25
                ]
                support = [item, *corroborating[:1]]
                n_sources = len({s["source_title"] for s in support})
                confidence = (
                    "high" if n_sources >= 2 and credible else ("moderate" if credible else "low")
                )
                caveats = []
                if not credible:
                    caveats.append(
                        "Supported only by lower-quality sources (blogs, forums or news)."
                    )
                if item["evidence_type"] == "expert_opinion":
                    caveats.append("Reflects opinion rather than measured evidence.")
                findings.append(
                    SynthesizedFinding(
                        subquestion_id=sq["id"],
                        statement=item["claim"],
                        evidence_ids=[s["id"] for s in support],
                        confidence=confidence,
                        caveats=caveats,
                    )
                )
                used_sources.add(item["source_title"])
        # Represent every contradiction in the most closely related finding.
        for contradiction in contradictions:
            related = [
                f for f in findings if f.subquestion_id == contradiction.get("subquestion_id")
            ] or findings
            if not related:
                continue
            target = max(
                related, key=lambda f: len(set(f.evidence_ids) & set(contradiction["evidence_ids"]))
            )
            target.contradiction_ids.append(contradiction["id"])
            target.caveats.append("Some sources make conflicting claims; see Conflicting Evidence.")
        n_sources = len({e["source_title"] for e in evidence})
        high = [f for f in findings if f.confidence == "high"]
        thin = [
            sq["id"]
            for sq in req.get("subquestions", [])
            if not any(f.subquestion_id == sq["id"] and f.confidence != "low" for f in findings)
        ]
        return SynthesisOutput(
            findings=findings,
            overall_assessment=(
                f"The analysis draws on {len(evidence)} evidence items from {n_sources} sources. "
                f"{len(high)} of {len(findings)} findings are corroborated by at least two credible sources."
            ),
            consensus_points=[f.statement for f in high[:3]],
            disagreements=[c["description"] for c in contradictions[:3]],
            open_questions=[
                f"Evidence for {sq_id} is limited to weaker or single sources." for sq_id in thin
            ],
        ).model_dump(mode="json")

    # -- critic ------------------------------------------------------------------------------------
    def _critic_output(self, request: MockRequest) -> dict[str, Any]:
        if not request.is_repair and self.faults.should_fail("invalid_output", "CriticOutput"):
            # Deliberately schema-invalid output (bad enum values) to exercise validation repair.
            return {
                "issues": [
                    {
                        "issue_type": "vibes",
                        "severity": "catastrophic",
                        "description": "Malformed review.",
                    }
                ],
                "summary": "Malformed review.",
                "verdict": "terrible",
            }
        req = extract_block(request.prompt_text, "review_request") or {}
        findings = extract_block(request.prompt_text, "findings") or []
        evidence = {e["id"]: e for e in extract_block(request.prompt_text, "evidence") or []}
        issues: list[CritiqueIssueOutput] = []
        for finding in findings:
            cited = [evidence[e] for e in finding["evidence_ids"] if e in evidence]
            support = " ".join(f"{c['claim']} {c['quote']}" for c in cited)
            if ABSOLUTE.search(finding["statement"]):
                issues.append(
                    CritiqueIssueOutput(
                        issue_type="overgeneralization",
                        severity="high",
                        finding_ids=[finding["id"]],
                        subquestion_id=finding["subquestion_id"],
                        description=f"{finding['id']} makes an absolute claim that the evidence cannot support.",
                        suggested_action="Qualify the claim with the conditions under which it was observed.",
                    )
                )
            if cited and coverage(finding["statement"], support) < 0.35:
                issues.append(
                    CritiqueIssueOutput(
                        issue_type="citation_mismatch",
                        severity="medium",
                        finding_ids=[finding["id"]],
                        subquestion_id=finding["subquestion_id"],
                        description=f"Cited evidence for {finding['id']} only partially matches its statement.",
                        suggested_action="Cite evidence that directly states the finding.",
                    )
                )
        coverage_rows = sorted(
            req.get("coverage", []),
            key=lambda c: (c["credible_evidence_count"], c["subquestion_id"]),
        )
        if coverage_rows:
            weakest = coverage_rows[0]
            sq = next(
                (s for s in req.get("subquestions", []) if s["id"] == weakest["subquestion_id"]),
                None,
            )
            terms = keywords(sq["question"] if sq else "", 5)
            if self.faults.should_fail("critic_rejection", "critique"):
                issues.append(
                    CritiqueIssueOutput(
                        issue_type="missing_coverage",
                        severity="critical",
                        subquestion_id=weakest["subquestion_id"],
                        description=f"{weakest['subquestion_id']} lacks independent quantitative evidence; conclusions for it may be wrong.",
                        suggested_action="Find independent benchmark or measurement studies.",
                        suggested_queries=[
                            f"{terms} benchmark evaluation",
                            f"{terms} measurement study",
                        ],
                    )
                )
            elif weakest["credible_evidence_count"] < 3:
                issues.append(
                    CritiqueIssueOutput(
                        issue_type="missing_coverage",
                        severity="low",
                        subquestion_id=weakest["subquestion_id"],
                        description=f"Only {weakest['credible_evidence_count']} credible evidence item(s) support {weakest['subquestion_id']}; independent replication would materially strengthen this conclusion.",
                        suggested_action="Treat this conclusion as provisional.",
                        suggested_queries=[f"{terms} replication"],
                    )
                )
        severities = {i.severity.value for i in issues}
        verdict = (
            "major_problems"
            if "critical" in severities
            else "needs_revision"
            if "high" in severities
            else "acceptable"
        )
        return CriticOutput(
            issues=issues,
            strengths=[
                "Every finding is tied to quoted evidence",
                "Primary research is weighted above blogs and forums",
            ],
            summary=f"Reviewed {len(findings)} findings against {len(evidence)} cited evidence items; {len(issues)} issue(s) raised.",
            verdict=verdict,
        ).model_dump(mode="json")

    # -- evaluation judge -------------------------------------------------------------------
    def _judge_output(self, request: MockRequest) -> dict[str, Any]:
        """Structural heuristics only — the mock judge is NOT a quality measurement."""
        report = str(extract_block(request.prompt_text, "report") or "")
        sections = report.count("## ")
        return {
            "faithfulness": 5 if "## References" in report else 2,
            "completeness": min(5, max(1, sections - 2)),
            "balance": 4
            if "## Conflicting Evidence" in report or "## Limitations" in report
            else 2,
            "actionability": 4 if "## Recommendations" in report else 2,
            "clarity": 3,
            "rationale": "Mock judge: scores derived from report structure only; not a measurement of quality.",
        }

    # -- report writer --------------------------------------------------------------------------------
    def _report_draft_output(self, request: MockRequest) -> dict[str, Any]:
        req = extract_block(request.prompt_text, "report_request") or {}
        findings = extract_block(request.prompt_text, "findings") or []
        evidence = extract_block(request.prompt_text, "evidence") or []
        contradictions = extract_block(request.prompt_text, "contradictions") or []
        ev_by_id = {e["id"]: e for e in evidence}
        question = str(req.get("question", ""))
        sample = match_sample(question)
        rank = {"high": 0, "moderate": 1, "low": 2}

        summary: list[DraftSentence] = []
        seen_sq: set[str] = set()
        ordered = sorted(
            findings,
            key=lambda f: (rank[f["confidence"]], not extract_quantities(f["statement"]), f["id"]),
        )
        for finding in ordered:
            if finding["subquestion_id"] in seen_sq or finding["confidence"] == "low":
                continue
            seen_sq.add(finding["subquestion_id"])
            summary.append(
                DraftSentence(text=finding["statement"], evidence_ids=finding["evidence_ids"])
            )
            if len(summary) == 5:
                break
        if not summary and findings:
            summary.append(
                DraftSentence(
                    text=findings[0]["statement"], evidence_ids=findings[0]["evidence_ids"]
                )
            )

        sections = []
        for sq in req.get("subquestions", []):
            sq_findings = [f for f in findings if f["subquestion_id"] == sq["id"]]
            if not sq_findings:
                continue
            sentences = [
                DraftSentence(text=f["statement"], evidence_ids=f["evidence_ids"])
                for f in sq_findings
            ]
            for contradiction in contradictions:
                if contradiction.get("subquestion_id") != sq["id"]:
                    continue
                other = next(
                    (
                        ev_by_id[e]
                        for e in contradiction["evidence_ids"]
                        if e in ev_by_id and not any(e in f["evidence_ids"] for f in sq_findings)
                    ),
                    None,
                )
                if other is not None:
                    sentences.append(
                        DraftSentence(
                            text=f"In contrast, one {other['source_type'].replace('_', ' ')} source claims that {_lower_first(other['claim'])}",
                            evidence_ids=[other["id"]],
                        )
                    )
            sections.append(
                DraftSection(
                    heading=_section_heading(sample, sq),
                    paragraphs=[DraftParagraph(sentences=sentences)],
                )
            )
        sections.append(
            DraftSection(
                heading="Comparative Synthesis",
                paragraphs=[
                    DraftParagraph(
                        sentences=[DraftSentence(text=SYNTHESIS_SENTENCE, evidence_ids=[])]
                    )
                ],
            )
        )

        recommendations = []
        for text in RECOMMENDATIONS.get(sample.key if sample else "", []):
            best = max(
                evidence, key=lambda e: coverage(text, f"{e['claim']} {e['quote']}"), default=None
            )
            if best is not None and coverage(text, f"{best['claim']} {best['quote']}") >= 0.3:
                recommendations.append(DraftSentence(text=text, evidence_ids=[best["id"]]))
        if not recommendations:
            for finding in [f for f in findings if f["confidence"] != "low"][:3]:
                recommendations.append(
                    DraftSentence(
                        text=f"Account for the evidence that {_lower_first(finding['statement'])}",
                        evidence_ids=finding["evidence_ids"],
                    )
                )

        review = req.get("quality_review") or {}
        limitations = [
            "This demo run used a synthetic offline corpus and a deterministic mock model; all figures are illustrative."
        ]
        limitations += [str(b) for b in review.get("blocking_issues") or []]
        limitations += [
            str(i["description"])
            for i in (review.get("open_issues") or [])
            if i["severity"] in ("medium", "high")
        ][:3]
        title = (
            f"Evidence Review: {question.split('.')[0]}"
            if sample is None
            else {
                "rag-vs-finetuning": "RAG vs. Fine-Tuning vs. Long-Context Prompting for Domain-Specific QA",
                "small-vs-large-models": "Small vs. Large Language Models for Production Inference",
                "rag-hallucinations": "Reducing Hallucinations in Retrieval-Augmented Generation",
            }[sample.key]
        )
        return ReportDraftOutput(
            title=truncate(title, 180),
            executive_summary=summary
            or [DraftSentence(text="No sufficiently supported findings were produced.")],
            sections=sections,
            recommendations=recommendations,
            limitations=limitations,
        ).model_dump(mode="json")
