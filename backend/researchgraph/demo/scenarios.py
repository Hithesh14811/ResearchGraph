"""Sample research questions and the curated plans the demo planner returns for them."""

from __future__ import annotations

from dataclasses import dataclass

from researchgraph.schemas.llm import PlannedSubquestion, PlannerOutput, SourcePref


@dataclass(frozen=True)
class SampleQuestion:
    key: str
    question: str
    keywords: tuple[str, ...]
    plan: PlannerOutput
    headings: tuple[str, ...] = ()


def _sq(
    question: str, rationale: str, requirements: list[str], queries: list[str], pref: SourcePref
) -> PlannedSubquestion:
    return PlannedSubquestion(
        question=question,
        rationale=rationale,
        information_requirements=requirements,
        search_queries=queries,
        preferred_sources=pref,
    )


SAMPLE_QUESTIONS: tuple[SampleQuestion, ...] = (
    SampleQuestion(
        key="rag-vs-finetuning",
        headings=(
            "Accuracy of Retrieval-Augmented Generation",
            "Fine-Tuning and Knowledge Injection",
            "Long-Context Prompting and Position Effects",
            "Cost, Latency and Operations",
            "Practical Guidance",
        ),
        question=(
            "Compare the effectiveness of RAG, fine-tuning, and long-context prompting for domain-specific "
            "question answering. Use recent academic papers and credible technical sources. Identify evidence, "
            "limitations, benchmarks, and practical recommendations."
        ),
        keywords=("rag", "fine-tuning", "long-context"),
        plan=PlannerOutput(
            objective="Determine when retrieval-augmented generation, fine-tuning and long-context prompting are most effective for domain-specific question answering, based on empirical evidence.",
            scope="Domain-specific QA (e.g. medical, legal, financial) with current LLMs; accuracy, knowledge freshness, cost and latency. General chat quality is out of scope.",
            subquestions=[
                _sq(
                    "How accurate is retrieval-augmented generation for domain-specific question answering compared with the alternatives?",
                    "Accuracy on domain QA is the primary effectiveness criterion.",
                    [
                        "exact-match accuracy on domain QA benchmarks",
                        "performance on recently updated facts",
                    ],
                    [
                        "retrieval-augmented generation domain-specific question answering accuracy",
                        "RAG exact match domain benchmark comparison",
                    ],
                    "academic",
                ),
                _sq(
                    "How effective is fine-tuning at injecting domain knowledge, and what are its side effects?",
                    "Fine-tuning is often assumed to add knowledge; this needs evidence.",
                    [
                        "accuracy on facts present only in fine-tuning data",
                        "catastrophic forgetting",
                    ],
                    [
                        "fine-tuning knowledge injection new facts accuracy",
                        "fine-tuning catastrophic forgetting domain adaptation",
                    ],
                    "academic",
                ),
                _sq(
                    "How does long-context prompting perform for domain QA, and how does the position of information affect accuracy?",
                    "Long context removes retrieval but may degrade with context length.",
                    [
                        "accuracy by position of relevant passage",
                        "latency and cost of long prompts",
                    ],
                    [
                        "long-context prompting accuracy position middle",
                        "long context question answering latency cost",
                    ],
                    "academic",
                ),
                _sq(
                    "What are the cost, latency and operational tradeoffs of RAG, fine-tuning and long-context approaches?",
                    "Production viability depends on cost and operations, not only accuracy.",
                    ["cost per 1,000 queries", "retraining or re-indexing frequency"],
                    [
                        "total cost of ownership LLM question answering retrieval",
                        "RAG fine-tuning long context cost per query",
                    ],
                    "mixed",
                ),
                _sq(
                    "What practical guidance do credible sources give for choosing between these approaches?",
                    "Practitioners need actionable decision criteria.",
                    ["documented decision criteria", "production case studies"],
                    [
                        "choosing retrieval fine-tuning long context guidance",
                        "retrieval-augmented generation production case study",
                    ],
                    "web",
                ),
            ],
            source_strategy=[
                "Peer-reviewed papers and preprints with controlled comparisons",
                "Official platform documentation",
                "Institutional cost studies",
                "Engineering case studies",
            ],
            stopping_criteria=[
                "Each subquestion has evidence from at least two independent credible sources",
                "Benchmarks and cost figures are cited with their conditions",
                "No critical issues remain after critique",
            ],
        ),
    ),
    SampleQuestion(
        key="small-vs-large-models",
        headings=(
            "Accuracy Across Model Sizes",
            "Compression and Specialisation",
            "Latency, Throughput and Energy",
            "Combining Small and Large Models",
            "Failure Modes of Small Models",
        ),
        question="Analyze the tradeoffs between small and large language models for production inference.",
        keywords=("small", "large", "inference"),
        plan=PlannerOutput(
            objective="Characterise the accuracy, latency, cost and reliability tradeoffs between small and large language models in production inference.",
            scope="Open-weight and hosted models from ~3B to ~70B parameters; serving, compression and routing strategies.",
            subquestions=[
                _sq(
                    "How does task accuracy differ between small and large language models across task types?",
                    "Accuracy gaps likely depend on the task.",
                    ["accuracy gap on reasoning vs classification tasks"],
                    [
                        "small vs large language model accuracy reasoning classification",
                        "model size benchmark production tasks",
                    ],
                    "academic",
                ),
                _sq(
                    "How much quality can distillation, fine-tuning or quantization recover for smaller or compressed models?",
                    "Compression and specialisation change the size tradeoff.",
                    ["retained performance after distillation", "accuracy drop from quantization"],
                    [
                        "distillation small model teacher performance",
                        "quantization accuracy drop model size",
                    ],
                    "academic",
                ),
                _sq(
                    "What are the latency, throughput, hardware and energy differences between small and large models?",
                    "Serving cost is driven by latency, throughput and hardware.",
                    ["median latency", "throughput per GPU", "energy per token"],
                    [
                        "inference latency throughput model size GPU",
                        "energy consumption LLM inference per token",
                    ],
                    "mixed",
                ),
                _sq(
                    "How do production systems combine small and large models, and with what cost-quality tradeoffs?",
                    "Routing/cascades are a common middle ground.",
                    ["share of traffic handled by small model", "cost reduction vs quality loss"],
                    [
                        "cascade routing small large model cost",
                        "model routing production case study",
                    ],
                    "web",
                ),
                _sq(
                    "What limitations and failure modes affect small models in production?",
                    "Failure modes determine where small models are unsafe.",
                    ["out-of-distribution degradation", "task types where small models fail"],
                    [
                        "small model out-of-distribution limitations",
                        "small language model production failure",
                    ],
                    "mixed",
                ),
            ],
            source_strategy=[
                "Benchmark papers",
                "Inference-server documentation",
                "Measurement studies",
                "Engineering case studies",
            ],
            stopping_criteria=[
                "Accuracy, latency and cost each supported by at least two credible sources",
                "Failure modes documented",
            ],
        ),
    ),
    SampleQuestion(
        key="rag-hallucinations",
        headings=(
            "Where RAG Hallucinations Come From",
            "Citation Grounding and Verification",
            "Retrieval Improvements",
            "Abstention",
            "Measuring and Operating Faithfulness",
        ),
        question="Evaluate current approaches for reducing hallucinations in retrieval-augmented generation.",
        keywords=("hallucination",),
        plan=PlannerOutput(
            objective="Evaluate which techniques measurably reduce hallucinations in retrieval-augmented generation, and at what cost.",
            scope="Retrieval improvements, grounding/citation, verification, abstention and evaluation practice for RAG systems.",
            subquestions=[
                _sq(
                    "Which failure sources cause hallucinations in retrieval-augmented generation?",
                    "Interventions should target the dominant causes.",
                    ["share of hallucinations caused by retrieval failures"],
                    [
                        "RAG hallucination causes retrieval failure analysis",
                        "retrieval errors unfaithful answers",
                    ],
                    "academic",
                ),
                _sq(
                    "How effective are citation grounding and claim verification at reducing unsupported statements?",
                    "Grounding and verification are the most direct mitigations.",
                    [
                        "unsupported statement rate before/after",
                        "verifier detection and false-positive rates",
                    ],
                    [
                        "citation grounded generation unsupported claims",
                        "entailment verification hallucination detection",
                    ],
                    "academic",
                ),
                _sq(
                    "How do retrieval improvements such as reranking, hybrid search and chunking affect faithfulness?",
                    "Better retrieval may prevent hallucinations upstream.",
                    ["faithfulness change from reranking", "chunk size effects"],
                    [
                        "reranker faithfulness RAG",
                        "chunk size faithfulness retrieval augmented generation",
                    ],
                    "academic",
                ),
                _sq(
                    "What role does abstention play in reducing hallucinations, and what does it cost?",
                    "Declining to answer trades coverage for reliability.",
                    ["hallucination reduction from abstention", "coverage loss"],
                    [
                        "abstention RAG hallucination coverage",
                        "calibrated refusal insufficient evidence",
                    ],
                    "academic",
                ),
                _sq(
                    "How should hallucination reduction be measured and operated in production?",
                    "Mitigations need monitoring to stay effective.",
                    [
                        "agreement of automated faithfulness metrics with humans",
                        "operational best practices",
                    ],
                    [
                        "RAG faithfulness evaluation production",
                        "grounding attribution best practices",
                    ],
                    "web",
                ),
            ],
            source_strategy=[
                "Controlled studies and error analyses",
                "Framework documentation",
                "Production engineering reports",
            ],
            stopping_criteria=[
                "Each mitigation has quantified evidence from credible sources",
                "Costs/tradeoffs of each mitigation are stated",
            ],
        ),
    ),
)


def match_sample(question: str) -> SampleQuestion | None:
    lowered = question.lower()
    for sample in SAMPLE_QUESTIONS:
        if lowered.strip() == sample.question.lower().strip():
            return sample
    for sample in SAMPLE_QUESTIONS:
        if all(k in lowered for k in sample.keywords):
            return sample
    return None
