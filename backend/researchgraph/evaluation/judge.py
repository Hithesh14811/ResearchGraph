"""LLM-as-judge rubric evaluation (optional and kept separate from deterministic metrics).

Judge scores are subjective model opinions. They are reported in their own section, never
blended into the deterministic metrics, and are explicitly labelled as non-meaningful when
the configured model is the offline mock.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from researchgraph.agents.base import UNTRUSTED_CONTENT_POLICY, make_prompt, render_block
from researchgraph.core.text import truncate
from researchgraph.llm.factory import ModelRegistry, ModelTier
from researchgraph.llm.structured import invoke_structured

RUBRIC = {
    "faithfulness": "Every statement is supported by the cited evidence; no invented facts or numbers.",
    "completeness": "All important aspects of the question are addressed.",
    "balance": "Conflicting evidence, uncertainty and limitations are represented fairly.",
    "actionability": "Recommendations are specific, justified and useful to a practitioner.",
    "clarity": "The report is well organised, precise and concise.",
}


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    balance: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    rationale: str = Field(description="Two or three sentences justifying the scores.")

    @property
    def mean(self) -> float:
        return round(
            (
                self.faithfulness
                + self.completeness
                + self.balance
                + self.actionability
                + self.clarity
            )
            / 5,
            3,
        )


SYSTEM = """You are a strict reviewer grading a technical research report. Score each criterion from 1 (poor) to 5 (excellent):
{rubric}
Use the evidence sample to spot-check faithfulness. Be critical: a 5 means you found no problems.
{policy}"""

PROMPT = make_prompt(SYSTEM, "{request}\n\n{report}\n\n{evidence}")


async def judge_report(
    models: ModelRegistry,
    *,
    question: str,
    report_markdown: str,
    evidence_sample: list[dict[str, str]],
) -> JudgeOutput:
    messages = (
        await PROMPT.ainvoke(
            {
                "rubric": "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items()),
                "policy": UNTRUSTED_CONTENT_POLICY,
                "request": render_block("judge_request", {"question": question}),
                "report": render_block("report", truncate(report_markdown, 30_000)),
                "evidence": render_block("evidence", evidence_sample[:25]),
            }
        )
    ).to_messages()
    result = await invoke_structured(
        models.get(ModelTier.STRONG),
        JudgeOutput,
        messages,
        operation="llm_judge",
        policy=models.policy,
    )
    return result.value
