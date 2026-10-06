---
title: [Synthetic] Does Fine-Tuning Inject New Knowledge? Evidence from Controlled Fact Insertion
url: https://papers.demo.example/fine-tuning-knowledge-injection
authors: M. Sample; J. Fixture
date: 2025-06-02
venue: Synthetic Transactions on Language Model Adaptation
kind: primary_research
channel: academic
format: markdown
tags: fine-tuning, knowledge injection, new facts, catastrophic forgetting, domain adaptation, hallucination, accuracy
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Does Fine-Tuning Inject New Knowledge? Evidence from Controlled Fact Insertion

## Abstract

We insert 5,000 synthetic facts into fine-tuning data and measure whether models can answer questions about them. Facts that appeared only in fine-tuning data were answered correctly 38% of the time, compared with 79% when the same facts were supplied in the prompt context. Fine-tuning on domain data reduced performance on a general-knowledge benchmark by 3.1 points, consistent with partial catastrophic forgetting.

## Results

Accuracy on inserted facts improved with repeated exposure: facts seen 10 times during fine-tuning reached 52% accuracy. Fine-tuned models answered confidently on 23% of questions about facts they had never seen, a form of hallucination that was rare when facts were provided in context. Fine-tuning was effective at teaching output style and domain terminology, improving terminology accuracy from 61% to 90%.

## Discussion

Fine-tuning is better suited to shaping model behavior and format than to adding frequently changing facts. Knowledge that changes often should be supplied at inference time.

## References

[1] Sample, M. (2023). Synthetic adaptation methods.
[2] Fixture, J. (2024). Synthetic forgetting analysis.
