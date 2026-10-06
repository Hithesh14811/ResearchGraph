---
title: [Synthetic] Retrieval, Fine-Tuning or Long Context? A Controlled Comparison on Domain-Specific QA
url: https://papers.demo.example/rag-ft-long-context-domain-qa
authors: R. Demo; L. Example; K. Placeholder
date: 2026-02-10
venue: Synthetic Proceedings of the Demo Conference on Knowledge-Intensive NLP
kind: primary_research
channel: academic
format: pdf
tags: retrieval-augmented generation, rag, fine-tuning, long-context prompting, domain-specific question answering, accuracy, exact match, benchmark, comparison
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Retrieval, Fine-Tuning or Long Context? A Controlled Comparison on Domain-Specific QA

## Abstract

We compare retrieval-augmented generation (RAG), supervised fine-tuning and long-context prompting on three domain-specific question answering benchmarks (medical, legal and financial) using the same 8B-parameter base model. Retrieval-augmented generation achieved the highest exact-match accuracy, 71.4% on average, compared with 64.2% for fine-tuning and 66.8% for long-context prompting. Combining retrieval with a fine-tuned model reached 74.9% exact-match accuracy, the best overall result.

## Method

All systems answered the same 3,000 held-out questions. The retrieval system indexed a knowledge base of 120,000 domain documents and supplied the top eight passages to the model. The long-context condition received up to 100,000 tokens of the most relevant documents. Fine-tuning used 40,000 question-answer pairs generated from the same knowledge base.

## Results

The advantage of retrieval was largest on questions about facts updated after the model's training cutoff, where RAG scored 68.0% exact match versus 31.5% for fine-tuning. Long-context prompting matched retrieval when the relevant passage appeared in the first 10% of the context window but fell 13 points behind retrieval when the passage appeared in the middle of the context. Fine-tuning produced the most consistent answer formatting, with 97% of answers following the required template versus 88% for RAG.

## Limitations

We evaluated a single model family, all benchmarks were in English, and costs were measured on one hardware configuration. Results may differ for much larger models.

## References

[1] Demo, R. (2024). Synthetic baseline methods. Demo Workshop.
[2] Example, L. et al. (2025). Synthetic retrieval study. Demo Journal.
