---
title: [Synthetic] Platform Guide: Choosing Between Retrieval, Fine-Tuning and Long Context
url: https://docs.demo.example/guides/retrieval-vs-fine-tuning
authors: Demo Platform Documentation Team
date: 2026-05-01
venue: Demo Platform Documentation
kind: official_documentation
channel: web
format: markdown
tags: choosing retrieval fine-tuning long context, guidance, retrieval-augmented generation, fine-tuning, best practices, operational, knowledge base updates
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Choosing Between Retrieval, Fine-Tuning and Long Context

Use retrieval-augmented generation when answers depend on a large or frequently changing knowledge base, because the index can be updated without retraining the model. Use fine-tuning to teach a model a consistent output format, tone or task-specific behavior; fine-tuning is not a reliable way to add new factual knowledge.

Long-context prompting is appropriate for small document sets that fit in the context window, but cost grows linearly with the number of input tokens on every request. Retrieval and fine-tuning can be combined: fine-tune for format and behavior, and use retrieval for facts.

## Operational notes

Retrieval pipelines require monitoring of retrieval quality, because a retrieval failure rate above 15% typically dominates answer errors. Re-index documents whenever the source of truth changes.
