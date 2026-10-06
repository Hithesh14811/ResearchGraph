---
title: [Synthetic] Total Cost of Ownership for Enterprise LLM Question Answering
url: https://institute.demo.example/reports/llm-question-answering-tco
authors: Demo Institute for Applied AI
date: 2026-04-08
venue: Demo Institute Technical Report
kind: institutional
channel: web
format: pdf
tags: total cost of ownership, cost per query, retrieval-augmented generation cost, fine-tuning retraining, long context cost, enterprise question answering
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Total Cost of Ownership for Enterprise LLM Question Answering

## Summary

Across five deployments, retrieval-augmented systems cost between $0.40 and $1.10 per 1,000 queries for retrieval infrastructure in addition to model inference. Fine-tuning required an up-front training cost and periodic retraining; teams retrained every 4 to 8 weeks to keep knowledge current.

Long-context prompting with 100,000-token prompts was the most expensive option, costing 10 to 20 times more per query than retrieval with 8,000-token prompts. The report recommends retrieval for large, changing corpora and fine-tuning for stable, behavior-focused tasks.

## Method

We interviewed engineering teams and analysed twelve months of infrastructure invoices for each deployment.
