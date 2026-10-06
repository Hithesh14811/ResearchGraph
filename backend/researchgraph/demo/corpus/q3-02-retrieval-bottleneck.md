---
title: [Synthetic] Retrieval Failures Explain Most RAG Hallucinations
url: https://papers.demo.example/retrieval-failures-rag-hallucinations
authors: R. Recall; P. Precision
date: 2025-09-09
venue: Synthetic Findings of Demo IR 2025
kind: primary_research
channel: academic
format: markdown
tags: rag hallucination causes retrieval failure analysis, retrieval errors, unfaithful answers, reranker, hybrid retrieval, faithfulness
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Retrieval Failures Explain Most RAG Hallucinations

## Abstract

In an error analysis of 2,000 retrieval-augmented answers, 52% of hallucinated answers were traced to retrieval failures in which relevant context was missing or irrelevant passages were retrieved. Adding a cross-encoder reranker improved answer faithfulness by 9 points.

## Results

Hybrid lexical and dense retrieval reduced retrieval failures by 18% relative to dense retrieval alone. The remaining hallucinations occurred when the model ignored or misread correctly retrieved context.
