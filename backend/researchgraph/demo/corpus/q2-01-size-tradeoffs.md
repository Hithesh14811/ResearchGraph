---
title: [Synthetic] Accuracy, Latency and Cost Across Model Sizes in Production Workloads
url: https://papers.demo.example/model-size-production-tradeoffs
authors: G. Benchmark; N. Synthetic
date: 2026-03-03
venue: Synthetic Proceedings of Demo Systems for ML 2026
kind: primary_research
channel: academic
format: pdf
tags: small vs large language models, model size, accuracy, reasoning, classification, extraction, latency, throughput, production tasks, benchmark
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Accuracy, Latency and Cost Across Model Sizes in Production Workloads

## Abstract

We benchmark 3B, 8B and 70B-parameter models on 12 production-style tasks. The 70B model scored 11.2 points higher than the 8B model on multi-step reasoning tasks, but only 2.4 points higher on classification and extraction tasks. After task-specific fine-tuning, the 8B model reached 94% of the 70B model's accuracy on narrow extraction tasks.

## Results

The 8B model served requests with 4.5 times lower median latency and 7 times higher throughput per GPU than the 70B model. The 3B model was fastest but lost 9.8 points on reasoning tasks relative to the 8B model. Gaps between sizes were largest for tasks requiring long chains of reasoning or broad world knowledge.

## Limitations

All models came from one model family and were served with the same inference engine.
