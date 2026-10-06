---
title: [Synthetic] Energy Consumption of LLM Inference: A Measurement Study
url: https://institute.demo.example/reports/llm-inference-energy
authors: Demo Institute for Sustainable Computing
date: 2025-08-20
venue: Demo Institute Technical Report
kind: institutional
channel: web
format: pdf
tags: energy consumption llm inference per token, energy, model size, batching, sustainability, hardware
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Energy Consumption of LLM Inference: A Measurement Study

## Findings

Measured energy per 1,000 generated tokens was 0.6 Wh for an 8B model and 5.1 Wh for a 70B model on the same hardware. Energy use scaled roughly linearly with the number of active parameters. Batching reduced energy per token by up to 40% at high load.

## Method

Energy was measured at the wall for a dedicated inference server over 72 hours of replayed production traffic.
