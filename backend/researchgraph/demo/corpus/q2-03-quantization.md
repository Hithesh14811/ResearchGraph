---
title: [Synthetic] The Cost of Compression: Quantization Effects Across Model Sizes
url: https://papers.demo.example/quantization-effects-model-size
authors: Q. Bits; W. Weights
date: 2025-12-01
venue: Synthetic Workshop on Efficient Inference
kind: primary_research
channel: academic
format: html
tags: quantization, 4-bit, accuracy drop, model size, memory, GPU, compression
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# The Cost of Compression: Quantization Effects Across Model Sizes

## Abstract

4-bit weight quantization reduced memory use by 3.6 times with an average accuracy drop of 1.2 points across our benchmark suite. Larger models were more robust to quantization: the 70B model lost 0.6 points while the 3B model lost 2.9 points.

## Results

Quantization allowed the 70B model to run on two GPUs instead of four, roughly halving serving cost per token at moderate load. Quantized small models showed the largest degradation on arithmetic and code generation.
