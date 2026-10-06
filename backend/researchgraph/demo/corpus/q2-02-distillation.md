---
title: [Synthetic] Distilling Large Language Models into Small Task Specialists
url: https://papers.demo.example/distilling-llms-small-specialists
authors: C. Teacher; D. Student
date: 2025-10-10
venue: Synthetic Journal of Efficient Machine Learning
kind: primary_research
channel: academic
format: markdown
tags: distillation, small model, teacher performance, out-of-distribution, task specialist, compression
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Distilling Large Language Models into Small Task Specialists

## Abstract

A 3B student model distilled from a 70B teacher retained 89% of the teacher's performance on in-distribution tasks. On out-of-distribution inputs, the student retained only 61% of teacher performance. Distillation required about 200,000 teacher-generated examples per task.

## Discussion

Distilled specialists are attractive when the input distribution is narrow and stable. When inputs drift, the gap to the teacher widens quickly, so production systems need monitoring and a fallback to a larger model.
