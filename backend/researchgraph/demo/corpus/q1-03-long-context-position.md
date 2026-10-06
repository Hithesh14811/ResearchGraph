---
title: [Synthetic] Position Effects and Cost in Long-Context Question Answering
url: https://papers.demo.example/long-context-position-effects
authors: S. Mockwell; T. Stub
date: 2025-11-20
venue: Synthetic Findings of Demo NLP 2025
kind: primary_research
channel: academic
format: html
tags: long-context prompting, long context, position, lost in the middle, latency, cost per query, multi-document question answering
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Position Effects and Cost in Long-Context Question Answering

## Abstract

We study how answer accuracy depends on the position of relevant information in contexts of 32K to 128K tokens. Accuracy was 81% when the relevant passage was at the beginning of the context and 58% when it was in the middle, recovering to 74% when it was at the end. Processing a 100K-token context cost roughly 12 times more per query than a retrieval pipeline that supplied 8K tokens.

## Results

Longer contexts increased median latency from 1.1 seconds to 6.4 seconds. Long-context prompting avoids building a retrieval index and handles questions that require synthesizing many passages, where it outperformed top-5 retrieval by 6 points on multi-document questions. Position effects were smaller for the largest model we tested but did not disappear.

## Limitations

Our contexts were constructed from benchmark documents and may not reflect messy enterprise corpora.
