---
title: [Synthetic] Case Study: Building a Legal Research Assistant with Retrieval
url: https://engineering.demo.example/blog/legal-assistant-retrieval-case-study
authors: P. Placeholder
date: 2025-09-15
venue: Demo Engineering Blog
kind: reputable_technical
channel: web
format: html
tags: retrieval-augmented generation production case study, legal, fine-tuned model, unsupported answers, latency, index refresh
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Case Study: Building a Legal Research Assistant with Retrieval

Our first legal assistant used a fine-tuned model trained on case summaries. Switching from the fine-tuned model to a retrieval-augmented design reduced unsupported answers from 17% to 6% in our internal evaluation of 1,200 questions.

The retrieval system added latency: p95 response time rose from 0.9 seconds to 1.8 seconds. Updating the knowledge base became a daily index refresh instead of a monthly fine-tuning job.

The main failure mode was retrieval of outdated statutes, which accounted for most of the remaining errors. We now attach effective dates to every indexed document and filter on them at query time.
