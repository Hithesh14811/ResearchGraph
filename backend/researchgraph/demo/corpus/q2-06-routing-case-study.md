---
title: [Synthetic] Cascading Small and Large Models: A Production Case Study
url: https://engineering.demo.example/blog/model-cascade-routing
authors: E. Router
date: 2026-02-25
venue: Demo Engineering Blog
kind: reputable_technical
channel: web
format: html
tags: cascade routing small large model cost, model routing production case study, escalation, confidence, cost reduction
---
> Synthetic demo document. Created for the ResearchGraph offline demo; not a real publication. All figures are illustrative.

# Cascading Small and Large Models: A Production Case Study

Routing each request to a small model first and escalating to a large model only when the small model's confidence was low handled 78% of requests with the small model. The cascade reduced inference cost by 64% with a 1.5-point drop in task success rate.

The router misclassified 6% of hard requests as easy, which caused most of the quality loss. We retrain the router monthly on escalation outcomes.
