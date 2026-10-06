"""LLM-backed reasoning roles.

Only steps that need judgement are agents (planning, query writing, tool selection,
extraction, contradiction adjudication, synthesis, critique, writing). Scoring, gating,
citation numbering and verification are deterministic services by design.
"""
