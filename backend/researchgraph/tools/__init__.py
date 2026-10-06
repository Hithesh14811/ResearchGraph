"""Tool layer.

* LLM-callable tools (``research_tools``): web search, academic search, calculator.
* Pipeline tools (called by graph nodes, never by the model): safe fetching, document
  extraction and metadata extraction. Keeping fetch out of the model's reach means the LLM
  can never request an arbitrary URL.
"""
