"""Strict checkpoint serialisation.

LangGraph's default serializer will import and construct any type named in checkpoint data,
which turns write access to the checkpoint database into code execution. We instead pass an
explicit allowlist: every Pydantic model and enum in ``researchgraph.schemas`` (LangChain
message types and stdlib value types are allowed by LangGraph's built-in safe list).
"""

from __future__ import annotations

import inspect
from enum import Enum
from functools import cache

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from researchgraph.schemas import common, events, evidence, report, research, review, sources

_SCHEMA_MODULES = (common, events, evidence, report, research, review, sources)


@cache
def state_types() -> tuple[type, ...]:
    found: dict[str, type] = {}
    for module in _SCHEMA_MODULES:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__ and issubclass(obj, BaseModel | Enum):
                found[f"{obj.__module__}.{obj.__name__}"] = obj
    return tuple(found.values())


def state_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=list(state_types()))
