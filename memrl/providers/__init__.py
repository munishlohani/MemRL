"""
Provider implementations for LLM and embedding services.

This package contains abstract base classes and concrete implementations
for various AI service providers used in the Memp system.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Base classes
    "BaseLLM",
    "BaseEmbedder", 
    "ProviderError",
    "LLMError",
    "EmbedderError",
    
    # LLM providers
    "OpenAILLM",
    "MockLLM",
    
    # Embedding providers
    "OpenAIEmbedder",
    "LocalEmbedder",
    "MockEmbedder",
    "AverageEmbedder",
]


def __getattr__(name: str) -> Any:
    if name in {"BaseLLM", "BaseEmbedder", "ProviderError", "LLMError", "EmbedderError"}:
        module = import_module(".base", __name__)
        return getattr(module, name)
    if name in {"OpenAILLM", "MockLLM"}:
        module = import_module(".llm", __name__)
        return getattr(module, name)
    if name in {"OpenAIEmbedder", "LocalEmbedder", "MockEmbedder", "AverageEmbedder"}:
        module = import_module(".embedding", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
