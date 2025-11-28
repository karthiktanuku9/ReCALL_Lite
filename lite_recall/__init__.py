"""
ReCALL Lite
A lightweight, AGI-inspired memory layer for LLMs.
"""

# Expose main classes for easy import
from .recall_core import (
    recall_lite,
    LiteAgent,
    ModelConnector,
    OpenAIConnector,
    GeminiAPIConnector,
    HuggingFaceConnector,
    OllamaConnector
)

__all__ = [
    "recall_lite",
    "LiteAgent",
    "ModelConnector",
    "OpenAIConnector",
    "GeminiAPIConnector",
    "HuggingFaceConnector",
    "OllamaConnector",
]
