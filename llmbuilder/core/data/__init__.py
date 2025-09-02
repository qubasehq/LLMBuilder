"""
Data processing modules for LLMBuilder.

This module contains all data-related functionality including ingestion,
preprocessing, deduplication, and dataset management.
"""

# Import key classes for easy access
from llmbuilder.core.data.ingest import DocumentIngester
from llmbuilder.core.data.dedup import HashDeduplicator, EmbeddingDeduplicator
# DataPreprocessor is in training module, not data module

__all__ = [
    "DocumentIngester",
    "HashDeduplicator", 
    "EmbeddingDeduplicator",
]