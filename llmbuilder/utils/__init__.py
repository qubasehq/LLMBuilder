"""
Utility modules for LLMBuilder.

This module contains utility functions and classes used throughout the package.
"""

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import setup_logging

# Lazy import vocab_sync to avoid circular imports
def _get_vocab_sync():
    from llmbuilder.utils.vocab_sync import (
        VocabSyncManager,
        auto_sync_vocab,
        get_vocab_analysis,
        sync_config_with_tokenizer
    )
    return VocabSyncManager, auto_sync_vocab, get_vocab_analysis, sync_config_with_tokenizer

# Export for direct access
def get_vocab_sync_manager():
    VocabSyncManager, _, _, _ = _get_vocab_sync()
    return VocabSyncManager

__all__ = [
    "ConfigManager",
    "setup_logging",
    "get_vocab_sync_manager",
]