"""
Advanced Cybersecurity Dataset Cleaning Module

This module provides comprehensive text cleaning capabilities specifically designed
for cybersecurity training datasets. It includes boilerplate removal, language filtering,
domain relevance assessment, quality evaluation, entity preservation, and repetition handling.
"""

from .data_models import (
    CleaningResult, CleaningStats, Entity, CleaningOperation,
    EntityType, CleaningOperationType
)
from .base_module import CleaningModule
from .config_manager import AdvancedCleaningConfig, ConfigManager
from .advanced_text_cleaner import AdvancedTextCleaner
from .modules import BoilerplateRemover

__version__ = "1.0.0"
__all__ = [
    "CleaningResult",
    "CleaningStats", 
    "Entity",
    "EntityType",
    "CleaningOperation",
    "CleaningOperationType",
    "CleaningModule",
    "AdvancedCleaningConfig",
    "ConfigManager",
    "AdvancedTextCleaner",
    "BoilerplateRemover"
]