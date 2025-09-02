"""
LLMBuilder - Complete LLM Training and Deployment Pipeline

A comprehensive, production-ready implementation for training and fine-tuning 
Large Language Models from scratch with enhanced document ingestion, intelligent 
deduplication, model training, automated GGUF conversion, and comprehensive testing.
"""

import sys
from pathlib import Path

# Version information
__version__ = "1.0.0"
__author__ = "Qubase"
__email__ = "contact@qubase.in"
__license__ = "MIT"
__description__ = "Complete LLM Training and Deployment Pipeline with CLI"

# Package metadata
__title__ = "llmbuilder"
__url__ = "https://github.com/qubase/llmbuilder"
__download_url__ = "https://pypi.org/project/llmbuilder/"
__docs_url__ = "https://llmbuilder.readthedocs.io"

# Minimum Python version check
MIN_PYTHON_VERSION = (3, 8)
if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError(
        f"LLMBuilder requires Python {'.'.join(map(str, MIN_PYTHON_VERSION))} or higher. "
        f"You are using Python {'.'.join(map(str, sys.version_info[:2]))}."
    )

# Package root directory
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Lazy imports for backward compatibility - only import when needed
# This prevents slow startup times from heavy ML libraries
def __getattr__(name):
    """Lazy import core modules to avoid startup delays."""
    if name in ['data', 'training', 'model', 'finetune', 'tools', 'eval']:
        from llmbuilder import core
        return getattr(core, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Version info for backward compatibility
VERSION = __version__

# Export version for setuptools dynamic versioning
def get_version():
    """Get package version."""
    return __version__