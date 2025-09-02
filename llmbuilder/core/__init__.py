"""
Core functionality for LLMBuilder.

This module contains all the core components for LLM training, data processing,
model management, and deployment. It provides a clean API for both CLI and
programmatic usage.
"""

__all__ = [
    'data',
    'training', 
    'model',
    'finetune',
    'tools',
    'eval'
]

def __getattr__(name):
    """Lazy import core modules to avoid startup delays from heavy ML libraries."""
    if name == 'data':
        from . import data
        return data
    elif name == 'training':
        from . import training
        return training
    elif name == 'model':
        from . import model
        return model
    elif name == 'finetune':
        from . import finetune
        return finetune
    elif name == 'tools':
        from . import tools
        return tools
    elif name == 'eval':
        from . import eval
        return eval
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")