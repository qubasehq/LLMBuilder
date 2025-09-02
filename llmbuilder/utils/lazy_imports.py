"""
Lazy import utilities to improve startup performance.

This module provides utilities for lazy importing of heavy dependencies
like torch, transformers, etc. to avoid slow startup times.
"""

import importlib
import sys
from typing import Any, Optional, Dict, Callable
from functools import wraps


class LazyImport:
    """Lazy import wrapper that imports modules only when accessed."""
    
    def __init__(self, module_name: str, attribute: Optional[str] = None):
        self.module_name = module_name
        self.attribute = attribute
        self._module = None
    
    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        
        if self.attribute:
            attr = getattr(self._module, self.attribute)
            return getattr(attr, name)
        else:
            return getattr(self._module, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        
        if self.attribute:
            attr = getattr(self._module, self.attribute)
            return attr(*args, **kwargs)
        else:
            return self._module(*args, **kwargs)


def lazy_import(module_name: str, attribute: Optional[str] = None) -> LazyImport:
    """Create a lazy import for a module or module attribute."""
    return LazyImport(module_name, attribute)


def requires_torch(func: Callable) -> Callable:
    """Decorator that ensures torch is available before function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
from llmbuilder.utils.lazy_imports import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for this functionality. "
                "Install it with: pip install torch"
            )
        return func(*args, **kwargs)
    return wrapper


def requires_transformers(func: Callable) -> Callable:
    """Decorator that ensures transformers is available before function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "Transformers is required for this functionality. "
                "Install it with: pip install transformers"
            )
        return func(*args, **kwargs)
    return wrapper


def requires_sentence_transformers(func: Callable) -> Callable:
    """Decorator that ensures sentence-transformers is available before function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import sentence_transformers
        except ImportError:
            raise ImportError(
                "Sentence Transformers is required for this functionality. "
                "Install it with: pip install sentence-transformers"
            )
        return func(*args, **kwargs)
    return wrapper


# Common lazy imports for heavy ML libraries
torch = lazy_import('torch')
transformers = lazy_import('transformers')
sentence_transformers = lazy_import('sentence_transformers')
numpy = lazy_import('numpy')
pandas = lazy_import('pandas')

# Specific commonly used classes/functions
AutoTokenizer = lazy_import('transformers', 'AutoTokenizer')
AutoModel = lazy_import('transformers', 'AutoModel')
SentenceTransformer = lazy_import('sentence_transformers', 'SentenceTransformer')


def check_ml_dependencies() -> Dict[str, bool]:
    """Check which ML dependencies are available without importing them."""
    dependencies = {
        'torch': False,
        'transformers': False,
        'sentence_transformers': False,
        'numpy': False,
        'pandas': False,
    }
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            dependencies[dep] = True
        except ImportError:
            pass
    
    return dependencies


def get_torch_device() -> str:
    """Get the best available torch device without importing torch at module level."""
    try:
from llmbuilder.utils.lazy_imports import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'