"""
Import caching utilities to further improve performance.

This module provides caching mechanisms for expensive imports
and module loading operations.
"""

import sys
import time
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps

# Global import cache
_IMPORT_CACHE: Dict[str, Any] = {}
_CACHE_FILE = Path.home() / ".llmbuilder" / "import_cache.pkl"


def get_cache_key(module_name: str, *args, **kwargs) -> str:
    """Generate a cache key for a module import."""
    key_data = f"{module_name}:{args}:{sorted(kwargs.items())}"
    return hashlib.md5(key_data.encode()).hexdigest()


def cached_import(module_name: str, attribute: Optional[str] = None):
    """Cache expensive imports to improve subsequent startup times."""
    cache_key = get_cache_key(module_name, attribute)
    
    if cache_key in _IMPORT_CACHE:
        return _IMPORT_CACHE[cache_key]
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        
        if attribute:
            result = getattr(module, attribute)
        else:
            result = module
        
        _IMPORT_CACHE[cache_key] = result
        return result
        
    except ImportError as e:
        # Cache the failure to avoid repeated attempts
        _IMPORT_CACHE[cache_key] = None
        raise e


def preload_common_imports():
    """Preload commonly used imports in the background."""
    common_imports = [
        ('click', None),
        ('pathlib', 'Path'),
        ('json', None),
        ('os', None),
        ('sys', None),
    ]
    
    for module_name, attribute in common_imports:
        try:
            cached_import(module_name, attribute)
        except ImportError:
            pass  # Skip unavailable modules


def timed_import(func: Callable) -> Callable:
    """Decorator to measure import times for debugging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log slow imports (> 100ms)
        if end_time - start_time > 0.1:
            print(f"Slow import detected: {func.__name__} took {end_time - start_time:.3f}s")
        
        return result
    return wrapper


def save_cache():
    """Save the import cache to disk."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, 'wb') as f:
            pickle.dump(_IMPORT_CACHE, f)
    except Exception:
        pass  # Fail silently


def load_cache():
    """Load the import cache from disk."""
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                _IMPORT_CACHE.update(cached_data)
    except Exception:
        pass  # Fail silently


def clear_cache():
    """Clear the import cache."""
    global _IMPORT_CACHE
    _IMPORT_CACHE.clear()
    
    try:
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()
    except Exception:
        pass


# Initialize cache on module import
load_cache()


class LazyModule:
    """A lazy module loader that caches results."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None
    
    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = cached_import(self.module_name)
        return getattr(self._module, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        if self._module is None:
            self._module = cached_import(self.module_name)
        return self._module(*args, **kwargs)


# Commonly used lazy modules
torch = LazyModule('torch')
transformers = LazyModule('transformers')
numpy = LazyModule('numpy')
pandas = LazyModule('pandas')