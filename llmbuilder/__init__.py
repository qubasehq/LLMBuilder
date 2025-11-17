"""
LLMBuilder - A comprehensive toolkit for building, training, and deploying language models.

This package provides a modular approach to language model development with support for:
- Data loading and preprocessing
- Tokenizer training and management
- Model building and configuration
- Training and fine-tuning pipelines
- Text generation and inference
- Model export in various formats
"""

__version__ = "2.0.0"
__author__ = "Qub△se"
__email__ = "contact@qubase.in"

import importlib
from pathlib import Path

# Lazy import machinery to keep import time low
from typing import Any, Dict, List, Optional, Union

# Main API exports for easy access
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core modules
    "config",
    "data",
    "tokenizer",
    "model",
    "training",
    "finetune",
    "inference",
    "export",
    "tools",
    "utils",
    "pipeline",
    # High-level API
    "train",
    "generate_text",
    "interactive_cli",
]

# Lazy-load submodules on first attribute access (PEP 562)
_SUBMODULES = {
    "config",
    "data",
    "tokenizer",
    "model",
    "training",
    "finetune",
    "inference",
    "export",
    "tools",
    "utils",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Convenience functions for common operations
def load_config(path: Optional[str] = None, preset: Optional[str] = None) -> Any:
    """Load configuration from file or use preset.

    Args:
        path: Optional path to a JSON/YAML configuration file.
        preset: Optional name of a built-in preset.

    Returns:
        A configuration object suitable for model/training builders.
    """
    from .config import load_config as _load_config

    return _load_config(path, preset)


def build_model(config: Any) -> Any:
    """Build a model from configuration."""
    from .model import build_model as _build_model

    return _build_model(config)


def train_model(model: Any, dataset: Any, config: Any) -> Any:
    """Train a model with the given dataset and configuration."""
    from .training import train_model as _train_model

    return _train_model(model, dataset, config)


def train(
    data_path: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    clean: bool = False,
) -> Any:
    """
    High-level training function that handles the complete training pipeline.

    Args:
        data_path: Path to input data file(s) or directory
        output_dir: Directory to save outputs (tokenizer, checkpoints, etc.)
        config: Optional configuration dictionary
        clean: If True, clean up previous outputs before starting

    Returns:
        TrainingPipeline: The trained pipeline instance

    Example:
        >>> import llmbuilder
        >>>
        >>> # Train with default settings
        >>> pipeline = llmbuilder.train(
        ...     data_path="./my_data/",
        ...     output_dir="./output/"
        ... )
        >>>
        >>> # Generate text after training
        >>> text = pipeline.generate("The future of AI is")
    """
    from .pipeline import train as _train

    return _train(data_path, output_dir, config or {}, clean)


def generate_text(
    model_path: str, tokenizer_path: str, prompt: str, **kwargs: Any
) -> str:
    """
    Generate text using a trained model.

    Args:
        model_path: Path to trained model checkpoint
        tokenizer_path: Path to tokenizer directory
        prompt: Input text prompt
        **kwargs: Additional generation parameters (temperature, top_k, top_p, etc.)

    Returns:
        Generated text string

    Example:
        >>> import llmbuilder
        >>>
        >>> text = llmbuilder.generate_text(
        ...     model_path="./output/checkpoints/model.pt",
        ...     tokenizer_path="./output/tokenizer/",
        ...     prompt="The future of AI is",
        ...     max_new_tokens=100,
        ...     temperature=0.8
        ... )
    """
    from .inference import generate_text as _generate_text

    return _generate_text(model_path, tokenizer_path, prompt, **kwargs)


def interactive_cli(model_path: str, tokenizer_path: str, **kwargs: Any) -> None:
    """
    Start an interactive CLI for text generation.

    Args:
        model_path: Path to trained model checkpoint
        tokenizer_path: Path to tokenizer directory
        **kwargs: Additional configuration parameters

    Example:
        >>> import llmbuilder
        >>>
        >>> llmbuilder.interactive_cli(
        ...     model_path="./output/checkpoints/model.pt",
        ...     tokenizer_path="./output/tokenizer/",
        ...     temperature=0.8
        ... )
    """
    from .inference import interactive_cli as _interactive_cli

    _interactive_cli(model_path, tokenizer_path, **kwargs)


def finetune_model(model: Any, dataset: Any, config: Any, **kwargs: Any) -> Any:
    """Fine-tune a model with the given dataset and configuration."""
    from .finetune import finetune_model as _finetune_model

    return _finetune_model(model, dataset, config, **kwargs)
