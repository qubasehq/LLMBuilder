"""
Legacy Import Compatibility Layer

This module ensures that existing Python imports continue to work after
the package migration, providing backward compatibility for user code.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
import importlib.util

from llmbuilder.compat.deprecation import warn_legacy_import

# Track which legacy imports have been warned about
_warned_imports = set()


def setup_legacy_import_hooks():
    """
    Set up import hooks to handle legacy module imports.
    This allows old import statements to continue working.
    """
    # Add legacy import paths to sys.path if they exist
    project_root = Path.cwd()
    legacy_paths = [
        project_root / "training",
        project_root / "data", 
        project_root / "model",
        project_root / "finetune",
        project_root / "eval",
        project_root / "tools",
        project_root / "scripts"
    ]
    
    for path in legacy_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))


class LegacyImportHandler:
    """
    Handles legacy imports and provides compatibility mappings.
    """
    
    def __init__(self):
        # Mapping of legacy imports to new imports
        self.import_mappings = {
            # Training modules
            'training.train': 'llmbuilder.core.training.train',
            'training.preprocess': 'llmbuilder.core.data.preprocess',
            'training.dataset': 'llmbuilder.core.training.dataset',
            'training.utils': 'llmbuilder.core.training.utils',
            'training.tokenize_corpus': 'llmbuilder.core.data.tokenize_corpus',
            'training.train_tokenizer': 'llmbuilder.core.data.train_tokenizer',
            
            # Data modules
            'data.ingest': 'llmbuilder.core.data.ingest',
            'data.dedup': 'llmbuilder.core.data.dedup',
            
            # Model modules
            'model.gpt_model': 'llmbuilder.core.model.gpt_model',
            'model.simple_transformer': 'llmbuilder.core.model.simple_transformer',
            
            # Fine-tuning modules
            'finetune.finetune': 'llmbuilder.core.finetune.finetune',
            'finetune.peft_config': 'llmbuilder.core.finetune.peft_config',
            'finetune.peft_model_utils': 'llmbuilder.core.finetune.peft_model_utils',
            'finetune.peft_model_wrapper': 'llmbuilder.core.finetune.peft_model_wrapper',
            
            # Evaluation modules
            'eval.eval': 'llmbuilder.core.eval.eval',
            
            # Tools modules
            'tools.conversion_pipeline': 'llmbuilder.core.tools.conversion_pipeline',
            'tools.quantization_manager': 'llmbuilder.core.tools.quantization_manager',
            'tools.export_gguf': 'llmbuilder.core.tools.export_gguf',
            'tools.generate_model_card': 'llmbuilder.core.tools.generate_model_card',
            'tools.download_hf_model': 'llmbuilder.core.tools.download_hf_model',
        }
        
        # Common function/class aliases for backward compatibility
        self.symbol_mappings = {
            # Training functions
            'train_model': 'llmbuilder.core.training.train.train_model',
            'preprocess_data': 'llmbuilder.core.data.preprocess.preprocess_data',
            'create_dataset': 'llmbuilder.core.training.dataset.create_dataset',
            
            # Model classes
            'GPTModel': 'llmbuilder.core.model.gpt_model.GPTModel',
            'SimpleTransformer': 'llmbuilder.core.model.simple_transformer.SimpleTransformer',
            
            # Fine-tuning classes
            'PEFTConfig': 'llmbuilder.core.finetune.peft_config.PEFTConfig',
            'PEFTModelWrapper': 'llmbuilder.core.finetune.peft_model_wrapper.PEFTModelWrapper',
            
            # Utility functions
            'ingest_documents': 'llmbuilder.core.data.ingest.ingest_documents',
            'deduplicate_data': 'llmbuilder.core.data.dedup.deduplicate_data',
        }
    
    def handle_legacy_import(self, module_name: str) -> Optional[Any]:
        """
        Handle a legacy import by mapping it to the new module structure.
        
        Args:
            module_name: Name of the legacy module being imported
            
        Returns:
            The imported module or None if not found
        """
        # Check if this is a known legacy import
        if module_name in self.import_mappings:
            new_module_name = self.import_mappings[module_name]
            
            # Warn about deprecated import (only once per module)
            if module_name not in _warned_imports:
                warn_legacy_import(module_name, new_module_name)
                _warned_imports.add(module_name)
            
            try:
                # Try to import the new module
                return importlib.import_module(new_module_name)
            except ImportError:
                # Fall back to legacy import if new module doesn't exist
                return self._try_legacy_import(module_name)
        
        return None
    
    def _try_legacy_import(self, module_name: str) -> Optional[Any]:
        """
        Try to import a module using the legacy path structure.
        
        Args:
            module_name: Name of the module to import
            
        Returns:
            The imported module or None if not found
        """
        try:
            # Try direct import (legacy modules should be in sys.path)
            return importlib.import_module(module_name)
        except ImportError:
            # Try with different path variations
            variations = [
                module_name.replace('.', '/') + '.py',
                module_name.replace('.', '\\') + '.py'
            ]
            
            project_root = Path.cwd()
            for variation in variations:
                module_path = project_root / variation
                if module_path.exists():
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return module
            
            return None


# Global import handler instance
_import_handler = LegacyImportHandler()


# Legacy import compatibility functions
def import_training_module(module_name: str = 'train') -> Any:
    """Import training module with backward compatibility."""
    full_name = f'training.{module_name}'
    return _import_handler.handle_legacy_import(full_name)


def import_data_module(module_name: str = 'ingest') -> Any:
    """Import data module with backward compatibility."""
    full_name = f'data.{module_name}'
    return _import_handler.handle_legacy_import(full_name)


def import_model_module(module_name: str = 'gpt_model') -> Any:
    """Import model module with backward compatibility."""
    full_name = f'model.{module_name}'
    return _import_handler.handle_legacy_import(full_name)


def import_finetune_module(module_name: str = 'finetune') -> Any:
    """Import finetune module with backward compatibility."""
    full_name = f'finetune.{module_name}'
    return _import_handler.handle_legacy_import(full_name)


def import_eval_module(module_name: str = 'eval') -> Any:
    """Import evaluation module with backward compatibility."""
    full_name = f'eval.{module_name}'
    return _import_handler.handle_legacy_import(full_name)


def import_tools_module(module_name: str = 'conversion_pipeline') -> Any:
    """Import tools module with backward compatibility."""
    full_name = f'tools.{module_name}'
    return _import_handler.handle_legacy_import(full_name)


# Convenience imports for common legacy usage patterns
# Create placeholder functions that show migration messages
def _create_migration_placeholder(old_name: str, new_name: str):
    def placeholder(*args, **kwargs):
        raise ImportError(f"""
        The function '{old_name}' has been moved to '{new_name}'.
        
        Please update your import:
        OLD: from {old_name.split('.')[0]} import {old_name.split('.')[-1]}
        NEW: from {new_name} import {new_name.split('.')[-1]}
        
        Or use the new CLI: llmbuilder <command>
        """)
    return placeholder

# Try to import new modules, fall back to placeholders
try:
    from llmbuilder.core.training.train import train_model
except ImportError:
    train_model = _create_migration_placeholder('training.train_model', 'llmbuilder.core.training.train.train_model')

try:
    from llmbuilder.core.data.preprocess import preprocess_data
except ImportError:
    preprocess_data = _create_migration_placeholder('training.preprocess_data', 'llmbuilder.core.data.preprocess.preprocess_data')

try:
    from llmbuilder.core.training.dataset import create_dataset
except ImportError:
    create_dataset = _create_migration_placeholder('training.create_dataset', 'llmbuilder.core.training.dataset.create_dataset')

try:
    from llmbuilder.core.model.gpt_model import GPTModel
except ImportError:
    GPTModel = _create_migration_placeholder('model.GPTModel', 'llmbuilder.core.model.gpt_model.GPTModel')

try:
    from llmbuilder.core.model.simple_transformer import SimpleTransformer
except ImportError:
    SimpleTransformer = _create_migration_placeholder('model.SimpleTransformer', 'llmbuilder.core.model.simple_transformer.SimpleTransformer')

try:
    from llmbuilder.core.data.ingest import ingest_documents
except ImportError:
    ingest_documents = _create_migration_placeholder('data.ingest_documents', 'llmbuilder.core.data.ingest.ingest_documents')

try:
    from llmbuilder.core.data.dedup import deduplicate_data
except ImportError:
    deduplicate_data = _create_migration_placeholder('data.deduplicate_data', 'llmbuilder.core.data.dedup.deduplicate_data')

try:
    from llmbuilder.core.finetune.peft_config import PEFTConfig
except ImportError:
    PEFTConfig = _create_migration_placeholder('finetune.PEFTConfig', 'llmbuilder.core.finetune.peft_config.PEFTConfig')

try:
    from llmbuilder.core.finetune.peft_model_wrapper import PEFTModelWrapper
except ImportError:
    PEFTModelWrapper = _create_migration_placeholder('finetune.PEFTModelWrapper', 'llmbuilder.core.finetune.peft_model_wrapper.PEFTModelWrapper')

try:
    from llmbuilder.core.eval.eval import evaluate_model
except ImportError:
    evaluate_model = _create_migration_placeholder('eval.evaluate_model', 'llmbuilder.core.eval.eval.evaluate_model')

try:
    from llmbuilder.core.tools.conversion_pipeline import convert_to_gguf
except ImportError:
    convert_to_gguf = _create_migration_placeholder('tools.convert_to_gguf', 'llmbuilder.core.tools.conversion_pipeline.convert_to_gguf')

try:
    from llmbuilder.core.tools.quantization_manager import quantize_model
except ImportError:
    quantize_model = _create_migration_placeholder('tools.quantize_model', 'llmbuilder.core.tools.quantization_manager.quantize_model')


# Module-level compatibility setup
def setup_module_compatibility():
    """Set up module-level compatibility for legacy imports."""
    
    # Add legacy import paths
    setup_legacy_import_hooks()
    
    # Create module aliases in sys.modules for common imports
    legacy_aliases = {
        'train': 'llmbuilder.core.training.train',
        'preprocess': 'llmbuilder.core.data.preprocess', 
        'gpt_model': 'llmbuilder.core.model.gpt_model',
        'finetune': 'llmbuilder.core.finetune.finetune',
        'eval': 'llmbuilder.core.eval.eval'
    }
    
    for alias, target in legacy_aliases.items():
        if alias not in sys.modules:
            try:
                module = importlib.import_module(target)
                sys.modules[alias] = module
            except ImportError:
                # Create a warning module
                class WarningModule:
                    def __getattr__(self, name):
                        warn_legacy_import(alias, target)
                        raise ImportError(f"Module '{alias}' has been moved to '{target}'")
                
                sys.modules[alias] = WarningModule()


# Auto-setup when module is imported
setup_module_compatibility()


# Export commonly used legacy symbols
__all__ = [
    'import_training_module',
    'import_data_module', 
    'import_model_module',
    'import_finetune_module',
    'import_eval_module',
    'import_tools_module',
    'setup_module_compatibility',
    'train_model',
    'preprocess_data',
    'GPTModel',
    'SimpleTransformer',
    'ingest_documents',
    'deduplicate_data',
    'PEFTConfig',
    'PEFTModelWrapper'
]

# Add create_dataset if it was successfully imported
try:
    create_dataset
    __all__.append('create_dataset')
except NameError:
    pass