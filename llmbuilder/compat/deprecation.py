"""
Deprecation Warnings and Migration Guides

This module provides deprecation warnings and migration guidance for users
transitioning from legacy scripts to the new CLI interface.
"""

import warnings
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import wraps
import inspect

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


class DeprecationManager:
    """
    Manages deprecation warnings and migration guidance.
    """
    
    def __init__(self):
        self.shown_warnings = set()  # Track which warnings have been shown
        
    def warn_legacy_usage(self, 
                         feature: str, 
                         replacement: str, 
                         version: str = "2.0.0",
                         show_once: bool = True) -> None:
        """
        Show deprecation warning for legacy feature usage.
        
        Args:
            feature: Name of deprecated feature
            replacement: Recommended replacement
            version: Version when feature will be removed
            show_once: Whether to show warning only once per session
        """
        warning_key = f"{feature}:{replacement}"
        
        if show_once and warning_key in self.shown_warnings:
            return
            
        warning_msg = f"""
        ⚠️  DEPRECATION WARNING: {feature}
        
        This feature is deprecated and will be removed in version {version}.
        Please migrate to: {replacement}
        
        For migration help, run: llmbuilder migrate --help
        """
        
        warnings.warn(warning_msg, DeprecationWarning, stacklevel=3)
        logger.warning(f"Deprecated feature used: {feature}")
        
        if show_once:
            self.shown_warnings.add(warning_key)
    
    def show_script_migration_guide(self, script_name: str) -> None:
        """
        Show migration guide for specific legacy script.
        
        Args:
            script_name: Name of the legacy script (e.g., 'run.sh')
        """
        guides = {
            'run.sh': self._get_bash_migration_guide(),
            'run.ps1': self._get_powershell_migration_guide(),
            'train.py': self._get_training_migration_guide(),
            'preprocess.py': self._get_preprocessing_migration_guide(),
            'eval.py': self._get_evaluation_migration_guide()
        }
        
        guide = guides.get(script_name, self._get_generic_migration_guide())
        print(guide)
    
    def _get_bash_migration_guide(self) -> str:
        """Get migration guide for run.sh script."""
        return """
        🔄 Bash Script Migration Guide (run.sh)
        =======================================
        
        Your run.sh script can be replaced with these CLI commands:
        
        Legacy Command              →  New CLI Command
        ──────────────────────────────────────────────────────────
        ./run.sh preprocess         →  llmbuilder data prepare
        ./run.sh tokenizer          →  llmbuilder data prepare --tokenizer-only
        ./run.sh train              →  llmbuilder train start
        ./run.sh eval               →  llmbuilder eval run
        ./run.sh finetune           →  llmbuilder train start --mode finetune
        ./run.sh inference          →  llmbuilder inference
        ./run.sh gguf               →  llmbuilder optimize quantize --format gguf
        ./run.sh all                →  llmbuilder train start --full-pipeline
        
        Benefits of New CLI:
        ───────────────────
        ✓ Better error handling and recovery
        ✓ Progress bars and real-time status
        ✓ Cross-platform compatibility
        ✓ Integrated help system
        ✓ Configuration validation
        ✓ Session management
        
        Quick Start:
        ───────────
        1. Install package: pip install -e .
        2. Initialize project: llmbuilder init my-project
        3. Migrate configs: llmbuilder migrate --from-legacy
        4. Start training: llmbuilder train start
        
        For detailed help: llmbuilder --help
        """
    
    def _get_powershell_migration_guide(self) -> str:
        """Get migration guide for run.ps1 script."""
        return """
        🔄 PowerShell Script Migration Guide (run.ps1)
        ==============================================
        
        Your run.ps1 script can be replaced with these CLI commands:
        
        Legacy Command                    →  New CLI Command
        ────────────────────────────────────────────────────────────
        .\\run.ps1 -Stage preprocess      →  llmbuilder data prepare
        .\\run.ps1 -Stage tokenizer       →  llmbuilder data prepare --tokenizer-only
        .\\run.ps1 -Stage train           →  llmbuilder train start
        .\\run.ps1 -Stage eval            →  llmbuilder eval run
        .\\run.ps1 -Stage finetune        →  llmbuilder train start --mode finetune
        .\\run.ps1 -Stage inference       →  llmbuilder inference
        .\\run.ps1 -Stage gguf            →  llmbuilder optimize quantize --format gguf
        .\\run.ps1 -Stage all             →  llmbuilder train start --full-pipeline
        
        Configuration Migration:
        ──────────────────────
        .\\run.ps1 -Config custom.json    →  llmbuilder --config custom.json <command>
        
        Benefits of New CLI:
        ───────────────────
        ✓ Native Windows support (no PowerShell execution policy issues)
        ✓ Better error messages with recovery suggestions
        ✓ Integrated progress tracking
        ✓ Configuration validation
        ✓ Session management and resumption
        
        Quick Start:
        ───────────
        1. Install package: pip install -e .
        2. Initialize project: llmbuilder init my-project
        3. Migrate configs: llmbuilder migrate --from-legacy
        4. Start training: llmbuilder train start
        
        For detailed help: llmbuilder --help
        """
    
    def _get_training_migration_guide(self) -> str:
        """Get migration guide for training scripts."""
        return """
        🔄 Training Script Migration Guide
        =================================
        
        Direct Python script usage is being replaced with CLI commands:
        
        Legacy Usage                      →  New CLI Command
        ────────────────────────────────────────────────────────────
        python training/train.py          →  llmbuilder train start
        python training/preprocess.py     →  llmbuilder data prepare
        python training/train_tokenizer.py →  llmbuilder data prepare --tokenizer-only
        
        Configuration:
        ─────────────
        Legacy: python train.py --config config.json
        New:    llmbuilder train start --config config.json
        
        Advanced Options:
        ────────────────
        Resume training:     llmbuilder train resume --checkpoint <path>
        Monitor progress:    llmbuilder monitor dashboard
        Fine-tuning:         llmbuilder train start --mode finetune
        
        Benefits:
        ────────
        ✓ Automatic checkpoint management
        ✓ Real-time progress monitoring
        ✓ Better error handling
        ✓ Session management
        ✓ Integrated logging
        
        Migration Steps:
        ───────────────
        1. Migrate configs: llmbuilder migrate --from-legacy
        2. Test with: llmbuilder config validate
        3. Start training: llmbuilder train start
        """
    
    def _get_preprocessing_migration_guide(self) -> str:
        """Get migration guide for preprocessing scripts."""
        return """
        🔄 Data Processing Migration Guide
        =================================
        
        Data processing scripts are now unified under the CLI:
        
        Legacy Usage                      →  New CLI Command
        ────────────────────────────────────────────────────────────
        python training/preprocess.py     →  llmbuilder data prepare
        python data/ingest.py             →  llmbuilder data prepare --ingest-only
        python data/dedup.py              →  llmbuilder data prepare --dedup-only
        
        Enhanced Features:
        ─────────────────
        Data validation:     llmbuilder data validate
        Dataset splitting:   llmbuilder data split --ratios 0.8,0.1,0.1
        Statistics:          llmbuilder data stats
        
        Pipeline Integration:
        ───────────────────
        Full pipeline:       llmbuilder data prepare --full-pipeline
        Custom pipeline:     llmbuilder data prepare --steps ingest,clean,dedup
        
        Benefits:
        ────────
        ✓ Unified data processing interface
        ✓ Better progress tracking
        ✓ Automatic validation
        ✓ Configurable pipelines
        ✓ Error recovery
        
        Quick Start:
        ───────────
        1. Place data in data/raw/
        2. Run: llmbuilder data prepare
        3. Validate: llmbuilder data validate
        """
    
    def _get_evaluation_migration_guide(self) -> str:
        """Get migration guide for evaluation scripts."""
        return """
        🔄 Evaluation Migration Guide
        ============================
        
        Evaluation scripts are now integrated into the CLI:
        
        Legacy Usage                      →  New CLI Command
        ────────────────────────────────────────────────────────────
        python eval/eval.py               →  llmbuilder eval run
        
        Enhanced Evaluation:
        ──────────────────
        Custom datasets:     llmbuilder eval run --dataset custom.json
        Benchmarks:          llmbuilder eval benchmark --suite standard
        Model comparison:    llmbuilder eval compare --models model1,model2
        
        Reporting:
        ─────────
        Generate reports:    llmbuilder eval report --format html
        Export metrics:      llmbuilder eval export --format csv
        
        Benefits:
        ────────
        ✓ Standardized evaluation metrics
        ✓ Automated report generation
        ✓ Model comparison tools
        ✓ Custom dataset support
        ✓ Export capabilities
        
        Quick Start:
        ───────────
        1. Ensure model is trained
        2. Run: llmbuilder eval run
        3. View report: llmbuilder eval report --open
        """
    
    def _get_generic_migration_guide(self) -> str:
        """Get generic migration guide."""
        return """
        🔄 LLMBuilder Migration Guide
        ============================
        
        You're using legacy functionality that has been improved in the new CLI.
        
        Key Benefits of New CLI:
        ──────────────────────
        ✓ Unified interface for all operations
        ✓ Better error handling and recovery
        ✓ Progress tracking and monitoring
        ✓ Configuration validation
        ✓ Cross-platform compatibility
        ✓ Integrated help system
        
        Migration Steps:
        ───────────────
        1. Install package: pip install -e .
        2. Initialize project: llmbuilder init my-project
        3. Migrate configs: llmbuilder migrate --from-legacy
        4. Explore commands: llmbuilder --help
        
        Common Commands:
        ───────────────
        llmbuilder init              # Initialize new project
        llmbuilder data prepare      # Process training data
        llmbuilder train start       # Start model training
        llmbuilder eval run          # Evaluate model
        llmbuilder inference         # Interactive testing
        
        Get Help:
        ────────
        llmbuilder --help            # General help
        llmbuilder <command> --help  # Command-specific help
        llmbuilder migrate --help    # Migration assistance
        """


# Global deprecation manager instance
_deprecation_manager = DeprecationManager()


def deprecated(replacement: str, version: str = "2.0.0", show_guide: bool = True):
    """
    Decorator to mark functions as deprecated.
    
    Args:
        replacement: Recommended replacement
        version: Version when function will be removed
        show_guide: Whether to show migration guide
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            feature_name = f"{func.__module__}.{func.__name__}"
            _deprecation_manager.warn_legacy_usage(
                feature_name, replacement, version
            )
            
            if show_guide:
                # Try to determine script name from call stack
                frame = inspect.currentframe()
                try:
                    caller_file = frame.f_back.f_globals.get('__file__', '')
                    script_name = Path(caller_file).name if caller_file else 'unknown'
                    _deprecation_manager.show_script_migration_guide(script_name)
                finally:
                    del frame
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def warn_legacy_import(module_name: str, new_import: str):
    """
    Warn about legacy import usage.
    
    Args:
        module_name: Name of legacy module being imported
        new_import: Recommended new import path
    """
    _deprecation_manager.warn_legacy_usage(
        f"import {module_name}",
        f"import {new_import}",
        show_once=True
    )


def show_migration_guide(script_name: Optional[str] = None):
    """
    Show migration guide for current or specified script.
    
    Args:
        script_name: Name of script to show guide for (auto-detect if None)
    """
    if script_name is None:
        # Try to detect script name from call stack
        frame = inspect.currentframe()
        try:
            caller_file = frame.f_back.f_globals.get('__file__', '')
            script_name = Path(caller_file).name if caller_file else 'generic'
        finally:
            del frame
    
    _deprecation_manager.show_script_migration_guide(script_name)


def check_legacy_usage() -> Dict[str, Any]:
    """
    Check for legacy usage patterns in the current project.
    
    Returns:
        Dictionary with legacy usage information
    """
    project_root = Path.cwd()
    
    legacy_indicators = {
        'scripts': [],
        'configs': [],
        'imports': [],
        'recommendations': []
    }
    
    # Check for legacy scripts
    legacy_scripts = ['run.sh', 'run.ps1']
    for script in legacy_scripts:
        script_path = project_root / script
        if script_path.exists():
            legacy_indicators['scripts'].append(str(script_path))
    
    # Check for legacy config files
    legacy_configs = ['config.json', 'config_gpu.json', 'config_cpu_small.json']
    for config in legacy_configs:
        config_path = project_root / config
        if config_path.exists():
            legacy_indicators['configs'].append(str(config_path))
    
    # Generate recommendations
    if legacy_indicators['scripts']:
        legacy_indicators['recommendations'].append(
            "Migrate shell scripts to CLI: llmbuilder migrate --scripts"
        )
    
    if legacy_indicators['configs']:
        legacy_indicators['recommendations'].append(
            "Migrate configuration files: llmbuilder migrate --configs"
        )
    
    if not legacy_indicators['scripts'] and not legacy_indicators['configs']:
        legacy_indicators['recommendations'].append(
            "Project appears to be already migrated or is new"
        )
    
    return legacy_indicators