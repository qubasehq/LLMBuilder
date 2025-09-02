"""
Script Wrapper for Legacy Compatibility

This module provides wrappers for existing shell scripts (run.sh, run.ps1)
to maintain backward compatibility while providing migration paths.
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
import platform

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


class ScriptWrapper:
    """
    Wrapper for legacy shell scripts with deprecation warnings and migration guidance.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.config_manager = ConfigManager()
        self.is_windows = platform.system() == "Windows"
        
    def run_legacy_script(self, stage: str = "all", **kwargs) -> int:
        """
        Run legacy shell script with deprecation warning.
        
        Args:
            stage: Pipeline stage to run
            **kwargs: Additional arguments
            
        Returns:
            Exit code from script execution
        """
        self._show_deprecation_warning()
        
        if self.is_windows:
            return self._run_powershell_script(stage, **kwargs)
        else:
            return self._run_bash_script(stage, **kwargs)
    
    def _run_bash_script(self, stage: str, **kwargs) -> int:
        """Run the legacy bash script."""
        script_path = self.project_root / "run.sh"
        
        if not script_path.exists():
            logger.error(f"Legacy script not found: {script_path}")
            return 1
            
        # Build command arguments
        cmd = ["bash", str(script_path), stage]
        
        # Add common options
        if kwargs.get("cpu_only"):
            cmd.append("--cpu-only")
        if kwargs.get("verbose"):
            cmd.append("--verbose")
            
        logger.info(f"Running legacy script: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except Exception as e:
            logger.error(f"Failed to run legacy script: {e}")
            return 1
    
    def _run_powershell_script(self, stage: str, **kwargs) -> int:
        """Run the legacy PowerShell script."""
        script_path = self.project_root / "run.ps1"
        
        if not script_path.exists():
            logger.error(f"Legacy script not found: {script_path}")
            return 1
            
        # Build PowerShell command
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
        cmd.extend(["-Stage", stage])
        
        # Add config if specified
        config_file = kwargs.get("config", "config.json")
        if config_file:
            cmd.extend(["-Config", config_file])
            
        logger.info(f"Running legacy script: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except Exception as e:
            logger.error(f"Failed to run legacy script: {e}")
            return 1
    
    def _show_deprecation_warning(self):
        """Show deprecation warning for legacy script usage."""
        warning_msg = """
        ⚠️  DEPRECATION WARNING ⚠️
        
        You are using the legacy shell script interface (run.sh/run.ps1).
        While this will continue to work, we recommend migrating to the new CLI:
        
        New CLI Commands:
        ================
        llmbuilder init          # Initialize new project
        llmbuilder data prepare  # Data preprocessing  
        llmbuilder train start   # Start training
        llmbuilder eval run      # Run evaluation
        
        Migration Help:
        ==============
        llmbuilder migrate --help    # See migration options
        llmbuilder help             # View all commands
        
        The new CLI provides better error handling, progress tracking,
        and more powerful features for your ML workflows.
        """
        
        warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
        print(warning_msg)
    
    def get_cli_equivalent(self, stage: str) -> str:
        """
        Get the equivalent CLI command for a legacy script stage.
        
        Args:
            stage: Legacy script stage
            
        Returns:
            Equivalent CLI command
        """
        stage_mapping = {
            "ingest": "llmbuilder data prepare --ingest-only",
            "dedup": "llmbuilder data prepare --dedup-only", 
            "preprocess": "llmbuilder data prepare",
            "tokenizer": "llmbuilder data prepare --tokenizer-only",
            "train": "llmbuilder train start",
            "eval": "llmbuilder eval run",
            "finetune": "llmbuilder train start --mode finetune",
            "inference": "llmbuilder inference",
            "gguf": "llmbuilder optimize quantize --format gguf",
            "test": "llmbuilder test",
            "all": "llmbuilder train start --full-pipeline"
        }
        
        return stage_mapping.get(stage, f"# No direct equivalent for '{stage}'")
    
    def show_migration_suggestions(self, stage: str):
        """Show specific migration suggestions for a stage."""
        cli_cmd = self.get_cli_equivalent(stage)
        
        print(f"""
        💡 Migration Suggestion for '{stage}':
        =====================================
        
        Instead of: ./run.sh {stage}
        Use:        {cli_cmd}
        
        Benefits of the new CLI:
        - Better error messages and recovery suggestions
        - Progress bars and real-time status
        - Configuration validation
        - Integrated help system
        - Cross-platform compatibility
        """)


def create_legacy_wrapper_scripts():
    """
    Create wrapper scripts that call the new CLI but maintain legacy interface.
    This provides a smooth transition path for users.
    """
    project_root = Path.cwd()
    
    # Create bash wrapper
    bash_wrapper = project_root / "run_legacy.sh"
    bash_content = '''#!/bin/bash
# Legacy wrapper for LLMBuilder CLI
# This script maintains backward compatibility while encouraging migration

echo "🔄 Using legacy compatibility wrapper"
echo "💡 Consider migrating to: llmbuilder <command>"
echo ""

# Map legacy stages to new CLI commands
case "$1" in
    "ingest")
        llmbuilder data prepare --ingest-only "${@:2}"
        ;;
    "dedup") 
        llmbuilder data prepare --dedup-only "${@:2}"
        ;;
    "preprocess")
        llmbuilder data prepare "${@:2}"
        ;;
    "tokenizer")
        llmbuilder data prepare --tokenizer-only "${@:2}"
        ;;
    "train")
        llmbuilder train start "${@:2}"
        ;;
    "eval")
        llmbuilder eval run "${@:2}"
        ;;
    "finetune")
        llmbuilder train start --mode finetune "${@:2}"
        ;;
    "inference")
        llmbuilder inference "${@:2}"
        ;;
    "gguf")
        llmbuilder optimize quantize --format gguf "${@:2}"
        ;;
    "test")
        llmbuilder test "${@:2}"
        ;;
    "all")
        llmbuilder train start --full-pipeline "${@:2}"
        ;;
    *)
        echo "Usage: $0 [stage] [options]"
        echo "Stages: ingest, dedup, preprocess, tokenizer, train, eval, finetune, inference, gguf, test, all"
        echo ""
        echo "💡 For more options, try: llmbuilder --help"
        exit 1
        ;;
esac
'''
    
    # Create PowerShell wrapper  
    ps_wrapper = project_root / "run_legacy.ps1"
    ps_content = '''# Legacy wrapper for LLMBuilder CLI
# This script maintains backward compatibility while encouraging migration

param(
    [string]$Stage = "all",
    [string]$Config = "config.json"
)

Write-Host "🔄 Using legacy compatibility wrapper" -ForegroundColor Yellow
Write-Host "💡 Consider migrating to: llmbuilder <command>" -ForegroundColor Cyan
Write-Host ""

# Map legacy stages to new CLI commands
switch ($Stage) {
    "ingest" {
        & llmbuilder data prepare --ingest-only --config $Config
    }
    "dedup" {
        & llmbuilder data prepare --dedup-only --config $Config  
    }
    "preprocess" {
        & llmbuilder data prepare --config $Config
    }
    "tokenizer" {
        & llmbuilder data prepare --tokenizer-only --config $Config
    }
    "train" {
        & llmbuilder train start --config $Config
    }
    "eval" {
        & llmbuilder eval run --config $Config
    }
    "finetune" {
        & llmbuilder train start --mode finetune --config $Config
    }
    "inference" {
        & llmbuilder inference --config $Config
    }
    "gguf" {
        & llmbuilder optimize quantize --format gguf --config $Config
    }
    "test" {
        & llmbuilder test
    }
    "all" {
        & llmbuilder train start --full-pipeline --config $Config
    }
    default {
        Write-Host "Usage: .\run_legacy.ps1 -Stage <stage> [-Config <config_file>]"
        Write-Host "Stages: ingest, dedup, preprocess, tokenizer, train, eval, finetune, inference, gguf, test, all"
        Write-Host ""
        Write-Host "💡 For more options, try: llmbuilder --help" -ForegroundColor Cyan
        exit 1
    }
}
'''
    
    # Write wrapper scripts
    with open(bash_wrapper, 'w') as f:
        f.write(bash_content)
    bash_wrapper.chmod(0o755)  # Make executable
    
    with open(ps_wrapper, 'w') as f:
        f.write(ps_content)
    
    logger.info(f"Created legacy wrapper scripts: {bash_wrapper}, {ps_wrapper}")
    return bash_wrapper, ps_wrapper