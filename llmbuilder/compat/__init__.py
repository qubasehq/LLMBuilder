"""
Backward Compatibility Layer for LLMBuilder

This module provides backward compatibility for existing scripts and imports,
ensuring that existing code continues to work after the package migration.
"""

import warnings
from pathlib import Path
import sys
import os

# Add project root to path for backward compatibility
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Compatibility imports - maintain existing module structure
from llmbuilder.compat.legacy_imports import *
from llmbuilder.compat.script_wrapper import ScriptWrapper
from llmbuilder.compat.config_migration import ConfigMigrator
from llmbuilder.compat.project_migrator import ProjectMigrator, migrate_legacy_project
from llmbuilder.compat.deprecation import DeprecationManager, show_migration_guide, check_legacy_usage

__all__ = [
    'ScriptWrapper',
    'ConfigMigrator', 
    'ProjectMigrator',
    'DeprecationManager',
    'migrate_legacy_project',
    'setup_legacy_environment',
    'check_migration_needed',
    'show_migration_guide',
    'check_legacy_usage'
]

def setup_legacy_environment():
    """
    Set up environment for legacy script compatibility.
    This ensures existing scripts can find their dependencies.
    """
    # Add common paths that legacy scripts expect
    legacy_paths = [
        PROJECT_ROOT,
        PROJECT_ROOT / "training",
        PROJECT_ROOT / "data", 
        PROJECT_ROOT / "model",
        PROJECT_ROOT / "finetune",
        PROJECT_ROOT / "eval",
        PROJECT_ROOT / "tools",
        PROJECT_ROOT / "scripts"
    ]
    
    for path in legacy_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))

def check_migration_needed():
    """
    Check if the project needs migration to the new CLI structure.
    Returns True if migration is recommended.
    """
    # Use the more comprehensive check from deprecation module
    legacy_info = check_legacy_usage()
    return bool(legacy_info['scripts'] or legacy_info['configs'])

# Auto-setup legacy environment when module is imported
setup_legacy_environment()