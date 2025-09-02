"""
Configuration Migration Utilities

This module handles migration of legacy configuration files to the new
CLI-compatible format, ensuring smooth transition for existing projects.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MigrationResult:
    """Result of a configuration migration operation."""
    success: bool
    migrated_files: List[Path]
    backup_files: List[Path]
    warnings: List[str]
    errors: List[str]


class ConfigMigrator:
    """
    Handles migration of legacy configuration files to new CLI format.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.config_manager = ConfigManager()
        
        # Legacy config file patterns
        self.legacy_configs = [
            "config.json",
            "config_cpu_small.json", 
            "config_gpu.json",
            "training_config.json",
            "model_config.json"
        ]
    
    def migrate_project_configs(self, backup: bool = True) -> MigrationResult:
        """
        Migrate all legacy configuration files in the project.
        
        Args:
            backup: Whether to create backup copies of original files
            
        Returns:
            MigrationResult with details of the migration
        """
        result = MigrationResult(
            success=True,
            migrated_files=[],
            backup_files=[],
            warnings=[],
            errors=[]
        )
        
        logger.info("Starting configuration migration...")
        
        # Find legacy config files
        legacy_files = self._find_legacy_configs()
        
        if not legacy_files:
            result.warnings.append("No legacy configuration files found")
            logger.info("No legacy configuration files to migrate")
            return result
        
        # Create .llmbuilder directory for new configs
        new_config_dir = self.project_root / ".llmbuilder"
        new_config_dir.mkdir(exist_ok=True)
        
        # Migrate each config file
        for legacy_file in legacy_files:
            try:
                migrated_file, backup_file = self._migrate_single_config(
                    legacy_file, new_config_dir, backup
                )
                
                result.migrated_files.append(migrated_file)
                if backup_file:
                    result.backup_files.append(backup_file)
                    
            except Exception as e:
                error_msg = f"Failed to migrate {legacy_file}: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)
                result.success = False
        
        # Create main project config
        try:
            main_config = self._create_main_project_config(legacy_files)
            main_config_path = new_config_dir / "config.json"
            
            with open(main_config_path, 'w') as f:
                json.dump(main_config, f, indent=2)
            
            result.migrated_files.append(main_config_path)
            logger.info(f"Created main project config: {main_config_path}")
            
        except Exception as e:
            error_msg = f"Failed to create main project config: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg)
            result.success = False
        
        # Log migration summary
        if result.success:
            logger.info(f"Successfully migrated {len(result.migrated_files)} configuration files")
        else:
            logger.warning(f"Migration completed with {len(result.errors)} errors")
        
        return result
    
    def _find_legacy_configs(self) -> List[Path]:
        """Find all legacy configuration files in the project."""
        found_configs = []
        
        for config_pattern in self.legacy_configs:
            config_path = self.project_root / config_pattern
            if config_path.exists():
                found_configs.append(config_path)
        
        # Also check common subdirectories
        for subdir in ["configs", "config", "settings"]:
            subdir_path = self.project_root / subdir
            if subdir_path.exists():
                for config_file in subdir_path.glob("*.json"):
                    found_configs.append(config_file)
        
        return found_configs
    
    def _migrate_single_config(self, legacy_file: Path, target_dir: Path, backup: bool) -> Tuple[Path, Optional[Path]]:
        """
        Migrate a single configuration file.
        
        Args:
            legacy_file: Path to legacy config file
            target_dir: Directory for migrated config
            backup: Whether to create backup
            
        Returns:
            Tuple of (migrated_file_path, backup_file_path)
        """
        logger.info(f"Migrating config file: {legacy_file}")
        
        # Load legacy config
        with open(legacy_file, 'r') as f:
            legacy_config = json.load(f)
        
        # Transform to new format
        new_config = self._transform_config_format(legacy_config, legacy_file.stem)
        
        # Determine new filename
        new_filename = self._get_new_config_filename(legacy_file)
        new_file_path = target_dir / new_filename
        
        # Create backup if requested
        backup_file_path = None
        if backup:
            backup_file_path = legacy_file.with_suffix(f"{legacy_file.suffix}.backup")
            shutil.copy2(legacy_file, backup_file_path)
            logger.info(f"Created backup: {backup_file_path}")
        
        # Write new config
        with open(new_file_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        logger.info(f"Migrated to: {new_file_path}")
        return new_file_path, backup_file_path
    
    def _transform_config_format(self, legacy_config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """
        Transform legacy configuration format to new CLI format.
        
        Args:
            legacy_config: Legacy configuration dictionary
            config_type: Type of config (from filename)
            
        Returns:
            Transformed configuration dictionary
        """
        # Base new config structure
        new_config = {
            "version": "1.0.0",
            "project": {
                "name": legacy_config.get("project_name", "migrated-project"),
                "description": "Migrated from legacy configuration",
                "created_by": "config_migration"
            },
            "model": {},
            "training": {},
            "data": {},
            "deployment": {},
            "monitoring": {}
        }
        
        # Map legacy fields to new structure
        field_mappings = {
            # Model configuration
            "vocab_size": ("model", "vocab_size"),
            "embedding_dim": ("model", "embedding_dim"), 
            "num_layers": ("model", "num_layers"),
            "num_heads": ("model", "num_heads"),
            "max_seq_length": ("model", "max_seq_length"),
            "dropout": ("model", "dropout"),
            "model_type": ("model", "architecture"),
            
            # Training configuration
            "batch_size": ("training", "batch_size"),
            "learning_rate": ("training", "learning_rate"),
            "num_epochs": ("training", "num_epochs"),
            "optimizer": ("training", "optimizer"),
            "scheduler": ("training", "scheduler"),
            "mixed_precision": ("training", "mixed_precision"),
            "gradient_checkpointing": ("training", "gradient_checkpointing"),
            "warmup_steps": ("training", "warmup_steps"),
            "weight_decay": ("training", "weight_decay"),
            
            # Data configuration
            "data_dir": ("data", "input_dir"),
            "output_dir": ("data", "output_dir"),
            "tokenizer_path": ("data", "tokenizer_path"),
            "max_length": ("data", "max_sequence_length"),
            
            # Deployment configuration
            "serve_host": ("deployment", "host"),
            "serve_port": ("deployment", "port"),
            "api_framework": ("deployment", "framework"),
        }
        
        # Apply field mappings
        for legacy_key, (section, new_key) in field_mappings.items():
            if legacy_key in legacy_config:
                new_config[section][new_key] = legacy_config[legacy_key]
        
        # Handle special cases based on config type
        if "gpu" in config_type.lower():
            new_config["training"]["device"] = "cuda"
            new_config["training"]["mixed_precision"] = True
        elif "cpu" in config_type.lower():
            new_config["training"]["device"] = "cpu"
            new_config["training"]["mixed_precision"] = False
        
        # Add any unmapped fields to a legacy section
        unmapped_fields = {}
        for key, value in legacy_config.items():
            if not any(key == legacy_key for legacy_key, _ in field_mappings.items()):
                unmapped_fields[key] = value
        
        if unmapped_fields:
            new_config["legacy"] = unmapped_fields
            logger.warning(f"Unmapped legacy fields preserved in 'legacy' section: {list(unmapped_fields.keys())}")
        
        return new_config
    
    def _get_new_config_filename(self, legacy_file: Path) -> str:
        """Get appropriate filename for migrated config."""
        stem = legacy_file.stem.lower()
        
        # Map legacy filenames to new names
        filename_mappings = {
            "config": "default.json",
            "config_gpu": "gpu.json", 
            "config_cpu_small": "cpu.json",
            "training_config": "training.json",
            "model_config": "model.json"
        }
        
        return filename_mappings.get(stem, f"{stem}_migrated.json")
    
    def _create_main_project_config(self, legacy_files: List[Path]) -> Dict[str, Any]:
        """
        Create a main project configuration by merging legacy configs.
        
        Args:
            legacy_files: List of legacy configuration files
            
        Returns:
            Merged project configuration
        """
        main_config = {
            "version": "1.0.0",
            "project": {
                "name": self.project_root.name,
                "description": f"Migrated LLM project from {self.project_root}",
                "created_by": "config_migration",
                "legacy_configs": [str(f.relative_to(self.project_root)) for f in legacy_files]
            },
            "profiles": {}
        }
        
        # Create profiles from different config files
        for legacy_file in legacy_files:
            try:
                with open(legacy_file, 'r') as f:
                    legacy_config = json.load(f)
                
                profile_name = legacy_file.stem
                if profile_name == "config":
                    profile_name = "default"
                
                main_config["profiles"][profile_name] = {
                    "description": f"Migrated from {legacy_file.name}",
                    "config_file": f".llmbuilder/{self._get_new_config_filename(legacy_file)}"
                }
                
            except Exception as e:
                logger.warning(f"Could not process {legacy_file} for main config: {e}")
        
        return main_config
    
    def validate_migration(self, result: MigrationResult) -> bool:
        """
        Validate that the migration was successful and configs are valid.
        
        Args:
            result: Migration result to validate
            
        Returns:
            True if migration is valid, False otherwise
        """
        if not result.success:
            return False
        
        # Validate each migrated config file
        for config_file in result.migrated_files:
            try:
                # Try to load and validate the config
                config = self.config_manager.load_config(config_file)
                validation_errors = self.config_manager.validate_config(config)
                
                if validation_errors:
                    logger.warning(f"Validation warnings for {config_file}: {validation_errors}")
                
            except Exception as e:
                logger.error(f"Failed to validate migrated config {config_file}: {e}")
                return False
        
        return True
    
    def create_migration_report(self, result: MigrationResult) -> str:
        """
        Create a detailed migration report.
        
        Args:
            result: Migration result
            
        Returns:
            Formatted migration report
        """
        report = f"""
Configuration Migration Report
============================

Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}

Migrated Files ({len(result.migrated_files)}):
{chr(10).join(f"  ✓ {f}" for f in result.migrated_files)}

Backup Files ({len(result.backup_files)}):
{chr(10).join(f"  📁 {f}" for f in result.backup_files)}

Warnings ({len(result.warnings)}):
{chr(10).join(f"  ⚠️  {w}" for w in result.warnings)}

Errors ({len(result.errors)}):
{chr(10).join(f"  ❌ {e}" for e in result.errors)}

Next Steps:
----------
1. Review migrated configurations in .llmbuilder/ directory
2. Test with new CLI: llmbuilder config list
3. Update any custom scripts to use new config paths
4. Remove backup files once migration is verified

For help with the new CLI, run: llmbuilder --help
        """
        
        return report.strip()


def migrate_legacy_project(project_path: Optional[Path] = None, backup: bool = True) -> MigrationResult:
    """
    Convenience function to migrate a legacy project.
    
    Args:
        project_path: Path to project directory (default: current directory)
        backup: Whether to create backup files
        
    Returns:
        Migration result
    """
    migrator = ConfigMigrator(project_path)
    result = migrator.migrate_project_configs(backup=backup)
    
    # Print migration report
    report = migrator.create_migration_report(result)
    print(report)
    
    return result