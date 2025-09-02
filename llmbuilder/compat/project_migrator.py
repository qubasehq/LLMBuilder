"""
Automated Project Migration Tools

This module provides comprehensive project migration capabilities,
automatically converting legacy LLM projects to the new CLI structure.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import sys

from llmbuilder.compat.config_migration import ConfigMigrator, MigrationResult
from llmbuilder.compat.script_wrapper import create_legacy_wrapper_scripts
from llmbuilder.utils.logging import get_logger
from llmbuilder.utils.project import ProjectManager

logger = get_logger(__name__)


@dataclass
class ProjectMigrationResult:
    """Result of a complete project migration."""
    success: bool
    project_path: Path
    migrated_components: List[str]
    config_migration: Optional[MigrationResult]
    backup_directory: Optional[Path]
    warnings: List[str]
    errors: List[str]
    next_steps: List[str]


class ProjectMigrator:
    """
    Comprehensive project migration tool for legacy LLM projects.
    """
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path or Path.cwd()
        self.config_migrator = ConfigMigrator(self.project_path)
        self.project_manager = ProjectManager()
        
    def migrate_project(self, 
                       backup: bool = True,
                       create_wrappers: bool = True,
                       validate: bool = True) -> ProjectMigrationResult:
        """
        Perform complete project migration from legacy to CLI structure.
        
        Args:
            backup: Whether to create backup of original files
            create_wrappers: Whether to create legacy wrapper scripts
            validate: Whether to validate migration results
            
        Returns:
            ProjectMigrationResult with migration details
        """
        logger.info(f"Starting project migration for: {self.project_path}")
        
        result = ProjectMigrationResult(
            success=True,
            project_path=self.project_path,
            migrated_components=[],
            config_migration=None,
            backup_directory=None,
            warnings=[],
            errors=[],
            next_steps=[]
        )
        
        try:
            # Step 1: Create backup if requested
            if backup:
                result.backup_directory = self._create_project_backup()
                logger.info(f"Created project backup: {result.backup_directory}")
            
            # Step 2: Analyze project structure
            project_info = self._analyze_project_structure()
            logger.info(f"Analyzed project: {len(project_info['legacy_files'])} legacy files found")
            
            # Step 3: Migrate configuration files
            config_result = self.config_migrator.migrate_project_configs(backup=False)  # Already backed up
            result.config_migration = config_result
            
            if config_result.success:
                result.migrated_components.append("configurations")
            else:
                result.errors.extend(config_result.errors)
            
            # Step 4: Create CLI project structure
            if self._create_cli_structure():
                result.migrated_components.append("project_structure")
            else:
                result.errors.append("Failed to create CLI project structure")
            
            # Step 5: Migrate data directories
            if self._migrate_data_structure():
                result.migrated_components.append("data_structure")
            else:
                result.warnings.append("Could not migrate data structure")
            
            # Step 6: Create legacy wrapper scripts
            if create_wrappers:
                if self._create_wrapper_scripts():
                    result.migrated_components.append("wrapper_scripts")
                else:
                    result.warnings.append("Could not create wrapper scripts")
            
            # Step 7: Update Python imports
            if self._update_import_compatibility():
                result.migrated_components.append("import_compatibility")
            else:
                result.warnings.append("Could not update import compatibility")
            
            # Step 8: Create migration documentation
            if self._create_migration_docs(result):
                result.migrated_components.append("documentation")
            
            # Step 9: Validate migration if requested
            if validate:
                validation_success = self._validate_migration(result)
                if not validation_success:
                    result.warnings.append("Migration validation found issues")
            
            # Step 10: Generate next steps
            result.next_steps = self._generate_next_steps(result)
            
        except Exception as e:
            logger.error(f"Migration failed with error: {e}")
            result.success = False
            result.errors.append(str(e))
        
        # Determine overall success
        result.success = len(result.errors) == 0
        
        logger.info(f"Migration completed. Success: {result.success}")
        return result
    
    def _create_project_backup(self) -> Path:
        """Create a backup of the entire project."""
        backup_dir = self.project_path / f".migration_backup_{self._get_timestamp()}"
        backup_dir.mkdir(exist_ok=True)
        
        # Files and directories to backup
        backup_items = [
            "config.json", "config_gpu.json", "config_cpu_small.json",
            "run.sh", "run.ps1", 
            "training/", "data/", "model/", "finetune/", "eval/", "tools/",
            "requirements.txt", "pyproject.toml"
        ]
        
        for item in backup_items:
            item_path = self.project_path / item
            if item_path.exists():
                if item_path.is_file():
                    shutil.copy2(item_path, backup_dir / item_path.name)
                else:
                    shutil.copytree(item_path, backup_dir / item_path.name, dirs_exist_ok=True)
        
        return backup_dir
    
    def _analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze the current project structure."""
        analysis = {
            "legacy_files": [],
            "data_directories": [],
            "config_files": [],
            "python_modules": [],
            "has_package_structure": False
        }
        
        # Check for legacy files
        legacy_patterns = [
            "run.sh", "run.ps1", "config*.json", 
            "training/*.py", "data/*.py", "model/*.py"
        ]
        
        for pattern in legacy_patterns:
            for file_path in self.project_path.glob(pattern):
                analysis["legacy_files"].append(str(file_path.relative_to(self.project_path)))
        
        # Check for data directories
        data_dirs = ["data/raw", "data/cleaned", "data/tokens", "data/deduped"]
        for data_dir in data_dirs:
            dir_path = self.project_path / data_dir
            if dir_path.exists():
                analysis["data_directories"].append(str(dir_path.relative_to(self.project_path)))
        
        # Check for existing package structure
        llmbuilder_dir = self.project_path / "llmbuilder"
        analysis["has_package_structure"] = llmbuilder_dir.exists()
        
        return analysis
    
    def _create_cli_structure(self) -> bool:
        """Create the CLI project structure."""
        try:
            # Create .llmbuilder directory for project configs
            config_dir = self.project_path / ".llmbuilder"
            config_dir.mkdir(exist_ok=True)
            
            # Create project metadata file
            project_metadata = {
                "version": "1.0.0",
                "migrated_from": "legacy_scripts",
                "migration_date": self._get_timestamp(),
                "project_type": "llm_training"
            }
            
            with open(config_dir / "project.json", 'w') as f:
                json.dump(project_metadata, f, indent=2)
            
            # Ensure data directories exist with proper structure
            data_dirs = [
                "data/raw", "data/cleaned", "data/tokens", 
                "data/deduped", "data/finetune"
            ]
            
            for data_dir in data_dirs:
                (self.project_path / data_dir).mkdir(parents=True, exist_ok=True)
            
            # Ensure export directories exist
            export_dirs = [
                "exports/checkpoints", "exports/tokenizer", 
                "exports/gguf", "exports/onnx"
            ]
            
            for export_dir in export_dirs:
                (self.project_path / export_dir).mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create CLI structure: {e}")
            return False
    
    def _migrate_data_structure(self) -> bool:
        """Migrate data directory structure to CLI expectations."""
        try:
            # Create README files for data directories
            data_readmes = {
                "data/raw/README.md": """# Raw Data Directory

Place your training data files here:
- PDF documents (.pdf)
- Text files (.txt)
- Word documents (.docx)
- HTML files (.html)
- EPUB files (.epub)

Use: `llmbuilder data prepare` to process this data.
""",
                "data/cleaned/README.md": """# Cleaned Data Directory

This directory contains processed and cleaned training data.
Generated by: `llmbuilder data prepare`
""",
                "data/tokens/README.md": """# Tokenized Data Directory

This directory contains tokenized data ready for training.
Generated by: `llmbuilder data prepare --tokenizer-only`
""",
                "data/finetune/README.md": """# Fine-tuning Data Directory

Place your fine-tuning datasets here:
- JSON files with instruction-response pairs
- CSV files with training examples

Use: `llmbuilder train start --mode finetune` to fine-tune models.
"""
            }
            
            for readme_path, content in data_readmes.items():
                readme_file = self.project_path / readme_path
                readme_file.parent.mkdir(parents=True, exist_ok=True)
                
                if not readme_file.exists():
                    with open(readme_file, 'w') as f:
                        f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate data structure: {e}")
            return False
    
    def _create_wrapper_scripts(self) -> bool:
        """Create wrapper scripts for legacy compatibility."""
        try:
            # Create wrapper scripts that call the new CLI
            bash_wrapper, ps_wrapper = create_legacy_wrapper_scripts()
            logger.info(f"Created wrapper scripts: {bash_wrapper}, {ps_wrapper}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create wrapper scripts: {e}")
            return False
    
    def _update_import_compatibility(self) -> bool:
        """Update Python imports for backward compatibility."""
        try:
            # Create compatibility imports in __init__.py if package exists
            llmbuilder_init = self.project_path / "llmbuilder" / "__init__.py"
            
            if llmbuilder_init.exists():
                # Add compatibility imports
                compatibility_imports = '''
# Backward compatibility imports
try:
    # Legacy module imports
    from llmbuilder.core.training import *
    from llmbuilder.core.data import *
    from llmbuilder.core.model import *
    from llmbuilder.core.finetune import *
    from llmbuilder.core.eval import *
    
    # Legacy script compatibility
    from llmbuilder.compat import setup_legacy_environment
    setup_legacy_environment()
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some legacy imports failed: {e}", ImportWarning)
'''
                
                with open(llmbuilder_init, 'a') as f:
                    f.write(compatibility_imports)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update import compatibility: {e}")
            return False
    
    def _create_migration_docs(self, result: ProjectMigrationResult) -> bool:
        """Create migration documentation."""
        try:
            docs_dir = self.project_path / "docs"
            docs_dir.mkdir(exist_ok=True)
            
            migration_doc = f"""# Project Migration Report

## Overview
This project has been migrated from legacy scripts to the new LLMBuilder CLI.

## Migration Summary
- **Date**: {self._get_timestamp()}
- **Success**: {result.success}
- **Components Migrated**: {', '.join(result.migrated_components)}

## Migrated Components
{chr(10).join(f"- {component}" for component in result.migrated_components)}

## Configuration Files
{chr(10).join(f"- {f}" for f in (result.config_migration.migrated_files if result.config_migration else []))}

## Backup Location
{result.backup_directory or "No backup created"}

## Warnings
{chr(10).join(f"- {w}" for w in result.warnings) if result.warnings else "None"}

## Errors
{chr(10).join(f"- {e}" for e in result.errors) if result.errors else "None"}

## Next Steps
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(result.next_steps))}

## New CLI Commands

### Data Processing
```bash
llmbuilder data prepare          # Process training data
llmbuilder data validate         # Validate data quality
llmbuilder data split            # Split into train/val/test
```

### Training
```bash
llmbuilder train start           # Start training
llmbuilder train resume          # Resume from checkpoint
llmbuilder monitor dashboard     # Monitor training
```

### Evaluation
```bash
llmbuilder eval run              # Run evaluation
llmbuilder eval benchmark        # Run benchmarks
llmbuilder eval compare          # Compare models
```

### Deployment
```bash
llmbuilder optimize quantize     # Quantize model
llmbuilder serve start           # Start API server
llmbuilder package               # Create deployment package
```

## Legacy Compatibility
Your original scripts (run.sh, run.ps1) will continue to work through wrapper scripts.
However, we recommend migrating to the new CLI for better features and support.

## Getting Help
```bash
llmbuilder --help               # General help
llmbuilder <command> --help     # Command-specific help
llmbuilder migrate --help       # Migration assistance
```
"""
            
            with open(docs_dir / "MIGRATION.md", 'w') as f:
                f.write(migration_doc)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create migration docs: {e}")
            return False
    
    def _validate_migration(self, result: ProjectMigrationResult) -> bool:
        """Validate the migration results."""
        try:
            validation_success = True
            
            # Check if CLI is accessible
            try:
                subprocess.run([sys.executable, "-m", "llmbuilder", "--version"], 
                             check=True, capture_output=True, cwd=self.project_path)
            except subprocess.CalledProcessError:
                result.warnings.append("CLI not accessible after migration")
                validation_success = False
            
            # Validate configuration files
            if result.config_migration:
                config_valid = self.config_migrator.validate_migration(result.config_migration)
                if not config_valid:
                    result.warnings.append("Configuration validation failed")
                    validation_success = False
            
            # Check directory structure
            required_dirs = [".llmbuilder", "data", "exports"]
            for req_dir in required_dirs:
                if not (self.project_path / req_dir).exists():
                    result.warnings.append(f"Required directory missing: {req_dir}")
                    validation_success = False
            
            return validation_success
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False
    
    def _generate_next_steps(self, result: ProjectMigrationResult) -> List[str]:
        """Generate recommended next steps after migration."""
        steps = []
        
        if result.success:
            steps.extend([
                "Review migrated configuration files in .llmbuilder/ directory",
                "Test CLI functionality: llmbuilder --help",
                "Validate your data: llmbuilder data validate",
                "Try a test run: llmbuilder train start --dry-run"
            ])
            
            if result.backup_directory:
                steps.append(f"Review backup files in {result.backup_directory}")
            
            steps.extend([
                "Update any custom scripts to use new CLI commands",
                "Read migration documentation in docs/MIGRATION.md",
                "Consider removing legacy scripts once migration is verified"
            ])
        else:
            steps.extend([
                "Review migration errors and warnings",
                "Check project structure and dependencies",
                "Retry migration with: llmbuilder migrate --retry",
                "Seek help: llmbuilder migrate --help"
            ])
        
        return steps
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for backup naming."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def migrate_legacy_project(project_path: Optional[Path] = None, 
                          backup: bool = True,
                          interactive: bool = True) -> ProjectMigrationResult:
    """
    Convenience function to migrate a legacy project with user interaction.
    
    Args:
        project_path: Path to project (default: current directory)
        backup: Whether to create backup
        interactive: Whether to prompt user for confirmation
        
    Returns:
        ProjectMigrationResult
    """
    migrator = ProjectMigrator(project_path)
    
    if interactive:
        print("🔄 LLMBuilder Project Migration")
        print("=" * 40)
        print(f"Project: {migrator.project_path}")
        print(f"Backup: {'Yes' if backup else 'No'}")
        print()
        
        response = input("Proceed with migration? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Migration cancelled.")
            return ProjectMigrationResult(
                success=False,
                project_path=migrator.project_path,
                migrated_components=[],
                config_migration=None,
                backup_directory=None,
                warnings=["Migration cancelled by user"],
                errors=[],
                next_steps=[]
            )
    
    # Perform migration
    result = migrator.migrate_project(backup=backup)
    
    # Print results
    print("\n" + "=" * 50)
    print("MIGRATION RESULTS")
    print("=" * 50)
    
    if result.success:
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration completed with errors")
    
    print(f"\nMigrated components: {', '.join(result.migrated_components)}")
    
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  ❌ {error}")
    
    print(f"\nNext steps:")
    for i, step in enumerate(result.next_steps, 1):
        print(f"  {i}. {step}")
    
    return result