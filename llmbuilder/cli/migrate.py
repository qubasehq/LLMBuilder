"""
Migration CLI Commands

This module provides CLI commands for migrating legacy projects
to the new LLMBuilder CLI structure.
"""

import click
from pathlib import Path
from typing import Optional

from llmbuilder.compat.project_migrator import migrate_legacy_project
from llmbuilder.compat.config_migration import migrate_legacy_project as migrate_configs
from llmbuilder.compat.script_wrapper import create_legacy_wrapper_scripts
from llmbuilder.compat.deprecation import show_migration_guide, check_legacy_usage
from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
def migrate():
    """Migration tools for legacy LLMBuilder projects."""
    pass


@migrate.command()
@click.option('--project-path', '-p', type=click.Path(exists=True, path_type=Path),
              help='Path to project directory (default: current directory)')
@click.option('--backup/--no-backup', default=True,
              help='Create backup of original files (default: True)')
@click.option('--interactive/--non-interactive', default=True,
              help='Interactive mode with user prompts (default: True)')
@click.option('--dry-run', is_flag=True,
              help='Show what would be migrated without making changes')
def project(project_path: Optional[Path], backup: bool, interactive: bool, dry_run: bool):
    """
    Migrate a complete legacy project to the new CLI structure.
    
    This command performs a comprehensive migration including:
    - Configuration files
    - Project structure
    - Wrapper scripts
    - Import compatibility
    - Documentation
    """
    if dry_run:
        click.echo("🔍 DRY RUN MODE - No changes will be made")
        click.echo()
    
    project_path = project_path or Path.cwd()
    
    if dry_run:
        # Show what would be migrated
        legacy_info = check_legacy_usage()
        
        click.echo(f"Project: {project_path}")
        click.echo(f"Legacy scripts found: {len(legacy_info['scripts'])}")
        click.echo(f"Legacy configs found: {len(legacy_info['configs'])}")
        click.echo()
        
        if legacy_info['scripts']:
            click.echo("Scripts to be wrapped:")
            for script in legacy_info['scripts']:
                click.echo(f"  - {script}")
        
        if legacy_info['configs']:
            click.echo("Configs to be migrated:")
            for config in legacy_info['configs']:
                click.echo(f"  - {config}")
        
        click.echo()
        click.echo("Run without --dry-run to perform migration")
        return
    
    # Perform actual migration
    try:
        result = migrate_legacy_project(
            project_path=project_path,
            backup=backup,
            interactive=interactive
        )
        
        if result.success:
            click.echo("✅ Project migration completed successfully!")
            click.echo()
            click.echo("Next steps:")
            for i, step in enumerate(result.next_steps, 1):
                click.echo(f"  {i}. {step}")
        else:
            click.echo("❌ Project migration completed with errors")
            for error in result.errors:
                click.echo(f"  Error: {error}", err=True)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        click.echo(f"❌ Migration failed: {e}", err=True)


@migrate.command()
@click.option('--project-path', '-p', type=click.Path(exists=True, path_type=Path),
              help='Path to project directory (default: current directory)')
@click.option('--backup/--no-backup', default=True,
              help='Create backup of original config files')
def configs(project_path: Optional[Path], backup: bool):
    """
    Migrate only configuration files to the new format.
    
    This command migrates legacy configuration files (config.json, etc.)
    to the new CLI-compatible format without changing other project files.
    """
    project_path = project_path or Path.cwd()
    
    click.echo(f"🔄 Migrating configuration files in: {project_path}")
    
    try:
        result = migrate_configs(project_path, backup=backup)
        
        if result.success:
            click.echo("✅ Configuration migration completed!")
            click.echo(f"Migrated {len(result.migrated_files)} files")
            
            for migrated_file in result.migrated_files:
                click.echo(f"  ✓ {migrated_file}")
        else:
            click.echo("❌ Configuration migration failed")
            for error in result.errors:
                click.echo(f"  Error: {error}", err=True)
                
    except Exception as e:
        logger.error(f"Config migration failed: {e}")
        click.echo(f"❌ Config migration failed: {e}", err=True)


@migrate.command()
@click.option('--project-path', '-p', type=click.Path(exists=True, path_type=Path),
              help='Path to project directory (default: current directory)')
def wrappers(project_path: Optional[Path]):
    """
    Create wrapper scripts for legacy compatibility.
    
    This command creates run_legacy.sh and run_legacy.ps1 scripts that
    provide backward compatibility for existing shell script usage.
    """
    project_path = project_path or Path.cwd()
    
    click.echo(f"🔄 Creating wrapper scripts in: {project_path}")
    
    try:
        # Change to project directory for wrapper creation
        import os
        original_cwd = os.getcwd()
        os.chdir(project_path)
        
        try:
            bash_wrapper, ps_wrapper = create_legacy_wrapper_scripts()
            click.echo("✅ Wrapper scripts created successfully!")
            click.echo(f"  ✓ {bash_wrapper}")
            click.echo(f"  ✓ {ps_wrapper}")
            
            click.echo()
            click.echo("Usage:")
            click.echo("  ./run_legacy.sh <stage>     # Bash wrapper")
            click.echo("  .\\run_legacy.ps1 -Stage <stage>  # PowerShell wrapper")
            
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        logger.error(f"Wrapper creation failed: {e}")
        click.echo(f"❌ Wrapper creation failed: {e}", err=True)


@migrate.command()
@click.argument('script_name', required=False)
def guide(script_name: Optional[str]):
    """
    Show migration guide for legacy scripts.
    
    SCRIPT_NAME: Name of script to show guide for (e.g., run.sh, train.py)
    If not provided, shows general migration guide.
    """
    if script_name:
        click.echo(f"📖 Migration guide for: {script_name}")
    else:
        click.echo("📖 General migration guide")
    
    click.echo()
    show_migration_guide(script_name)


@migrate.command()
@click.option('--project-path', '-p', type=click.Path(exists=True, path_type=Path),
              help='Path to project directory (default: current directory)')
def check(project_path: Optional[Path]):
    """
    Check project for legacy usage patterns.
    
    This command analyzes the project to identify legacy scripts,
    configuration files, and other patterns that can be migrated.
    """
    project_path = project_path or Path.cwd()
    
    click.echo(f"🔍 Checking for legacy usage in: {project_path}")
    click.echo()
    
    # Change to project directory for analysis
    import os
    original_cwd = os.getcwd()
    os.chdir(project_path)
    
    try:
        legacy_info = check_legacy_usage()
        
        # Display results
        if legacy_info['scripts']:
            click.echo("📜 Legacy Scripts Found:")
            for script in legacy_info['scripts']:
                click.echo(f"  - {script}")
            click.echo()
        
        if legacy_info['configs']:
            click.echo("⚙️  Legacy Configuration Files Found:")
            for config in legacy_info['configs']:
                click.echo(f"  - {config}")
            click.echo()
        
        if legacy_info['recommendations']:
            click.echo("💡 Recommendations:")
            for rec in legacy_info['recommendations']:
                click.echo(f"  - {rec}")
            click.echo()
        
        if not legacy_info['scripts'] and not legacy_info['configs']:
            click.echo("✅ No legacy usage patterns detected")
            click.echo("Project appears to be using the new CLI structure")
        else:
            click.echo("🔄 To migrate this project, run:")
            click.echo("   llmbuilder migrate project")
            
    finally:
        os.chdir(original_cwd)


@migrate.command()
def status():
    """
    Show migration status and available commands.
    
    This command provides an overview of migration tools and their usage.
    """
    click.echo("🔄 LLMBuilder Migration Tools")
    click.echo("=" * 40)
    click.echo()
    
    click.echo("Available Commands:")
    click.echo("  check      - Analyze project for legacy patterns")
    click.echo("  project    - Migrate complete project")
    click.echo("  configs    - Migrate configuration files only")
    click.echo("  wrappers   - Create legacy wrapper scripts")
    click.echo("  guide      - Show migration guides")
    click.echo()
    
    click.echo("Migration Workflow:")
    click.echo("  1. llmbuilder migrate check")
    click.echo("  2. llmbuilder migrate project --dry-run")
    click.echo("  3. llmbuilder migrate project")
    click.echo("  4. Test with: llmbuilder --help")
    click.echo()
    
    click.echo("For detailed help on any command:")
    click.echo("  llmbuilder migrate <command> --help")


if __name__ == '__main__':
    migrate()