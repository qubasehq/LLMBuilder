"""
Configuration management commands for LLMBuilder CLI.

This module provides commands for viewing, setting, and managing
LLMBuilder configuration settings.
"""

import click
import json
from pathlib import Path
from typing import Any, Dict, Optional

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
def config():
    """Manage LLMBuilder configuration settings."""
    pass


@config.command()
@click.argument('key', required=False)
@click.option(
    '--format', '-f',
    type=click.Choice(['json', 'table']),
    default='json',
    help='Output format'
)
@click.option(
    '--config-file', '-c',
    type=click.Path(path_type=Path),
    help='Specific config file to read from'
)
@click.pass_context
def get(ctx: click.Context, key: Optional[str], format: str, config_file: Optional[Path]):
    """
    Get configuration values.
    
    Display configuration settings. If no key is specified, shows all settings.
    Use dot notation to access nested values (e.g., model.vocab_size).
    
    Examples:
        llmbuilder config get                    # Show all config
        llmbuilder config get model.vocab_size  # Show specific value
        llmbuilder config get training --format table
    """
    try:
        config_manager = ConfigManager()
        
        # Load configuration
        if config_file:
            config = config_manager._load_config_file(config_file)
            if config is None:
                click.echo(f"✗ Could not load config file: {config_file}")
                raise click.Abort()
        else:
            config = config_manager.load_config()
        
        # Get specific key or all config
        if key:
            value = _get_nested_value(config, key)
            if value is None:
                click.echo(f"✗ Configuration key not found: {key}")
                raise click.Abort()
            
            # Display single value
            if format == 'json':
                click.echo(json.dumps(value, indent=2))
            else:
                click.echo(f"{key}: {value}")
        else:
            # Display all configuration
            _display_config(config, format)
            
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        click.echo(f"✗ Failed to get configuration: {e}")
        raise click.Abort()


@config.command()
@click.argument('key')
@click.argument('value')
@click.option(
    '--config-file', '-c',
    type=click.Path(path_type=Path),
    help='Specific config file to update'
)
@click.option(
    '--type', '-t',
    type=click.Choice(['string', 'int', 'float', 'bool', 'json']),
    default='string',
    help='Value type'
)
@click.option(
    '--create', 
    is_flag=True,
    help='Create config file if it does not exist'
)
def set(key: str, value: str, config_file: Optional[Path], type: str, create: bool):
    """
    Set configuration values.
    
    Set configuration settings using dot notation for nested keys.
    The value will be parsed according to the specified type.
    
    Examples:
        llmbuilder config set model.vocab_size 32000 --type int
        llmbuilder config set training.learning_rate 0.001 --type float
        llmbuilder config set training.use_mixed_precision true --type bool
        llmbuilder config set project.name "My Project"
    """
    try:
        config_manager = ConfigManager()
        
        # Determine config file path
        if config_file is None:
            # Find project config file
            config_file = _find_project_config_file()
            if config_file is None:
                if create:
                    config_file = Path.cwd() / "llmbuilder.json"
                    click.echo(f"Creating new config file: {config_file}")
                else:
                    click.echo("✗ No config file found. Use --create to create one or specify --config-file")
                    raise click.Abort()
        
        # Load existing config or create new one
        if config_file.exists():
            config = config_manager._load_config_file(config_file)
            if config is None:
                config = {}
        else:
            if not create:
                click.echo(f"✗ Config file does not exist: {config_file}")
                click.echo("Use --create to create it")
                raise click.Abort()
            config = config_manager.get_default_config()
        
        # Parse value according to type
        parsed_value = _parse_value(value, type)
        
        # Set nested value
        _set_nested_value(config, key, parsed_value)
        
        # Validate configuration
        validation_errors = config_manager.validate_config(config)
        if validation_errors:
            click.echo("!  Configuration validation warnings:")
            for error in validation_errors:
                click.echo(f"  - {error}")
            
            if not click.confirm("Continue anyway?", default=False):
                raise click.Abort()
        
        # Save configuration
        config_manager.save_config(config, config_file)
        
        click.echo(f"✓ Configuration updated: {key} = {parsed_value}")
        click.echo(f"File: Config file: {config_file}")
        
    except Exception as e:
        logger.error(f"Failed to set configuration: {e}")
        click.echo(f"✗ Failed to set configuration: {e}")
        raise click.Abort()


@config.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(path_type=Path),
    help='Specific config file to validate'
)
def validate(config_file: Optional[Path]):
    """
    Validate configuration file.
    
    Check configuration file for errors and display validation results.
    
    Examples:
        llmbuilder config validate
        llmbuilder config validate --config-file custom-config.json
    """
    try:
        config_manager = ConfigManager()
        
        # Determine config file
        if config_file is None:
            config = config_manager.load_config()
            config_file = _find_project_config_file() or Path("llmbuilder.json")
        else:
            config = config_manager._load_config_file(config_file)
            if config is None:
                click.echo(f"✗ Could not load config file: {config_file}")
                raise click.Abort()
        
        click.echo(f"Validating configuration: {config_file}")
        
        # Validate configuration
        validation_errors = config_manager.validate_config(config)
        
        if validation_errors:
            click.echo("✗ Configuration validation failed:")
            for error in validation_errors:
                click.echo(f"  - {error}")
            raise click.Abort()
        else:
            click.echo("✓ Configuration is valid!")
            
            # Show summary
            click.echo()
            click.echo("Configuration Summary:")
            click.echo(f"  Project: {config.get('project', {}).get('name', 'Unknown')}")
            click.echo(f"  Model: {config.get('model', {}).get('architecture', 'Unknown')}")
            click.echo(f"  Vocab Size: {config.get('model', {}).get('vocab_size', 'Unknown')}")
            click.echo(f"  Batch Size: {config.get('training', {}).get('batch_size', 'Unknown')}")
            
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        click.echo(f"✗ Failed to validate configuration: {e}")
        raise click.Abort()


@config.command()
@click.option(
    '--template',
    type=click.Choice(['default', 'research', 'production', 'minimal', 'gpu', 'cpu']),
    default='default',
    help='Configuration template to use'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output file path (default: llmbuilder.json)'
)
@click.option(
    '--force', '-f',
    is_flag=True,
    help='Overwrite existing file'
)
def reset(template: str, output: Optional[Path], force: bool):
    """
    Reset configuration to defaults.
    
    Create a new configuration file with default settings based on
    the specified template.
    
    Examples:
        llmbuilder config reset
        llmbuilder config reset --template production
        llmbuilder config reset --output my-config.json
    """
    try:
        config_manager = ConfigManager()
        
        # Determine output path
        if output is None:
            output = Path.cwd() / "llmbuilder.json"
        
        # Check if file exists
        if output.exists() and not force:
            click.echo(f"✗ File already exists: {output}")
            click.echo("Use --force to overwrite")
            raise click.Abort()
        
        # Create configuration from template
        config_manager.create_config_from_template(template, output)
        
        click.echo(f"✓ Configuration reset to {template} template")
        click.echo(f"File: Config file: {output}")
        
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        click.echo(f"✗ Failed to reset configuration: {e}")
        raise click.Abort()


@config.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(path_type=Path),
    help='Specific config file to list'
)
def list(config_file: Optional[Path]):
    """
    List all configuration keys.
    
    Display all available configuration keys in a hierarchical format.
    
    Examples:
        llmbuilder config list
        llmbuilder config list --config-file custom-config.json
    """
    try:
        config_manager = ConfigManager()
        
        # Load configuration
        if config_file:
            config = config_manager._load_config_file(config_file)
            if config is None:
                click.echo(f"✗ Could not load config file: {config_file}")
                raise click.Abort()
        else:
            config = config_manager.load_config()
        
        click.echo("Configuration Keys:")
        click.echo("=" * 40)
        
        _list_keys(config)
        
    except Exception as e:
        logger.error(f"Failed to list configuration: {e}")
        click.echo(f"✗ Failed to list configuration: {e}")
        raise click.Abort()


def _get_nested_value(config: Dict[str, Any], key: str) -> Any:
    """Get nested configuration value using dot notation."""
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    
    return value


def _set_nested_value(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set nested configuration value using dot notation."""
    keys = key.split('.')
    current = config
    
    # Navigate to parent of target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Set the final value
    current[keys[-1]] = value


def _parse_value(value: str, value_type: str) -> Any:
    """Parse string value according to specified type."""
    if value_type == 'int':
        return int(value)
    elif value_type == 'float':
        return float(value)
    elif value_type == 'bool':
        return value.lower() in ('true', '1', 'yes', 'on')
    elif value_type == 'json':
        return json.loads(value)
    else:  # string
        return value


def _display_config(config: Dict[str, Any], format: str) -> None:
    """Display configuration in specified format."""
    if format == 'json':
        click.echo(json.dumps(config, indent=2))

    else:  # table
        click.echo("Configuration Settings:")
        click.echo("=" * 50)
        _print_config_table(config)


def _print_config_table(config: Dict[str, Any], prefix: str = "") -> None:
    """Print configuration in table format."""
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            click.echo(f"{full_key}:")
            _print_config_table(value, full_key)
        else:
            click.echo(f"  {full_key:<30} {value}")


def _list_keys(config: Dict[str, Any], prefix: str = "", indent: int = 0) -> None:
    """List all configuration keys recursively."""
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        indent_str = "  " * indent
        
        if isinstance(value, dict):
            click.echo(f"{indent_str}{key}/")
            _list_keys(value, full_key, indent + 1)
        else:
            type_str = type(value).__name__
            click.echo(f"{indent_str}{key} ({type_str})")


@config.command()
def templates():
    """
    List available configuration templates.
    
    Show all available configuration templates with descriptions.
    
    Examples:
        llmbuilder config templates
    """
    try:
        config_manager = ConfigManager()
        templates = config_manager.get_config_templates()
        
        click.echo("Available Configuration Templates:")
        click.echo("=" * 50)
        
        template_descriptions = {
            'default': 'Balanced settings for general use',
            'research': 'Lightweight settings for experimentation',
            'production': 'Optimized settings for production training',
            'minimal': 'Minimal configuration for testing',
            'gpu': 'GPU-optimized settings',
            'cpu': 'CPU-optimized settings'
        }
        
        for name, config in templates.items():
            description = template_descriptions.get(name, 'No description')
            model_info = config.get('model', {})
            training_info = config.get('training', {})
            
            click.echo(f"\n* {name}")
            click.echo(f"   {description}")
            click.echo(f"   Model: {model_info.get('num_layers', 'N/A')} layers, "
                      f"{model_info.get('embedding_dim', 'N/A')} dim")
            click.echo(f"   Training: batch_size={training_info.get('batch_size', 'N/A')}, "
                      f"epochs={training_info.get('num_epochs', 'N/A')}")
        
        click.echo(f"\nUse 'llmbuilder config reset --template <name>' to create from template")
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        click.echo(f"✗ Failed to list templates: {e}")
        raise click.Abort()


@config.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file to migrate'
)
@click.option(
    '--target-version',
    default='latest',
    help='Target configuration version (default: latest)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be migrated without making changes'
)
def migrate(config_file: Optional[Path], target_version: str, dry_run: bool):
    """
    Migrate configuration file to newer format.
    
    Upgrade configuration files to the latest format, preserving
    existing settings while adding new defaults.
    
    Examples:
        llmbuilder config migrate
        llmbuilder config migrate --config-file old-config.json
        llmbuilder config migrate --dry-run
    """
    try:
        config_manager = ConfigManager()
        
        # Find config file if not specified
        if config_file is None:
            config_file = _find_project_config_file()
            if config_file is None:
                click.echo("✗ No configuration file found to migrate")
                click.echo("Specify --config-file or create a config file first")
                raise click.Abort()
        
        click.echo(f"Migrating configuration: {config_file}")
        
        if dry_run:
            # Load and show what would be migrated
            current_config = config_manager._load_config_file(config_file)
            current_version = current_config.get("_version", "1.0.0")
            
            click.echo(f"Current version: {current_version}")
            click.echo(f"Target version: {target_version}")
            click.echo("\nDry run - no changes will be made")
            
            # Show migration preview
            migrated_config = config_manager._apply_migrations(
                current_config, current_version, target_version
            )
            
            click.echo("\nMigration preview:")
            click.echo("New fields that would be added:")
            
            def show_new_fields(original, migrated, prefix=""):
                for key, value in migrated.items():
                    if key not in original:
                        click.echo(f"  + {prefix}{key}: {value}")
                    elif isinstance(value, dict) and isinstance(original.get(key), dict):
                        show_new_fields(original[key], value, f"{prefix}{key}.")
            
            show_new_fields(current_config, migrated_config)
        else:
            # Perform actual migration
            migrated_config = config_manager.migrate_config(config_file, target_version)
            click.echo(f"✓ Configuration migrated successfully")
            click.echo(f"File: Updated file: {config_file}")
        
    except Exception as e:
        logger.error(f"Failed to migrate configuration: {e}")
        click.echo(f"✗ Failed to migrate configuration: {e}")
        raise click.Abort()


@config.command()
@click.option(
    '--show-values',
    is_flag=True,
    help='Show environment variable values'
)
def env(show_values: bool):
    """
    Show environment variable configuration.
    
    Display all LLMBuilder environment variables and their values.
    Environment variables use the prefix LLMBUILDER_ and double
    underscores for nested keys.
    
    Examples:
        llmbuilder config env
        llmbuilder config env --show-values
        
    Environment variable examples:
        LLMBUILDER_MODEL__VOCAB_SIZE=32000
        LLMBUILDER_TRAINING__BATCH_SIZE=8
        LLMBUILDER_TRAINING__USE_MIXED_PRECISION=true
    """
    try:
        config_manager = ConfigManager()
        env_config = config_manager._load_env_config()
        
        click.echo("Environment Variable Configuration:")
        click.echo("=" * 50)
        
        if not env_config:
            click.echo("No LLMBuilder environment variables found")
            click.echo()
            click.echo("Set environment variables with LLMBUILDER_ prefix:")
            click.echo("  LLMBUILDER_MODEL__VOCAB_SIZE=32000")
            click.echo("  LLMBUILDER_TRAINING__BATCH_SIZE=8")
            click.echo("  LLMBUILDER_TRAINING__USE_MIXED_PRECISION=true")
            return
        
        def show_env_config(config, prefix=""):
            for key, value in config.items():
                if isinstance(value, dict):
                    click.echo(f"{prefix}{key}/")
                    show_env_config(value, prefix + "  ")
                else:
                    if show_values:
                        click.echo(f"{prefix}{key}: {value}")
                    else:
                        click.echo(f"{prefix}{key}: <set>")
        
        show_env_config(env_config)
        
        if not show_values:
            click.echo()
            click.echo("Use --show-values to display actual values")
        
    except Exception as e:
        logger.error(f"Failed to show environment configuration: {e}")
        click.echo(f"✗ Failed to show environment configuration: {e}")
        raise click.Abort()


@config.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file to backup'
)
def backup(config_file: Optional[Path]):
    """
    Create a backup of configuration file.
    
    Create a timestamped backup of the configuration file.
    
    Examples:
        llmbuilder config backup
        llmbuilder config backup --config-file custom-config.json
    """
    try:
        config_manager = ConfigManager()
        
        # Find config file if not specified
        if config_file is None:
            config_file = _find_project_config_file()
            if config_file is None:
                click.echo("✗ No configuration file found to backup")
                raise click.Abort()
        
        backup_path = config_manager.backup_config(config_file)
        
        click.echo(f"✓ Configuration backed up")
        click.echo(f"File: Original: {config_file}")
        click.echo(f"File: Backup: {backup_path}")
        
    except Exception as e:
        logger.error(f"Failed to backup configuration: {e}")
        click.echo(f"✗ Failed to backup configuration: {e}")
        raise click.Abort()


def _find_project_config_file() -> Optional[Path]:
    """Find project configuration file."""
    config_files = [
        "llmbuilder.json",
        "config.json"
    ]
    
    for config_file in config_files:
        path = Path.cwd() / config_file
        if path.exists():
            return path
    
    return None


# Register commands with main CLI
def register_commands(cli_group):
    """Register config commands with the main CLI group."""
    cli_group.add_command(config)