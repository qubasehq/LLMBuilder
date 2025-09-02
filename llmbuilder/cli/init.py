"""
Project initialization commands for LLMBuilder.

This module provides commands for creating new LLM projects with
proper directory structure, configuration files, and templates.
"""

import click
import shutil
from pathlib import Path
from typing import Optional
import yaml

from llmbuilder.utils.logging import get_logger
from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.project import ProjectManager
from llmbuilder.utils.colors import (
    ColorFormatter, Color, print_header, print_success, print_error, 
    print_warning, print_info, confirm_action
)
from llmbuilder.utils.progress import (
    spinner, long_running_task, show_step_progress
)

logger = get_logger(__name__)


@click.group()
def init():
    """Initialize new LLMBuilder projects and manage project structure."""
    pass


@init.command()
@click.argument('project_name')
@click.option(
    '--template', 
    '-t', 
    default='default',
    help='Project template to use (default, research, production, fine-tuning)'
)
@click.option(
    '--description', 
    '-d', 
    help='Project description'
)
@click.option(
    '--force', 
    '-f', 
    is_flag=True, 
    help='Overwrite existing directory'
)
@click.option(
    '--git', 
    is_flag=True, 
    default=True,
    help='Initialize git repository'
)
@click.pass_context
def project(ctx, project_name: str, template: str, description: Optional[str], force: bool, git: bool):
    """
    Create a new LLM project with proper structure and configuration.
    
    PROJECT_NAME: Name of the project directory to create
    
    Examples:
        llmbuilder init project my-llm-model
        llmbuilder init project research-project --template research
        llmbuilder init project production-model --template production --description "Production LLM"
    """
    try:
        print_header(f"Creating Project: {project_name}")
        print_info(f"Template: {template}")
        if description:
            print_info(f"Description: {description}")
        
        # Check if directory exists and confirm overwrite
        project_path = Path(project_name)
        if project_path.exists() and not force:
            if not confirm_action(f"Directory '{project_name}' already exists. Overwrite?"):
                print_info("Project creation cancelled")
                return
        
        def create_project():
            project_manager = ProjectManager()
            return project_manager.create_project(
                name=project_name,
                template=template,
                description=description,
                force=force,
                initialize_git=git
            )
        
        # Create project with progress indication
        project_path = long_running_task(
            create_project,
            message="Creating project structure",
            success_message=f"Project '{project_name}' created successfully",
            timeout_message="Setting up project files and directories..."
        )
        
        print_info(f"Project location: {project_path}")
        
        print_header("Next Steps")
        steps = [
            f"cd {project_name}",
            "Add your training data to data/raw/",
            "Run: llmbuilder data prepare",
            "Run: llmbuilder train start"
        ]
        
        for i, step in enumerate(steps, 1):
            click.echo(f"  {ColorFormatter.format(f'{i}.', Color.BLUE)} {step}")
        
        print_info("\nFor help: llmbuilder --help")
        
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        print_error(f"Error creating project: {e}")
        raise click.Abort()


@init.command()
@click.option(
    '--template', 
    '-t', 
    default='default',
    help='Configuration template to use'
)
@click.option(
    '--output', 
    '-o', 
    type=click.Path(path_type=Path),
    default='llmbuilder.yaml',
    help='Output configuration file path'
)
@click.option(
    '--format', 
    type=click.Choice(['yaml', 'json']),
    default='yaml',
    help='Configuration file format'
)
@click.pass_context
def config(ctx, template: str, output: Path, format: str):
    """
    Initialize configuration file in current directory.
    
    Examples:
        llmbuilder init config
        llmbuilder init config --template research --format json
        llmbuilder init config --output my-config.yaml
    """
    try:
        config_manager = ConfigManager()
        
        # Get template configuration
        if template == 'default':
            config_data = config_manager.get_default_config()
        else:
            # Load template-specific configuration
            template_path = Path(__file__).parent.parent / 'templates' / 'configs' / f'{template}.yaml'
            if template_path.exists():
                with open(template_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                click.echo(f"❌ Template '{template}' not found. Using default.", err=True)
                config_data = config_manager.get_default_config()
        
        # Adjust output file extension based on format
        if format == 'json' and output.suffix not in ['.json']:
            output = output.with_suffix('.json')
        elif format == 'yaml' and output.suffix not in ['.yaml', '.yml']:
            output = output.with_suffix('.yaml')
        
        # Check if file exists
        if output.exists():
            if not click.confirm(f"Configuration file '{output}' already exists. Overwrite?"):
                click.echo("Operation cancelled.")
                return
        
        # Save configuration
        config_manager.save_config(config_data, output)
        
        click.echo(f"✅ Configuration file created: {output}")
        click.echo(f"📝 Format: {format}")
        click.echo(f"📋 Template: {template}")
        click.echo()
        click.echo("You can now customize the configuration and run:")
        click.echo(f"  llmbuilder --config {output} <command>")
        
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        click.echo(f"❌ Error creating configuration: {e}", err=True)
        raise click.Abort()


@init.command()
@click.pass_context
def templates(ctx):
    """List available project and configuration templates."""
    try:
        project_manager = ProjectManager()
        templates = project_manager.list_templates()
        
        click.echo("📋 Available Templates:")
        click.echo()
        
        # Group templates by type
        builtin_templates = {k: v for k, v in templates.items() if v["type"] == "built-in"}
        user_templates = {k: v for k, v in templates.items() if v["type"] == "user"}
        
        if builtin_templates:
            click.echo("🏗️  Built-in Templates:")
            for name, info in builtin_templates.items():
                click.echo(f"  • {name:<15} - {info['description']}")
        
        if user_templates:
            click.echo()
            click.echo("👤 User Templates:")
            for name, info in user_templates.items():
                click.echo(f"  • {name:<15} - {info['description']}")
        
        click.echo()
        click.echo("Usage:")
        click.echo("  llmbuilder init project my-project --template research")
        click.echo("  llmbuilder init template create my-template --base default")
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        click.echo(f"❌ Error listing templates: {e}", err=True)
        raise click.Abort()


@init.command()
@click.argument('name')
@click.option('--base', '-b', default='default', help='Base template to extend')
@click.option('--description', '-d', help='Template description')
@click.pass_context
def template(ctx, name: str, base: str, description: Optional[str]):
    """Create a custom project template."""
    try:
        project_manager = ProjectManager()
        
        if not description:
            description = click.prompt("Template description")
        
        # Get customizations interactively
        click.echo("\nCustomize your template (press Enter to skip):")
        
        customizations = {}
        
        # Ask for directory customizations
        if click.confirm("Add custom directories?"):
            dirs = click.prompt("Enter directories (comma-separated)", default="").strip()
            if dirs:
                custom_dirs = [d.strip() for d in dirs.split(",")]
                customizations["directories"] = custom_dirs
        
        # Create template
        template_file = project_manager.create_custom_template(
            name=name,
            description=description,
            base_template=base,
            customizations=customizations if customizations else None
        )
        
        click.echo(f"✅ Custom template '{name}' created successfully!")
        click.echo(f"📁 Template file: {template_file}")
        click.echo()
        click.echo("Usage:")
        click.echo(f"  llmbuilder init project my-project --template {name}")
        
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        click.echo(f"❌ Error creating template: {e}", err=True)
        raise click.Abort()


@init.command()
@click.option('--path', '-p', type=click.Path(exists=True, path_type=Path), 
              default=Path.cwd(), help='Project path to validate')
@click.pass_context
def validate(ctx, path: Path):
    """Validate project structure and configuration."""
    try:
        project_manager = ProjectManager()
        results = project_manager.validate_project(path)
        
        click.echo(f"🔍 Validating project: {path.name}")
        click.echo()
        
        if results["valid"]:
            click.echo("✅ Project structure is valid!")
        else:
            click.echo("❌ Project validation failed!")
        
        # Show errors
        if results["errors"]:
            click.echo()
            click.echo("🚨 Errors:")
            for error in results["errors"]:
                click.echo(f"  • {error}")
        
        # Show warnings
        if results["warnings"]:
            click.echo()
            click.echo("⚠️  Warnings:")
            for warning in results["warnings"]:
                click.echo(f"  • {warning}")
        
        # Show suggestions
        if results["suggestions"]:
            click.echo()
            click.echo("💡 Suggestions:")
            for suggestion in results["suggestions"]:
                click.echo(f"  • {suggestion}")
        
    except Exception as e:
        logger.error(f"Failed to validate project: {e}")
        click.echo(f"❌ Error validating project: {e}", err=True)
        raise click.Abort()


@init.command()
@click.option('--path', '-p', type=click.Path(exists=True, path_type=Path), 
              default=Path.cwd(), help='Project path to check')
@click.pass_context
def health(ctx, path: Path):
    """Perform comprehensive project health check."""
    try:
        project_manager = ProjectManager()
        health = project_manager.health_check(path)
        
        click.echo(f"🏥 Health check for project: {path.name}")
        click.echo()
        
        # Overall health
        health_emoji = {
            "good": "💚",
            "fair": "💛", 
            "poor": "❤️"
        }
        
        overall = health["overall_health"]
        click.echo(f"Overall Health: {health_emoji.get(overall, '❓')} {overall.upper()}")
        click.echo()
        
        # Individual checks
        for check_name, check_result in health["checks"].items():
            status_emoji = "✅" if check_result["status"] else "❌"
            click.echo(f"{status_emoji} {check_name.replace('_', ' ').title()}: {check_result['message']}")
            if check_result.get("details"):
                click.echo(f"   {check_result['details']}")
        
        # Recommendations
        if health.get("recommendations"):
            click.echo()
            click.echo("💡 Recommendations:")
            for rec in health["recommendations"]:
                click.echo(f"  • {rec}")
        
    except Exception as e:
        logger.error(f"Failed to perform health check: {e}")
        click.echo(f"❌ Error performing health check: {e}", err=True)
        raise click.Abort()