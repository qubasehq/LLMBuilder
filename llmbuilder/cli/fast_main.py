"""
Fast CLI entry point for LLMBuilder.

This module provides a lightweight entry point that avoids importing
heavy dependencies until they're actually needed.
"""

import sys
import click
from pathlib import Path
from typing import Optional


@click.group(invoke_without_command=True)
@click.version_option(prog_name="llmbuilder")
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool, quiet: bool, no_color: bool):
    """
    LLMBuilder - Complete LLM Training and Deployment Pipeline
    
    Fast startup version - commands are loaded on demand.
    """
    ctx.ensure_object(dict)
    ctx.obj.update({
        'config_path': config,
        'verbose': verbose,
        'quiet': quiet,
        'no_color': no_color
    })
    
    if no_color:
        ctx.color = False
    
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Lazy command registration
_COMMANDS = {
    'init': 'llmbuilder.cli.init:init',
    'config': 'llmbuilder.cli.config:config',
    'data': 'llmbuilder.cli.data:data',
    'model': 'llmbuilder.cli.model:model',
    'train': 'llmbuilder.cli.train:train',
    'eval': 'llmbuilder.cli.eval:eval',
    'optimize': 'llmbuilder.cli.optimize:optimize',
    'inference': 'llmbuilder.cli.inference:inference',
    'deploy': 'llmbuilder.cli.deploy:deploy',
    'monitor': 'llmbuilder.cli.monitor:monitor',
    'vocab': 'llmbuilder.cli.vocab:vocab',
    'tools': 'llmbuilder.cli.tools:tools',
    'help': 'llmbuilder.cli.help:help',
    'upgrade': 'llmbuilder.cli.upgrade:upgrade',
    'migrate': 'llmbuilder.cli.migrate:migrate',
    'pipeline': 'llmbuilder.cli.pipeline:pipeline',
}


def _load_command(name: str):
    """Dynamically load a command when needed."""
    if name not in _COMMANDS:
        return None
    
    module_path, attr_name = _COMMANDS[name].split(':')
    
    try:
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        click.echo(f"Error loading command '{name}': {e}", err=True)
        return None


# Override click's command resolution to load commands dynamically
class LazyGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        # First check if command is already registered
        rv = super().get_command(ctx, cmd_name)
        if rv is not None:
            return rv
        
        # Try to load command dynamically
        command = _load_command(cmd_name)
        if command is not None:
            self.add_command(command)
            return command
        
        return None
    
    def list_commands(self, ctx):
        # Return all available commands
        return sorted(_COMMANDS.keys())


# Create a new CLI instance with lazy loading
cli = LazyGroup(
    name='llmbuilder',
    invoke_without_command=True,
    help="""
    LLMBuilder - Complete LLM Training and Deployment Pipeline
    
    Fast startup version - commands are loaded on demand.
    """
)


@cli.command()
def status():
    """Show system status quickly."""
    click.echo("LLMBuilder Status: Ready")
    
    # Quick dependency check without importing
    deps = ['torch', 'transformers', 'click', 'rich']
    for dep in deps:
        try:
            __import__(dep)
            status = "✓"
        except ImportError:
            status = "✗"
        click.echo(f"  {dep}: {status}")


def main():
    """Fast main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()