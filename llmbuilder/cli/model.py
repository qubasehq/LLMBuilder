"""
Model management CLI commands for LLMBuilder.

This module provides commands for model registry, selection, downloading,
and management of both local and remote models.
"""

import click
import json
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any
from loguru import logger
import os

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.colors import (
    ColorFormatter, Color, print_header, print_success, print_error, 
    print_warning, print_info, print_table, confirm_action
)
from llmbuilder.utils.progress import (
    progress_bar, spinner, long_running_task, show_step_progress
)
from llmbuilder.utils.interactive import interactive_menu


class ModelRegistry:
    """Model registry for tracking available models."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize model registry."""
        if registry_path is None:
            # Default to user's home directory
            registry_path = Path.home() / ".llmbuilder" / "model_registry.json"
        
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry if it doesn't exist
        if not self.registry_path.exists():
            self._create_empty_registry()
    
    def _create_empty_registry(self):
        """Create an empty model registry."""
        empty_registry = {
            "version": "1.0",
            "models": {},
            "local_paths": [],
            "huggingface_cache": {}
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(empty_registry, f, indent=2)
    
    def load_registry(self) -> Dict[str, Any]:
        """Load the model registry."""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            self._create_empty_registry()
            return self.load_registry()
    
    def save_registry(self, registry: Dict[str, Any]):
        """Save the model registry."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """Register a model in the registry."""
        registry = self.load_registry()
        registry["models"][model_id] = model_info
        self.save_registry(registry)
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information from registry."""
        registry = self.load_registry()
        return registry["models"].get(model_id)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models."""
        registry = self.load_registry()
        return registry["models"]
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        registry = self.load_registry()
        if model_id in registry["models"]:
            del registry["models"][model_id]
            self.save_registry(registry)
            return True
        return False
    
    def scan_local_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Scan a local directory for GGUF and other model files."""
        found_models = []
        
        # Ensure we have a proper Path object
        if not isinstance(directory, Path):
            directory = Path(directory)
            
        if not directory.exists():
            return found_models
        
        # Look for GGUF files
        for gguf_file in directory.rglob("*.gguf"):
            model_info = {
                "model_id": gguf_file.stem,
                "type": "gguf",
                "path": str(gguf_file),
                "size": gguf_file.stat().st_size,
                "source": "local"
            }
            found_models.append(model_info)
        
        # Look for PyTorch model directories (containing pytorch_model.bin or model.safetensors)
        import os
        import fnmatch
        
        for model_dir in directory.iterdir():
            if model_dir.is_dir():
                # Use os.listdir and fnmatch instead of glob to avoid Click conflicts
                pytorch_files = []
                try:
                    for filename in os.listdir(str(model_dir)):
                        if (fnmatch.fnmatch(filename, "pytorch_model*.bin") or 
                            fnmatch.fnmatch(filename, "model*.safetensors")):
                            pytorch_files.append(model_dir / filename)
                except OSError:
                    continue
                    
                config_file = model_dir / "config.json"
                
                if pytorch_files and config_file.exists():
                    try:
                        # Read config safely
                        architecture = "unknown"
                        config = {}
                        
                        try:
                            with open(str(config_file), 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            
                            # Extract architecture safely
                            if "architectures" in config:
                                arch_list = config["architectures"]
                                if arch_list and len(arch_list) > 0:
                                    architecture = str(arch_list[0])
                        except Exception as config_error:
                            logger.warning(f"Could not read config for {model_dir}: {config_error}")
                        
                        model_info = {
                            "model_id": model_dir.name,
                            "type": "pytorch",
                            "path": str(model_dir),
                            "architecture": architecture,
                            "source": "local",
                            "config": config
                        }
                        found_models.append(model_info)
                        
                    except Exception as e:
                        logger.warning(f"Error processing model in {model_dir}: {e}")
        
        return found_models


@click.group()
def model():
    """
    Model management commands for downloading, registering, and managing models.
    
    This command group provides tools for:
    - Downloading models from Hugging Face Hub
    - Managing local model registry
    - Scanning local directories for models
    - Displaying model information and statistics
    
    Examples:
        llmbuilder model select microsoft/DialoGPT-medium
        llmbuilder model list
        llmbuilder model info microsoft/DialoGPT-medium
        llmbuilder model scan ./models
    """
    pass


@model.command()
@click.argument('model_name')
@click.option(
    '--cache-dir',
    type=click.Path(path_type=Path),
    help='Directory to cache downloaded models'
)
@click.option(
    '--revision',
    default='main',
    help='Model revision/branch to download'
)
@click.option(
    '--token',
    help='Hugging Face API token for private models'
)
@click.option(
    '--force/--no-force',
    default=False,
    help='Force re-download even if model exists'
)
@click.pass_context
def select(
    ctx: click.Context,
    model_name: str,
    cache_dir: Optional[Path],
    revision: str,
    token: Optional[str],
    force: bool
):
    """
    Download and register a model from Hugging Face Hub.
    
    Downloads the specified model and registers it in the local model registry
    for use with other LLMBuilder commands.
    
    MODEL_NAME: Hugging Face model identifier (e.g., microsoft/DialoGPT-medium)
    
    Examples:
        llmbuilder model select microsoft/DialoGPT-medium
        llmbuilder model select gpt2 --cache-dir ./models
        llmbuilder model select private/model --token hf_xxx
    """
    try:
        console.print(f"[bold blue]Downloading model: {model_name}[/bold blue]")
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists and not forcing
        registry = ModelRegistry()
        existing_model = registry.get_model(model_name)
        
        if existing_model and not force:
            console.print(f"[yellow]Model {model_name} already registered[/yellow]")
            console.print("Use --force to re-download")
            return
        
        # Try to download using huggingface_hub if available
        try:
            from huggingface_hub import snapshot_download, model_info
            
            console.print("Fetching model information...")
            
            # Get model info
            info = model_info(model_name, revision=revision, token=token)
            
            console.print(f"Model: {info.modelId}")
            console.print(f"Downloads: {info.downloads:,}")
            console.print(f"Size: {info.safetensors.total if info.safetensors else 'Unknown'}")
            
            # Download model
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Downloading model files...", total=None)
                
                local_path = snapshot_download(
                    repo_id=model_name,
                    revision=revision,
                    cache_dir=str(cache_dir),
                    token=token,
                    resume_download=True
                )
                
                progress.update(task, description="Download completed!")
            
            # Register the model
            model_info_dict = {
                "model_id": model_name,
                "type": "huggingface",
                "path": local_path,
                "revision": revision,
                "source": "huggingface_hub",
                "downloads": info.downloads,
                "architecture": getattr(info, 'pipeline_tag', 'unknown'),
                "registered_at": str(Path().cwd())
            }
            
            registry.register_model(model_name, model_info_dict)
            
            console.print(f"[green]✓ Model {model_name} downloaded and registered successfully![/green]")
            console.print(f"Local path: {local_path}")
            
        except ImportError:
            console.print("[yellow]huggingface_hub not available. Using fallback downloader...[/yellow]")
            
            # Use the existing download utility as fallback
            try:
                from llmbuilder.core.tools.download_hf_model import fetch_model_file_list, download_file
                
                console.print("Fetching model file list...")
                
                # Create model directory
                model_dir = cache_dir / model_name.replace('/', '_')
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Fetch file list
                files_data = fetch_model_file_list(model_name)
                all_files = [item['path'] for item in files_data if item['type'] == 'file']
                
                console.print(f"Found {len(all_files)} files to download")
                
                # Download files with progress
                base_url = f"https://huggingface.co/{model_name}/resolve/main"
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Downloading files...", total=len(all_files))
                    
                    for filename in all_files:
                        progress.update(task, description=f"Downloading {filename}")
                        url = f"{base_url}/{filename}"
                        save_path = model_dir / filename
                        download_file(url, save_path)
                        progress.update(task, advance=1)
                
                # Register the model
                model_info_dict = {
                    "model_id": model_name,
                    "type": "huggingface",
                    "path": str(model_dir),
                    "revision": revision,
                    "source": "huggingface_fallback",
                    "file_count": len(all_files),
                    "registered_at": str(Path().cwd())
                }
                
                registry.register_model(model_name, model_info_dict)
                
                console.print(f"[green]✓ Model {model_name} downloaded and registered successfully![/green]")
                console.print(f"Local path: {model_dir}")
                
            except Exception as fallback_error:
                console.print(f"[red]Fallback download also failed: {fallback_error}[/red]")
                console.print("Please install huggingface_hub: pip install huggingface_hub")
                raise click.ClickException("Model download failed")
        
    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        logger.error(f"Model download failed: {e}")
        raise click.ClickException(f"Model download failed: {e}")


@model.command()
@click.option(
    '--format',
    type=click.Choice(['table', 'json']),
    default='table',
    help='Output format'
)
@click.option(
    '--filter-type',
    type=click.Choice(['all', 'local', 'huggingface', 'gguf', 'pytorch']),
    default='all',
    help='Filter models by type'
)
@click.pass_context
def list(ctx: click.Context, format: str, filter_type: str):
    """
    List all registered models.
    
    Shows all models in the local registry with their details including
    type, source, and local paths.
    
    Examples:
        llmbuilder model list
        llmbuilder model list --format json
        llmbuilder model list --filter-type gguf
    """
    try:
        registry = ModelRegistry()
        models = registry.list_models()
        
        if not models:
            console.print("[yellow]No models registered[/yellow]")
            console.print("Use 'llmbuilder model select <model_name>' to download models")
            return
        
        # Filter models if requested
        if filter_type != 'all':
            models = {k: v for k, v in models.items() if v.get('type') == filter_type}
        
        if format == 'json':
            console.print(json.dumps(models, indent=2))
            return
        
        # Display as table
        table = Table(title=f"Registered Models ({len(models)} total)")
        table.add_column("Model ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Source", style="yellow")
        table.add_column("Path", style="blue")
        
        for model_id, model_info in models.items():
            model_type = model_info.get('type', 'unknown')
            source = model_info.get('source', 'unknown')
            path = model_info.get('path', 'N/A')
            
            # Truncate long paths
            path_str = str(path)
            if len(path_str) > 50:
                path_str = "..." + path_str[-47:]
            
            table.add_row(model_id, model_type, source, path_str)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        logger.error(f"Model listing failed: {e}")
        raise click.ClickException(f"Model listing failed: {e}")


@model.command()
@click.argument('model_name')
@click.pass_context
def info(ctx: click.Context, model_name: str):
    """
    Display detailed information about a registered model.
    
    Shows comprehensive information about the specified model including
    configuration, size, architecture, and local paths.
    
    MODEL_NAME: Name of the registered model
    
    Examples:
        llmbuilder model info microsoft/DialoGPT-medium
        llmbuilder model info my-local-model
    """
    try:
        registry = ModelRegistry()
        model_info = registry.get_model(model_name)
        
        if not model_info:
            console.print(f"[red]Model '{model_name}' not found in registry[/red]")
            console.print("Use 'llmbuilder model list' to see available models")
            return
        
        # Create info panel
        info_lines = []
        info_lines.append(f"[bold]Model ID:[/bold] {model_name}")
        info_lines.append(f"[bold]Type:[/bold] {model_info.get('type', 'unknown')}")
        info_lines.append(f"[bold]Source:[/bold] {model_info.get('source', 'unknown')}")
        info_lines.append(f"[bold]Path:[/bold] {model_info.get('path', 'N/A')}")
        
        if 'revision' in model_info:
            info_lines.append(f"[bold]Revision:[/bold] {model_info['revision']}")
        
        if 'architecture' in model_info:
            info_lines.append(f"[bold]Architecture:[/bold] {model_info['architecture']}")
        
        if 'downloads' in model_info:
            info_lines.append(f"[bold]Downloads:[/bold] {model_info['downloads']:,}")
        
        if 'size' in model_info:
            size_mb = model_info['size'] / (1024 * 1024)
            info_lines.append(f"[bold]Size:[/bold] {size_mb:.1f} MB")
        
        # Check if path exists
        model_path = Path(model_info.get('path', ''))
        if model_path.exists():
            info_lines.append(f"[bold]Status:[/bold] [green]Available[/green]")
        else:
            info_lines.append(f"[bold]Status:[/bold] [red]Path not found[/red]")
        
        console.print(Panel(
            "\n".join(info_lines),
            title=f"Model Information: {model_name}",
            border_style="blue"
        ))
        
        # Show config if available
        if 'config' in model_info:
            config_lines = []
            config = model_info['config']
            
            for key in ['model_type', 'vocab_size', 'hidden_size', 'num_attention_heads', 'num_hidden_layers']:
                if key in config:
                    config_lines.append(f"[bold]{key}:[/bold] {config[key]}")
            
            if config_lines:
                console.print(Panel(
                    "\n".join(config_lines),
                    title="Model Configuration",
                    border_style="green"
                ))
        
    except Exception as e:
        console.print(f"[red]Error getting model info: {e}[/red]")
        logger.error(f"Model info failed: {e}")
        raise click.ClickException(f"Model info failed: {e}")


@model.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    '--register/--no-register',
    default=True,
    help='Automatically register found models'
)
@click.option(
    '--recursive/--no-recursive',
    default=True,
    help='Scan subdirectories recursively'
)
@click.pass_context
def scan(
    ctx: click.Context,
    directory: str,
    register: bool,
    recursive: bool
):
    """
    Scan a local directory for GGUF and PyTorch models.
    
    Searches the specified directory for model files and optionally
    registers them in the local model registry.
    
    DIRECTORY: Directory to scan for model files
    
    Examples:
        llmbuilder model scan ./models
        llmbuilder model scan /path/to/models --no-register
        llmbuilder model scan ./gguf_models --no-recursive
    """
    try:
        # Convert string path to Path object
        directory_path = Path(directory)
        console.print(f"[bold blue]Scanning directory: {directory_path}[/bold blue]")
        
        registry = ModelRegistry()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scanning for models...", total=None)
            
            try:
                found_models = registry.scan_local_directory(directory_path)
            except Exception as e:
                logger.error(f"Error in scan_local_directory: {e}")
                console.print(f"[red]Error scanning directory: {e}[/red]")
                raise
            
            progress.update(task, description=f"Found {len(found_models)} models")
        
        if not found_models:
            console.print("[yellow]No models found in the specified directory[/yellow]")
            return
        
        # Display found models
        table = Table(title=f"Found Models ({len(found_models)} total)")
        table.add_column("Model ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Path", style="blue")
        table.add_column("Size", style="yellow")
        
        registered_count = 0
        
        for model_info in found_models:
            model_id = model_info['model_id']
            model_type = model_info['type']
            path = str(model_info['path'])  # Ensure path is always a string
            
            # Format size
            if 'size' in model_info:
                size_mb = model_info['size'] / (1024 * 1024)
                size_str = f"{size_mb:.1f} MB"
            else:
                size_str = "N/A"
            
            # Truncate long paths
            display_path = path
            if len(display_path) > 40:
                display_path = "..." + display_path[-37:]
            
            table.add_row(model_id, model_type, display_path, size_str)
            
            # Register if requested
            if register:
                try:
                    registry.register_model(model_id, model_info)
                    registered_count += 1
                except Exception as e:
                    logger.warning(f"Failed to register {model_id}: {e}")
        
        console.print(table)
        
        if register:
            console.print(f"\n[green]✓ Registered {registered_count} models in the registry[/green]")
        else:
            console.print(f"\n[yellow]Found {len(found_models)} models (not registered)[/yellow]")
            console.print("Use --register to add them to the registry")
        
    except Exception as e:
        console.print(f"[red]Error scanning directory: {e}[/red]")
        logger.error(f"Directory scan failed: {e}")
        raise click.ClickException(f"Directory scan failed: {e}")


@model.command()
@click.argument('model_name')
@click.option(
    '--confirm/--no-confirm',
    default=True,
    help='Ask for confirmation before removing'
)
@click.pass_context
def remove(ctx: click.Context, model_name: str, confirm: bool):
    """
    Remove a model from the registry.
    
    Removes the model entry from the local registry. This does not delete
    the actual model files, only the registry entry.
    
    MODEL_NAME: Name of the model to remove from registry
    
    Examples:
        llmbuilder model remove microsoft/DialoGPT-medium
        llmbuilder model remove old-model --no-confirm
    """
    try:
        registry = ModelRegistry()
        model_info = registry.get_model(model_name)
        
        if not model_info:
            console.print(f"[red]Model '{model_name}' not found in registry[/red]")
            return
        
        # Confirm removal
        if confirm:
            if not click.confirm(f"Remove model '{model_name}' from registry?"):
                console.print("Cancelled")
                return
        
        # Remove from registry
        if registry.remove_model(model_name):
            console.print(f"[green]✓ Model '{model_name}' removed from registry[/green]")
            console.print("[yellow]Note: Model files were not deleted[/yellow]")
        else:
            console.print(f"[red]Failed to remove model '{model_name}'[/red]")
        
    except Exception as e:
        console.print(f"[red]Error removing model: {e}[/red]")
        logger.error(f"Model removal failed: {e}")
        raise click.ClickException(f"Model removal failed: {e}")


if __name__ == "__main__":
    model()