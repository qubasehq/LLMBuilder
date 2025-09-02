"""
Deployment and serving CLI commands for LLMBuilder.

This module provides commands for model serving, packaging, and deployment
to various platforms including web APIs and mobile devices.
"""

import click
import json
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import shutil
import zipfile
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import get_logger
from llmbuilder.core.deployment.server import ModelServer
from llmbuilder.core.deployment.packager import ModelPackager
from llmbuilder.core.deployment.mobile import MobileExporter

logger = get_logger(__name__)
console = Console()


@click.group()
def deploy():
    """Model deployment and serving commands."""
    pass


@deploy.command()
@click.option(
    '--model', '-m',
    type=click.Path(exists=True, path_type=Path),
    help='Path to model checkpoint or GGUF file'
)
@click.option(
    '--tokenizer', '-t',
    type=click.Path(exists=True, path_type=Path),
    help='Path to tokenizer directory'
)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Path to model configuration file'
)
@click.option(
    '--host',
    default='127.0.0.1',
    help='Host to bind the server to (default: 127.0.0.1)'
)
@click.option(
    '--port',
    type=int,
    default=8000,
    help='Port to bind the server to (default: 8000)'
)
@click.option(
    '--workers',
    type=int,
    default=1,
    help='Number of worker processes (default: 1)'
)
@click.option(
    '--max-tokens',
    type=int,
    default=512,
    help='Maximum tokens per request (default: 512)'
)
@click.option(
    '--timeout',
    type=int,
    default=30,
    help='Request timeout in seconds (default: 30)'
)
@click.option(
    '--cors/--no-cors',
    default=True,
    help='Enable CORS support (default: enabled)'
)
@click.option(
    '--auth-token',
    help='Authentication token for API access'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Logging level (default: INFO)'
)
@click.option(
    '--background', '-d',
    is_flag=True,
    help='Run server in background (daemon mode)'
)
@click.pass_context
def serve(
    ctx: click.Context,
    model: Optional[Path],
    tokenizer: Optional[Path],
    config: Optional[Path],
    host: str,
    port: int,
    workers: int,
    max_tokens: int,
    timeout: int,
    cors: bool,
    auth_token: Optional[str],
    log_level: str,
    background: bool
):
    """
    Start a FastAPI model serving endpoint.
    
    This command creates a REST API server for model inference with support
    for authentication, CORS, rate limiting, and health monitoring.
    
    Examples:
        llmbuilder deploy serve --model model.pt --port 8080
        llmbuilder deploy serve --background --auth-token secret123
        llmbuilder deploy serve --workers 4 --max-tokens 1024
    """
    try:
        console.print(f"[bold blue]Starting model serving endpoint[/bold blue]")
        
        # Load configuration
        config_manager = ConfigManager()
        project_config = config_manager.get_project_config()
        
        # Resolve paths from config if not provided
        if not model:
            model = Path(project_config.get('model_path', 'model.pt'))
        if not tokenizer:
            tokenizer = Path(project_config.get('tokenizer_path', 'tokenizer/'))
        if not config:
            config = Path(project_config.get('config_path', 'config.json'))
        
        # Validate required files
        if not model.exists():
            raise click.ClickException(f"Model file not found: {model}")
        if not config.exists():
            raise click.ClickException(f"Config file not found: {config}")
        
        # Server configuration
        server_config = {
            'model_path': str(model),
            'tokenizer_path': str(tokenizer),
            'config_path': str(config),
            'host': host,
            'port': port,
            'workers': workers,
            'max_tokens': max_tokens,
            'timeout': timeout,
            'cors': cors,
            'auth_token': auth_token,
            'log_level': log_level
        }
        
        # Initialize model server
        console.print("[yellow]Initializing model server...[/yellow]")
        server = ModelServer(server_config)
        
        if background:
            # Start server in background
            console.print(f"[green]Starting server in background on {host}:{port}[/green]")
            server.start_background()
            
            # Save server info for management
            server_info = {
                'pid': server.get_pid(),
                'host': host,
                'port': port,
                'started_at': datetime.now().isoformat(),
                'config': server_config
            }
            
            server_info_file = Path('.llmbuilder') / 'server.json'
            server_info_file.parent.mkdir(exist_ok=True)
            with open(server_info_file, 'w') as f:
                json.dump(server_info, f, indent=2)
            
            console.print(f"[green]✓ Server started successfully![/green]")
            console.print(f"Server PID: {server_info['pid']}")
            console.print(f"API URL: http://{host}:{port}")
            console.print(f"Health check: http://{host}:{port}/health")
            console.print(f"Use 'llmbuilder deploy status' to check server status")
            console.print(f"Use 'llmbuilder deploy stop' to stop the server")
        else:
            # Start server in foreground
            console.print(f"[green]Starting server on {host}:{port}[/green]")
            console.print(f"API URL: http://{host}:{port}")
            console.print(f"Health check: http://{host}:{port}/health")
            console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")
            
            try:
                server.start_foreground()
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down server...[/yellow]")
                server.stop()
                console.print("[green]Server stopped[/green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to start server: {e}[/bold red]")
        logger.error(f"Server startup error: {e}", exc_info=True)
        raise click.ClickException(f"Failed to start server: {e}")


@deploy.command()
def status():
    """
    Check the status of running model servers.
    
    Shows information about currently running servers including PID,
    uptime, and health status.
    """
    try:
        server_info_file = Path('.llmbuilder') / 'server.json'
        
        if not server_info_file.exists():
            console.print("[yellow]No server information found[/yellow]")
            console.print("Use 'llmbuilder deploy serve --background' to start a server")
            return
        
        with open(server_info_file, 'r') as f:
            server_info = json.load(f)
        
        # Check if server is still running
        pid = server_info.get('pid')
        if pid and _is_process_running(pid):
            status_text = "[green]Running[/green]"
            
            # Try to get health status
            host = server_info.get('host', 'localhost')
            port = server_info.get('port', 8000)
            health_status = _check_server_health(host, port)
        else:
            status_text = "[red]Stopped[/red]"
            health_status = "N/A"
        
        # Display server information
        table = Table(title="Model Server Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", status_text)
        table.add_row("PID", str(pid) if pid else "N/A")
        table.add_row("Host", server_info.get('host', 'N/A'))
        table.add_row("Port", str(server_info.get('port', 'N/A')))
        table.add_row("Started", server_info.get('started_at', 'N/A'))
        table.add_row("Health", health_status)
        
        if pid and _is_process_running(pid):
            started_at = server_info.get('started_at')
            if started_at:
                try:
                    start_time = datetime.fromisoformat(started_at)
                    uptime = datetime.now() - start_time
                    table.add_row("Uptime", str(uptime).split('.')[0])
                except:
                    pass
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to check server status: {e}[/bold red]")
        raise click.ClickException(f"Failed to check server status: {e}")


@deploy.command()
def stop():
    """
    Stop the running model server.
    
    Gracefully shuts down the background model server and cleans up
    server information files.
    """
    try:
        server_info_file = Path('.llmbuilder') / 'server.json'
        
        if not server_info_file.exists():
            console.print("[yellow]No running server found[/yellow]")
            return
        
        with open(server_info_file, 'r') as f:
            server_info = json.load(f)
        
        pid = server_info.get('pid')
        if not pid:
            console.print("[yellow]No server PID found[/yellow]")
            return
        
        if not _is_process_running(pid):
            console.print("[yellow]Server is not running[/yellow]")
            server_info_file.unlink()
            return
        
        console.print(f"[yellow]Stopping server (PID: {pid})...[/yellow]")
        
        # Try graceful shutdown first
        try:
            import psutil
            process = psutil.Process(pid)
            process.terminate()
            process.wait(timeout=10)
        except ImportError:
            # Fallback to os.kill
            import os
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except Exception as e:
            logger.warning(f"Graceful shutdown failed: {e}")
            # Force kill
            try:
                import os
                os.kill(pid, signal.SIGKILL)
            except:
                pass
        
        # Clean up server info file
        server_info_file.unlink()
        
        console.print("[green]✓ Server stopped successfully[/green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to stop server: {e}[/bold red]")
        raise click.ClickException(f"Failed to stop server: {e}")


@deploy.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option(
    '--tokenizer',
    type=click.Path(exists=True, path_type=Path),
    help='Path to tokenizer directory'
)
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to model configuration file'
)
@click.option(
    '--format',
    type=click.Choice(['zip', 'tar', 'docker']),
    default='zip',
    help='Package format (default: zip)'
)
@click.option(
    '--include-server',
    is_flag=True,
    help='Include FastAPI server code in package'
)
@click.option(
    '--include-dependencies',
    is_flag=True,
    help='Include Python dependencies in package'
)
@click.option(
    '--compression',
    type=click.Choice(['none', 'gzip', 'bzip2']),
    default='gzip',
    help='Compression method (default: gzip)'
)
def package(
    model_path: Path,
    output_path: Path,
    tokenizer: Optional[Path],
    config: Optional[Path],
    format: str,
    include_server: bool,
    include_dependencies: bool,
    compression: str
):
    """
    Package model for deployment.
    
    Creates a deployable package containing the model, tokenizer, configuration,
    and optionally server code and dependencies.
    
    Examples:
        llmbuilder deploy package model.pt deployment.zip
        llmbuilder deploy package model.pt package.tar --include-server
        llmbuilder deploy package model.pt docker-image --format docker
    """
    try:
        console.print(f"[bold blue]Creating deployment package[/bold blue]")
        console.print(f"Input: {model_path}")
        console.print(f"Output: {output_path}")
        console.print(f"Format: {format}")
        
        # Load configuration
        config_manager = ConfigManager()
        project_config = config_manager.get_project_config()
        
        # Resolve paths from config if not provided
        if not tokenizer:
            tokenizer = Path(project_config.get('tokenizer_path', 'tokenizer/'))
        if not config:
            config = Path(project_config.get('config_path', 'config.json'))
        
        # Package configuration
        package_config = {
            'model_path': model_path,
            'tokenizer_path': tokenizer,
            'config_path': config,
            'output_path': output_path,
            'format': format,
            'include_server': include_server,
            'include_dependencies': include_dependencies,
            'compression': compression
        }
        
        # Initialize packager
        packager = ModelPackager(package_config)
        
        # Create package
        console.print("[yellow]Creating package...[/yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Packaging model...", total=100)
            
            def progress_callback(step: str, percentage: float):
                progress.update(task, description=step, completed=percentage)
            
            packager.set_progress_callback(progress_callback)
            result = packager.create_package()
        
        # Display results
        _display_package_results(result)
        
        console.print(f"[bold green]✓ Package created successfully![/bold green]")
        console.print(f"Output: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Packaging failed: {e}[/bold red]")
        logger.error(f"Packaging error: {e}", exc_info=True)
        raise click.ClickException(f"Packaging failed: {e}")


@deploy.group()
def mobile():
    """Mobile deployment commands."""
    pass


@mobile.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option(
    '--platform',
    type=click.Choice(['android', 'ios', 'both']),
    default='both',
    help='Target mobile platform (default: both)'
)
@click.option(
    '--quantization',
    type=click.Choice(['f16', 'q8_0', 'q4_0']),
    default='q8_0',
    help='Quantization for mobile deployment (default: q8_0)'
)
@click.option(
    '--max-model-size',
    type=int,
    default=100,
    help='Maximum model size in MB (default: 100)'
)
@click.option(
    '--include-examples',
    is_flag=True,
    help='Include example integration code'
)
def export(
    model_path: Path,
    output_dir: Path,
    platform: str,
    quantization: str,
    max_model_size: int,
    include_examples: bool
):
    """
    Export model for mobile deployment.
    
    Prepares model weights and generates integration code for Android/iOS
    deployment with optimizations for mobile devices.
    
    Examples:
        llmbuilder deploy mobile export model.pt mobile/ --platform android
        llmbuilder deploy mobile export model.pt mobile/ --quantization q4_0
        llmbuilder deploy mobile export model.pt mobile/ --include-examples
    """
    try:
        console.print(f"[bold blue]Exporting model for mobile deployment[/bold blue]")
        console.print(f"Model: {model_path}")
        console.print(f"Output: {output_dir}")
        console.print(f"Platform: {platform}")
        console.print(f"Quantization: {quantization}")
        
        # Export configuration
        export_config = {
            'model_path': model_path,
            'output_dir': output_dir,
            'platform': platform,
            'quantization': quantization,
            'max_model_size': max_model_size,
            'include_examples': include_examples
        }
        
        # Initialize mobile exporter
        exporter = MobileExporter(export_config)
        
        # Export for mobile
        console.print("[yellow]Preparing mobile export...[/yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Exporting for mobile...", total=100)
            
            def progress_callback(step: str, percentage: float):
                progress.update(task, description=step, completed=percentage)
            
            exporter.set_progress_callback(progress_callback)
            result = exporter.export()
        
        # Display results
        _display_mobile_export_results(result)
        
        console.print(f"[bold green]✓ Mobile export completed successfully![/bold green]")
        console.print(f"Output directory: {output_dir}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Mobile export failed: {e}[/bold red]")
        logger.error(f"Mobile export error: {e}", exc_info=True)
        raise click.ClickException(f"Mobile export failed: {e}")


# Helper functions

def _is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback method
        try:
            import os
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _check_server_health(host: str, port: int) -> str:
    """Check server health status."""
    try:
        import requests
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        if response.status_code == 200:
            return "[green]Healthy[/green]"
        else:
            return f"[yellow]Unhealthy ({response.status_code})[/yellow]"
    except Exception:
        return "[red]Unreachable[/red]"


def _display_package_results(result: Dict[str, Any]):
    """Display packaging results."""
    table = Table(title="Package Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size", style="yellow")
    
    for component, info in result.get('components', {}).items():
        status = "✓ Included" if info.get('included', False) else "✗ Skipped"
        size = info.get('size', 'N/A')
        table.add_row(component, status, size)
    
    console.print(table)
    
    # Summary
    total_size = result.get('total_size', 'Unknown')
    console.print(f"\n[bold]Package Summary:[/bold]")
    console.print(f"Total size: {total_size}")
    console.print(f"Components: {len(result.get('components', {}))}")


def _display_mobile_export_results(result: Dict[str, Any]):
    """Display mobile export results."""
    table = Table(title="Mobile Export Results")
    table.add_column("Platform", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Model Size", style="yellow")
    table.add_column("Files", style="blue")
    
    for platform, info in result.get('platforms', {}).items():
        status = "✓ Success" if info.get('success', False) else "✗ Failed"
        model_size = info.get('model_size', 'N/A')
        file_count = len(info.get('files', []))
        table.add_row(platform, status, model_size, str(file_count))
    
    console.print(table)


# Add the mobile group to deploy
deploy.add_command(mobile)