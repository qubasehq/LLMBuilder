"""
Monitoring and debugging CLI commands for LLMBuilder.

This module provides commands for real-time monitoring, system diagnostics,
log aggregation, and performance metrics collection.
"""

import click
import json
import time
import psutil
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import subprocess
import sys
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.tree import Tree

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import get_logger
from llmbuilder.core.monitoring.dashboard import MonitoringDashboard
from llmbuilder.core.monitoring.diagnostics import SystemDiagnostics
from llmbuilder.core.monitoring.metrics import MetricsCollector
from llmbuilder.core.monitoring.logs import LogAggregator

logger = get_logger(__name__)
console = Console()


@click.group()
def monitor():
    """Monitoring and debugging commands."""
    pass


@monitor.command()
@click.option(
    '--refresh-rate',
    type=float,
    default=1.0,
    help='Dashboard refresh rate in seconds (default: 1.0)'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Minimum log level to display (default: INFO)'
)
@click.option(
    '--max-logs',
    type=int,
    default=100,
    help='Maximum number of log entries to display (default: 100)'
)
@click.option(
    '--export-metrics',
    type=click.Path(path_type=Path),
    help='Export metrics to file (JSON format)'
)
@click.option(
    '--training-logs',
    type=click.Path(exists=True, path_type=Path),
    help='Path to training logs directory'
)
@click.option(
    '--gpu-monitoring/--no-gpu-monitoring',
    default=True,
    help='Enable GPU monitoring (default: enabled)'
)
def dashboard(
    refresh_rate: float,
    log_level: str,
    max_logs: int,
    export_metrics: Optional[Path],
    training_logs: Optional[Path],
    gpu_monitoring: bool
):
    """
    Launch real-time monitoring dashboard.
    
    Displays live metrics including training progress, system resources,
    GPU usage, and recent logs in a comprehensive dashboard interface.
    
    Examples:
        llmbuilder monitor dashboard
        llmbuilder monitor dashboard --refresh-rate 0.5 --gpu-monitoring
        llmbuilder monitor dashboard --export-metrics metrics.json
    """
    try:
        console.print(f"[bold blue]Starting monitoring dashboard[/bold blue]")
        console.print(f"Refresh rate: {refresh_rate}s")
        console.print(f"Log level: {log_level}")
        console.print("[yellow]Press Ctrl+C to exit[/yellow]")
        
        # Load configuration
        config_manager = ConfigManager()
        project_config = config_manager.get_project_config()
        
        # Resolve training logs path
        if not training_logs:
            training_logs = Path(project_config.get('paths', {}).get('log_dir', 'logs'))
        
        # Initialize dashboard
        dashboard_config = {
            'refresh_rate': refresh_rate,
            'log_level': log_level,
            'max_logs': max_logs,
            'training_logs': training_logs,
            'gpu_monitoring': gpu_monitoring,
            'export_metrics': export_metrics
        }
        
        dashboard = MonitoringDashboard(dashboard_config)
        
        # Start dashboard
        dashboard.start()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]✗ Dashboard failed: {e}[/bold red]")
        logger.error(f"Dashboard error: {e}", exc_info=True)
        raise click.ClickException(f"Dashboard failed: {e}")


@monitor.command()
@click.option(
    '--check-dependencies/--no-check-dependencies',
    default=True,
    help='Check for missing dependencies (default: enabled)'
)
@click.option(
    '--check-data/--no-check-data',
    default=True,
    help='Check for data corruption (default: enabled)'
)
@click.option(
    '--check-models/--no-check-models',
    default=True,
    help='Check model files integrity (default: enabled)'
)
@click.option(
    '--check-config/--no-check-config',
    default=True,
    help='Check configuration validity (default: enabled)'
)
@click.option(
    '--check-system/--no-check-system',
    default=True,
    help='Check system resources (default: enabled)'
)
@click.option(
    '--fix-issues/--no-fix-issues',
    default=False,
    help='Attempt to automatically fix detected issues (default: disabled)'
)
@click.option(
    '--report',
    type=click.Path(path_type=Path),
    help='Save diagnostic report to file'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed diagnostic information'
)
def debug(
    check_dependencies: bool,
    check_data: bool,
    check_models: bool,
    check_config: bool,
    check_system: bool,
    fix_issues: bool,
    report: Optional[Path],
    verbose: bool
):
    """
    Run automated system diagnostics.
    
    Performs comprehensive system checks including dependencies,
    data integrity, model files, configuration, and system resources.
    Provides automated suggestions for fixing detected issues.
    
    Examples:
        llmbuilder monitor debug
        llmbuilder monitor debug --fix-issues --report debug_report.json
        llmbuilder monitor debug --no-check-data --verbose
    """
    try:
        console.print(f"[bold blue]Running system diagnostics[/bold blue]")
        
        # Initialize diagnostics
        diagnostics_config = {
            'check_dependencies': check_dependencies,
            'check_data': check_data,
            'check_models': check_models,
            'check_config': check_config,
            'check_system': check_system,
            'fix_issues': fix_issues,
            'verbose': verbose
        }
        
        diagnostics = SystemDiagnostics(diagnostics_config)
        
        # Run diagnostics
        console.print("[yellow]Running diagnostic checks...[/yellow]")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            def progress_callback(step: str, percentage: float):
                task = getattr(progress_callback, 'task', None)
                if task is None:
                    progress_callback.task = progress.add_task("Running diagnostics...", total=100)
                    task = progress_callback.task
                progress.update(task, description=step, completed=percentage)
            
            diagnostics.set_progress_callback(progress_callback)
            results = diagnostics.run_all_checks()
        
        # Display results
        _display_diagnostic_results(results, verbose)
        
        # Save report if requested
        if report:
            diagnostics.save_report(results, report)
            console.print(f"[green]Diagnostic report saved to {report}[/green]")
        
        # Exit with error code if critical issues found
        if results.get('critical_issues', 0) > 0:
            console.print(f"[bold red]Found {results['critical_issues']} critical issues[/bold red]")
            raise click.ClickException("Critical issues detected")
        elif results.get('warnings', 0) > 0:
            console.print(f"[bold yellow]Found {results['warnings']} warnings[/bold yellow]")
        else:
            console.print(f"[bold green]✓ All diagnostic checks passed![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Diagnostics failed: {e}[/bold red]")
        logger.error(f"Diagnostics error: {e}", exc_info=True)
        raise click.ClickException(f"Diagnostics failed: {e}")


@monitor.command()
@click.option(
    '--log-dir',
    type=click.Path(exists=True, path_type=Path),
    help='Directory containing log files'
)
@click.option(
    '--pattern',
    default='*.log',
    help='Log file pattern to search (default: *.log)'
)
@click.option(
    '--level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    help='Filter by log level'
)
@click.option(
    '--since',
    help='Show logs since time (e.g., "1h", "30m", "2024-01-01")'
)
@click.option(
    '--tail',
    type=int,
    default=100,
    help='Number of recent log entries to show (default: 100)'
)
@click.option(
    '--follow', '-f',
    is_flag=True,
    help='Follow log files for new entries'
)
@click.option(
    '--search',
    help='Search for specific text in logs'
)
@click.option(
    '--export',
    type=click.Path(path_type=Path),
    help='Export filtered logs to file'
)
def logs(
    log_dir: Optional[Path],
    pattern: str,
    level: Optional[str],
    since: Optional[str],
    tail: int,
    follow: bool,
    search: Optional[str],
    export: Optional[Path]
):
    """
    Aggregate and search log files.
    
    Provides powerful log aggregation, filtering, and search capabilities
    across multiple log files with real-time following support.
    
    Examples:
        llmbuilder monitor logs --tail 50
        llmbuilder monitor logs --level ERROR --since "1h"
        llmbuilder monitor logs --search "training" --follow
        llmbuilder monitor logs --export filtered_logs.txt
    """
    try:
        console.print(f"[bold blue]Log aggregation and search[/bold blue]")
        
        # Load configuration
        config_manager = ConfigManager()
        project_config = config_manager.get_project_config()
        
        # Resolve log directory
        if not log_dir:
            log_dir = Path(project_config.get('paths', {}).get('log_dir', 'logs'))
        
        if not log_dir.exists():
            console.print(f"[yellow]Log directory not found: {log_dir}[/yellow]")
            console.print("Use --log-dir to specify a different directory")
            return
        
        # Initialize log aggregator
        aggregator_config = {
            'log_dir': log_dir,
            'pattern': pattern,
            'level': level,
            'since': since,
            'tail': tail,
            'search': search
        }
        
        aggregator = LogAggregator(aggregator_config)
        
        if follow:
            # Follow mode - real-time log monitoring
            console.print(f"[yellow]Following logs in {log_dir} (Press Ctrl+C to stop)[/yellow]")
            aggregator.follow_logs()
        else:
            # Static mode - show filtered logs
            console.print(f"[yellow]Searching logs in {log_dir}[/yellow]")
            logs_data = aggregator.get_logs()
            
            if not logs_data:
                console.print("[yellow]No logs found matching criteria[/yellow]")
                return
            
            # Display logs
            _display_logs(logs_data)
            
            # Export if requested
            if export:
                aggregator.export_logs(logs_data, export)
                console.print(f"[green]Logs exported to {export}[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Log monitoring stopped[/yellow]")
    except Exception as e:
        console.print(f"[bold red]✗ Log aggregation failed: {e}[/bold red]")
        logger.error(f"Log aggregation error: {e}", exc_info=True)
        raise click.ClickException(f"Log aggregation failed: {e}")


@monitor.command()
@click.option(
    '--duration',
    type=int,
    default=60,
    help='Monitoring duration in seconds (default: 60)'
)
@click.option(
    '--interval',
    type=float,
    default=1.0,
    help='Sampling interval in seconds (default: 1.0)'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Save metrics to file (JSON format)'
)
@click.option(
    '--include-gpu/--no-include-gpu',
    default=True,
    help='Include GPU metrics (default: enabled)'
)
@click.option(
    '--include-network/--no-include-network',
    default=False,
    help='Include network metrics (default: disabled)'
)
@click.option(
    '--alert-cpu',
    type=float,
    help='CPU usage alert threshold (percentage)'
)
@click.option(
    '--alert-memory',
    type=float,
    help='Memory usage alert threshold (percentage)'
)
@click.option(
    '--alert-disk',
    type=float,
    help='Disk usage alert threshold (percentage)'
)
def metrics(
    duration: int,
    interval: float,
    output: Optional[Path],
    include_gpu: bool,
    include_network: bool,
    alert_cpu: Optional[float],
    alert_memory: Optional[float],
    alert_disk: Optional[float]
):
    """
    Collect system performance metrics.
    
    Monitors CPU, memory, disk, and optionally GPU and network usage
    with configurable alerts and data export capabilities.
    
    Examples:
        llmbuilder monitor metrics --duration 300 --interval 0.5
        llmbuilder monitor metrics --output metrics.json --alert-cpu 80
        llmbuilder monitor metrics --include-network --alert-memory 90
    """
    try:
        console.print(f"[bold blue]Collecting system metrics[/bold blue]")
        console.print(f"Duration: {duration}s, Interval: {interval}s")
        
        # Initialize metrics collector
        collector_config = {
            'duration': duration,
            'interval': interval,
            'include_gpu': include_gpu,
            'include_network': include_network,
            'alerts': {
                'cpu': alert_cpu,
                'memory': alert_memory,
                'disk': alert_disk
            }
        }
        
        collector = MetricsCollector(collector_config)
        
        # Start collection
        console.print("[yellow]Starting metrics collection...[/yellow]")
        console.print("[dim]Press Ctrl+C to stop early[/dim]")
        
        try:
            metrics_data = collector.collect_metrics()
            
            # Display summary
            _display_metrics_summary(metrics_data)
            
            # Save to file if requested
            if output:
                collector.save_metrics(metrics_data, output)
                console.print(f"[green]Metrics saved to {output}[/green]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Metrics collection stopped early[/yellow]")
            # Still try to get partial data
            metrics_data = collector.get_current_metrics()
            if metrics_data and output:
                collector.save_metrics(metrics_data, output)
                console.print(f"[green]Partial metrics saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Metrics collection failed: {e}[/bold red]")
        logger.error(f"Metrics collection error: {e}", exc_info=True)
        raise click.ClickException(f"Metrics collection failed: {e}")


@monitor.command()
@click.option(
    '--process-name',
    help='Monitor specific process by name'
)
@click.option(
    '--pid',
    type=int,
    help='Monitor specific process by PID'
)
@click.option(
    '--refresh-rate',
    type=float,
    default=2.0,
    help='Refresh rate in seconds (default: 2.0)'
)
def processes(
    process_name: Optional[str],
    pid: Optional[int],
    refresh_rate: float
):
    """
    Monitor running processes.
    
    Shows real-time information about system processes with filtering
    options for specific processes or PIDs.
    
    Examples:
        llmbuilder monitor processes
        llmbuilder monitor processes --process-name python
        llmbuilder monitor processes --pid 1234 --refresh-rate 1.0
    """
    try:
        console.print(f"[bold blue]Process monitoring[/bold blue]")
        console.print("[yellow]Press Ctrl+C to exit[/yellow]")
        
        def update_process_table():
            table = Table(title="Running Processes")
            table.add_column("PID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("CPU %", style="yellow")
            table.add_column("Memory %", style="blue")
            table.add_column("Status", style="magenta")
            
            try:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                    try:
                        pinfo = proc.info
                        
                        # Filter by process name if specified
                        if process_name and process_name.lower() not in pinfo['name'].lower():
                            continue
                        
                        # Filter by PID if specified
                        if pid and pinfo['pid'] != pid:
                            continue
                        
                        processes.append(pinfo)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Sort by CPU usage
                processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
                
                # Show top 20 processes
                for pinfo in processes[:20]:
                    table.add_row(
                        str(pinfo['pid']),
                        pinfo['name'][:30],
                        f"{pinfo['cpu_percent']:.1f}" if pinfo['cpu_percent'] else "0.0",
                        f"{pinfo['memory_percent']:.1f}" if pinfo['memory_percent'] else "0.0",
                        pinfo['status']
                    )
                
            except Exception as e:
                table.add_row("Error", str(e), "", "", "")
            
            return table
        
        # Live monitoring
        with Live(update_process_table(), refresh_per_second=1/refresh_rate, console=console) as live:
            try:
                while True:
                    time.sleep(refresh_rate)
                    live.update(update_process_table())
            except KeyboardInterrupt:
                pass
        
        console.print("\n[yellow]Process monitoring stopped[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Process monitoring failed: {e}[/bold red]")
        logger.error(f"Process monitoring error: {e}", exc_info=True)
        raise click.ClickException(f"Process monitoring failed: {e}")


# Helper functions

def _display_diagnostic_results(results: Dict[str, Any], verbose: bool):
    """Display diagnostic results in a formatted table."""
    
    # Summary panel
    summary_text = f"""
[bold]Diagnostic Summary[/bold]

✓ Checks passed: {results.get('checks_passed', 0)}
⚠ Warnings: {results.get('warnings', 0)}  
✗ Critical issues: {results.get('critical_issues', 0)}
[Config] Auto-fixes applied: {results.get('fixes_applied', 0)}
"""
    
    console.print(Panel(summary_text, title="System Diagnostics", border_style="blue"))
    
    # Detailed results
    if verbose or results.get('issues'):
        table = Table(title="Detailed Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        table.add_column("Suggestion", style="blue")
        
        for check_name, check_result in results.get('checks', {}).items():
            status = "✓ Pass" if check_result.get('passed', False) else "✗ Fail"
            if check_result.get('warning', False):
                status = "⚠ Warning"
            
            details = check_result.get('details', '')
            suggestion = check_result.get('suggestion', '')
            
            table.add_row(check_name, status, details, suggestion)
        
        console.print(table)


def _display_logs(logs_data: List[Dict[str, Any]]):
    """Display log entries in a formatted way."""
    
    for log_entry in logs_data[-50:]:  # Show last 50 entries
        timestamp = log_entry.get('timestamp', '')
        level = log_entry.get('level', 'INFO')
        message = log_entry.get('message', '')
        source = log_entry.get('source', '')
        
        # Color code by level
        level_colors = {
            'DEBUG': 'dim',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold red'
        }
        
        level_color = level_colors.get(level, 'white')
        
        console.print(f"[dim]{timestamp}[/dim] [{level_color}]{level}[/{level_color}] [cyan]{source}[/cyan]: {message}")


def _display_metrics_summary(metrics_data: Dict[str, Any]):
    """Display metrics summary."""
    
    summary_table = Table(title="Metrics Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Average", style="green")
    summary_table.add_column("Peak", style="yellow")
    summary_table.add_column("Current", style="blue")
    
    # CPU metrics
    cpu_data = metrics_data.get('cpu', {})
    summary_table.add_row(
        "CPU Usage (%)",
        f"{cpu_data.get('avg', 0):.1f}",
        f"{cpu_data.get('max', 0):.1f}",
        f"{cpu_data.get('current', 0):.1f}"
    )
    
    # Memory metrics
    memory_data = metrics_data.get('memory', {})
    summary_table.add_row(
        "Memory Usage (%)",
        f"{memory_data.get('avg', 0):.1f}",
        f"{memory_data.get('max', 0):.1f}",
        f"{memory_data.get('current', 0):.1f}"
    )
    
    # Disk metrics
    disk_data = metrics_data.get('disk', {})
    summary_table.add_row(
        "Disk Usage (%)",
        f"{disk_data.get('avg', 0):.1f}",
        f"{disk_data.get('max', 0):.1f}",
        f"{disk_data.get('current', 0):.1f}"
    )
    
    # GPU metrics if available
    if 'gpu' in metrics_data:
        gpu_data = metrics_data['gpu']
        summary_table.add_row(
            "GPU Usage (%)",
            f"{gpu_data.get('avg', 0):.1f}",
            f"{gpu_data.get('max', 0):.1f}",
            f"{gpu_data.get('current', 0):.1f}"
        )
    
    console.print(summary_table)
    
    # Alerts if any
    alerts = metrics_data.get('alerts', [])
    if alerts:
        console.print("\n[bold red]Alerts:[/bold red]")
        for alert in alerts:
            console.print(f"  ⚠ {alert}")


# Add monitor to main CLI
if __name__ == "__main__":
    monitor()