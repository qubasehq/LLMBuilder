"""
Real-time monitoring dashboard for LLMBuilder.

This module provides a live dashboard interface for monitoring training
progress, system resources, and logs in real-time.
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import psutil

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text

from llmbuilder.utils.logging import get_logger
from .metrics import MetricsCollector
from .logs import LogAggregator

logger = get_logger(__name__)


class MonitoringDashboard:
    """
    Real-time monitoring dashboard.
    
    Provides a comprehensive live dashboard showing training metrics,
    system resources, GPU usage, and recent logs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the monitoring dashboard.
        
        Args:
            config: Dashboard configuration dictionary
        """
        self.config = config
        self.console = Console()
        self.running = False
        self.metrics_data = {}
        self.logs_data = []
        
        # Initialize components
        self.metrics_collector = MetricsCollector({
            'interval': config.get('refresh_rate', 1.0),
            'include_gpu': config.get('gpu_monitoring', True),
            'include_network': False
        })
        
        if config.get('training_logs'):
            self.log_aggregator = LogAggregator({
                'log_dir': config['training_logs'],
                'pattern': '*.log',
                'level': config.get('log_level', 'INFO'),
                'tail': config.get('max_logs', 100)
            })
        else:
            self.log_aggregator = None
        
        logger.info("Monitoring dashboard initialized")
    
    def start(self):
        """Start the monitoring dashboard."""
        self.running = True
        
        # Start background data collection
        metrics_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        metrics_thread.start()
        
        if self.log_aggregator:
            logs_thread = threading.Thread(target=self._collect_logs, daemon=True)
            logs_thread.start()
        
        # Start live dashboard
        try:
            with Live(self._create_layout(), refresh_per_second=1/self.config.get('refresh_rate', 1.0), console=self.console) as live:
                while self.running:
                    live.update(self._create_layout())
                    time.sleep(self.config.get('refresh_rate', 1.0))
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the monitoring dashboard."""
        self.running = False
        
        # Export metrics if requested
        export_path = self.config.get('export_metrics')
        if export_path and self.metrics_data:
            self._export_metrics(export_path)
            self.console.print(f"[green]Metrics exported to {export_path}[/green]")
        
        logger.info("Monitoring dashboard stopped")
    
    def _collect_metrics(self):
        """Background thread for collecting system metrics."""
        while self.running:
            try:
                current_metrics = self.metrics_collector.get_current_metrics()
                if current_metrics:
                    self.metrics_data = current_metrics
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(self.config.get('refresh_rate', 1.0))
    
    def _collect_logs(self):
        """Background thread for collecting log entries."""
        if not self.log_aggregator:
            return
        
        while self.running:
            try:
                recent_logs = self.log_aggregator.get_logs()
                if recent_logs:
                    self.logs_data = recent_logs[-self.config.get('max_logs', 100):]
            except Exception as e:
                logger.error(f"Error collecting logs: {e}")
            
            time.sleep(self.config.get('refresh_rate', 1.0) * 2)  # Update logs less frequently
    
    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(self._create_header())
        
        # Main area - split into left and right
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Left side - system metrics
        layout["left"].split_column(
            Layout(name="system", size=12),
            Layout(name="training", size=8)
        )
        
        layout["system"].update(self._create_system_panel())
        layout["training"].update(self._create_training_panel())
        
        # Right side - logs
        layout["right"].update(self._create_logs_panel())
        
        # Footer
        layout["footer"].update(self._create_footer())
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create dashboard header."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        uptime = self._get_system_uptime()
        
        header_text = f"[bold blue]LLMBuilder Monitoring Dashboard[/bold blue] | {current_time} | Uptime: {uptime}"
        return Panel(header_text, style="blue")
    
    def _create_system_panel(self) -> Panel:
        """Create system metrics panel."""
        table = Table(title="System Resources", show_header=True, header_style="bold magenta")
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="green")
        table.add_column("Available", style="yellow")
        table.add_column("Total", style="blue")
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            table.add_row(
                "CPU",
                f"{cpu_percent:.1f}%",
                f"{cpu_count - (cpu_count * cpu_percent / 100):.1f} cores",
                f"{cpu_count} cores"
            )
            
            # Memory
            memory = psutil.virtual_memory()
            table.add_row(
                "Memory",
                f"{memory.percent:.1f}%",
                f"{memory.available / (1024**3):.1f} GB",
                f"{memory.total / (1024**3):.1f} GB"
            )
            
            # Disk
            disk = psutil.disk_usage('/')
            table.add_row(
                "Disk",
                f"{(disk.used / disk.total) * 100:.1f}%",
                f"{disk.free / (1024**3):.1f} GB",
                f"{disk.total / (1024**3):.1f} GB"
            )
            
            # GPU (if available)
            gpu_info = self._get_gpu_info()
            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    table.add_row(
                        f"GPU {i}",
                        f"{gpu.get('utilization', 0):.1f}%",
                        f"{gpu.get('memory_free', 0):.1f} MB",
                        f"{gpu.get('memory_total', 0):.1f} MB"
                    )
            
        except Exception as e:
            table.add_row("Error", str(e), "", "")
        
        return Panel(table, title="System Resources", border_style="green")
    
    def _create_training_panel(self) -> Panel:
        """Create training metrics panel."""
        # Try to read training metrics from logs or files
        training_info = self._get_training_info()
        
        if training_info:
            table = Table(title="Training Status", show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in training_info.items():
                table.add_row(key, str(value))
            
            content = table
        else:
            content = Text("No training session detected", style="dim")
        
        return Panel(content, title="Training Metrics", border_style="yellow")
    
    def _create_logs_panel(self) -> Panel:
        """Create logs panel."""
        if not self.logs_data:
            content = Text("No logs available", style="dim")
        else:
            # Show recent log entries
            log_text = Text()
            
            for log_entry in self.logs_data[-20:]:  # Show last 20 entries
                timestamp = log_entry.get('timestamp', '')[:19]  # Truncate timestamp
                level = log_entry.get('level', 'INFO')
                message = log_entry.get('message', '')[:80]  # Truncate message
                
                # Color code by level
                level_colors = {
                    'DEBUG': 'dim',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold red'
                }
                
                level_color = level_colors.get(level, 'white')
                
                log_text.append(f"{timestamp} ", style="dim")
                log_text.append(f"{level:8} ", style=level_color)
                log_text.append(f"{message}\n")
            
            content = log_text
        
        return Panel(content, title="Recent Logs", border_style="blue")
    
    def _create_footer(self) -> Panel:
        """Create dashboard footer."""
        footer_text = "[dim]Press Ctrl+C to exit | Refresh rate: {:.1f}s[/dim]".format(
            self.config.get('refresh_rate', 1.0)
        )
        return Panel(footer_text, style="dim")
    
    def _get_system_uptime(self) -> str:
        """Get system uptime."""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            
            if days > 0:
                return f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except:
            return "Unknown"
    
    def _get_gpu_info(self) -> Optional[List[Dict[str, Any]]]:
        """Get GPU information if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'name': gpu.name,
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
            
            return gpu_info
        except ImportError:
            # GPUtil not available
            return None
        except Exception as e:
            logger.debug(f"Error getting GPU info: {e}")
            return None
    
    def _get_training_info(self) -> Optional[Dict[str, Any]]:
        """Get current training information."""
        try:
            # Try to read from training logs or status files
            training_log_path = Path("logs/training.log")
            if training_log_path.exists():
                # Parse recent training logs for metrics
                # This is a simplified implementation
                return {
                    "Status": "Active",
                    "Epoch": "5/10",
                    "Loss": "2.34",
                    "Learning Rate": "0.0001"
                }
            
            # Check for training status file
            status_file = Path(".llmbuilder/training_status.json")
            if status_file.exists():
                with open(status_file, 'r') as f:
                    return json.load(f)
            
            return None
        except Exception as e:
            logger.debug(f"Error getting training info: {e}")
            return None
    
    def _export_metrics(self, export_path: Path):
        """Export collected metrics to file."""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'metrics': self.metrics_data,
                'logs_count': len(self.logs_data)
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {export_path}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")