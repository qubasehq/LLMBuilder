"""
Status display utilities for CLI operations.

Provides consistent status displays, dashboards, and real-time updates.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import click

from .colors import ColorFormatter, Color, print_info, print_success, print_error


@dataclass
class StatusItem:
    """Represents a status item with name, value, and color."""
    name: str
    value: str
    color: Color = Color.WHITE
    status: str = "ok"  # ok, warning, error, info


class StatusDisplay:
    """Real-time status display for CLI operations."""
    
    def __init__(self, title: str = "Status", refresh_rate: float = 1.0):
        self.title = title
        self.refresh_rate = refresh_rate
        self.items: Dict[str, StatusItem] = {}
        self.running = False
        self.thread = None
        self.last_update = None
        
    def add_item(self, key: str, name: str, value: str, color: Color = Color.WHITE, status: str = "ok"):
        """Add or update a status item."""
        self.items[key] = StatusItem(name=name, value=value, color=color, status=status)
        self.last_update = datetime.now()
        
    def update_item(self, key: str, value: str, color: Optional[Color] = None, status: Optional[str] = None):
        """Update an existing status item."""
        if key in self.items:
            self.items[key].value = value
            if color is not None:
                self.items[key].color = color
            if status is not None:
                self.items[key].status = status
            self.last_update = datetime.now()
            
    def remove_item(self, key: str):
        """Remove a status item."""
        if key in self.items:
            del self.items[key]
            self.last_update = datetime.now()
            
    def start(self):
        """Start the status display."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._display_loop)
            self.thread.daemon = True
            self.thread.start()
            
    def stop(self):
        """Stop the status display."""
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _display_loop(self):
        """Main display loop."""
        while self.running:
            self._render_status()
            time.sleep(self.refresh_rate)
            
    def _render_status(self):
        """Render the current status."""
        if not self.items:
            return
            
        # Clear screen and move cursor to top
        click.echo("\033[2J\033[H", nl=False)
        
        # Display title
        title_text = ColorFormatter.header(self.title)
        click.echo(title_text)
        click.echo(ColorFormatter.format("=" * len(self.title), Color.BLUE))
        
        # Display timestamp
        if self.last_update:
            timestamp = self.last_update.strftime("%Y-%m-%d %H:%M:%S")
            click.echo(ColorFormatter.info(f"Last updated: {timestamp}"))
        
        click.echo()
        
        # Display status items
        max_name_length = max(len(item.name) for item in self.items.values()) if self.items else 0
        
        for item in self.items.values():
            # Status indicator
            if item.status == "ok":
                indicator = ColorFormatter.success("●")
            elif item.status == "warning":
                indicator = ColorFormatter.warning("●")
            elif item.status == "error":
                indicator = ColorFormatter.error("●")
            else:
                indicator = ColorFormatter.info("●")
            
            # Format name and value
            name_padded = item.name.ljust(max_name_length)
            name_colored = ColorFormatter.format(name_padded, Color.WHITE)
            value_colored = ColorFormatter.format(item.value, item.color)
            
            click.echo(f"{indicator} {name_colored}: {value_colored}")


class ProgressTracker:
    """Track progress across multiple operations."""
    
    def __init__(self, title: str = "Progress"):
        self.title = title
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.display = StatusDisplay(title)
        
    def add_operation(self, key: str, name: str, total: int = 100):
        """Add a new operation to track."""
        self.operations[key] = {
            "name": name,
            "current": 0,
            "total": total,
            "status": "running",
            "start_time": datetime.now()
        }
        self._update_display()
        
    def update_progress(self, key: str, current: int, status: str = "running"):
        """Update progress for an operation."""
        if key in self.operations:
            self.operations[key]["current"] = current
            self.operations[key]["status"] = status
            self._update_display()
            
    def complete_operation(self, key: str, success: bool = True):
        """Mark an operation as complete."""
        if key in self.operations:
            self.operations[key]["status"] = "completed" if success else "failed"
            self.operations[key]["current"] = self.operations[key]["total"]
            self._update_display()
            
    def start_display(self):
        """Start the progress display."""
        self.display.start()
        
    def stop_display(self):
        """Stop the progress display."""
        self.display.stop()
        
    def _update_display(self):
        """Update the status display with current progress."""
        for key, op in self.operations.items():
            current = op["current"]
            total = op["total"]
            percentage = (current / total * 100) if total > 0 else 0
            
            # Create progress bar
            bar_width = 20
            filled = int(percentage / 100 * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            # Determine color and status based on operation status
            if op["status"] == "completed":
                color = Color.GREEN
                status = "ok"
                value = f"[{bar}] 100% - Completed"
            elif op["status"] == "failed":
                color = Color.RED
                status = "error"
                value = f"[{bar}] {percentage:.0f}% - Failed"
            elif op["status"] == "running":
                color = Color.BLUE
                status = "info"
                value = f"[{bar}] {percentage:.0f}% - Running"
            else:
                color = Color.YELLOW
                status = "warning"
                value = f"[{bar}] {percentage:.0f}% - {op['status']}"
            
            self.display.add_item(key, op["name"], value, color, status)


def show_system_status():
    """Display system status information."""
    import psutil
    import platform
    
    print_header("System Status")
    
    # System information
    system_info = [
        ["Platform", platform.system()],
        ["Architecture", platform.machine()],
        ["Python Version", platform.python_version()],
        ["CPU Cores", str(psutil.cpu_count())],
        ["Memory Total", f"{psutil.virtual_memory().total / (1024**3):.1f} GB"],
        ["Memory Available", f"{psutil.virtual_memory().available / (1024**3):.1f} GB"],
        ["Disk Free", f"{psutil.disk_usage('/').free / (1024**3):.1f} GB"]
    ]
    
    from .colors import print_table
    print_table(["Component", "Value"], system_info, header_color=Color.BLUE)
    
    # GPU information if available
    try:
from llmbuilder.utils.lazy_imports import torch
        if torch.cuda.is_available():
            print_info(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print_info(f"CUDA Version: {torch.version.cuda}")
            print_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print_info("\nGPU: Not available")
    except ImportError:
        print_info("\nGPU: PyTorch not installed")


def show_training_dashboard(session_data: Dict[str, Any]):
    """Show a training dashboard with real-time metrics."""
    display = StatusDisplay("Training Dashboard", refresh_rate=2.0)
    
    try:
        display.start()
        
        # Add initial status items
        display.add_item("session", "Session ID", session_data.get("session_id", "N/A"))
        display.add_item("model", "Model", session_data.get("model_name", "N/A"))
        display.add_item("status", "Status", session_data.get("status", "Unknown"), Color.BLUE)
        display.add_item("progress", "Progress", "0%", Color.YELLOW)
        display.add_item("loss", "Loss", "N/A", Color.GREEN)
        display.add_item("lr", "Learning Rate", "N/A", Color.BLUE_LIGHT)
        display.add_item("epoch", "Epoch", "0/0", Color.WHITE)
        
        # Keep dashboard running until interrupted
        try:
            while True:
                time.sleep(1)
                # Here you would update with real training metrics
                # This is a placeholder for demonstration
                
        except KeyboardInterrupt:
            print_info("\nDashboard stopped by user")
            
    finally:
        display.stop()


def create_loading_animation(message: str, duration: float = 3.0):
    """Create a loading animation with dots."""
    def animate():
        start_time = time.time()
        dots = 0
        
        while time.time() - start_time < duration:
            dots_str = "." * (dots % 4)
            display_text = f"\r{ColorFormatter.info(message)}{dots_str}   "
            click.echo(display_text, nl=False)
            time.sleep(0.5)
            dots += 1
        
        # Clear the line
        click.echo("\r" + " " * (len(message) + 10) + "\r", nl=False)
    
    thread = threading.Thread(target=animate)
    thread.daemon = True
    thread.start()
    return thread