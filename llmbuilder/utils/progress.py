"""
Enhanced progress bar utilities for CLI operations.

Provides progress bars with status messages and time estimates.
"""

import time
import threading
from typing import Optional, Callable, Any
from contextlib import contextmanager

import click
from tqdm import tqdm

from .colors import ColorFormatter, Color, print_info, print_success, print_error


class ProgressBar:
    """Enhanced progress bar with status messages."""
    
    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        unit: str = "items",
        show_eta: bool = True,
        color: Color = Color.BLUE
    ):
        self.total = total
        self.description = description
        self.unit = unit
        self.show_eta = show_eta
        self.color = color
        self.pbar = None
        self.start_time = None
        self.current_status = ""
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def start(self):
        """Start the progress bar."""
        self.start_time = time.time()
        
        # Configure tqdm with our color scheme
        bar_format = (
            f"{ColorFormatter.format('{desc}', self.color)}: "
            f"{ColorFormatter.format('{percentage:3.0f}%', Color.WHITE)}|"
            f"{ColorFormatter.format('{bar}', self.color)}| "
            f"{ColorFormatter.format('{n_fmt}/{total_fmt}', Color.WHITE)} "
            f"[{ColorFormatter.format('{elapsed}<{remaining}', Color.BLUE_LIGHT)}, "
            f"{ColorFormatter.format('{rate_fmt}', Color.BLUE_LIGHT)}]"
        )
        
        self.pbar = tqdm(
            total=self.total,
            desc=self.description,
            unit=self.unit,
            bar_format=bar_format,
            ncols=80,
            leave=True
        )
        
    def update(self, n: int = 1, status: Optional[str] = None):
        """Update progress bar."""
        if self.pbar:
            if status:
                self.current_status = status
                self.pbar.set_postfix_str(ColorFormatter.format(status, Color.BLUE_LIGHT))
            self.pbar.update(n)
            
    def set_description(self, desc: str):
        """Update the description."""
        self.description = desc
        if self.pbar:
            self.pbar.set_description(ColorFormatter.format(desc, self.color))
            
    def set_status(self, status: str):
        """Update the status message."""
        self.current_status = status
        if self.pbar:
            self.pbar.set_postfix_str(ColorFormatter.format(status, Color.BLUE_LIGHT))
            
    def close(self, success_message: Optional[str] = None):
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()
            
        if success_message:
            print_success(success_message)


class SpinnerProgress:
    """Spinner for indeterminate progress with status messages."""
    
    def __init__(self, message: str = "Processing", color: Color = Color.BLUE):
        self.message = message
        self.color = color
        self.spinner_chars = "|/-\\"
        self.spinner_index = 0
        self.running = False
        self.thread = None
        self.current_status = ""
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def start(self):
        """Start the spinner."""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self, success_message: Optional[str] = None):
        """Stop the spinner."""
        self.running = False
        if self.thread:
            self.thread.join()
            
        # Clear the spinner line
        click.echo("\r" + " " * 80 + "\r", nl=False)
        
        if success_message:
            print_success(success_message)
            
    def set_status(self, status: str):
        """Update the status message."""
        self.current_status = status
        
    def _spin(self):
        """Internal spinner animation."""
        while self.running:
            spinner_char = self.spinner_chars[self.spinner_index]
            status_text = f" - {self.current_status}" if self.current_status else ""
            
            display_text = (
                f"\r{ColorFormatter.format(spinner_char, self.color)} "
                f"{ColorFormatter.format(self.message, Color.WHITE)}"
                f"{ColorFormatter.format(status_text, Color.BLUE_LIGHT)}"
            )
            
            click.echo(display_text, nl=False)
            
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
            time.sleep(0.1)


@contextmanager
def progress_bar(
    total: Optional[int] = None,
    description: str = "Processing",
    unit: str = "items",
    color: Color = Color.BLUE
):
    """Context manager for progress bar."""
    pbar = ProgressBar(total=total, description=description, unit=unit, color=color)
    try:
        yield pbar
    finally:
        pbar.close()


@contextmanager
def spinner(message: str = "Processing", color: Color = Color.BLUE):
    """Context manager for spinner."""
    spin = SpinnerProgress(message=message, color=color)
    try:
        yield spin
    finally:
        spin.stop()


def long_running_task(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    message: str = "Processing",
    success_message: Optional[str] = None,
    timeout_message: str = "This is taking longer than expected...",
    timeout_seconds: int = 5
) -> Any:
    """
    Execute a long-running task with progress indication.
    
    Shows a spinner initially, then adds timeout message if task takes too long.
    """
    if kwargs is None:
        kwargs = {}
        
    result = None
    exception = None
    
    def run_task():
        nonlocal result, exception
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = e
    
    # Start the task in a separate thread
    task_thread = threading.Thread(target=run_task)
    task_thread.daemon = True
    task_thread.start()
    
    # Show progress with timeout handling
    with spinner(message) as spin:
        start_time = time.time()
        timeout_shown = False
        
        while task_thread.is_alive():
            elapsed = time.time() - start_time
            
            if elapsed > timeout_seconds and not timeout_shown:
                spin.set_status(timeout_message)
                timeout_shown = True
            elif elapsed > timeout_seconds * 2 and timeout_shown:
                spin.set_status("Still processing, please wait...")
                
            time.sleep(0.1)
    
    # Wait for task to complete
    task_thread.join()
    
    if exception:
        raise exception
        
    if success_message:
        print_success(success_message)
        
    return result


def show_step_progress(steps: list, current_step: int, step_name: str):
    """Show progress through a series of steps."""
    total_steps = len(steps)
    progress_percent = (current_step / total_steps) * 100
    
    # Create progress bar visualization
    bar_width = 30
    filled_width = int((current_step / total_steps) * bar_width)
    bar = "█" * filled_width + "░" * (bar_width - filled_width)
    
    # Format the display
    step_info = f"Step {current_step}/{total_steps}: {step_name}"
    progress_display = f"[{ColorFormatter.format(bar, Color.BLUE)}] {progress_percent:.0f}%"
    
    click.echo(f"\n{ColorFormatter.header(step_info)}")
    click.echo(f"{progress_display}\n")


def confirm_with_progress(message: str, task_func: Callable, *args, **kwargs) -> bool:
    """Confirm action and execute with progress indication."""
    if not click.confirm(ColorFormatter.warning(f"{message} Continue?")):
        print_info("Operation cancelled")
        return False
        
    try:
        result = long_running_task(
            task_func,
            args=args,
            kwargs=kwargs,
            message="Executing operation",
            success_message="Operation completed successfully"
        )
        return True
    except Exception as e:
        print_error(f"Operation failed: {str(e)}")
        return False