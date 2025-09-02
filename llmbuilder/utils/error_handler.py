"""
Enhanced error handling system with user-friendly messages and recovery suggestions.
"""

import sys
import traceback
import click
from typing import Optional, Dict, Any, List, Callable
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

console = Console()


class LLMBuilderError(Exception):
    """Base exception for LLMBuilder errors."""
    
    def __init__(self, message: str, suggestions: Optional[List[str]] = None, 
                 recovery_commands: Optional[List[str]] = None):
        super().__init__(message)
        self.suggestions = suggestions or []
        self.recovery_commands = recovery_commands or []


class DataProcessingError(LLMBuilderError):
    """Error in data processing operations."""
    pass


class ModelError(LLMBuilderError):
    """Error in model operations."""
    pass


class TrainingError(LLMBuilderError):
    """Error in training operations."""
    pass


class ConfigurationError(LLMBuilderError):
    """Error in configuration."""
    pass


class DependencyError(LLMBuilderError):
    """Error with missing or incompatible dependencies."""
    pass


class ErrorHandler:
    """Enhanced error handler with user-friendly messages and recovery suggestions."""
    
    def __init__(self):
        self.error_patterns = self._load_error_patterns()
        self.recovery_suggestions = self._load_recovery_suggestions()
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions with enhanced error reporting."""
        if exc_type == KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            return
        
        if exc_type == click.ClickException:
            # Let Click handle its own exceptions
            exc_value.show()
            return
        
        # Check if it's a known LLMBuilder error
        if isinstance(exc_value, LLMBuilderError):
            self._show_llmbuilder_error(exc_value)
        else:
            self._show_generic_error(exc_type, exc_value, exc_traceback)
    
    def _show_llmbuilder_error(self, error: LLMBuilderError):
        """Show formatted LLMBuilder error with suggestions."""
        error_type = type(error).__name__
        
        # Create error panel
        error_text = f"[bold red]{error_type}[/bold red]\n\n{str(error)}"
        
        if error.suggestions:
            error_text += "\n\n[bold yellow]Suggestions:[/bold yellow]"
            for i, suggestion in enumerate(error.suggestions, 1):
                error_text += f"\n  {i}. {suggestion}"
        
        if error.recovery_commands:
            error_text += "\n\n[bold cyan]Try these commands:[/bold cyan]"
            for cmd in error.recovery_commands:
                error_text += f"\n  [green]$ {cmd}[/green]"
        
        console.print(Panel(
            error_text,
            title="Error",
            border_style="red",
            expand=False
        ))
    
    def _show_generic_error(self, exc_type, exc_value, exc_traceback):
        """Show generic error with pattern matching for suggestions."""
        error_message = str(exc_value)
        error_type = exc_type.__name__
        
        # Try to match error patterns
        suggestions = self._match_error_patterns(error_message, error_type)
        
        # Create error display
        error_text = f"[bold red]{error_type}[/bold red]\n\n{error_message}"
        
        if suggestions:
            error_text += "\n\n[bold yellow]Possible solutions:[/bold yellow]"
            for i, suggestion in enumerate(suggestions, 1):
                error_text += f"\n  {i}. {suggestion['message']}"
                if 'command' in suggestion:
                    error_text += f"\n     [green]$ {suggestion['command']}[/green]"
        
        # Add debug information
        error_text += "\n\n[dim]For more help, run: llmbuilder help troubleshooting[/dim]"
        
        console.print(Panel(
            error_text,
            title="Unexpected Error",
            border_style="red",
            expand=False
        ))
        
        # Optionally show traceback in verbose mode
        if console.options.legacy_windows or "--verbose" in sys.argv:
            console.print("\n[dim]Full traceback:[/dim]")
            console.print_exception(show_locals=True)
    
    def _match_error_patterns(self, error_message: str, error_type: str) -> List[Dict[str, str]]:
        """Match error message against known patterns."""
        suggestions = []
        
        error_lower = error_message.lower()
        
        # CUDA/GPU errors
        if "cuda" in error_lower and "out of memory" in error_lower:
            suggestions.append({
                "message": "Reduce batch size to use less GPU memory",
                "command": "llmbuilder config set training.batch_size 2"
            })
            suggestions.append({
                "message": "Enable gradient checkpointing to save memory",
                "command": "llmbuilder config set training.gradient_checkpointing true"
            })
        
        # File not found errors
        elif "no such file" in error_lower or "file not found" in error_lower:
            suggestions.append({
                "message": "Check if the file path is correct and the file exists"
            })
            suggestions.append({
                "message": "Use absolute paths or ensure you're in the correct directory"
            })
        
        # Permission errors
        elif "permission denied" in error_lower:
            suggestions.append({
                "message": "Check file permissions and ensure you have write access"
            })
            suggestions.append({
                "message": "Try running with appropriate permissions or change file ownership"
            })
        
        # Import/module errors
        elif "no module named" in error_lower or "importerror" in error_type.lower():
            module_name = self._extract_module_name(error_message)
            if module_name:
                suggestions.append({
                    "message": f"Install the missing module: {module_name}",
                    "command": f"pip install {module_name}"
                })
            suggestions.append({
                "message": "Install LLMBuilder with all dependencies",
                "command": "pip install llmbuilder[all]"
            })
        
        # Network/connection errors
        elif "connection" in error_lower or "network" in error_lower:
            suggestions.append({
                "message": "Check your internet connection"
            })
            suggestions.append({
                "message": "Try again later or use cached/offline mode if available"
            })
        
        # Configuration errors
        elif "config" in error_lower or "configuration" in error_lower:
            suggestions.append({
                "message": "Validate your configuration file",
                "command": "llmbuilder config validate"
            })
            suggestions.append({
                "message": "Reset to default configuration",
                "command": "llmbuilder config reset"
            })
        
        # Data format errors
        elif "json" in error_lower or "yaml" in error_lower or "format" in error_lower:
            suggestions.append({
                "message": "Check the format of your data files"
            })
            suggestions.append({
                "message": "Validate your data",
                "command": "llmbuilder data validate"
            })
        
        return suggestions
    
    def _extract_module_name(self, error_message: str) -> Optional[str]:
        """Extract module name from import error message."""
        import re
        
        # Pattern to match "No module named 'module_name'"
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_message)
        if match:
            return match.group(1)
        
        return None
    
    def _load_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load error patterns and their solutions."""
        return {
            "cuda_oom": {
                "patterns": ["cuda", "out of memory", "gpu"],
                "suggestions": [
                    "Reduce batch size",
                    "Use gradient checkpointing",
                    "Try CPU training instead"
                ],
                "commands": [
                    "llmbuilder config set training.batch_size 2",
                    "llmbuilder config set device.use_cuda false"
                ]
            },
            "file_not_found": {
                "patterns": ["no such file", "file not found", "does not exist"],
                "suggestions": [
                    "Check file path spelling",
                    "Ensure file exists in the specified location",
                    "Use absolute paths"
                ]
            },
            "dependency_missing": {
                "patterns": ["no module named", "importerror", "cannot import"],
                "suggestions": [
                    "Install missing dependencies",
                    "Update package installation"
                ],
                "commands": [
                    "pip install llmbuilder[all]",
                    "pip install --upgrade llmbuilder"
                ]
            }
        }
    
    def _load_recovery_suggestions(self) -> Dict[str, List[str]]:
        """Load recovery suggestions for different error types."""
        return {
            "DataProcessingError": [
                "Check input data format and encoding",
                "Verify file permissions and accessibility",
                "Try processing smaller batches of data"
            ],
            "ModelError": [
                "Verify model path and format",
                "Check model compatibility",
                "Try downloading the model again"
            ],
            "TrainingError": [
                "Check training configuration",
                "Verify data preprocessing completed successfully",
                "Monitor system resources (GPU/CPU/memory)"
            ],
            "ConfigurationError": [
                "Validate configuration syntax",
                "Check for required configuration fields",
                "Reset to default configuration if needed"
            ],
            "DependencyError": [
                "Update package installation",
                "Install missing dependencies",
                "Check Python version compatibility"
            ]
        }


def setup_error_handler():
    """Set up the enhanced error handler."""
    handler = ErrorHandler()
    sys.excepthook = handler.handle_exception


def create_error(error_type: str, message: str, suggestions: Optional[List[str]] = None,
                recovery_commands: Optional[List[str]] = None) -> LLMBuilderError:
    """Create a specific LLMBuilder error with suggestions."""
    error_classes = {
        "data": DataProcessingError,
        "model": ModelError,
        "training": TrainingError,
        "config": ConfigurationError,
        "dependency": DependencyError
    }
    
    error_class = error_classes.get(error_type, LLMBuilderError)
    return error_class(message, suggestions, recovery_commands)


def handle_click_exception(func: Callable) -> Callable:
    """Decorator to handle Click command exceptions with enhanced error reporting."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LLMBuilderError as e:
            handler = ErrorHandler()
            handler._show_llmbuilder_error(e)
            sys.exit(1)
        except Exception as e:
            handler = ErrorHandler()
            handler._show_generic_error(type(e), e, e.__traceback__)
            sys.exit(1)
    
    return wrapper


# Context manager for better error handling
class ErrorContext:
    """Context manager for enhanced error handling in specific operations."""
    
    def __init__(self, operation: str, suggestions: Optional[List[str]] = None):
        self.operation = operation
        self.suggestions = suggestions or []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # Enhance the error with context
            if isinstance(exc_value, LLMBuilderError):
                # Add operation context to existing error
                exc_value.suggestions.extend(self.suggestions)
            else:
                # Convert to LLMBuilder error with context
                message = f"Error during {self.operation}: {str(exc_value)}"
                raise LLMBuilderError(message, self.suggestions) from exc_value
        
        return False  # Don't suppress exceptions