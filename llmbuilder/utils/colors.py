"""
Color utilities for CLI output.

Provides consistent color scheme across all CLI commands.
Colors: RED, YELLOW, BLUE (THICK), BLUE (LIGHT), GREEN, WHITE
"""

import click
from typing import Optional, Any
from enum import Enum


class Color(Enum):
    """Defined color palette for CLI output."""
    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"
    BLUE_LIGHT = "cyan"  # Using cyan for light blue
    GREEN = "green"
    WHITE = "white"


class ColorFormatter:
    """Handles colored output formatting."""
    
    @staticmethod
    def format(text: str, color: Color, bold: bool = False) -> str:
        """Format text with specified color."""
        return click.style(text, fg=color.value, bold=bold)
    
    @staticmethod
    def success(text: str) -> str:
        """Format success message in green."""
        return click.style(text, fg=Color.GREEN.value, bold=True)
    
    @staticmethod
    def error(text: str) -> str:
        """Format error message in red."""
        return click.style(text, fg=Color.RED.value, bold=True)
    
    @staticmethod
    def warning(text: str) -> str:
        """Format warning message in yellow."""
        return click.style(text, fg=Color.YELLOW.value, bold=True)
    
    @staticmethod
    def info(text: str) -> str:
        """Format info message in light blue."""
        return click.style(text, fg=Color.BLUE_LIGHT.value)
    
    @staticmethod
    def header(text: str) -> str:
        """Format header text in thick blue."""
        return click.style(text, fg=Color.BLUE.value, bold=True)
    
    @staticmethod
    def emphasis(text: str) -> str:
        """Format emphasized text in white bold."""
        return click.style(text, fg=Color.WHITE.value, bold=True)


def print_colored(text: str, color: Color, bold: bool = False, **kwargs):
    """Print colored text to console."""
    formatted_text = ColorFormatter.format(text, color, bold)
    click.echo(formatted_text, **kwargs)


def print_success(text: str, **kwargs):
    """Print success message."""
    click.echo(ColorFormatter.success(f"✓ {text}"), **kwargs)


def print_error(text: str, **kwargs):
    """Print error message."""
    click.echo(ColorFormatter.error(f"✗ {text}"), **kwargs)


def print_warning(text: str, **kwargs):
    """Print warning message."""
    click.echo(ColorFormatter.warning(f"⚠ {text}"), **kwargs)


def print_info(text: str, **kwargs):
    """Print info message."""
    click.echo(ColorFormatter.info(f"ℹ {text}"), **kwargs)


def print_header(text: str, **kwargs):
    """Print header text."""
    click.echo(ColorFormatter.header(f"\n{text}"), **kwargs)
    click.echo(ColorFormatter.header("=" * len(text)), **kwargs)


def print_section(text: str, **kwargs):
    """Print section header."""
    click.echo(ColorFormatter.header(f"\n{text}"), **kwargs)
    click.echo(ColorFormatter.header("-" * len(text)), **kwargs)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation with colored prompt."""
    prompt_text = ColorFormatter.warning(f"{message} [y/N]" if not default else f"{message} [Y/n]")
    return click.confirm(prompt_text, default=default)


def prompt_choice(message: str, choices: list, default: Optional[str] = None) -> str:
    """Prompt user to choose from a list of options."""
    formatted_message = ColorFormatter.info(message)
    return click.prompt(formatted_message, type=click.Choice(choices), default=default)


def format_table_row(columns: list, widths: list, colors: Optional[list] = None) -> str:
    """Format a table row with proper spacing and colors."""
    if colors is None:
        colors = [Color.WHITE] * len(columns)
    
    formatted_columns = []
    for i, (col, width, color) in enumerate(zip(columns, widths, colors)):
        formatted_col = str(col).ljust(width)
        formatted_columns.append(ColorFormatter.format(formatted_col, color))
    
    return " | ".join(formatted_columns)


def print_table(headers: list, rows: list, header_color: Color = Color.BLUE):
    """Print a formatted table with colors."""
    if not rows:
        print_info("No data to display")
        return
    
    # Calculate column widths
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_colors = [header_color] * len(headers)
    click.echo(format_table_row(headers, widths, header_colors))
    click.echo(ColorFormatter.format("-" * (sum(widths) + 3 * (len(widths) - 1)), Color.BLUE))
    
    # Print rows
    for row in rows:
        click.echo(format_table_row(row, widths))