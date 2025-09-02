"""
Inference and interactive testing CLI commands for LLMBuilder.

This module provides commands for interactive model testing, prompt management,
and conversation history tracking.
"""

import click
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import get_logger
from llmbuilder.core.inference.engine import InferenceEngine
from llmbuilder.core.inference.templates import PromptTemplateManager
from llmbuilder.core.inference.history import ConversationHistory

logger = get_logger(__name__)
console = Console()


@click.group()
def inference():
    """Interactive model testing and inference commands."""
    pass


@inference.command()
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
    '--device',
    type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
    default='auto',
    help='Device to use for inference (default: auto)'
)
@click.option(
    '--max-tokens',
    type=int,
    default=100,
    help='Maximum tokens to generate (default: 100)'
)
@click.option(
    '--temperature',
    type=float,
    default=0.8,
    help='Sampling temperature (default: 0.8)'
)
@click.option(
    '--top-k',
    type=int,
    default=50,
    help='Top-k sampling parameter (default: 50)'
)
@click.option(
    '--top-p',
    type=float,
    default=0.9,
    help='Top-p (nucleus) sampling parameter (default: 0.9)'
)
@click.option(
    '--save-history',
    type=click.Path(path_type=Path),
    help='Save conversation history to file'
)
@click.option(
    '--load-history',
    type=click.Path(exists=True, path_type=Path),
    help='Load previous conversation history'
)
@click.option(
    '--template',
    help='Use a prompt template by name'
)
@click.pass_context
def chat(
    ctx: click.Context,
    model: Optional[Path],
    tokenizer: Optional[Path],
    config: Optional[Path],
    device: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    save_history: Optional[Path],
    load_history: Optional[Path],
    template: Optional[str]
):
    """
    Start an interactive chat session with the model.
    
    This command provides a conversational interface for testing model responses
    with adjustable generation parameters and conversation history management.
    
    Examples:
        llmbuilder inference chat --model model.pt --tokenizer tokenizer/
        llmbuilder inference chat --template creative-writing --save-history chat.json
        llmbuilder inference chat --load-history previous-chat.json
    """
    try:
        console.print(f"[bold blue]Starting interactive chat session[/bold blue]")
        
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
        
        # Initialize inference engine
        console.print("[yellow]Loading model...[/yellow]")
        engine = InferenceEngine(
            model_path=model,
            tokenizer_path=tokenizer,
            config_path=config,
            device=device
        )
        
        # Initialize prompt template manager
        template_manager = PromptTemplateManager()
        
        # Initialize conversation history
        history = ConversationHistory()
        if load_history:
            history.load_from_file(load_history)
            console.print(f"[green]Loaded conversation history from {load_history}[/green]")
        
        # Generation parameters
        gen_params = {
            'max_new_tokens': max_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'do_sample': True
        }
        
        # Apply template if specified
        if template:
            template_config = template_manager.get_template(template)
            if template_config:
                gen_params.update(template_config.get('parameters', {}))
                console.print(f"[green]Applied template: {template}[/green]")
            else:
                console.print(f"[yellow]Template '{template}' not found, using defaults[/yellow]")
        
        # Start interactive session
        _run_interactive_chat(engine, history, template_manager, gen_params, save_history)
        
    except Exception as e:
        console.print(f"[bold red]✗ Chat session failed: {e}[/bold red]")
        logger.error(f"Chat session error: {e}", exc_info=True)
        raise click.ClickException(f"Chat session failed: {e}")


@inference.command()
@click.argument('prompt', type=str)
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
    '--max-tokens',
    type=int,
    default=100,
    help='Maximum tokens to generate (default: 100)'
)
@click.option(
    '--temperature',
    type=float,
    default=0.8,
    help='Sampling temperature (default: 0.8)'
)
@click.option(
    '--top-k',
    type=int,
    default=50,
    help='Top-k sampling parameter (default: 50)'
)
@click.option(
    '--top-p',
    type=float,
    default=0.9,
    help='Top-p (nucleus) sampling parameter (default: 0.9)'
)
@click.option(
    '--template',
    help='Use a prompt template by name'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Save output to file'
)
def generate(
    prompt: str,
    model: Optional[Path],
    tokenizer: Optional[Path],
    config: Optional[Path],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    template: Optional[str],
    output: Optional[Path]
):
    """
    Generate text from a single prompt.
    
    This command generates text from a single prompt without interactive mode,
    useful for batch processing or scripting.
    
    Examples:
        llmbuilder inference generate "Once upon a time" --max-tokens 200
        llmbuilder inference generate "Explain quantum computing" --template technical
        llmbuilder inference generate "Write a poem" --output poem.txt
    """
    try:
        console.print(f"[bold blue]Generating text for prompt[/bold blue]")
        
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
        
        # Initialize inference engine
        console.print("[yellow]Loading model...[/yellow]")
        engine = InferenceEngine(
            model_path=model,
            tokenizer_path=tokenizer,
            config_path=config
        )
        
        # Apply template if specified
        gen_params = {
            'max_new_tokens': max_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'do_sample': True
        }
        
        if template:
            template_manager = PromptTemplateManager()
            template_config = template_manager.get_template(template)
            if template_config:
                gen_params.update(template_config.get('parameters', {}))
                # Apply template formatting to prompt
                prompt = template_config.get('format', '{prompt}').format(prompt=prompt)
                console.print(f"[green]Applied template: {template}[/green]")
        
        # Generate text
        console.print("[yellow]Generating...[/yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating text...", total=None)
            
            generated_text = engine.generate_text(prompt, **gen_params)
            
            progress.update(task, completed=True)
        
        # Display results
        console.print(Panel(
            f"[bold]Prompt:[/bold] {prompt}\n\n[bold]Generated:[/bold]\n{generated_text}",
            title="Generation Result",
            border_style="green"
        ))
        
        # Save to file if requested
        if output:
            result = {
                'prompt': prompt,
                'generated_text': generated_text,
                'parameters': gen_params,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]Output saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Text generation failed: {e}[/bold red]")
        logger.error(f"Text generation error: {e}", exc_info=True)
        raise click.ClickException(f"Text generation failed: {e}")


@inference.group()
def templates():
    """Manage prompt templates."""
    pass


@templates.command()
def list():
    """List available prompt templates."""
    try:
        template_manager = PromptTemplateManager()
        available_templates = template_manager.list_templates()
        
        if not available_templates:
            console.print("[yellow]No prompt templates found[/yellow]")
            return
        
        table = Table(title="Available Prompt Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Parameters", style="yellow")
        
        for name, template in available_templates.items():
            params = ", ".join(f"{k}={v}" for k, v in template.get('parameters', {}).items())
            table.add_row(
                name,
                template.get('description', 'No description'),
                params or 'Default'
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to list templates: {e}[/bold red]")
        raise click.ClickException(f"Failed to list templates: {e}")


@templates.command()
@click.argument('name')
@click.option(
    '--description',
    help='Template description'
)
@click.option(
    '--format',
    help='Template format string (use {prompt} for prompt placeholder)'
)
@click.option(
    '--temperature',
    type=float,
    help='Default temperature for this template'
)
@click.option(
    '--max-tokens',
    type=int,
    help='Default max tokens for this template'
)
@click.option(
    '--top-k',
    type=int,
    help='Default top-k for this template'
)
@click.option(
    '--top-p',
    type=float,
    help='Default top-p for this template'
)
def create(
    name: str,
    description: Optional[str],
    format: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_k: Optional[int],
    top_p: Optional[float]
):
    """
    Create a new prompt template.
    
    Templates allow you to save prompt formats and generation parameters
    for reuse across different inference sessions.
    
    Examples:
        llmbuilder inference templates create creative --description "Creative writing" --temperature 1.0
        llmbuilder inference templates create technical --format "Explain {prompt} in technical terms"
    """
    try:
        template_manager = PromptTemplateManager()
        
        # Build template configuration
        template_config = {}
        
        if description:
            template_config['description'] = description
        
        if format:
            template_config['format'] = format
        
        # Build parameters
        parameters = {}
        if temperature is not None:
            parameters['temperature'] = temperature
        if max_tokens is not None:
            parameters['max_new_tokens'] = max_tokens
        if top_k is not None:
            parameters['top_k'] = top_k
        if top_p is not None:
            parameters['top_p'] = top_p
        
        if parameters:
            template_config['parameters'] = parameters
        
        # Save template
        template_manager.save_template(name, template_config)
        
        console.print(f"[green]✓ Template '{name}' created successfully[/green]")
        
        # Show template details
        table = Table(title=f"Template: {name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in template_config.items():
            if key == 'parameters':
                for param_key, param_value in value.items():
                    table.add_row(f"  {param_key}", str(param_value))
            else:
                table.add_row(key, str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to create template: {e}[/bold red]")
        raise click.ClickException(f"Failed to create template: {e}")


@templates.command()
@click.argument('name')
def show(name: str):
    """Show details of a specific template."""
    try:
        template_manager = PromptTemplateManager()
        template = template_manager.get_template(name)
        
        if not template:
            console.print(f"[red]Template '{name}' not found[/red]")
            return
        
        table = Table(title=f"Template: {name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in template.items():
            if key == 'parameters':
                for param_key, param_value in value.items():
                    table.add_row(f"  {param_key}", str(param_value))
            else:
                table.add_row(key, str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to show template: {e}[/bold red]")
        raise click.ClickException(f"Failed to show template: {e}")


@templates.command()
@click.argument('name')
def delete(name: str):
    """Delete a prompt template."""
    try:
        template_manager = PromptTemplateManager()
        
        if not template_manager.get_template(name):
            console.print(f"[red]Template '{name}' not found[/red]")
            return
        
        if Confirm.ask(f"Delete template '{name}'?"):
            template_manager.delete_template(name)
            console.print(f"[green]✓ Template '{name}' deleted[/green]")
        else:
            console.print("Cancelled")
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to delete template: {e}[/bold red]")
        raise click.ClickException(f"Failed to delete template: {e}")


@inference.group()
def history():
    """Manage conversation history."""
    pass


@history.command()
@click.option(
    '--limit',
    type=int,
    default=10,
    help='Number of recent conversations to show (default: 10)'
)
def list(limit: int):
    """List recent conversation histories."""
    try:
        history_manager = ConversationHistory()
        histories = history_manager.list_histories(limit=limit)
        
        if not histories:
            console.print("[yellow]No conversation histories found[/yellow]")
            return
        
        table = Table(title="Recent Conversation Histories")
        table.add_column("File", style="cyan")
        table.add_column("Date", style="green")
        table.add_column("Messages", style="yellow")
        table.add_column("Size", style="blue")
        
        for hist in histories:
            table.add_row(
                hist['filename'],
                hist['date'],
                str(hist['message_count']),
                hist['size']
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to list histories: {e}[/bold red]")
        raise click.ClickException(f"Failed to list histories: {e}")


@history.command()
@click.argument('filename', type=click.Path(exists=True, path_type=Path))
def show(filename: Path):
    """Show conversation history from a file."""
    try:
        history_manager = ConversationHistory()
        history_manager.load_from_file(filename)
        
        messages = history_manager.get_messages()
        
        if not messages:
            console.print("[yellow]No messages in history file[/yellow]")
            return
        
        console.print(f"[bold]Conversation History: {filename}[/bold]\n")
        
        for i, message in enumerate(messages, 1):
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            timestamp = message.get('timestamp', '')
            
            if role == 'user':
                console.print(f"[bold blue]User ({timestamp}):[/bold blue]")
                console.print(f"  {content}\n")
            elif role == 'assistant':
                console.print(f"[bold green]Assistant ({timestamp}):[/bold green]")
                console.print(f"  {content}\n")
            else:
                console.print(f"[bold yellow]{role.title()} ({timestamp}):[/bold yellow]")
                console.print(f"  {content}\n")
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to show history: {e}[/bold red]")
        raise click.ClickException(f"Failed to show history: {e}")


@history.command()
@click.argument('filename', type=click.Path(exists=True, path_type=Path))
def delete(filename: Path):
    """Delete a conversation history file."""
    try:
        if Confirm.ask(f"Delete conversation history '{filename}'?"):
            filename.unlink()
            console.print(f"[green]✓ History file '{filename}' deleted[/green]")
        else:
            console.print("Cancelled")
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to delete history: {e}[/bold red]")
        raise click.ClickException(f"Failed to delete history: {e}")


# Helper functions

def _run_interactive_chat(
    engine,
    history,
    template_manager,
    gen_params: Dict[str, Any],
    save_history: Optional[Path]
):
    """Run the interactive chat session."""
    console.print(Panel(
        "[bold]Interactive Chat Session[/bold]\n\n"
        "Commands:\n"
        "  /help - Show help\n"
        "  /settings - Adjust generation parameters\n"
        "  /template <name> - Apply a template\n"
        "  /save <filename> - Save conversation\n"
        "  /clear - Clear conversation history\n"
        "  /quit - Exit chat\n\n"
        "Type your message and press Enter to chat!",
        title="Welcome",
        border_style="blue"
    ))
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                if _handle_chat_command(user_input, engine, history, template_manager, gen_params):
                    break  # Exit chat
                continue
            
            # Add user message to history
            history.add_message('user', user_input)
            
            # Generate response
            console.print("[yellow]Thinking...[/yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating response...", total=None)
                
                response = engine.generate_text(user_input, **gen_params)
                
                progress.update(task, completed=True)
            
            # Extract just the new generated part (remove the original prompt)
            if response.startswith(user_input):
                response = response[len(user_input):].strip()
            
            # Display response
            console.print(f"\n[bold green]Assistant:[/bold green]")
            console.print(Panel(response, border_style="green"))
            
            # Add assistant response to history
            history.add_message('assistant', response)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Chat interrupted. Type /quit to exit or continue chatting.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.error(f"Chat error: {e}", exc_info=True)
    
    # Save history if requested
    if save_history:
        history.save_to_file(save_history)
        console.print(f"[green]Conversation saved to {save_history}[/green]")


def _handle_chat_command(
    command: str,
    engine,
    history,
    template_manager,
    gen_params: Dict[str, Any]
) -> bool:
    """Handle chat commands. Returns True if should exit chat."""
    parts = command[1:].split()
    cmd = parts[0].lower() if parts else ''
    
    if cmd == 'help':
        console.print(Panel(
            "[bold]Available Commands:[/bold]\n\n"
            "/help - Show this help message\n"
            "/settings - Adjust generation parameters\n"
            "/template <name> - Apply a prompt template\n"
            "/save <filename> - Save conversation to file\n"
            "/clear - Clear conversation history\n"
            "/quit - Exit chat session",
            title="Help",
            border_style="blue"
        ))
        
    elif cmd == 'settings':
        _adjust_settings(gen_params)
        
    elif cmd == 'template':
        if len(parts) > 1:
            template_name = parts[1]
            template_config = template_manager.get_template(template_name)
            if template_config:
                gen_params.update(template_config.get('parameters', {}))
                console.print(f"[green]Applied template: {template_name}[/green]")
            else:
                console.print(f"[red]Template '{template_name}' not found[/red]")
        else:
            console.print("[yellow]Usage: /template <name>[/yellow]")
    
    elif cmd == 'save':
        if len(parts) > 1:
            filename = Path(parts[1])
            history.save_to_file(filename)
            console.print(f"[green]Conversation saved to {filename}[/green]")
        else:
            console.print("[yellow]Usage: /save <filename>[/yellow]")
    
    elif cmd == 'clear':
        if Confirm.ask("Clear conversation history?"):
            history.clear()
            console.print("[green]Conversation history cleared[/green]")
    
    elif cmd == 'quit':
        console.print("[blue]Goodbye![/blue]")
        return True
    
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[yellow]Type /help for available commands[/yellow]")
    
    return False


def _adjust_settings(gen_params: Dict[str, Any]):
    """Interactive settings adjustment."""
    console.print("\n[bold]Current Generation Settings:[/bold]")
    
    table = Table()
    table.add_column("Parameter", style="cyan")
    table.add_column("Current Value", style="green")
    
    for key, value in gen_params.items():
        if key != 'do_sample':  # Skip internal parameter
            table.add_row(key, str(value))
    
    console.print(table)
    
    console.print("\n[yellow]Enter new values (press Enter to keep current):[/yellow]")
    
    for key in ['max_new_tokens', 'temperature', 'top_k', 'top_p']:
        if key in gen_params:
            current_value = gen_params[key]
            new_value = Prompt.ask(f"{key} [{current_value}]", default="")
            
            if new_value:
                try:
                    if key == 'max_new_tokens' or key == 'top_k':
                        gen_params[key] = int(new_value)
                    else:
                        gen_params[key] = float(new_value)
                    console.print(f"[green]Updated {key} to {gen_params[key]}[/green]")
                except ValueError:
                    console.print(f"[red]Invalid value for {key}, keeping current value[/red]")
    
    console.print("[green]Settings updated![/green]")


# Add the inference group to the main CLI
inference.add_command(templates)
inference.add_command(history)