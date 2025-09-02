"""
Interactive CLI utilities for enhanced user experience.

Provides interactive prompts, menus, and user guidance.
"""

import click
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path

from .colors import ColorFormatter, Color, print_header, print_info, print_success, print_error


def interactive_menu(
    title: str,
    options: List[Dict[str, Any]],
    allow_back: bool = True,
    allow_quit: bool = True
) -> Optional[Any]:
    """
    Display an interactive menu and return the selected option.
    
    Args:
        title: Menu title
        options: List of option dictionaries with 'name', 'description', and 'value' keys
        allow_back: Whether to show a "Back" option
        allow_quit: Whether to show a "Quit" option
    
    Returns:
        The value of the selected option, or None if back/quit was selected
    """
    while True:
        print_header(title)
        
        # Display options
        for i, option in enumerate(options, 1):
            name = ColorFormatter.format(f"{i}. {option['name']}", Color.BLUE, bold=True)
            description = ColorFormatter.format(f"   {option.get('description', '')}", Color.BLUE_LIGHT)
            click.echo(name)
            if option.get('description'):
                click.echo(description)
        
        # Add navigation options
        nav_options = []
        if allow_back:
            nav_options.append(f"{len(options) + 1}. Back")
        if allow_quit:
            nav_options.append(f"{len(options) + len(nav_options) + 1}. Quit")
        
        if nav_options:
            click.echo()
            for nav_option in nav_options:
                click.echo(ColorFormatter.format(nav_option, Color.YELLOW))
        
        # Get user choice
        max_choice = len(options) + len(nav_options)
        try:
            choice = click.prompt(
                ColorFormatter.format("\nSelect an option", Color.WHITE),
                type=click.IntRange(1, max_choice)
            )
            
            if choice <= len(options):
                return options[choice - 1]['value']
            elif allow_back and choice == len(options) + 1:
                return None
            elif allow_quit and choice == max_choice:
                print_info("Goodbye!")
                raise click.Abort()
                
        except click.Abort:
            raise
        except Exception:
            print_error("Invalid selection. Please try again.")
            continue


def interactive_config_setup() -> Dict[str, Any]:
    """Interactive configuration setup wizard."""
    print_header("LLMBuilder Configuration Setup")
    print_info("This wizard will help you set up your LLMBuilder configuration.")
    
    config = {}
    
    # Model configuration
    print_header("Model Configuration")
    
    model_types = [
        {"name": "GPT-style (Decoder-only)", "description": "Best for text generation", "value": "gpt"},
        {"name": "BERT-style (Encoder-only)", "description": "Best for classification", "value": "bert"},
        {"name": "T5-style (Encoder-decoder)", "description": "Best for translation/summarization", "value": "t5"}
    ]
    
    model_type = interactive_menu("Select model architecture:", model_types)
    if model_type:
        config['model_type'] = model_type
    
    # Training configuration
    print_header("Training Configuration")
    
    batch_size = click.prompt(
        ColorFormatter.format("Batch size", Color.WHITE),
        type=int,
        default=8
    )
    config['batch_size'] = batch_size
    
    learning_rate = click.prompt(
        ColorFormatter.format("Learning rate", Color.WHITE),
        type=float,
        default=5e-5
    )
    config['learning_rate'] = learning_rate
    
    epochs = click.prompt(
        ColorFormatter.format("Number of epochs", Color.WHITE),
        type=int,
        default=3
    )
    config['epochs'] = epochs
    
    # Data configuration
    print_header("Data Configuration")
    
    data_dir = click.prompt(
        ColorFormatter.format("Training data directory", Color.WHITE),
        type=click.Path(exists=True, path_type=Path),
        default=Path("./data")
    )
    config['data_dir'] = str(data_dir)
    
    return config


def guided_project_setup(project_name: str) -> Dict[str, Any]:
    """Guided project setup with interactive prompts."""
    print_header(f"Setting up project: {project_name}")
    
    # Project type selection
    project_types = [
        {
            "name": "Research Project",
            "description": "For experimentation and research",
            "value": "research"
        },
        {
            "name": "Production Model",
            "description": "For production deployment",
            "value": "production"
        },
        {
            "name": "Fine-tuning Project",
            "description": "For fine-tuning existing models",
            "value": "fine-tuning"
        },
        {
            "name": "Custom Project",
            "description": "Custom configuration",
            "value": "custom"
        }
    ]
    
    project_type = interactive_menu("What type of project are you creating?", project_types)
    
    setup_config = {
        "project_name": project_name,
        "project_type": project_type
    }
    
    # Get additional details based on project type
    if project_type == "fine-tuning":
        print_header("Fine-tuning Configuration")
        
        base_models = [
            {"name": "GPT-2 Small", "description": "124M parameters", "value": "gpt2"},
            {"name": "GPT-2 Medium", "description": "355M parameters", "value": "gpt2-medium"},
            {"name": "DistilBERT", "description": "66M parameters", "value": "distilbert-base-uncased"},
            {"name": "Custom Model", "description": "Specify your own", "value": "custom"}
        ]
        
        base_model = interactive_menu("Select base model:", base_models)
        setup_config["base_model"] = base_model
        
        if base_model == "custom":
            custom_model = click.prompt(
                ColorFormatter.format("Enter model name or path", Color.WHITE),
                type=str
            )
            setup_config["custom_model"] = custom_model
    
    elif project_type == "production":
        print_header("Production Configuration")
        
        deployment_targets = [
            {"name": "API Server", "description": "REST API deployment", "value": "api"},
            {"name": "Mobile", "description": "Mobile app integration", "value": "mobile"},
            {"name": "Edge Device", "description": "Edge computing deployment", "value": "edge"},
            {"name": "Cloud", "description": "Cloud platform deployment", "value": "cloud"}
        ]
        
        deployment_target = interactive_menu("Select deployment target:", deployment_targets)
        setup_config["deployment_target"] = deployment_target
    
    # Common configuration
    print_header("Additional Configuration")
    
    use_gpu = click.confirm(
        ColorFormatter.format("Do you have GPU support available?", Color.WHITE),
        default=True
    )
    setup_config["use_gpu"] = use_gpu
    
    if use_gpu:
        gpu_memory = click.prompt(
            ColorFormatter.format("GPU memory (GB)", Color.WHITE),
            type=int,
            default=8
        )
        setup_config["gpu_memory"] = gpu_memory
    
    enable_monitoring = click.confirm(
        ColorFormatter.format("Enable monitoring and logging?", Color.WHITE),
        default=True
    )
    setup_config["enable_monitoring"] = enable_monitoring
    
    return setup_config


def interactive_data_preparation() -> Dict[str, Any]:
    """Interactive data preparation wizard."""
    print_header("Data Preparation Wizard")
    
    config = {}
    
    # Data source
    data_sources = [
        {"name": "Local Files", "description": "Files on your computer", "value": "local"},
        {"name": "URL/Web", "description": "Download from web", "value": "web"},
        {"name": "Database", "description": "Connect to database", "value": "database"},
        {"name": "API", "description": "Fetch from API", "value": "api"}
    ]
    
    data_source = interactive_menu("Where is your training data?", data_sources)
    config["data_source"] = data_source
    
    if data_source == "local":
        data_dir = click.prompt(
            ColorFormatter.format("Data directory path", Color.WHITE),
            type=click.Path(exists=True, path_type=Path)
        )
        config["data_dir"] = str(data_dir)
        
        # File formats
        formats = ["pdf", "txt", "docx", "json", "csv", "html"]
        selected_formats = []
        
        print_info("Select file formats to process (press Enter when done):")
        for fmt in formats:
            if click.confirm(f"Process .{fmt} files?", default=True):
                selected_formats.append(fmt)
        
        config["formats"] = selected_formats
    
    # Processing options
    print_header("Processing Options")
    
    config["deduplicate"] = click.confirm("Enable deduplication?", default=True)
    config["clean_text"] = click.confirm("Enable text cleaning?", default=True)
    config["split_data"] = click.confirm("Split into train/validation/test?", default=True)
    
    if config["split_data"]:
        train_ratio = click.prompt("Training set ratio", type=float, default=0.8)
        val_ratio = click.prompt("Validation set ratio", type=float, default=0.1)
        test_ratio = 1.0 - train_ratio - val_ratio
        
        config["split_ratios"] = [train_ratio, val_ratio, test_ratio]
    
    return config


def show_command_help(command_name: str, commands: Dict[str, Any]):
    """Show detailed help for a specific command."""
    if command_name not in commands:
        print_error(f"Command '{command_name}' not found")
        return
    
    command_info = commands[command_name]
    
    print_header(f"Help: {command_name}")
    print_info(command_info.get("description", "No description available"))
    
    if "usage" in command_info:
        click.echo(f"\n{ColorFormatter.format('Usage:', Color.BLUE, bold=True)}")
        click.echo(f"  {command_info['usage']}")
    
    if "examples" in command_info:
        click.echo(f"\n{ColorFormatter.format('Examples:', Color.BLUE, bold=True)}")
        for example in command_info["examples"]:
            click.echo(f"  {ColorFormatter.format(example, Color.GREEN)}")
    
    if "options" in command_info:
        click.echo(f"\n{ColorFormatter.format('Options:', Color.BLUE, bold=True)}")
        for option, desc in command_info["options"].items():
            click.echo(f"  {ColorFormatter.format(option, Color.YELLOW)}  {desc}")


def confirm_destructive_action(action: str, target: str) -> bool:
    """Confirm a potentially destructive action with extra safety."""
    print_error(f"WARNING: This will {action} {target}")
    print_info("This action cannot be undone.")
    
    if not click.confirm("Are you sure you want to continue?"):
        return False
    
    # Double confirmation for very destructive actions
    if "delete" in action.lower() or "remove" in action.lower():
        confirmation = click.prompt(
            f"Type '{target}' to confirm",
            type=str
        )
        if confirmation != target:
            print_error("Confirmation failed. Action cancelled.")
            return False
    
    return True


def progress_callback(current: int, total: int, message: str = "Processing"):
    """Callback function for progress updates."""
    percentage = (current / total) * 100 if total > 0 else 0
    bar_width = 30
    filled = int(percentage / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    status_text = f"\r{ColorFormatter.format(message, Color.BLUE)}: [{bar}] {percentage:.1f}%"
    click.echo(status_text, nl=False)
    
    if current >= total:
        click.echo()  # New line when complete