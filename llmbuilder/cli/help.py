"""
Enhanced help and documentation system for LLMBuilder CLI.
"""

import click
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.columns import Columns
from rich.text import Text

console = Console()


class HelpSystem:
    """Enhanced help system with examples and interactive features."""
    
    def __init__(self):
        self.examples = self._load_examples()
        self.workflows = self._load_workflows()
        self.troubleshooting = self._load_troubleshooting()
    
    def show_command_help(self, command_name: str, context: Optional[click.Context] = None) -> None:
        """Show enhanced help for a specific command."""
        if command_name in self.examples:
            self._show_command_examples(command_name)
        
        if command_name in self.workflows:
            self._show_related_workflows(command_name)
        
        if command_name in self.troubleshooting:
            self._show_troubleshooting_tips(command_name)
    
    def show_interactive_help(self) -> None:
        """Show interactive help menu."""
        console.print("\n[bold cyan]LLMBuilder Interactive Help[/bold cyan]")
        console.print("Choose a topic to learn more:")
        
        options = [
            ("1", "Getting Started", "Basic setup and first steps"),
            ("2", "Data Preparation", "Processing and preparing training data"),
            ("3", "Model Training", "Training and fine-tuning models"),
            ("4", "Model Evaluation", "Testing and benchmarking models"),
            ("5", "Model Deployment", "Serving and deploying models"),
            ("6", "Tool Integration", "Custom tools and extensions"),
            ("7", "Configuration", "Managing settings and configurations"),
            ("8", "Troubleshooting", "Common issues and solutions"),
            ("9", "Workflows", "Complete end-to-end workflows"),
            ("0", "Exit", "Return to command line")
        ]
        
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="bold cyan", width=8)
        table.add_column("Topic", style="bold", width=20)
        table.add_column("Description", style="dim")
        
        for option, topic, description in options:
            table.add_row(option, topic, description)
        
        console.print(table)
        
        while True:
            choice = console.input("\n[bold]Select an option (0-9): [/bold]")
            
            if choice == "0":
                console.print("Goodbye!")
                break
            elif choice == "1":
                self._show_getting_started()
            elif choice == "2":
                self._show_data_preparation_help()
            elif choice == "3":
                self._show_training_help()
            elif choice == "4":
                self._show_evaluation_help()
            elif choice == "5":
                self._show_deployment_help()
            elif choice == "6":
                self._show_tools_help()
            elif choice == "7":
                self._show_configuration_help()
            elif choice == "8":
                self._show_troubleshooting_help()
            elif choice == "9":
                self._show_workflows_help()
            else:
                console.print("[red]Invalid option. Please choose 0-9.[/red]")
    
    def show_command_discovery(self) -> None:
        """Show command discovery interface."""
        console.print("\n[bold cyan]Command Discovery[/bold cyan]")
        console.print("Find the right command for your task:")
        
        categories = {
            "Project Setup": ["init", "config"],
            "Data Management": ["data prepare", "data split", "data validate"],
            "Model Management": ["model select", "model list", "model info"],
            "Training": ["train start", "train resume", "train status"],
            "Evaluation": ["eval run", "eval compare", "eval benchmark"],
            "Optimization": ["quantize", "optimize", "export"],
            "Inference": ["inference", "serve"],
            "Deployment": ["deploy", "package", "serve"],
            "Monitoring": ["monitor", "debug", "logs"],
            "Tools": ["tools register", "tools list", "tools search"],
            "Help": ["help", "docs", "examples"]
        }
        
        for category, commands in categories.items():
            console.print(f"\n[bold green]{category}[/bold green]")
            for cmd in commands:
                console.print(f"  • [cyan]llmbuilder {cmd}[/cyan]")
        
        console.print(f"\n[dim]Use 'llmbuilder COMMAND --help' for detailed help on any command.[/dim]")
    
    def show_usage_examples(self, category: Optional[str] = None) -> None:
        """Show usage examples by category."""
        if category:
            if category in self.examples:
                self._show_category_examples(category)
            else:
                console.print(f"[red]No examples found for category: {category}[/red]")
        else:
            console.print("\n[bold cyan]Usage Examples[/bold cyan]")
            for cat in self.examples.keys():
                console.print(f"\n[bold green]{cat.title()}[/bold green]")
                examples = self.examples[cat][:3]  # Show first 3 examples
                for example in examples:
                    console.print(f"  [cyan]{example['command']}[/cyan]")
                    console.print(f"    {example['description']}")
    
    def _show_command_examples(self, command_name: str) -> None:
        """Show examples for a specific command."""
        examples = self.examples.get(command_name, [])
        if not examples:
            return
        
        console.print(f"\n[bold cyan]Examples for '{command_name}'[/bold cyan]")
        
        for i, example in enumerate(examples, 1):
            console.print(f"\n[bold]Example {i}:[/bold] {example['description']}")
            console.print(f"[green]$ {example['command']}[/green]")
            
            if 'output' in example:
                console.print("[dim]Expected output:[/dim]")
                console.print(example['output'])
            
            if 'notes' in example:
                console.print(f"[dim]Note: {example['notes']}[/dim]")
    
    def _show_related_workflows(self, command_name: str) -> None:
        """Show workflows that include this command."""
        related = []
        for workflow_name, workflow in self.workflows.items():
            if any(command_name in step.get('command', '') for step in workflow.get('steps', [])):
                related.append(workflow_name)
        
        if related:
            console.print(f"\n[bold cyan]Related Workflows[/bold cyan]")
            for workflow_name in related:
                console.print(f"  • {workflow_name}")
            console.print(f"[dim]Use 'llmbuilder help workflows' to see complete workflows.[/dim]")
    
    def _show_troubleshooting_tips(self, command_name: str) -> None:
        """Show troubleshooting tips for a command."""
        tips = self.troubleshooting.get(command_name, [])
        if not tips:
            return
        
        console.print(f"\n[bold cyan]Troubleshooting Tips[/bold cyan]")
        for tip in tips:
            console.print(f"[yellow]Issue:[/yellow] {tip['issue']}")
            console.print(f"[green]Solution:[/green] {tip['solution']}")
            if 'command' in tip:
                console.print(f"[cyan]Try:[/cyan] {tip['command']}")
            console.print()
    
    def _show_getting_started(self) -> None:
        """Show getting started guide."""
        guide = """
# Getting Started with LLMBuilder

## 1. Create a New Project
```bash
llmbuilder init my-llm-project
cd my-llm-project
```

## 2. Prepare Your Data
```bash
# Add your text files to data/raw/
llmbuilder data prepare --input data/raw --output data/processed
```

## 3. Select a Base Model
```bash
llmbuilder model select --name microsoft/DialoGPT-small
```

## 4. Start Training
```bash
llmbuilder train start --config config.json
```

## 5. Test Your Model
```bash
llmbuilder inference --model exports/checkpoints/latest
```

That's it! You've trained your first model with LLMBuilder.
"""
        
        markdown = Markdown(guide)
        console.print(Panel(markdown, title="Getting Started", border_style="green"))
    
    def _show_data_preparation_help(self) -> None:
        """Show data preparation help."""
        help_text = """
# Data Preparation

## Supported Formats
- Text files (.txt, .md)
- PDFs (.pdf)
- Word documents (.docx)
- Web pages (.html)
- E-books (.epub)
- JSON and CSV files

## Common Commands
```bash
# Basic data preparation
llmbuilder data prepare --input data/raw

# With custom settings
llmbuilder data prepare --input data/raw --max-length 1000 --min-length 50

# Split data for training
llmbuilder data split --input data/processed --train-ratio 0.8

# Validate data quality
llmbuilder data validate --input data/processed
```

## Tips
- Keep raw data in data/raw/ directory
- Use --preview flag to see what will be processed
- Check data statistics after preparation
"""
        
        markdown = Markdown(help_text)
        console.print(Panel(markdown, title="Data Preparation", border_style="blue"))
    
    def _show_training_help(self) -> None:
        """Show training help."""
        help_text = """
# Model Training

## Training Methods
- **Full Fine-tuning**: Complete model retraining
- **LoRA**: Low-Rank Adaptation (memory efficient)
- **QLoRA**: Quantized LoRA (even more efficient)

## Common Commands
```bash
# Start training with default settings
llmbuilder train start

# Resume interrupted training
llmbuilder train resume

# Monitor training progress
llmbuilder monitor

# Custom training configuration
llmbuilder train start --method lora --batch-size 4 --learning-rate 0.0001
```

## Configuration
Edit config.json to customize:
- Learning rate and batch size
- Number of epochs
- Model architecture
- Training method (full/lora/qlora)
"""
        
        markdown = Markdown(help_text)
        console.print(Panel(markdown, title="Model Training", border_style="yellow"))
    
    def _show_evaluation_help(self) -> None:
        """Show evaluation help."""
        help_text = """
# Model Evaluation

## Evaluation Metrics
- Perplexity
- BLEU score
- Accuracy
- Custom metrics

## Common Commands
```bash
# Run standard evaluation
llmbuilder eval run --model exports/checkpoints/latest

# Compare multiple models
llmbuilder eval compare --models model1,model2,model3

# Benchmark against standard datasets
llmbuilder eval benchmark --dataset wikitext-2

# Custom evaluation dataset
llmbuilder eval run --dataset data/test.json --metrics perplexity,bleu
```

## Custom Datasets
Prepare evaluation data in JSON format:
```json
[
  {"input": "Question or prompt", "expected": "Expected output"},
  {"input": "Another prompt", "expected": "Another expected output"}
]
```
"""
        
        markdown = Markdown(help_text)
        console.print(Panel(markdown, title="Model Evaluation", border_style="magenta"))
    
    def _show_deployment_help(self) -> None:
        """Show deployment help."""
        help_text = """
# Model Deployment

## Deployment Options
- **Local Server**: FastAPI endpoint
- **Mobile Export**: Android/iOS compatible
- **Cloud Package**: Containerized deployment

## Common Commands
```bash
# Start local server
llmbuilder serve --model exports/checkpoints/latest --port 8000

# Create deployment package
llmbuilder package --model exports/checkpoints/latest --output deploy/

# Export for mobile
llmbuilder deploy mobile --model exports/checkpoints/latest

# Quantize for production
llmbuilder quantize --model exports/checkpoints/latest --format gguf
```

## Server Management
```bash
# Check server status
llmbuilder serve status

# Stop server
llmbuilder serve stop

# View server logs
llmbuilder logs server
```
"""
        
        markdown = Markdown(help_text)
        console.print(Panel(markdown, title="Model Deployment", border_style="red"))
    
    def _show_tools_help(self) -> None:
        """Show tools help."""
        help_text = """
# Tool Integration

## Tool Categories
- **Alarms**: Scheduling and notifications
- **Messaging**: Communication tools
- **Data Processing**: Data manipulation
- **Custom**: Your own tools

## Common Commands
```bash
# Register a custom tool
llmbuilder tools register my_tool.py my_function --category custom

# List available tools
llmbuilder tools list

# Search marketplace
llmbuilder tools search --query "data processing"

# Install from marketplace
llmbuilder tools install csv_processor

# Create tool template
llmbuilder tools template --category messaging
```

## Creating Tools
1. Write a Python function with type hints
2. Add docstring with parameter descriptions
3. Register with LLMBuilder
4. Test and validate
"""
        
        markdown = Markdown(help_text)
        console.print(Panel(markdown, title="Tool Integration", border_style="cyan"))
    
    def _show_configuration_help(self) -> None:
        """Show configuration help."""
        help_text = """
# Configuration Management

## Configuration Hierarchy
1. Command-line arguments (highest priority)
2. Project config file (llmbuilder.json)
3. User config (~/.llmbuilder/config.json)
4. Default settings (lowest priority)

## Common Commands
```bash
# View current configuration
llmbuilder config get

# Set configuration values
llmbuilder config set training.batch_size 8
llmbuilder config set model.architecture gpt

# Reset to defaults
llmbuilder config reset

# Create from template
llmbuilder config template --type production
```

## Configuration Files
- Project: `llmbuilder.json` in project root
- User: `~/.llmbuilder/config.json`
- Templates: Use `llmbuilder config template`
"""
        
        markdown = Markdown(help_text)
        console.print(Panel(markdown, title="Configuration", border_style="green"))
    
    def _show_troubleshooting_help(self) -> None:
        """Show troubleshooting help."""
        help_text = """
# Troubleshooting

## Common Issues

### CUDA Out of Memory
- Reduce batch size: `llmbuilder config set training.batch_size 2`
- Use gradient checkpointing
- Try QLoRA instead of full fine-tuning

### Training Not Starting
- Check data format: `llmbuilder data validate`
- Verify model path: `llmbuilder model list`
- Check configuration: `llmbuilder config get`

### Poor Model Performance
- Increase training epochs
- Adjust learning rate
- Check data quality and quantity
- Try different training methods

### Installation Issues
- Update pip: `pip install --upgrade pip`
- Install with all dependencies: `pip install llmbuilder[all]`
- Check Python version (3.8+ required)

## Debug Commands
```bash
# System diagnostics
llmbuilder debug

# Check dependencies
llmbuilder debug deps

# Validate configuration
llmbuilder debug config

# Check GPU availability
llmbuilder debug gpu
```
"""
        
        markdown = Markdown(help_text)
        console.print(Panel(markdown, title="Troubleshooting", border_style="red"))
    
    def _show_workflows_help(self) -> None:
        """Show complete workflows."""
        console.print("\n[bold cyan]Complete Workflows[/bold cyan]")
        
        for workflow_name, workflow in self.workflows.items():
            console.print(f"\n[bold green]{workflow_name}[/bold green]")
            console.print(workflow['description'])
            
            for i, step in enumerate(workflow['steps'], 1):
                console.print(f"  {i}. [cyan]{step['command']}[/cyan]")
                console.print(f"     {step['description']}")
    
    def _load_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load command examples."""
        return {
            "init": [
                {
                    "command": "llmbuilder init my-project",
                    "description": "Create a new project with default structure",
                    "notes": "Creates directories and config files"
                },
                {
                    "command": "llmbuilder init my-project --template research",
                    "description": "Create project with research template",
                    "notes": "Optimized for experimentation"
                }
            ],
            "data": [
                {
                    "command": "llmbuilder data prepare --input data/raw",
                    "description": "Process all files in data/raw directory",
                    "output": "Processed 150 files, 45MB of text data"
                },
                {
                    "command": "llmbuilder data split --train-ratio 0.8 --val-ratio 0.1",
                    "description": "Split data into 80% train, 10% validation, 10% test",
                }
            ],
            "train": [
                {
                    "command": "llmbuilder train start",
                    "description": "Start training with default configuration",
                },
                {
                    "command": "llmbuilder train start --method lora --batch-size 4",
                    "description": "Start LoRA training with custom batch size",
                }
            ],
            "tools": [
                {
                    "command": "llmbuilder tools register my_tool.py process_data",
                    "description": "Register a custom data processing tool",
                },
                {
                    "command": "llmbuilder tools search --category messaging",
                    "description": "Search for messaging tools in marketplace",
                }
            ]
        }
    
    def _load_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Load complete workflows."""
        return {
            "Basic Training Workflow": {
                "description": "Complete workflow from data to trained model",
                "steps": [
                    {"command": "llmbuilder init my-project", "description": "Create new project"},
                    {"command": "llmbuilder data prepare", "description": "Process training data"},
                    {"command": "llmbuilder model select", "description": "Choose base model"},
                    {"command": "llmbuilder train start", "description": "Start training"},
                    {"command": "llmbuilder eval run", "description": "Evaluate trained model"}
                ]
            },
            "Production Deployment": {
                "description": "Deploy model to production environment",
                "steps": [
                    {"command": "llmbuilder quantize --format gguf", "description": "Optimize model size"},
                    {"command": "llmbuilder package", "description": "Create deployment package"},
                    {"command": "llmbuilder serve --port 8000", "description": "Start production server"},
                    {"command": "llmbuilder monitor", "description": "Monitor performance"}
                ]
            },
            "Custom Tool Development": {
                "description": "Create and deploy custom tools",
                "steps": [
                    {"command": "llmbuilder tools template --category custom", "description": "Create tool template"},
                    {"command": "# Edit the generated template file", "description": "Implement your tool logic"},
                    {"command": "llmbuilder tools register my_tool.py", "description": "Register the tool"},
                    {"command": "llmbuilder tools test my_tool", "description": "Test tool functionality"}
                ]
            }
        }
    
    def _load_troubleshooting(self) -> Dict[str, List[Dict[str, str]]]:
        """Load troubleshooting information."""
        return {
            "train": [
                {
                    "issue": "CUDA out of memory error",
                    "solution": "Reduce batch size or use gradient checkpointing",
                    "command": "llmbuilder config set training.batch_size 2"
                },
                {
                    "issue": "Training loss not decreasing",
                    "solution": "Check learning rate and data quality",
                    "command": "llmbuilder data validate"
                }
            ],
            "data": [
                {
                    "issue": "No files found in input directory",
                    "solution": "Check file paths and supported formats",
                    "command": "llmbuilder data prepare --preview"
                },
                {
                    "issue": "Data processing is very slow",
                    "solution": "Use parallel processing or reduce file size",
                    "command": "llmbuilder data prepare --workers 8"
                }
            ],
            "serve": [
                {
                    "issue": "Server won't start",
                    "solution": "Check if port is already in use",
                    "command": "llmbuilder serve --port 8001"
                },
                {
                    "issue": "Model loading errors",
                    "solution": "Verify model path and format",
                    "command": "llmbuilder model info"
                }
            ]
        }


@click.group()
def help():
    """Enhanced help and documentation system."""
    pass


@help.command()
def interactive():
    """Start interactive help system."""
    help_system = HelpSystem()
    help_system.show_interactive_help()


@help.command()
def discover():
    """Discover commands by category."""
    help_system = HelpSystem()
    help_system.show_command_discovery()


@help.command()
@click.option('--category', '-c', help='Show examples for specific category')
def examples(category):
    """Show usage examples."""
    help_system = HelpSystem()
    help_system.show_usage_examples(category)


@help.command()
def workflows():
    """Show complete end-to-end workflows."""
    help_system = HelpSystem()
    help_system._show_workflows_help()


@help.command()
def troubleshooting():
    """Show troubleshooting guide."""
    help_system = HelpSystem()
    help_system._show_troubleshooting_help()


@help.command()
@click.argument('topic', required=False)
def docs(topic):
    """Show documentation for specific topics."""
    help_system = HelpSystem()
    
    if not topic:
        console.print("[bold cyan]Available Documentation Topics:[/bold cyan]")
        topics = [
            "getting-started", "data-preparation", "training", 
            "evaluation", "deployment", "tools", "configuration"
        ]
        for t in topics:
            console.print(f"  • {t}")
        console.print(f"\n[dim]Use 'llmbuilder help docs TOPIC' for specific documentation.[/dim]")
        return
    
    topic_methods = {
        "getting-started": help_system._show_getting_started,
        "data-preparation": help_system._show_data_preparation_help,
        "training": help_system._show_training_help,
        "evaluation": help_system._show_evaluation_help,
        "deployment": help_system._show_deployment_help,
        "tools": help_system._show_tools_help,
        "configuration": help_system._show_configuration_help,
    }
    
    if topic in topic_methods:
        topic_methods[topic]()
    else:
        console.print(f"[red]Unknown topic: {topic}[/red]")
        console.print("Available topics: " + ", ".join(topic_methods.keys()))


# Add the help group to the main CLI
def register_commands(cli_group):
    """Register help commands with the main CLI."""
    cli_group.add_command(help)