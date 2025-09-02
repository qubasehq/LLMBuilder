"""
CLI commands for tool integration and extension system.
"""

import click
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from ..core.tools.registry import get_registry, ToolMetadata
from ..core.tools.validator import ToolValidator, ValidationLevel
from ..core.tools.marketplace import get_marketplace, MarketplaceTool
from ..utils.config import ConfigManager

console = Console()


@click.group()
def tools():
    """Tool integration and extension system."""
    pass


@tools.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.argument('function_name')
@click.option('--name', '-n', help='Tool name (defaults to function name)')
@click.option('--description', '-d', help='Tool description')
@click.option('--category', '-c', default='custom', 
              type=click.Choice(['alarm', 'messaging', 'data_processing', 'custom']),
              help='Tool category')
@click.option('--version', '-v', default='1.0.0', help='Tool version')
@click.option('--author', '-a', default='unknown', help='Tool author')
@click.option('--validate/--no-validate', default=True, help='Validate tool before registration')
@click.option('--test/--no-test', default=False, help='Test tool before registration')
def register(file_path, function_name, name, description, category, version, author, validate, test):
    """Register a custom tool from a Python file."""
    registry = get_registry()
    validator = ToolValidator()
    
    file_path = Path(file_path)
    
    try:
        # Load and validate the function first
        import importlib.util
        spec = importlib.util.spec_from_file_location("tool_module", file_path)
        if spec is None or spec.loader is None:
            console.print(f"[red]Error: Cannot load module from {file_path}[/red]")
            return
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, function_name):
            console.print(f"[red]Error: Function '{function_name}' not found in {file_path}[/red]")
            return
        
        func = getattr(module, function_name)
        
        # Validate the function
        if validate:
            console.print("[yellow]Validating function...[/yellow]")
            validation_results = validator.validate_function(func)
            
            # Display validation results
            if validation_results:
                table = Table(title="Validation Results")
                table.add_column("Level", style="bold")
                table.add_column("Message")
                table.add_column("Details")
                
                for result in validation_results:
                    level_color = {
                        ValidationLevel.INFO: "blue",
                        ValidationLevel.WARNING: "yellow", 
                        ValidationLevel.ERROR: "red"
                    }[result.level]
                    
                    table.add_row(
                        f"[{level_color}]{result.level.value.upper()}[/{level_color}]",
                        result.message,
                        result.details or ""
                    )
                
                console.print(table)
                
                # Check for errors
                has_errors = any(r.level == ValidationLevel.ERROR for r in validation_results)
                if has_errors:
                    console.print("[red]Cannot register tool due to validation errors.[/red]")
                    return
        
        # Test the function
        if test:
            console.print("[yellow]Testing function...[/yellow]")
            test_result = validator.test_function(func)
            
            if test_result.success:
                console.print(f"[green]✓ Test passed in {test_result.execution_time:.3f}s[/green]")
                if test_result.output is not None:
                    console.print(f"Output: {test_result.output}")
            else:
                console.print(f"[red]✗ Test failed: {test_result.error}[/red]")
                if not click.confirm("Register tool anyway?"):
                    return
        
        # Register the tool
        tool_name = registry.register_from_file(
            file_path=file_path,
            function_name=function_name,
            name=name,
            description=description,
            category=category,
            version=version,
            author=author
        )
        
        console.print(f"[green]✓ Tool '{tool_name}' registered successfully![/green]")
        
        # Display tool schema
        tool_metadata = registry.get_tool(tool_name)
        if tool_metadata:
            schema_json = json.dumps(tool_metadata.schema, indent=2)
            syntax = Syntax(schema_json, "json", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Generated Tool Schema"))
        
    except Exception as e:
        console.print(f"[red]Error registering tool: {e}[/red]")


@tools.command()
@click.option('--category', '-c', help='Filter by category')
@click.option('--enabled/--all', default=True, help='Show only enabled tools')
def list(category, enabled):
    """List registered tools."""
    registry = get_registry()
    tools_list = registry.list_tools(category=category, enabled_only=enabled)
    
    if not tools_list:
        console.print("[yellow]No tools found.[/yellow]")
        return
    
    table = Table(title="Registered Tools")
    table.add_column("Name", style="bold cyan")
    table.add_column("Category", style="green")
    table.add_column("Version", style="blue")
    table.add_column("Author", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Description")
    
    for tool in tools_list:
        status = "[green]Enabled[/green]" if tool.enabled else "[red]Disabled[/red]"
        table.add_row(
            tool.name,
            tool.category,
            tool.version,
            tool.author,
            status,
            tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
        )
    
    console.print(table)


@tools.command()
@click.argument('tool_name')
def info(tool_name):
    """Show detailed information about a tool."""
    registry = get_registry()
    tool = registry.get_tool(tool_name)
    
    if not tool:
        console.print(f"[red]Tool '{tool_name}' not found.[/red]")
        return
    
    # Display tool information
    info_text = f"""
[bold cyan]Name:[/bold cyan] {tool.name}
[bold cyan]Description:[/bold cyan] {tool.description}
[bold cyan]Category:[/bold cyan] {tool.category}
[bold cyan]Version:[/bold cyan] {tool.version}
[bold cyan]Author:[/bold cyan] {tool.author}
[bold cyan]Status:[/bold cyan] {'Enabled' if tool.enabled else 'Disabled'}
[bold cyan]Created:[/bold cyan] {tool.created_at.strftime('%Y-%m-%d %H:%M:%S')}
[bold cyan]Updated:[/bold cyan] {tool.updated_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    if tool.file_path:
        info_text += f"[bold cyan]File Path:[/bold cyan] {tool.file_path}\n"
    if tool.module_name:
        info_text += f"[bold cyan]Module:[/bold cyan] {tool.module_name}\n"
    
    console.print(Panel(info_text.strip(), title=f"Tool Information: {tool_name}"))
    
    # Display function signature
    if tool.function_signature:
        sig_info = tool.function_signature
        sig_text = f"[bold]Function:[/bold] {sig_info['name']}\n"
        
        if sig_info['parameters']:
            sig_text += "[bold]Parameters:[/bold]\n"
            for param_name, param_info in sig_info['parameters'].items():
                annotation = param_info.get('annotation', 'Any')
                default = param_info.get('default')
                default_str = f" = {default}" if default and default != 'None' else ""
                sig_text += f"  • {param_name}: {annotation}{default_str}\n"
        
        if sig_info['return_annotation']:
            sig_text += f"[bold]Returns:[/bold] {sig_info['return_annotation']}\n"
        
        console.print(Panel(sig_text.strip(), title="Function Signature"))
    
    # Display tool schema
    if tool.schema:
        schema_json = json.dumps(tool.schema, indent=2)
        syntax = Syntax(schema_json, "json", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Tool Schema"))


@tools.command()
@click.argument('tool_name')
@click.option('--test-args', help='JSON string of test arguments')
def test(tool_name, test_args):
    """Test a registered tool."""
    registry = get_registry()
    tool = registry.get_tool(tool_name)
    
    if not tool:
        console.print(f"[red]Tool '{tool_name}' not found.[/red]")
        return
    
    if not tool.enabled:
        console.print(f"[yellow]Tool '{tool_name}' is disabled.[/yellow]")
        return
    
    # Load the function
    try:
        if tool.file_path:
            import importlib.util
            spec = importlib.util.spec_from_file_location("tool_module", tool.file_path)
            if spec is None or spec.loader is None:
                console.print(f"[red]Cannot load module from {tool.file_path}[/red]")
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            func_name = tool.function_signature['name']
            if not hasattr(module, func_name):
                console.print(f"[red]Function '{func_name}' not found in module[/red]")
                return
            
            func = getattr(module, func_name)
        else:
            console.print("[red]Tool file path not available[/red]")
            return
        
        # Parse test arguments
        test_kwargs = {}
        if test_args:
            try:
                test_kwargs = json.loads(test_args)
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON in test arguments: {e}[/red]")
                return
        
        # Test the function
        validator = ToolValidator()
        console.print(f"[yellow]Testing tool '{tool_name}'...[/yellow]")
        
        test_result = validator.test_function(func, test_kwargs=test_kwargs)
        
        if test_result.success:
            console.print(f"[green]✓ Test passed in {test_result.execution_time:.3f}s[/green]")
            if test_result.output is not None:
                console.print("[bold]Output:[/bold]")
                if isinstance(test_result.output, (dict, list)):
                    output_json = json.dumps(test_result.output, indent=2)
                    syntax = Syntax(output_json, "json", theme="monokai")
                    console.print(syntax)
                else:
                    console.print(str(test_result.output))
        else:
            console.print(f"[red]✗ Test failed: {test_result.error}[/red]")
    
    except Exception as e:
        console.print(f"[red]Error testing tool: {e}[/red]")


@tools.command()
@click.argument('tool_name')
def unregister(tool_name):
    """Unregister a tool."""
    registry = get_registry()
    
    if not registry.get_tool(tool_name):
        console.print(f"[red]Tool '{tool_name}' not found.[/red]")
        return
    
    if click.confirm(f"Are you sure you want to unregister tool '{tool_name}'?"):
        if registry.unregister_tool(tool_name):
            console.print(f"[green]✓ Tool '{tool_name}' unregistered successfully.[/green]")
        else:
            console.print(f"[red]Failed to unregister tool '{tool_name}'.[/red]")


@tools.command()
@click.argument('tool_name')
def enable(tool_name):
    """Enable a tool."""
    registry = get_registry()
    
    if registry.enable_tool(tool_name):
        console.print(f"[green]✓ Tool '{tool_name}' enabled.[/green]")
    else:
        console.print(f"[red]Tool '{tool_name}' not found.[/red]")


@tools.command()
@click.argument('tool_name')
def disable(tool_name):
    """Disable a tool."""
    registry = get_registry()
    
    if registry.disable_tool(tool_name):
        console.print(f"[yellow]Tool '{tool_name}' disabled.[/yellow]")
    else:
        console.print(f"[red]Tool '{tool_name}' not found.[/red]")


@tools.command()
@click.option('--query', '-q', help='Search query')
@click.option('--category', '-c', help='Filter by category')
@click.option('--tags', '-t', help='Filter by tags (comma-separated)')
@click.option('--limit', '-l', default=20, help='Maximum number of results')
def search(query, category, tags, limit):
    """Search for tools in the marketplace."""
    marketplace = get_marketplace()
    
    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(',')]
    
    try:
        console.print("[yellow]Searching marketplace...[/yellow]")
        tools_list = marketplace.search_tools(
            query=query,
            category=category,
            tags=tag_list,
            limit=limit
        )
        
        if not tools_list:
            console.print("[yellow]No tools found.[/yellow]")
            return
        
        table = Table(title="Marketplace Tools")
        table.add_column("Name", style="bold cyan")
        table.add_column("Category", style="green")
        table.add_column("Version", style="blue")
        table.add_column("Author", style="magenta")
        table.add_column("Rating", style="yellow")
        table.add_column("Downloads", style="red")
        table.add_column("Description")
        
        for tool in tools_list:
            table.add_row(
                tool.name,
                tool.category,
                tool.version,
                tool.author,
                f"{tool.rating:.1f}",
                str(tool.downloads),
                tool.description[:40] + "..." if len(tool.description) > 40 else tool.description
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error searching marketplace: {e}[/red]")


@tools.command()
@click.argument('tool_name')
def install(tool_name):
    """Install a tool from the marketplace."""
    marketplace = get_marketplace()
    registry = get_registry()
    
    try:
        console.print(f"[yellow]Getting tool details for '{tool_name}'...[/yellow]")
        tool = marketplace.get_tool_details(tool_name)
        
        if not tool:
            console.print(f"[red]Tool '{tool_name}' not found in marketplace.[/red]")
            return
        
        # Display tool information
        info_text = f"""
[bold]Name:[/bold] {tool.name}
[bold]Description:[/bold] {tool.description}
[bold]Category:[/bold] {tool.category}
[bold]Version:[/bold] {tool.version}
[bold]Author:[/bold] {tool.author}
[bold]Rating:[/bold] {tool.rating:.1f}
[bold]Downloads:[/bold] {tool.downloads}
"""
        
        if tool.license:
            info_text += f"[bold]License:[/bold] {tool.license}\n"
        if tool.tags:
            info_text += f"[bold]Tags:[/bold] {', '.join(tool.tags)}\n"
        
        console.print(Panel(info_text.strip(), title=f"Tool: {tool_name}"))
        
        if not click.confirm("Install this tool?"):
            return
        
        console.print("[yellow]Downloading tool...[/yellow]")
        tool_file = marketplace.download_tool(tool)
        
        if not tool_file:
            console.print("[red]Failed to download tool.[/red]")
            return
        
        console.print(f"[green]✓ Tool downloaded to {tool_file}[/green]")
        
        # Auto-register if it's a Python file
        if tool_file.suffix == '.py':
            if click.confirm("Auto-register the tool?"):
                # Try to find the main function in the file
                # This is a simple implementation - could be enhanced
                try:
                    import ast
                    with open(tool_file, 'r') as f:
                        tree = ast.parse(f.read())
                    
                    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    
                    if functions:
                        main_func = functions[0]  # Use first function as default
                        if len(functions) > 1:
                            console.print(f"Available functions: {', '.join(functions)}")
                            main_func = click.prompt("Select function to register", default=main_func)
                        
                        # Register the tool
                        registry.register_from_file(
                            file_path=tool_file,
                            function_name=main_func,
                            name=tool.name,
                            description=tool.description,
                            category=tool.category,
                            version=tool.version,
                            author=tool.author
                        )
                        
                        console.print(f"[green]✓ Tool '{tool.name}' registered successfully![/green]")
                    else:
                        console.print("[yellow]No functions found in the tool file.[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]Failed to auto-register tool: {e}[/red]")
                    console.print(f"You can manually register it using: llmbuilder tools register {tool_file} <function_name>")
        
    except Exception as e:
        console.print(f"[red]Error installing tool: {e}[/red]")


@tools.command()
@click.option('--category', '-c', 
              type=click.Choice(['alarm', 'messaging', 'data_processing']),
              help='Template category')
def template(category):
    """Create a tool template file."""
    if not category:
        console.print("Available template categories:")
        console.print("  • alarm - Alarm and notification tools")
        console.print("  • messaging - Messaging and communication tools") 
        console.print("  • data_processing - Data processing and analysis tools")
        category = click.prompt("Select category", type=click.Choice(['alarm', 'messaging', 'data_processing']))
    
    # Get template content
    template_files = {
        'alarm': 'alarm_tool.py',
        'messaging': 'messaging_tool.py',
        'data_processing': 'data_processing_tool.py'
    }
    
    template_file = template_files[category]
    template_path = Path(__file__).parent.parent / 'templates' / 'tools' / template_file
    
    if not template_path.exists():
        console.print(f"[red]Template file not found: {template_path}[/red]")
        return
    
    # Copy template to current directory
    output_file = Path(f"my_{template_file}")
    
    if output_file.exists():
        if not click.confirm(f"File {output_file} already exists. Overwrite?"):
            return
    
    try:
        with open(template_path, 'r') as src:
            content = src.read()
        
        with open(output_file, 'w') as dst:
            dst.write(content)
        
        console.print(f"[green]✓ Template created: {output_file}[/green]")
        console.print(f"Edit the file and then register it using:")
        console.print(f"  llmbuilder tools register {output_file} <function_name>")
        
    except Exception as e:
        console.print(f"[red]Error creating template: {e}[/red]")


# Add the tools group to the main CLI
def register_commands(cli_group):
    """Register tool commands with the main CLI."""
    cli_group.add_command(tools)