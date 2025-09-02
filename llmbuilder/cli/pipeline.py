"""
Pipeline execution commands for chaining multiple operations.
"""

import click
from pathlib import Path
from typing import Dict, Any, List

from llmbuilder.utils.workflow import get_workflow_manager, PipelineBuilder
from llmbuilder.utils.logging import get_logger
from llmbuilder.utils.config import ConfigManager

logger = get_logger(__name__)


@click.group()
def pipeline():
    """Execute predefined pipelines and workflows."""
    pass


@pipeline.command()
@click.argument('name')
@click.option('--data-path', '-d', required=True, help='Path to training data')
@click.option('--model', '-m', required=True, help='Base model name or path')
@click.option('--output-dir', '-o', required=True, help='Output directory')
@click.option('--split-ratios', default='0.8,0.1,0.1', help='Train/val/test split ratios')
@click.option('--epochs', default=3, help='Number of training epochs')
@click.option('--batch-size', default=4, help='Training batch size')
@click.option('--learning-rate', default=2e-5, help='Learning rate')
@click.option('--dry-run', is_flag=True, help='Show pipeline steps without executing')
@click.pass_context
def train(ctx, name: str, data_path: str, model: str, output_dir: str, 
          split_ratios: str, epochs: int, batch_size: int, learning_rate: float,
          dry_run: bool):
    """Execute a complete training pipeline."""
    
    # Parse split ratios
    ratios = [float(r.strip()) for r in split_ratios.split(',')]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 0.01:
        raise click.BadParameter("Split ratios must be three numbers that sum to 1.0")
    
    # Build pipeline steps
    steps = PipelineBuilder.full_training_pipeline(
        data_path=data_path,
        model_name=model,
        output_dir=output_dir,
        split_ratios=ratios,
        train_args={
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    )
    
    if dry_run:
        click.echo(f"Pipeline '{name}' would execute the following steps:")
        for i, step in enumerate(steps, 1):
            click.echo(f"  {i}. {step['command']}")
            for key, value in step['args'].items():
                click.echo(f"     --{key.replace('_', '-')} {value}")
        return
    
    # Create and execute workflow
    workflow_manager = get_workflow_manager()
    workflow_id = workflow_manager.create_workflow(name, steps)
    
    click.echo(f"Created training pipeline '{name}' (ID: {workflow_id})")
    click.echo("Use 'llmbuilder pipeline run' to execute the workflow")


@pipeline.command()
@click.argument('name')
@click.option('--model-path', '-m', required=True, help='Path to trained model')
@click.option('--type', 'deployment_type', default='api', 
              type=click.Choice(['api', 'mobile']), help='Deployment type')
@click.option('--optimize/--no-optimize', default=True, help='Optimize model before deployment')
@click.option('--quantization-format', default='gguf', 
              type=click.Choice(['gguf', 'int8', 'q4', 'q5']), help='Quantization format')
@click.option('--port', default=8000, help='API server port (for API deployment)')
@click.option('--dry-run', is_flag=True, help='Show pipeline steps without executing')
@click.pass_context
def deploy(ctx, name: str, model_path: str, deployment_type: str, optimize: bool,
           quantization_format: str, port: int, dry_run: bool):
    """Execute a deployment pipeline."""
    
    # Build pipeline steps
    steps = PipelineBuilder.deployment_pipeline(
        model_path=model_path,
        deployment_type=deployment_type,
        optimize=optimize,
        quantization_format=quantization_format,
        deploy_args={'port': port} if deployment_type == 'api' else {}
    )
    
    if dry_run:
        click.echo(f"Pipeline '{name}' would execute the following steps:")
        for i, step in enumerate(steps, 1):
            click.echo(f"  {i}. {step['command']}")
            for key, value in step['args'].items():
                click.echo(f"     --{key.replace('_', '-')} {value}")
        return
    
    # Create and execute workflow
    workflow_manager = get_workflow_manager()
    workflow_id = workflow_manager.create_workflow(name, steps)
    
    click.echo(f"Created deployment pipeline '{name}' (ID: {workflow_id})")
    click.echo("Use 'llmbuilder pipeline run' to execute the workflow")


@pipeline.command()
@click.argument('workflow_id')
@click.option('--step', type=int, help='Start from specific step (1-based)')
@click.option('--continue-on-error', is_flag=True, help='Continue execution on step failure')
@click.pass_context
def run(ctx, workflow_id: str, step: int, continue_on_error: bool):
    """Execute a workflow pipeline."""
    
    workflow_manager = get_workflow_manager()
    
    try:
        workflow_data = workflow_manager.load_workflow(workflow_id)
    except FileNotFoundError:
        click.echo(f"Error: Workflow '{workflow_id}' not found", err=True)
        return
    
    steps = workflow_data["steps"]
    start_step = (step - 1) if step else 0
    
    click.echo(f"Executing workflow: {workflow_data['name']}")
    click.echo(f"Steps: {len(steps)}, Starting from step: {start_step + 1}")
    
    for i in range(start_step, len(steps)):
        step_data = steps[i]
        command = step_data["command"]
        args = step_data["args"]
        
        click.echo(f"\n[{i+1}/{len(steps)}] Executing: {command}")
        
        # Update step status
        workflow_manager.update_step_status(workflow_id, i, "running")
        
        try:
            # Execute the command
            success = _execute_command(command, args, ctx)
            
            if success:
                workflow_manager.update_step_status(workflow_id, i, "completed")
                click.echo(f"✓ Step {i+1} completed successfully")
            else:
                workflow_manager.update_step_status(workflow_id, i, "failed", 
                                                  error="Command execution failed")
                click.echo(f"✗ Step {i+1} failed", err=True)
                
                if not continue_on_error:
                    click.echo("Pipeline execution stopped due to failure", err=True)
                    return
                    
        except Exception as e:
            workflow_manager.update_step_status(workflow_id, i, "failed", error=str(e))
            click.echo(f"✗ Step {i+1} failed with error: {e}", err=True)
            
            if not continue_on_error:
                click.echo("Pipeline execution stopped due to error", err=True)
                return
    
    click.echo(f"\n✓ Workflow '{workflow_data['name']}' execution completed")


@pipeline.command()
@click.option('--status', type=click.Choice(['all', 'running', 'completed', 'failed', 'pending']),
              default='all', help='Filter by status')
def list(status: str):
    """List all workflows."""
    
    workflow_manager = get_workflow_manager()
    workflows = workflow_manager.list_workflows()
    
    if status != 'all':
        workflows = [w for w in workflows if w['status'] == status]
    
    if not workflows:
        click.echo("No workflows found")
        return
    
    click.echo(f"{'ID':<20} {'Name':<20} {'Status':<10} {'Progress':<10} {'Created'}")
    click.echo("-" * 80)
    
    for workflow in workflows:
        click.echo(f"{workflow['id']:<20} {workflow['name']:<20} "
                  f"{workflow['status']:<10} {workflow['progress']:>6.1f}% "
                  f"{workflow['created_at']}")


@pipeline.command()
@click.argument('workflow_id')
def status(workflow_id: str):
    """Show detailed status of a workflow."""
    
    workflow_manager = get_workflow_manager()
    
    try:
        workflow_data = workflow_manager.load_workflow(workflow_id)
    except FileNotFoundError:
        click.echo(f"Error: Workflow '{workflow_id}' not found", err=True)
        return
    
    click.echo(f"Workflow: {workflow_data['name']}")
    click.echo(f"ID: {workflow_data['id']}")
    click.echo(f"Created: {workflow_data['created_at']}")
    click.echo(f"Steps: {len(workflow_data['steps'])}")
    click.echo()
    
    for i, step in enumerate(workflow_data['steps'], 1):
        status_icon = {
            'pending': '⏳',
            'running': '🔄',
            'completed': '✅',
            'failed': '❌'
        }.get(step['status'], '❓')
        
        click.echo(f"{status_icon} Step {i}: {step['command']} [{step['status']}]")
        
        if step.get('start_time'):
            click.echo(f"   Started: {step['start_time']}")
        if step.get('end_time'):
            click.echo(f"   Ended: {step['end_time']}")
        if step.get('error'):
            click.echo(f"   Error: {step['error']}")


@pipeline.command()
@click.argument('workflow_id')
@click.confirmation_option(prompt='Are you sure you want to delete this workflow?')
def delete(workflow_id: str):
    """Delete a workflow."""
    
    workflow_manager = get_workflow_manager()
    workflow_file = workflow_manager.workflow_dir / f"{workflow_id}.json"
    
    if not workflow_file.exists():
        click.echo(f"Error: Workflow '{workflow_id}' not found", err=True)
        return
    
    workflow_file.unlink()
    click.echo(f"Workflow '{workflow_id}' deleted successfully")


def _execute_command(command: str, args: Dict[str, Any], ctx: click.Context) -> bool:
    """Execute a CLI command with the given arguments."""
    
    # Import the main CLI to access subcommands
    from llmbuilder.cli.main import cli
    
    # Build command arguments
    cmd_args = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd_args.append(f"--{key.replace('_', '-')}")
        else:
            cmd_args.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Parse command parts
    command_parts = command.split()
    
    try:
        # Create a new context for the subcommand
        with cli.make_context('llmbuilder', command_parts + cmd_args, 
                             parent=ctx, resilient_parsing=True) as sub_ctx:
            # Execute the command
            cli.invoke(sub_ctx)
            return True
            
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return False


if __name__ == "__main__":
    pipeline()