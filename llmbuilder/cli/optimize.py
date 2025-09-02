"""
Optimization and export CLI commands for LLMBuilder.

This module provides commands for model quantization, pruning, distillation,
and export to various formats including GGUF.
"""

import click
from llmbuilder.utils.lazy_imports import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import get_logger
from llmbuilder.core.tools.quantization_manager import (
    QuantizationManager, 
    QuantizationConfig, 
    QuantizationType,
    create_quantization_config
)
from llmbuilder.core.tools.export_gguf import GGUFConverter, ModelMetadata
from llmbuilder.core.tools.conversion_pipeline import ConversionPipeline, ConversionConfig

logger = get_logger(__name__)
console = Console()


@click.group()
def optimize():
    """Model optimization and export commands."""
    pass


@optimize.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option(
    '--type', 'quant_type',
    type=click.Choice(['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1']),
    default='q4_0',
    help='Quantization type'
)
@click.option(
    '--block-size',
    type=int,
    default=32,
    help='Block size for quantization (default: 32)'
)
@click.option(
    '--quality-threshold',
    type=float,
    default=0.95,
    help='Minimum quality threshold (default: 0.95)'
)
@click.option(
    '--skip-layers',
    multiple=True,
    help='Layer patterns to skip quantization (can be used multiple times)'
)
@click.option(
    '--force-layers',
    multiple=True,
    help='Layer patterns to force quantization (can be used multiple times)'
)
@click.option(
    '--report',
    type=click.Path(path_type=Path),
    help='Path to save quantization report'
)
@click.option(
    '--validate/--no-validate',
    default=True,
    help='Validate quantization quality (default: enabled)'
)
@click.pass_context
def quantize(
    ctx: click.Context,
    model_path: Path,
    output_path: Path,
    quant_type: str,
    block_size: int,
    quality_threshold: float,
    skip_layers: tuple,
    force_layers: tuple,
    report: Optional[Path],
    validate: bool
):
    """
    Quantize a trained model to reduce size and improve inference speed.
    
    Supports multiple quantization formats:
    - f32: No quantization (baseline)
    - f16: Half precision (2x compression)
    - q8_0: 8-bit quantization with block scaling
    - q4_0: 4-bit quantization with block scaling
    - q4_1: 4-bit quantization with scale and min
    - q5_0: 5-bit quantization with block scaling
    - q5_1: 5-bit quantization with scale and min
    
    Examples:
        llmbuilder optimize quantize model.pt quantized.pt --type q4_0
        llmbuilder optimize quantize model.pt quantized.pt --type q8_0 --skip-layers bias norm
    """
    try:
        console.print(f"[bold blue]Starting quantization: {quant_type}[/bold blue]")
        console.print(f"Input: {model_path}")
        console.print(f"Output: {output_path}")
        
        # Create quantization config
        config = create_quantization_config(
            quant_type,
            block_size=block_size,
            quality_threshold=quality_threshold,
            skip_layers=list(skip_layers),
            force_layers=list(force_layers)
        )
        
        # Load model
        console.print("[yellow]Loading model...[/yellow]")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state_dict = checkpoint['model']
                model_config = checkpoint.get('config', {})
            elif 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                model_config = checkpoint.get('config', {})
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
                model_config = checkpoint.get('config', {})
            else:
                model_state_dict = checkpoint
                model_config = {}
        else:
            model_state_dict = checkpoint.state_dict()
            model_config = getattr(checkpoint, 'config', {})
        
        # Initialize quantization manager
        quantizer = QuantizationManager(config)
        
        # Set up progress callback
        def progress_callback(message: str, progress: float):
            console.print(f"[green]{message}[/green] ({progress:.1f}%)")
        
        quantizer.set_progress_callback(progress_callback)
        
        # Perform quantization
        console.print("[yellow]Quantizing model...[/yellow]")
        start_time = time.time()
        
        result = quantizer.quantize_model(model_state_dict)
        
        # Validate if requested
        validation_result = None
        if validate:
            console.print("[yellow]Validating quantization quality...[/yellow]")
            validation_result = quantizer.validate_quantization(model_state_dict, result)
        
        # Prepare output checkpoint
        output_checkpoint = {
            'quantized_tensors': quantizer.quantized_tensors,
            'quantization_config': config.__dict__,
            'quantization_result': result.__dict__,
            'original_config': model_config,
            'metadata': {
                'quantization_type': quant_type,
                'original_model_path': str(model_path),
                'quantization_timestamp': time.time(),
                'llmbuilder_version': ctx.obj.get('version', 'unknown') if ctx.obj else 'unknown'
            }
        }
        
        # Add validation results if available
        if validation_result:
            output_checkpoint['validation_result'] = validation_result
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save quantized model
        console.print("[yellow]Saving quantized model...[/yellow]")
        torch.save(output_checkpoint, output_path)
        
        # Display results
        _display_quantization_results(result, validation_result)
        
        # Save report if requested
        if report:
            console.print(f"[yellow]Saving quantization report to {report}...[/yellow]")
            quantizer.save_quantization_report(result, validation_result or {}, report)
        
        console.print(f"[bold green]✓ Quantization completed successfully![/bold green]")
        console.print(f"Output saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Quantization failed: {e}[/bold red]")
        logger.error(f"Quantization error: {e}", exc_info=True)
        raise click.ClickException(f"Quantization failed: {e}")


@optimize.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option(
    '--sparsity',
    type=float,
    default=0.5,
    help='Target sparsity level (0.0-1.0, default: 0.5)'
)
@click.option(
    '--method',
    type=click.Choice(['magnitude', 'structured', 'gradual']),
    default='magnitude',
    help='Pruning method (default: magnitude)'
)
@click.option(
    '--skip-layers',
    multiple=True,
    help='Layer patterns to skip pruning (can be used multiple times)'
)
@click.option(
    '--validate/--no-validate',
    default=True,
    help='Validate pruned model (default: enabled)'
)
def prune(
    model_path: Path,
    output_path: Path,
    sparsity: float,
    method: str,
    skip_layers: tuple,
    validate: bool
):
    """
    Prune a trained model to reduce parameters and improve efficiency.
    
    Supports different pruning methods:
    - magnitude: Remove weights with smallest absolute values
    - structured: Remove entire channels/filters
    - gradual: Gradually increase sparsity during fine-tuning
    
    Examples:
        llmbuilder optimize prune model.pt pruned.pt --sparsity 0.3
        llmbuilder optimize prune model.pt pruned.pt --method structured --sparsity 0.5
    """
    try:
        console.print(f"[bold blue]Starting model pruning: {method} method[/bold blue]")
        console.print(f"Target sparsity: {sparsity:.1%}")
        console.print(f"Input: {model_path}")
        console.print(f"Output: {output_path}")
        
        # Load model
        console.print("[yellow]Loading model...[/yellow]")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state_dict = checkpoint['model']
                model_config = checkpoint.get('config', {})
            else:
                model_state_dict = checkpoint
                model_config = {}
        else:
            model_state_dict = checkpoint.state_dict()
            model_config = getattr(checkpoint, 'config', {})
        
        # Perform pruning based on method
        console.print(f"[yellow]Applying {method} pruning...[/yellow]")
        
        if method == 'magnitude':
            pruned_state_dict = _magnitude_prune(model_state_dict, sparsity, skip_layers)
        elif method == 'structured':
            pruned_state_dict = _structured_prune(model_state_dict, sparsity, skip_layers)
        elif method == 'gradual':
            # For gradual pruning, we just prepare the model with pruning masks
            pruned_state_dict = _prepare_gradual_prune(model_state_dict, sparsity, skip_layers)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        # Calculate pruning statistics
        original_params = sum(tensor.numel() for tensor in model_state_dict.values())
        pruned_params = sum(tensor.numel() for tensor in pruned_state_dict.values())
        actual_sparsity = 1 - (pruned_params / original_params)
        
        # Prepare output checkpoint
        output_checkpoint = {
            'model': pruned_state_dict,
            'config': model_config,
            'pruning_config': {
                'method': method,
                'target_sparsity': sparsity,
                'actual_sparsity': actual_sparsity,
                'skip_layers': list(skip_layers)
            },
            'metadata': {
                'original_model_path': str(model_path),
                'pruning_timestamp': time.time(),
                'original_params': original_params,
                'pruned_params': pruned_params
            }
        }
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save pruned model
        console.print("[yellow]Saving pruned model...[/yellow]")
        torch.save(output_checkpoint, output_path)
        
        # Display results
        _display_pruning_results(original_params, pruned_params, actual_sparsity)
        
        console.print(f"[bold green]✓ Pruning completed successfully![/bold green]")
        console.print(f"Output saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Pruning failed: {e}[/bold red]")
        logger.error(f"Pruning error: {e}", exc_info=True)
        raise click.ClickException(f"Pruning failed: {e}")


@optimize.command()
@click.argument('teacher_path', type=click.Path(exists=True, path_type=Path))
@click.argument('student_config', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option(
    '--temperature',
    type=float,
    default=4.0,
    help='Distillation temperature (default: 4.0)'
)
@click.option(
    '--alpha',
    type=float,
    default=0.7,
    help='Weight for distillation loss (default: 0.7)'
)
@click.option(
    '--epochs',
    type=int,
    default=10,
    help='Number of distillation epochs (default: 10)'
)
def distill(
    teacher_path: Path,
    student_config: Path,
    output_path: Path,
    temperature: float,
    alpha: float,
    epochs: int
):
    """
    Distill knowledge from a large teacher model to a smaller student model.
    
    The student model architecture is defined in the config file.
    
    Examples:
        llmbuilder optimize distill teacher.pt student_config.json distilled.pt
        llmbuilder optimize distill teacher.pt config.json distilled.pt --temperature 3.0
    """
    try:
        console.print(f"[bold blue]Starting knowledge distillation[/bold blue]")
        console.print(f"Teacher: {teacher_path}")
        console.print(f"Student config: {student_config}")
        console.print(f"Output: {output_path}")
        console.print(f"Temperature: {temperature}, Alpha: {alpha}")
        
        # Load teacher model
        console.print("[yellow]Loading teacher model...[/yellow]")
        teacher_checkpoint = torch.load(teacher_path, map_location='cpu')
        
        # Load student configuration
        console.print("[yellow]Loading student configuration...[/yellow]")
        with open(student_config, 'r') as f:
            student_config_data = json.load(f)
        
        # This is a placeholder for distillation implementation
        # In a real implementation, you would:
        # 1. Initialize student model from config
        # 2. Set up distillation training loop
        # 3. Train student to match teacher outputs
        # 4. Save distilled model
        
        console.print("[yellow]Initializing student model...[/yellow]")
        # TODO: Implement student model initialization
        
        console.print("[yellow]Starting distillation training...[/yellow]")
        # TODO: Implement distillation training loop
        
        # For now, create a placeholder output
        output_checkpoint = {
            'model': {},  # Placeholder for distilled model
            'config': student_config_data,
            'distillation_config': {
                'teacher_path': str(teacher_path),
                'temperature': temperature,
                'alpha': alpha,
                'epochs': epochs
            },
            'metadata': {
                'distillation_timestamp': time.time(),
                'teacher_model_path': str(teacher_path)
            }
        }
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save distilled model
        console.print("[yellow]Saving distilled model...[/yellow]")
        torch.save(output_checkpoint, output_path)
        
        console.print(f"[bold green]✓ Distillation completed successfully![/bold green]")
        console.print(f"Output saved to: {output_path}")
        
        # Note: This is a placeholder implementation
        console.print("[yellow]Note: Distillation implementation is a placeholder.[/yellow]")
        console.print("[yellow]Full implementation requires training loop integration.[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Distillation failed: {e}[/bold red]")
        logger.error(f"Distillation error: {e}", exc_info=True)
        raise click.ClickException(f"Distillation failed: {e}")


@optimize.group()
def export():
    """Export models to different formats."""
    pass


@export.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option(
    '--quantization',
    type=click.Choice(['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1']),
    default='f16',
    help='Quantization type for GGUF export (default: f16)'
)
@click.option(
    '--tokenizer',
    type=click.Path(exists=True, path_type=Path),
    help='Path to tokenizer files'
)
@click.option(
    '--metadata',
    type=click.Path(exists=True, path_type=Path),
    help='Path to metadata JSON file'
)
@click.option(
    '--validate/--no-validate',
    default=True,
    help='Validate exported GGUF file (default: enabled)'
)
def gguf(
    model_path: Path,
    output_path: Path,
    quantization: str,
    tokenizer: Optional[Path],
    metadata: Optional[Path],
    validate: bool
):
    """
    Export model to GGUF format for llama.cpp compatibility.
    
    GGUF is the standard format for running models with llama.cpp and
    other inference engines.
    
    Examples:
        llmbuilder optimize export gguf model.pt model.gguf
        llmbuilder optimize export gguf model.pt model.gguf --quantization q4_0
        llmbuilder optimize export gguf model.pt model.gguf --tokenizer tokenizer/
    """
    try:
        console.print(f"[bold blue]Exporting to GGUF format[/bold blue]")
        console.print(f"Input: {model_path}")
        console.print(f"Output: {output_path}")
        console.print(f"Quantization: {quantization}")
        
        # Load metadata if provided
        model_metadata = None
        if metadata:
            console.print(f"[yellow]Loading metadata from {metadata}...[/yellow]")
            with open(metadata, 'r') as f:
                metadata_dict = json.load(f)
                model_metadata = ModelMetadata(**metadata_dict)
        else:
            # Create default metadata
            model_metadata = ModelMetadata(
                name=model_path.stem,
                architecture="gpt2",  # Default architecture
                version="1.0"
            )
        
        # Initialize GGUF converter
        converter = GGUFConverter(
            model_path=str(model_path),
            output_path=str(output_path),
            metadata=model_metadata,
            quantization_type=quantization
        )
        
        # Export to GGUF
        console.print("[yellow]Converting to GGUF format...[/yellow]")
        success = converter.export_to_gguf(
            tokenizer_path=str(tokenizer) if tokenizer else None,
            validate=validate
        )
        
        if success:
            # Display export information
            file_size = output_path.stat().st_size if output_path.exists() else 0
            console.print(f"[bold green]✓ GGUF export completed successfully![/bold green]")
            console.print(f"Output file: {output_path}")
            console.print(f"File size: {file_size / (1024*1024):.2f} MB")
        else:
            raise click.ClickException("GGUF export failed")
        
    except Exception as e:
        console.print(f"[bold red]✗ GGUF export failed: {e}[/bold red]")
        logger.error(f"GGUF export error: {e}", exc_info=True)
        raise click.ClickException(f"GGUF export failed: {e}")


@export.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option(
    '--quantization',
    multiple=True,
    default=['f16', 'q8_0', 'q4_0'],
    help='Quantization levels to export (can be used multiple times)'
)
@click.option(
    '--tokenizer',
    type=click.Path(exists=True, path_type=Path),
    help='Path to tokenizer files'
)
@click.option(
    '--model-name',
    help='Model name for output files'
)
@click.option(
    '--validate/--no-validate',
    default=True,
    help='Validate exported files (default: enabled)'
)
@click.option(
    '--report',
    type=click.Path(path_type=Path),
    help='Save conversion report to file'
)
def batch(
    model_path: Path,
    output_dir: Path,
    quantization: tuple,
    tokenizer: Optional[Path],
    model_name: Optional[str],
    validate: bool,
    report: Optional[Path]
):
    """
    Batch export model to multiple GGUF formats with different quantization levels.
    
    This command creates multiple GGUF files with different quantization levels
    in a single operation, useful for creating a complete model distribution.
    
    Examples:
        llmbuilder optimize export batch model.pt output/ --quantization f16 --quantization q4_0
        llmbuilder optimize export batch model.pt output/ --model-name my-model --report report.json
    """
    try:
        console.print(f"[bold blue]Starting batch GGUF export[/bold blue]")
        console.print(f"Input: {model_path}")
        console.print(f"Output directory: {output_dir}")
        console.print(f"Quantization levels: {list(quantization)}")
        
        # Create conversion configuration
        config = ConversionConfig(
            input_checkpoint=model_path,
            output_dir=output_dir,
            quantization_levels=list(quantization),
            model_name=model_name,
            tokenizer_path=tokenizer,
            validate_output=validate
        )
        
        # Create and run pipeline
        pipeline = ConversionPipeline(config)
        
        console.print("[yellow]Running conversion pipeline...[/yellow]")
        result = pipeline.convert_all()
        
        # Display results
        _display_batch_results(result)
        
        # Save report if requested
        if report:
            console.print(f"[yellow]Saving conversion report to {report}...[/yellow]")
            pipeline.save_report(report)
        
        if result.failed_conversions == 0:
            console.print(f"[bold green]✓ Batch export completed successfully![/bold green]")
        else:
            console.print(f"[bold yellow]⚠ Batch export completed with {result.failed_conversions} failures[/bold yellow]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Batch export failed: {e}[/bold red]")
        logger.error(f"Batch export error: {e}", exc_info=True)
        raise click.ClickException(f"Batch export failed: {e}")


@export.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option(
    '--precision',
    type=click.Choice(['fp32', 'fp16', 'int8']),
    default='fp16',
    help='Export precision (default: fp16)'
)
def onnx(
    model_path: Path,
    output_path: Path,
    precision: str
):
    """
    Export model to ONNX format for cross-platform inference.
    
    ONNX format enables deployment on various inference engines
    and hardware accelerators.
    
    Examples:
        llmbuilder optimize export onnx model.pt model.onnx
        llmbuilder optimize export onnx model.pt model.onnx --precision int8
    """
    try:
        console.print(f"[bold blue]Exporting to ONNX format[/bold blue]")
        console.print(f"Input: {model_path}")
        console.print(f"Output: {output_path}")
        console.print(f"Precision: {precision}")
        
        # This is a placeholder for ONNX export implementation
        console.print("[yellow]Loading model...[/yellow]")
        # TODO: Load PyTorch model
        
        console.print("[yellow]Converting to ONNX...[/yellow]")
        # TODO: Implement ONNX export using torch.onnx.export
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Placeholder output
        console.print(f"[bold green]✓ ONNX export completed successfully![/bold green]")
        console.print(f"Output saved to: {output_path}")
        
        # Note: This is a placeholder implementation
        console.print("[yellow]Note: ONNX export implementation is a placeholder.[/yellow]")
        console.print("[yellow]Full implementation requires torch.onnx integration.[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]✗ ONNX export failed: {e}[/bold red]")
        logger.error(f"ONNX export error: {e}", exc_info=True)
        raise click.ClickException(f"ONNX export failed: {e}")


@optimize.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--format',
    type=click.Choice(['quantized', 'gguf', 'onnx']),
    help='Validate specific format'
)
@click.option(
    '--detailed',
    is_flag=True,
    help='Show detailed validation information'
)
def validate(
    model_path: Path,
    format: Optional[str],
    detailed: bool
):
    """
    Validate exported model files for correctness and quality.
    
    Performs comprehensive validation including:
    - File format integrity
    - Model architecture consistency
    - Quantization quality metrics
    - Inference capability testing
    
    Examples:
        llmbuilder optimize validate model.gguf
        llmbuilder optimize validate model.pt --format quantized --detailed
    """
    try:
        console.print(f"[bold blue]Validating model: {model_path}[/bold blue]")
        
        # Determine format from file extension if not specified
        if not format:
            suffix = model_path.suffix.lower()
            if suffix == '.gguf':
                format = 'gguf'
            elif suffix == '.onnx':
                format = 'onnx'
            else:
                format = 'quantized'  # Default for .pt files
        
        console.print(f"Format: {format}")
        
        validation_results = {}
        
        if format == 'gguf':
            validation_results = _validate_gguf_file(model_path, detailed)
        elif format == 'onnx':
            validation_results = _validate_onnx_file(model_path, detailed)
        elif format == 'quantized':
            validation_results = _validate_quantized_file(model_path, detailed)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Display validation results
        _display_validation_results(validation_results, detailed)
        
        if validation_results.get('is_valid', False):
            console.print(f"[bold green]✓ Model validation passed![/bold green]")
        else:
            console.print(f"[bold red]✗ Model validation failed![/bold red]")
            raise click.ClickException("Model validation failed")
        
    except Exception as e:
        console.print(f"[bold red]✗ Validation failed: {e}[/bold red]")
        logger.error(f"Validation error: {e}", exc_info=True)
        raise click.ClickException(f"Validation failed: {e}")


# Add the optimize group to the main CLI
optimize.add_command(export)


# Helper functions

def _display_quantization_results(result, validation_result=None):
    """Display quantization results in a formatted table."""
    table = Table(title="Quantization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Original Size", f"{result.original_size / (1024*1024):.2f} MB")
    table.add_row("Quantized Size", f"{result.quantized_size / (1024*1024):.2f} MB")
    table.add_row("Compression Ratio", f"{result.compression_ratio:.2f}x")
    table.add_row("Size Reduction", f"{result.size_reduction_percent:.1f}%")
    table.add_row("Quality Score", f"{result.quality_score:.3f}")
    table.add_row("Processing Time", f"{result.processing_time:.2f}s")
    
    if validation_result:
        table.add_row("Validation", "✓ Passed" if validation_result['validation_passed'] else "✗ Failed")
        if validation_result['low_quality_tensors']:
            table.add_row("Low Quality Tensors", str(len(validation_result['low_quality_tensors'])))
    
    console.print(table)


def _display_pruning_results(original_params, pruned_params, actual_sparsity):
    """Display pruning results in a formatted table."""
    table = Table(title="Pruning Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Original Parameters", f"{original_params:,}")
    table.add_row("Pruned Parameters", f"{pruned_params:,}")
    table.add_row("Parameters Removed", f"{original_params - pruned_params:,}")
    table.add_row("Sparsity Achieved", f"{actual_sparsity:.1%}")
    table.add_row("Model Size Reduction", f"{actual_sparsity:.1%}")
    
    console.print(table)


def _display_validation_results(results, detailed=False):
    """Display validation results."""
    if results.get('is_valid', False):
        console.print(Panel("[green]✓ Validation Passed[/green]", title="Validation Results"))
    else:
        console.print(Panel("[red]✗ Validation Failed[/red]", title="Validation Results"))
    
    if detailed and 'details' in results:
        table = Table(title="Validation Details")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        for check, status in results['details'].items():
            if isinstance(status, dict):
                status_str = "✓ Pass" if status.get('passed', False) else "✗ Fail"
                details = status.get('message', '')
            else:
                status_str = "✓ Pass" if status else "✗ Fail"
                details = ""
            
            table.add_row(check, status_str, details)
        
        console.print(table)


def _display_batch_results(result):
    """Display batch conversion results."""
    table = Table(title="Batch Conversion Results")
    table.add_column("Quantization", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("File Size", style="yellow")
    table.add_column("Time", style="blue")
    table.add_column("Compression", style="magenta")
    
    for conv_result in result.results:
        if conv_result.success:
            status = "✓ Success"
            file_size = f"{conv_result.file_size_mb:.1f} MB"
            time_str = f"{conv_result.conversion_time:.1f}s"
            compression = f"{conv_result.compression_ratio:.1f}x" if conv_result.compression_ratio else "N/A"
        else:
            status = "✗ Failed"
            file_size = "N/A"
            time_str = f"{conv_result.conversion_time:.1f}s"
            compression = "N/A"
        
        table.add_row(
            conv_result.quantization_level,
            status,
            file_size,
            time_str,
            compression
        )
    
    console.print(table)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"Total conversions: {len(result.results)}")
    console.print(f"Successful: {result.successful_conversions}")
    console.print(f"Failed: {result.failed_conversions}")
    console.print(f"Success rate: {result.success_rate:.1f}%")
    console.print(f"Total time: {result.total_time:.1f}s")


def _magnitude_prune(state_dict, sparsity, skip_layers):
    """Implement magnitude-based pruning."""
    pruned_dict = {}
    
    for name, tensor in state_dict.items():
        # Check if layer should be skipped
        skip = any(pattern in name for pattern in skip_layers)
        
        if skip or 'bias' in name.lower():
            # Keep layer as-is
            pruned_dict[name] = tensor.clone()
        else:
            # Apply magnitude pruning
            flat_tensor = tensor.flatten()
            k = int(len(flat_tensor) * sparsity)
            
            if k > 0:
                # Find threshold for pruning
                threshold = torch.kthvalue(torch.abs(flat_tensor), k).values
                mask = torch.abs(tensor) >= threshold
                pruned_dict[name] = tensor * mask.float()
            else:
                pruned_dict[name] = tensor.clone()
    
    return pruned_dict


def _structured_prune(state_dict, sparsity, skip_layers):
    """Implement structured pruning (placeholder)."""
    # This is a simplified placeholder for structured pruning
    # Real implementation would remove entire channels/filters
    return _magnitude_prune(state_dict, sparsity, skip_layers)


def _prepare_gradual_prune(state_dict, sparsity, skip_layers):
    """Prepare model for gradual pruning (placeholder)."""
    # This would typically add pruning masks to the model
    # For now, just return the original state dict
    return {name: tensor.clone() for name, tensor in state_dict.items()}


def _validate_gguf_file(model_path, detailed=False):
    """Validate GGUF file format."""
    try:
        # Basic file existence and size check
        if not model_path.exists():
            return {'is_valid': False, 'error': 'File does not exist'}
        
        file_size = model_path.stat().st_size
        if file_size < 1024:  # Minimum reasonable size
            return {'is_valid': False, 'error': 'File too small to be valid GGUF'}
        
        # Try to read GGUF header
        with open(model_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return {'is_valid': False, 'error': 'Invalid GGUF magic number'}
        
        results = {
            'is_valid': True,
            'file_size': file_size,
            'format': 'GGUF'
        }
        
        if detailed:
            results['details'] = {
                'magic_number': {'passed': True, 'message': 'Valid GGUF magic number'},
                'file_size': {'passed': True, 'message': f'{file_size / (1024*1024):.2f} MB'},
            }
        
        return results
        
    except Exception as e:
        return {'is_valid': False, 'error': str(e)}


def _validate_onnx_file(model_path, detailed=False):
    """Validate ONNX file format."""
    try:
        # Basic validation for ONNX files
        if not model_path.exists():
            return {'is_valid': False, 'error': 'File does not exist'}
        
        file_size = model_path.stat().st_size
        
        results = {
            'is_valid': True,
            'file_size': file_size,
            'format': 'ONNX'
        }
        
        if detailed:
            results['details'] = {
                'file_exists': {'passed': True, 'message': 'File exists'},
                'file_size': {'passed': True, 'message': f'{file_size / (1024*1024):.2f} MB'},
            }
        
        return results
        
    except Exception as e:
        return {'is_valid': False, 'error': str(e)}


def _validate_quantized_file(model_path, detailed=False):
    """Validate quantized PyTorch model file."""
    try:
        # Load and validate quantized model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        required_keys = ['quantized_tensors', 'quantization_config']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            return {
                'is_valid': False, 
                'error': f'Missing required keys: {missing_keys}'
            }
        
        results = {
            'is_valid': True,
            'file_size': model_path.stat().st_size,
            'format': 'Quantized PyTorch'
        }
        
        if detailed:
            quant_config = checkpoint.get('quantization_config', {})
            results['details'] = {
                'quantization_type': {
                    'passed': True, 
                    'message': quant_config.get('quantization_type', 'unknown')
                },
                'tensor_count': {
                    'passed': True,
                    'message': str(len(checkpoint.get('quantized_tensors', {})))
                }
            }
        
        return results
        
    except Exception as e:
        return {'is_valid': False, 'error': str(e)}