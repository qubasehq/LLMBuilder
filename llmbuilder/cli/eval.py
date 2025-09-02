"""
Evaluation and benchmarking CLI commands for LLMBuilder.

This module provides commands for model evaluation, benchmarking,
and performance comparison.
"""

import click
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from loguru import logger
import os

from llmbuilder.utils.config import ConfigManager

console = Console()


class EvaluationManager:
    """Manages evaluation sessions and results."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize evaluation manager."""
        if results_dir is None:
            results_dir = Path.home() / ".llmbuilder" / "evaluation_results"
        
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: Dict[str, Any], name: str) -> Path:
        """Save evaluation results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return filepath
    
    def load_results(self, filepath: Path) -> Dict[str, Any]:
        """Load evaluation results from file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_results(self) -> List[Path]:
        """List all evaluation result files."""
        return list(self.results_dir.glob("*.json"))


@click.group()
def eval():
    """
    Evaluation and benchmarking commands for model assessment.
    
    This command group provides tools for:
    - Running comprehensive model evaluations
    - Benchmarking model performance
    - Comparing multiple models
    - Generating evaluation reports
    
    Examples:
        llmbuilder eval run --model-path ./model.pt
        llmbuilder eval benchmark --model-path ./model.pt
        llmbuilder eval compare model1.pt model2.pt
        llmbuilder eval report --results-dir ./results
    """
    pass


@eval.command()
@click.option(
    '--model-path',
    '-m',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to model checkpoint'
)
@click.option(
    '--tokenizer-path',
    '-t',
    type=click.Path(exists=True, path_type=Path),
    help='Path to tokenizer model'
)
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file path'
)
@click.option(
    '--test-data',
    type=click.Path(exists=True, path_type=Path),
    help='Test dataset path for perplexity evaluation'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(path_type=Path),
    help='Output directory for results'
)
@click.option(
    '--name',
    '-n',
    help='Name for this evaluation run'
)
@click.option(
    '--prompts-file',
    type=click.Path(exists=True, path_type=Path),
    help='File containing test prompts (one per line)'
)
@click.option(
    '--max-samples',
    type=int,
    default=1000,
    help='Maximum samples for perplexity evaluation'
)
@click.option(
    '--batch-size',
    type=int,
    default=16,
    help='Batch size for evaluation'
)
@click.pass_context
def run(
    ctx: click.Context,
    model_path: Path,
    tokenizer_path: Optional[Path],
    config: Optional[Path],
    test_data: Optional[Path],
    output_dir: Optional[Path],
    name: Optional[str],
    prompts_file: Optional[Path],
    max_samples: int,
    batch_size: int
):
    """
    Run comprehensive model evaluation.
    
    Evaluates model performance including perplexity, generation quality,
    and inference speed. Generates detailed reports with metrics and examples.
    
    Examples:
        llmbuilder eval run --model-path ./model.pt --test-data ./test.pt
        llmbuilder eval run -m ./model.pt -t ./tokenizer.model --name my-eval
        llmbuilder eval run -m ./model.pt --prompts-file ./prompts.txt
    """
    try:
        from llmbuilder.core.eval.eval import ModelEvaluator
        
        console.print(f"[bold blue]Starting model evaluation[/bold blue]")
        console.print(f"Model: {model_path}")
        
        # Set default paths if not provided
        if tokenizer_path is None:
            # Try common tokenizer locations
            possible_paths = [
                Path("exports/tokenizer/tokenizer.model"),
                Path("tokenizer/tokenizer.model"),
                model_path.parent / "tokenizer.model"
            ]
            for path in possible_paths:
                if path.exists():
                    tokenizer_path = path
                    break
            
            if tokenizer_path is None:
                console.print("[red]Tokenizer not found. Please specify --tokenizer-path[/red]")
                return
        
        console.print(f"Tokenizer: {tokenizer_path}")
        
        # Load test prompts
        test_prompts = []
        if prompts_file:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                test_prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default test prompts
            test_prompts = [
                "The future of artificial intelligence",
                "In a world where technology",
                "The most important thing to remember",
                "Once upon a time in a distant land",
                "Explain the concept of machine learning",
                "What are the benefits of renewable energy",
                "Describe the process of photosynthesis"
            ]
        
        console.print(f"Test prompts: {len(test_prompts)}")
        
        # Initialize evaluator
        with console.status("[bold green]Initializing evaluator...") as status:
            evaluator = ModelEvaluator(
                model_path=str(model_path),
                tokenizer_path=str(tokenizer_path),
                config_path=str(config) if config else None
            )
            
            status.update("[bold green]Loading model...")
            evaluator.load_model()
            
            status.update("[bold green]Loading tokenizer...")
            evaluator.load_tokenizer()
        
        console.print("[green]✓ Model and tokenizer loaded successfully[/green]")
        
        # Run comprehensive evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            eval_task = progress.add_task("Running evaluation...", total=None)
            
            results = evaluator.comprehensive_evaluation(
                test_data_path=str(test_data) if test_data else None,
                prompts=test_prompts
            )
            
            progress.update(eval_task, description="Evaluation completed!")
        
        # Display results
        _display_evaluation_results(results)
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            manager = EvaluationManager(output_dir)
        else:
            manager = EvaluationManager()
        
        eval_name = name or f"eval_{model_path.stem}"
        results_file = manager.save_results(results, eval_name)
        
        console.print(f"\n[green]✓ Evaluation completed successfully![/green]")
        console.print(f"Results saved to: {results_file}")
        
    except Exception as e:
        console.print(f"[red]Error running evaluation: {e}[/red]")
        logger.error(f"Evaluation failed: {e}")
        raise click.ClickException(f"Evaluation failed: {e}")


@eval.command()
@click.option(
    '--model-path',
    '-m',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to model checkpoint'
)
@click.option(
    '--tokenizer-path',
    '-t',
    type=click.Path(exists=True, path_type=Path),
    help='Path to tokenizer model'
)
@click.option(
    '--sequence-length',
    type=int,
    default=256,
    help='Sequence length for benchmarking'
)
@click.option(
    '--batch-size',
    type=int,
    default=1,
    help='Batch size for benchmarking'
)
@click.option(
    '--iterations',
    type=int,
    default=100,
    help='Number of benchmark iterations'
)
@click.option(
    '--warmup',
    type=int,
    default=10,
    help='Number of warmup iterations'
)
@click.pass_context
def benchmark(
    ctx: click.Context,
    model_path: Path,
    tokenizer_path: Optional[Path],
    sequence_length: int,
    batch_size: int,
    iterations: int,
    warmup: int
):
    """
    Benchmark model inference performance.
    
    Measures inference speed, memory usage, and throughput for the model
    under different conditions.
    
    Examples:
        llmbuilder eval benchmark --model-path ./model.pt
        llmbuilder eval benchmark -m ./model.pt --batch-size 4 --iterations 200
        llmbuilder eval benchmark -m ./model.pt --sequence-length 512
    """
    try:
        from llmbuilder.core.eval.eval import ModelEvaluator
        
        console.print(f"[bold blue]Benchmarking model performance[/bold blue]")
        console.print(f"Model: {model_path}")
        console.print(f"Sequence length: {sequence_length}")
        console.print(f"Batch size: {batch_size}")
        console.print(f"Iterations: {iterations}")
        
        # Set default tokenizer path if not provided
        if tokenizer_path is None:
            possible_paths = [
                Path("exports/tokenizer/tokenizer.model"),
                Path("tokenizer/tokenizer.model"),
                model_path.parent / "tokenizer.model"
            ]
            for path in possible_paths:
                if path.exists():
                    tokenizer_path = path
                    break
        
        # Initialize evaluator
        with console.status("[bold green]Loading model...") as status:
            evaluator = ModelEvaluator(
                model_path=str(model_path),
                tokenizer_path=str(tokenizer_path) if tokenizer_path else None
            )
            evaluator.load_model()
        
        console.print("[green]✓ Model loaded successfully[/green]")
        
        # Run benchmark
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            benchmark_task = progress.add_task("Benchmarking...", total=iterations + warmup)
            
            # Custom benchmark with progress tracking
            results = _run_detailed_benchmark(
                evaluator, sequence_length, batch_size, iterations, warmup, progress, benchmark_task
            )
        
        # Display benchmark results
        _display_benchmark_results(results)
        
        console.print(f"\n[green]✓ Benchmark completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error running benchmark: {e}[/red]")
        logger.error(f"Benchmark failed: {e}")
        raise click.ClickException(f"Benchmark failed: {e}")


@eval.command()
@click.argument('model_paths', nargs=-1, required=True)
@click.option(
    '--tokenizer-path',
    '-t',
    type=click.Path(exists=True, path_type=Path),
    help='Path to tokenizer model (shared for all models)'
)
@click.option(
    '--test-data',
    type=click.Path(exists=True, path_type=Path),
    help='Test dataset path for comparison'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(path_type=Path),
    help='Output directory for comparison results'
)
@click.option(
    '--metrics',
    multiple=True,
    default=['perplexity', 'speed'],
    help='Metrics to compare (perplexity, speed, generation)'
)
@click.pass_context
def compare(
    ctx: click.Context,
    model_paths: tuple,
    tokenizer_path: Optional[Path],
    test_data: Optional[Path],
    output_dir: Optional[Path],
    metrics: tuple
):
    """
    Compare multiple models side-by-side.
    
    Evaluates multiple models on the same metrics and generates
    a comparison report showing relative performance.
    
    MODEL_PATHS: Paths to model checkpoints to compare
    
    Examples:
        llmbuilder eval compare model1.pt model2.pt model3.pt
        llmbuilder eval compare *.pt --test-data ./test.pt
        llmbuilder eval compare model1.pt model2.pt --metrics perplexity speed
    """
    try:
        from llmbuilder.core.eval.eval import ModelEvaluator
        
        model_paths = [Path(p) for p in model_paths]
        
        console.print(f"[bold blue]Comparing {len(model_paths)} models[/bold blue]")
        for i, path in enumerate(model_paths, 1):
            console.print(f"  {i}. {path}")
        
        console.print(f"Metrics: {', '.join(metrics)}")
        
        # Set default tokenizer path if not provided
        if tokenizer_path is None:
            possible_paths = [
                Path("exports/tokenizer/tokenizer.model"),
                Path("tokenizer/tokenizer.model")
            ]
            for path in possible_paths:
                if path.exists():
                    tokenizer_path = path
                    break
        
        if tokenizer_path is None:
            console.print("[red]Tokenizer not found. Please specify --tokenizer-path[/red]")
            return
        
        # Evaluate each model
        comparison_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Comparing models...", total=len(model_paths))
            
            for i, model_path in enumerate(model_paths):
                progress.update(main_task, description=f"Evaluating {model_path.name}...")
                
                try:
                    evaluator = ModelEvaluator(
                        model_path=str(model_path),
                        tokenizer_path=str(tokenizer_path)
                    )
                    evaluator.load_model()
                    
                    # Run selected evaluations
                    model_results = {
                        'model_path': str(model_path),
                        'model_name': model_path.stem,
                        'model_info': {
                            'parameters': evaluator.model.numel() if hasattr(evaluator.model, 'numel') else 0
                        }
                    }
                    
                    if 'perplexity' in metrics and test_data:
                        ppl_results = evaluator.evaluate_perplexity(str(test_data), max_samples=500)
                        model_results['perplexity'] = ppl_results
                    
                    if 'speed' in metrics:
                        bench_results = evaluator.benchmark_inference(num_iterations=50)
                        model_results['benchmark'] = bench_results
                    
                    if 'generation' in metrics:
                        test_prompts = ["The future of AI", "Once upon a time"]
                        gen_results = []
                        for prompt in test_prompts:
                            generated = evaluator.generate_text(prompt, max_new_tokens=30, num_samples=1)
                            gen_results.append(generated[0])
                        model_results['generation'] = gen_results
                    
                    comparison_results.append(model_results)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_path}: {e}")
                    comparison_results.append({
                        'model_path': str(model_path),
                        'model_name': model_path.stem,
                        'error': str(e)
                    })
                
                progress.update(main_task, advance=1)
        
        # Display comparison results
        _display_comparison_results(comparison_results, metrics)
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            manager = EvaluationManager(output_dir)
        else:
            manager = EvaluationManager()
        
        results_file = manager.save_results({
            'comparison_results': comparison_results,
            'metrics': list(metrics),
            'timestamp': time.time()
        }, "model_comparison")
        
        console.print(f"\n[green]✓ Model comparison completed![/green]")
        console.print(f"Results saved to: {results_file}")
        
    except Exception as e:
        console.print(f"[red]Error comparing models: {e}[/red]")
        logger.error(f"Model comparison failed: {e}")
        raise click.ClickException(f"Model comparison failed: {e}")


@eval.command()
@click.option(
    '--results-dir',
    '-d',
    type=click.Path(exists=True, path_type=Path),
    help='Directory containing evaluation results'
)
@click.option(
    '--results-file',
    '-f',
    type=click.Path(exists=True, path_type=Path),
    help='Specific results file to generate report from'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    help='Output file for the report (HTML or JSON)'
)
@click.option(
    '--format',
    type=click.Choice(['html', 'json', 'markdown']),
    default='html',
    help='Report format'
)
@click.pass_context
def report(
    ctx: click.Context,
    results_dir: Optional[Path],
    results_file: Optional[Path],
    output: Optional[Path],
    format: str
):
    """
    Generate evaluation report from results.
    
    Creates comprehensive reports from evaluation results with
    visualizations, metrics summaries, and recommendations.
    
    Examples:
        llmbuilder eval report --results-dir ./eval_results
        llmbuilder eval report --results-file ./results.json --format html
        llmbuilder eval report -d ./results -o report.html
    """
    try:
        if results_file:
            # Generate report from specific file
            manager = EvaluationManager()
            results = manager.load_results(results_file)
            
            console.print(f"[bold blue]Generating report from {results_file}[/bold blue]")
            
            if format == 'html':
                report_content = _generate_html_report(results)
                output_file = output or results_file.with_suffix('.html')
            elif format == 'markdown':
                report_content = _generate_markdown_report(results)
                output_file = output or results_file.with_suffix('.md')
            else:  # json
                report_content = json.dumps(results, indent=2, default=str)
                output_file = output or results_file.with_suffix('.formatted.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            console.print(f"[green]✓ Report generated: {output_file}[/green]")
            
        elif results_dir:
            # Generate report from directory
            manager = EvaluationManager(results_dir)
            result_files = manager.list_results()
            
            if not result_files:
                console.print("[yellow]No evaluation results found in directory[/yellow]")
                return
            
            console.print(f"[bold blue]Generating summary report from {len(result_files)} result files[/bold blue]")
            
            # Load all results
            all_results = []
            for file in result_files:
                try:
                    results = manager.load_results(file)
                    results['_source_file'] = str(file)
                    all_results.append(results)
                except Exception as e:
                    logger.warning(f"Could not load {file}: {e}")
            
            # Generate summary report
            if format == 'html':
                report_content = _generate_summary_html_report(all_results)
                output_file = output or (results_dir / "summary_report.html")
            elif format == 'markdown':
                report_content = _generate_summary_markdown_report(all_results)
                output_file = output or (results_dir / "summary_report.md")
            else:  # json
                report_content = json.dumps(all_results, indent=2, default=str)
                output_file = output or (results_dir / "summary_report.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            console.print(f"[green]✓ Summary report generated: {output_file}[/green]")
            
        else:
            console.print("[red]Please specify either --results-dir or --results-file[/red]")
            return
        
    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        logger.error(f"Report generation failed: {e}")
        raise click.ClickException(f"Report generation failed: {e}")


def _display_evaluation_results(results: Dict[str, Any]):
    """Display evaluation results in a formatted way."""
    console.print("\n[bold cyan]═══ Evaluation Results ═══[/bold cyan]")
    
    # Model info
    if 'model_info' in results:
        info = results['model_info']
        model_panel = Panel(
            f"Parameters: {info.get('parameters', 'N/A'):,}\n"
            f"Architecture: {info.get('n_layer', 'N/A')}L-{info.get('n_head', 'N/A')}H-{info.get('n_embd', 'N/A')}D\n"
            f"Vocab Size: {info.get('vocab_size', 'N/A'):,}\n"
            f"Max Length: {info.get('block_size', 'N/A')}\n"
            f"Device: {info.get('device', 'N/A')}",
            title="Model Information",
            border_style="blue"
        )
        console.print(model_panel)
    
    # Perplexity results
    if 'perplexity' in results and results['perplexity']:
        ppl = results['perplexity']
        ppl_panel = Panel(
            f"Loss: {ppl.get('loss', 'N/A'):.4f}\n"
            f"Perplexity: {ppl.get('perplexity', 'N/A'):.2f}\n"
            f"Tokens/sec: {ppl.get('tokens_per_sec', 'N/A'):.0f}\n"
            f"Total tokens: {ppl.get('total_tokens', 'N/A'):,}\n"
            f"Eval time: {ppl.get('eval_time', 'N/A'):.1f}s",
            title="Perplexity Evaluation",
            border_style="green"
        )
        console.print(ppl_panel)
    
    # Benchmark results
    if 'benchmark' in results and results['benchmark']:
        bench = results['benchmark']
        bench_panel = Panel(
            f"Tokens/sec: {bench.get('tokens_per_sec', 'N/A'):.0f}\n"
            f"Avg time/batch: {bench.get('avg_time_per_batch', 'N/A')*1000:.2f}ms\n"
            f"Batch size: {bench.get('batch_size', 'N/A')}\n"
            f"Sequence length: {bench.get('sequence_length', 'N/A')}\n"
            f"Iterations: {bench.get('iterations', 'N/A')}",
            title="Performance Benchmark",
            border_style="yellow"
        )
        console.print(bench_panel)
    
    # Generation examples
    if 'generation' in results and results['generation']:
        console.print("\n[bold cyan]Generation Examples:[/bold cyan]")
        for i, gen in enumerate(results['generation'][:3], 1):  # Show first 3
            prompt = gen.get('prompt', 'N/A')
            generated = gen.get('generated', {})
            text = generated.get('text', 'N/A')
            quality = generated.get('quality_score', 0)
            
            gen_panel = Panel(
                f"[bold]Prompt:[/bold] {prompt}\n\n"
                f"[bold]Generated:[/bold] {text}\n\n"
                f"[bold]Quality Score:[/bold] {quality:.2f}",
                title=f"Example {i}",
                border_style="magenta"
            )
            console.print(gen_panel)


def _display_benchmark_results(results: Dict[str, Any]):
    """Display benchmark results in a formatted way."""
    console.print("\n[bold cyan]═══ Benchmark Results ═══[/bold cyan]")
    
    # Create metrics table
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="yellow")
    
    table.add_row("Tokens per second", f"{results.get('tokens_per_sec', 0):.0f}", "tokens/sec")
    table.add_row("Average time per batch", f"{results.get('avg_time_per_batch', 0)*1000:.2f}", "ms")
    table.add_row("Total time", f"{results.get('total_time', 0):.2f}", "seconds")
    table.add_row("Iterations", f"{results.get('iterations', 0)}", "count")
    table.add_row("Batch size", f"{results.get('batch_size', 0)}", "samples")
    table.add_row("Sequence length", f"{results.get('sequence_length', 0)}", "tokens")
    
    console.print(table)


def _display_comparison_results(results: List[Dict[str, Any]], metrics: tuple):
    """Display model comparison results."""
    console.print("\n[bold cyan]═══ Model Comparison ═══[/bold cyan]")
    
    # Create comparison table
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    
    if 'perplexity' in metrics:
        table.add_column("Perplexity", style="green")
        table.add_column("Loss", style="green")
    
    if 'speed' in metrics:
        table.add_column("Tokens/sec", style="yellow")
    
    table.add_column("Parameters", style="blue")
    table.add_column("Status", style="magenta")
    
    for result in results:
        row = [result.get('model_name', 'Unknown')]
        
        if 'perplexity' in metrics:
            if 'perplexity' in result:
                ppl = result['perplexity']
                row.append(f"{ppl.get('perplexity', 'N/A'):.2f}")
                row.append(f"{ppl.get('loss', 'N/A'):.4f}")
            else:
                row.extend(['N/A', 'N/A'])
        
        if 'speed' in metrics:
            if 'benchmark' in result:
                bench = result['benchmark']
                row.append(f"{bench.get('tokens_per_sec', 'N/A'):.0f}")
            else:
                row.append('N/A')
        
        # Parameters
        params = result.get('model_info', {}).get('parameters', 0)
        if params > 0:
            if params > 1e9:
                row.append(f"{params/1e9:.1f}B")
            elif params > 1e6:
                row.append(f"{params/1e6:.1f}M")
            else:
                row.append(f"{params/1e3:.1f}K")
        else:
            row.append('N/A')
        
        # Status
        if 'error' in result:
            row.append('[red]Error[/red]')
        else:
            row.append('[green]Success[/green]')
        
        table.add_row(*row)
    
    console.print(table)


def _run_detailed_benchmark(evaluator, sequence_length, batch_size, iterations, warmup, progress, task_id):
    """Run detailed benchmark with progress tracking."""
from llmbuilder.utils.lazy_imports import torch
    import time
    
    # Create dummy input
    dummy_input = torch.randint(
        0, evaluator.model.vocab_size,
        (batch_size, sequence_length),
        device=evaluator.device
    )
    
    # Warmup
    with torch.no_grad():
        for i in range(warmup):
            _ = evaluator.model(dummy_input)
            progress.update(task_id, advance=1)
    
    # Benchmark
    times = []
    
    with torch.no_grad():
        for i in range(iterations):
            start_time = time.time()
            _ = evaluator.model(dummy_input)
            end_time = time.time()
            
            times.append(end_time - start_time)
            progress.update(task_id, advance=1)
    
    # Calculate detailed metrics
    total_time = sum(times)
    avg_time = total_time / iterations
    min_time = min(times)
    max_time = max(times)
    tokens_per_sec = (batch_size * sequence_length * iterations) / total_time
    
    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time,
        'min_time_per_batch': min_time,
        'max_time_per_batch': max_time,
        'tokens_per_sec': tokens_per_sec,
        'iterations': iterations,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'warmup_iterations': warmup
    }


def _generate_html_report(results: Dict[str, Any]) -> str:
    """Generate HTML report from evaluation results."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLMBuilder Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .metric { display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }
            .generation { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 3px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>LLMBuilder Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        {model_info_section}
        {perplexity_section}
        {benchmark_section}
        {generation_section}
    </body>
    </html>
    """
    
    # Generate sections
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    model_info_section = ""
    if 'model_info' in results:
        info = results['model_info']
        model_info_section = f"""
        <div class="section">
            <h2>Model Information</h2>
            <div class="metric">Parameters: {info.get('parameters', 'N/A'):,}</div>
            <div class="metric">Architecture: {info.get('n_layer', 'N/A')}L-{info.get('n_head', 'N/A')}H-{info.get('n_embd', 'N/A')}D</div>
            <div class="metric">Vocab Size: {info.get('vocab_size', 'N/A'):,}</div>
            <div class="metric">Device: {info.get('device', 'N/A')}</div>
        </div>
        """
    
    perplexity_section = ""
    if 'perplexity' in results and results['perplexity']:
        ppl = results['perplexity']
        perplexity_section = f"""
        <div class="section">
            <h2>Perplexity Evaluation</h2>
            <div class="metric">Loss: {ppl.get('loss', 'N/A'):.4f}</div>
            <div class="metric">Perplexity: {ppl.get('perplexity', 'N/A'):.2f}</div>
            <div class="metric">Tokens/sec: {ppl.get('tokens_per_sec', 'N/A'):.0f}</div>
        </div>
        """
    
    benchmark_section = ""
    if 'benchmark' in results and results['benchmark']:
        bench = results['benchmark']
        benchmark_section = f"""
        <div class="section">
            <h2>Performance Benchmark</h2>
            <div class="metric">Tokens/sec: {bench.get('tokens_per_sec', 'N/A'):.0f}</div>
            <div class="metric">Avg time/batch: {bench.get('avg_time_per_batch', 'N/A')*1000:.2f}ms</div>
        </div>
        """
    
    generation_section = ""
    if 'generation' in results and results['generation']:
        generation_section = "<div class='section'><h2>Generation Examples</h2>"
        for i, gen in enumerate(results['generation'][:5], 1):
            prompt = gen.get('prompt', 'N/A')
            generated = gen.get('generated', {})
            text = generated.get('text', 'N/A')
            quality = generated.get('quality_score', 0)
            
            generation_section += f"""
            <div class="generation">
                <h4>Example {i}</h4>
                <p><strong>Prompt:</strong> {prompt}</p>
                <p><strong>Generated:</strong> {text}</p>
                <p><strong>Quality Score:</strong> {quality:.2f}</p>
            </div>
            """
        generation_section += "</div>"
    
    return html_template.format(
        timestamp=timestamp,
        model_info_section=model_info_section,
        perplexity_section=perplexity_section,
        benchmark_section=benchmark_section,
        generation_section=generation_section
    )


def _generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate Markdown report from evaluation results."""
    report = "# LLMBuilder Evaluation Report\n\n"
    report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if 'model_info' in results:
        info = results['model_info']
        report += "## Model Information\n\n"
        report += f"- **Parameters:** {info.get('parameters', 'N/A'):,}\n"
        report += f"- **Architecture:** {info.get('n_layer', 'N/A')}L-{info.get('n_head', 'N/A')}H-{info.get('n_embd', 'N/A')}D\n"
        report += f"- **Vocab Size:** {info.get('vocab_size', 'N/A'):,}\n"
        report += f"- **Device:** {info.get('device', 'N/A')}\n\n"
    
    if 'perplexity' in results and results['perplexity']:
        ppl = results['perplexity']
        report += "## Perplexity Evaluation\n\n"
        report += f"- **Loss:** {ppl.get('loss', 'N/A'):.4f}\n"
        report += f"- **Perplexity:** {ppl.get('perplexity', 'N/A'):.2f}\n"
        report += f"- **Tokens/sec:** {ppl.get('tokens_per_sec', 'N/A'):.0f}\n\n"
    
    if 'benchmark' in results and results['benchmark']:
        bench = results['benchmark']
        report += "## Performance Benchmark\n\n"
        report += f"- **Tokens/sec:** {bench.get('tokens_per_sec', 'N/A'):.0f}\n"
        report += f"- **Avg time/batch:** {bench.get('avg_time_per_batch', 'N/A')*1000:.2f}ms\n\n"
    
    if 'generation' in results and results['generation']:
        report += "## Generation Examples\n\n"
        for i, gen in enumerate(results['generation'][:5], 1):
            prompt = gen.get('prompt', 'N/A')
            generated = gen.get('generated', {})
            text = generated.get('text', 'N/A')
            quality = generated.get('quality_score', 0)
            
            report += f"### Example {i}\n\n"
            report += f"**Prompt:** {prompt}\n\n"
            report += f"**Generated:** {text}\n\n"
            report += f"**Quality Score:** {quality:.2f}\n\n"
    
    return report


def _generate_summary_html_report(all_results: List[Dict[str, Any]]) -> str:
    """Generate summary HTML report from multiple results."""
    # This would be a more complex implementation
    # For now, return a simple summary
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>LLMBuilder Summary Report</title></head>
    <body>
        <h1>LLMBuilder Summary Report</h1>
        <p>Summary of {len(all_results)} evaluation runs</p>
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """


def _generate_summary_markdown_report(all_results: List[Dict[str, Any]]) -> str:
    """Generate summary Markdown report from multiple results."""
    report = "# LLMBuilder Summary Report\n\n"
    report += f"Summary of {len(all_results)} evaluation runs\n\n"
    report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, result in enumerate(all_results, 1):
        report += f"## Evaluation {i}\n\n"
        if 'model_info' in result:
            info = result['model_info']
            report += f"- Parameters: {info.get('parameters', 'N/A'):,}\n"
        report += "\n"
    
    return report


if __name__ == "__main__":
    eval()