"""
Data management CLI commands for LLMBuilder.

This module provides commands for data preparation, splitting, validation,
and integration with existing deduplication and cleaning pipelines.
"""

import click
import json
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger

from llmbuilder.core.data.ingest import DocumentIngester
from llmbuilder.core.data.dedup import DeduplicationPipeline
from llmbuilder.core.training.preprocess import DataPreprocessor
from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.colors import (
    ColorFormatter, Color, print_header, print_success, print_error, 
    print_warning, print_info, print_table, confirm_action
)
from llmbuilder.utils.progress import (
    progress_bar, spinner, long_running_task, show_step_progress
)


@click.group()
def data():
    """
    Data management commands for processing and preparing training datasets.
    
    This command group provides tools for:
    - Preparing and cleaning raw data files
    - Splitting datasets into train/validation/test sets
    - Validating data quality and statistics
    - Deduplication and preprocessing
    
    Examples:
        llmbuilder data prepare ./raw_data ./processed_data
        llmbuilder data split ./processed_data --ratios 0.8 0.1 0.1
        llmbuilder data validate ./processed_data
    """
    pass


@data.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option(
    '--formats', 
    '-f',
    multiple=True,
    default=['pdf', 'txt', 'docx', 'json', 'csv'],
    help='File formats to process (can be specified multiple times)'
)
@click.option(
    '--deduplicate/--no-deduplicate',
    default=True,
    help='Enable or disable deduplication'
)
@click.option(
    '--clean/--no-clean',
    default=True,
    help='Enable or disable text cleaning'
)
@click.option(
    '--max-files',
    type=int,
    help='Maximum number of files to process'
)
@click.option(
    '--batch-size',
    type=int,
    default=100,
    help='Number of files to process in each batch'
)
@click.option(
    '--workers',
    type=int,
    default=4,
    help='Number of worker processes for parallel processing'
)
@click.pass_context
def prepare(
    ctx: click.Context,
    input_dir: Path,
    output_dir: Path,
    formats: Tuple[str, ...],
    deduplicate: bool,
    clean: bool,
    max_files: Optional[int],
    batch_size: int,
    workers: int
):
    """
    Prepare and process raw data files for training.
    
    This command loads and processes PDFs, DOCX, TXT, JSON, and CSV files,
    performing deduplication and text cleaning automatically.
    
    INPUT_DIR: Directory containing raw data files
    OUTPUT_DIR: Directory where processed data will be saved
    
    Examples:
        llmbuilder data prepare ./raw ./processed
        llmbuilder data prepare ./docs ./clean --formats pdf txt --no-deduplicate
        llmbuilder data prepare ./data ./output --max-files 1000 --workers 8
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print_header("Data Preparation")
        print_info(f"Input directory: {input_dir}")
        print_info(f"Output directory: {output_dir}")
        print_info(f"Formats: {', '.join(formats)}")
        print_info(f"Deduplication: {'enabled' if deduplicate else 'disabled'}")
        print_info(f"Cleaning: {'enabled' if clean else 'disabled'}")
        
        # Initialize components with progress
        with spinner("Initializing components") as spin:
            ingester = DocumentIngester()
            spin.set_status("Finding files to process...")
            
            # Find files to process
            files_to_process = []
            for format_ext in formats:
                pattern = f"*.{format_ext}"
                found_files = list(input_dir.rglob(pattern))
                files_to_process.extend(found_files)
            
            if max_files:
                files_to_process = files_to_process[:max_files]
        
        print_success(f"Found {len(files_to_process)} files to process")
        
        if not files_to_process:
            print_warning("No files found to process")
            return
        
        # Process files with progress bar
        processed_files = []
        failed_files = []
        total_chars = 0
        
        show_step_progress(["File Processing", "Deduplication", "Text Cleaning", "Summary"], 1, "File Processing")
        
        with progress_bar(
            total=len(files_to_process),
            description="Processing files",
            unit="files",
            color=Color.BLUE
        ) as pbar:
            
            for i, file_path in enumerate(files_to_process):
                try:
                    pbar.set_status(f"Processing {file_path.name}")
                    
                    # Extract text from file
                    extracted_text = ingester.ingest_file(file_path)
                    
                    if extracted_text and extracted_text.strip():
                        # Create output file path
                        relative_path = file_path.relative_to(input_dir)
                        output_file = output_dir / relative_path.with_suffix('.txt')
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Write processed text
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(extracted_text)
                        
                        processed_files.append(output_file)
                        total_chars += len(extracted_text)
                        
                        logger.debug(f"Processed {file_path} -> {output_file}")
                    else:
                        failed_files.append(file_path)
                        logger.warning(f"Failed to extract text from {file_path}")
                
                except Exception as e:
                    failed_files.append(file_path)
                    logger.error(f"Error processing {file_path}: {e}")
                
                pbar.update(1)
        
        # Deduplication if enabled
        if deduplicate and processed_files:
            show_step_progress(["File Processing", "Deduplication", "Text Cleaning", "Summary"], 2, "Deduplication")
            
            try:
                def run_deduplication():
                    dedup_pipeline = DeduplicationPipeline()
                    
                    # Create a temporary file list for deduplication
                    file_list_path = output_dir / "file_list.txt"
                    with open(file_list_path, 'w') as f:
                        for file_path in processed_files:
                            f.write(f"{file_path}\n")
                    
                    # Run deduplication
                    dedup_output_dir = output_dir / "deduped"
                    dedup_output_dir.mkdir(exist_ok=True)
                    
                    dedup_stats = dedup_pipeline.deduplicate_files(
                        str(file_list_path),
                        str(dedup_output_dir)
                    )
                    
                    # Clean up temporary file
                    file_list_path.unlink(missing_ok=True)
                    
                    return dedup_stats
                
                dedup_stats = long_running_task(
                    run_deduplication,
                    message="Running deduplication",
                    success_message="Deduplication completed",
                    timeout_message="Deduplication is processing large dataset..."
                )
                
                if dedup_stats:
                    print_info(f"Original files: {dedup_stats.get('original_count', 'N/A')}")
                    print_info(f"After deduplication: {dedup_stats.get('final_count', 'N/A')}")
                
            except Exception as e:
                print_error(f"Deduplication failed: {e}")
                logger.error(f"Deduplication error: {e}")
        
        # Additional cleaning if enabled
        if clean and processed_files:
            show_step_progress(["File Processing", "Deduplication", "Text Cleaning", "Summary"], 3, "Text Cleaning")
            
            try:
                preprocessor = DataPreprocessor()
                
                with progress_bar(
                    total=len(processed_files),
                    description="Cleaning text",
                    unit="files",
                    color=Color.GREEN
                ) as pbar:
                    
                    for file_path in processed_files:
                        try:
                            pbar.set_status(f"Cleaning {file_path.name}")
                            
                            # Read file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                            
                            # Clean text
                            cleaned_text = preprocessor.clean_text(text)
                            
                            # Write back cleaned text
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(cleaned_text)
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Error cleaning {file_path}: {e}")
                            pbar.update(1)
                
                print_success("Text cleaning completed")
                
            except Exception as e:
                print_error(f"Text cleaning failed: {e}")
                logger.error(f"Text cleaning error: {e}")
        
        # Generate summary
        summary_data = {
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'total_files_found': len(files_to_process),
            'successfully_processed': len(processed_files),
            'failed_files': len(failed_files),
            'total_characters': total_chars,
            'deduplication_enabled': deduplicate,
            'cleaning_enabled': clean,
            'formats_processed': list(formats)
        }
        
        # Save summary
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Display results
        show_step_progress(["File Processing", "Deduplication", "Text Cleaning", "Summary"], 4, "Summary")
        
        print_success("Data preparation completed!")
        
        # Create summary table
        headers = ["Metric", "Value"]
        rows = [
            ["Files Found", str(len(files_to_process))],
            ["Successfully Processed", str(len(processed_files))],
            ["Failed", str(len(failed_files))],
            ["Total Characters", f"{total_chars:,}"],
            ["Output Directory", str(output_dir)]
        ]
        
        print_table(headers, rows, header_color=Color.BLUE)
        
        if failed_files:
            print_warning(f"{len(failed_files)} files failed to process")
            print_info("Check logs for details on failed files")
        
    except Exception as e:
        print_error(f"Error during data preparation: {e}")
        logger.error(f"Data preparation failed: {e}")
        raise click.ClickException(f"Data preparation failed: {e}")


@data.command()
@click.argument('data_path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--ratios',
    '-r',
    type=float,
    multiple=True,
    default=[0.8, 0.1, 0.1],
    help='Split ratios for train/validation/test (must sum to 1.0)'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(path_type=Path),
    help='Output directory for split datasets (default: data_path/splits)'
)
@click.option(
    '--stratify/--no-stratify',
    default=False,
    help='Apply stratification when splitting (experimental)'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducible splits'
)
@click.option(
    '--shuffle/--no-shuffle',
    default=True,
    help='Shuffle data before splitting'
)
@click.pass_context
def split(
    ctx: click.Context,
    data_path: Path,
    ratios: Tuple[float, ...],
    output_dir: Optional[Path],
    stratify: bool,
    seed: int,
    shuffle: bool
):
    """
    Split datasets into train/validation/test sets.
    
    Automatically splits data into train/validation/test sets with configurable
    ratios while preserving data integrity and preventing data leakage.
    
    DATA_PATH: Directory containing processed data files
    
    Examples:
        llmbuilder data split ./processed_data
        llmbuilder data split ./data --ratios 0.7 0.2 0.1
        llmbuilder data split ./data --output-dir ./splits --seed 123
    """
    try:
        # Validate ratios
        if len(ratios) not in [2, 3]:
            raise click.BadParameter("Must provide 2 or 3 ratios (train/val or train/val/test)")
        
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise click.BadParameter(f"Ratios must sum to 1.0, got {sum(ratios)}")
        
        # Set output directory
        if output_dir is None:
            output_dir = data_path / "splits"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Splitting data from {data_path}[/bold blue]")
        console.print(f"Ratios: {ratios}")
        console.print(f"Output directory: {output_dir}")
        console.print(f"Seed: {seed}")
        console.print(f"Shuffle: {shuffle}")
        
        # Find all text files
        text_files = list(data_path.rglob("*.txt"))
        
        if not text_files:
            console.print("[yellow]No .txt files found to split[/yellow]")
            return
        
        console.print(f"Found {len(text_files)} files to split")
        
        # Import splitting functionality
        import random
        random.seed(seed)
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(text_files)
        
        # Calculate split indices
        total_files = len(text_files)
        
        if len(ratios) == 2:
            # Train/validation split
            train_size = int(total_files * ratios[0])
            val_size = total_files - train_size
            
            train_files = text_files[:train_size]
            val_files = text_files[train_size:]
            test_files = []
            
            split_names = ['train', 'validation']
            split_files = [train_files, val_files]
            
        else:
            # Train/validation/test split
            train_size = int(total_files * ratios[0])
            val_size = int(total_files * ratios[1])
            test_size = total_files - train_size - val_size
            
            train_files = text_files[:train_size]
            val_files = text_files[train_size:train_size + val_size]
            test_files = text_files[train_size + val_size:]
            
            split_names = ['train', 'validation', 'test']
            split_files = [train_files, val_files, test_files]
        
        # Create split directories and copy files
        split_stats = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            for split_name, files in zip(split_names, split_files):
                if not files:
                    continue
                
                split_dir = output_dir / split_name
                split_dir.mkdir(exist_ok=True)
                
                task = progress.add_task(f"Creating {split_name} split...", total=len(files))
                
                total_chars = 0
                
                for file_path in files:
                    try:
                        # Read original file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Create relative path structure
                        relative_path = file_path.relative_to(data_path)
                        output_file = split_dir / relative_path
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Write to split directory
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        total_chars += len(content)
                        progress.update(task, advance=1)
                        
                    except Exception as e:
                        logger.error(f"Error copying {file_path} to {split_name}: {e}")
                        progress.update(task, advance=1)
                
                split_stats[split_name] = {
                    'file_count': len(files),
                    'character_count': total_chars,
                    'directory': str(split_dir)
                }
        
        # Save split metadata
        metadata = {
            'original_data_path': str(data_path),
            'output_directory': str(output_dir),
            'ratios': list(ratios),
            'total_files': total_files,
            'seed': seed,
            'shuffle': shuffle,
            'stratify': stratify,
            'splits': split_stats
        }
        
        metadata_path = output_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Display results
        console.print("\n[bold green]Dataset splitting completed![/bold green]")
        
        table = Table(title="Split Summary")
        table.add_column("Split", style="cyan")
        table.add_column("Files", style="green")
        table.add_column("Characters", style="green")
        table.add_column("Percentage", style="yellow")
        
        for split_name, stats in split_stats.items():
            percentage = (stats['file_count'] / total_files) * 100
            table.add_row(
                split_name.title(),
                str(stats['file_count']),
                f"{stats['character_count']:,}",
                f"{percentage:.1f}%"
            )
        
        console.print(table)
        console.print(f"\nSplit datasets saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"[red]Error during dataset splitting: {e}[/red]")
        logger.error(f"Dataset splitting failed: {e}")
        raise click.ClickException(f"Dataset splitting failed: {e}")


@data.command()
@click.argument('data_path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-file',
    '-o',
    type=click.Path(path_type=Path),
    help='Output file for validation report (default: data_path/validation_report.json)'
)
@click.option(
    '--sample-size',
    type=int,
    default=1000,
    help='Number of files to sample for detailed analysis'
)
@click.option(
    '--check-encoding/--no-check-encoding',
    default=True,
    help='Check for encoding issues'
)
@click.option(
    '--check-duplicates/--no-check-duplicates',
    default=True,
    help='Check for potential duplicates'
)
@click.pass_context
def validate(
    ctx: click.Context,
    data_path: Path,
    output_file: Optional[Path],
    sample_size: int,
    check_encoding: bool,
    check_duplicates: bool
):
    """
    Validate data quality and provide statistics.
    
    Analyzes the dataset to provide comprehensive statistics, quality checks,
    and identifies potential issues with the processed data.
    
    DATA_PATH: Directory containing data files to validate
    
    Examples:
        llmbuilder data validate ./processed_data
        llmbuilder data validate ./splits/train --sample-size 500
        llmbuilder data validate ./data --output-file ./report.json
    """
    try:
        console.print(f"[bold blue]Validating data in {data_path}[/bold blue]")
        
        # Find all text files
        text_files = list(data_path.rglob("*.txt"))
        
        if not text_files:
            console.print("[yellow]No .txt files found to validate[/yellow]")
            return
        
        console.print(f"Found {len(text_files)} files to validate")
        
        # Initialize validation stats
        validation_stats = {
            'total_files': len(text_files),
            'total_characters': 0,
            'total_words': 0,
            'total_lines': 0,
            'file_sizes': [],
            'character_counts': [],
            'word_counts': [],
            'line_counts': [],
            'encoding_issues': [],
            'empty_files': [],
            'potential_duplicates': [],
            'language_distribution': {},
            'sample_analysis': {}
        }
        
        # Sample files for detailed analysis
        import random
        sample_files = random.sample(text_files, min(sample_size, len(text_files)))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            # Basic statistics task
            basic_task = progress.add_task("Collecting basic statistics...", total=len(text_files))
            
            for file_path in text_files:
                try:
                    # Get file size
                    file_size = file_path.stat().st_size
                    validation_stats['file_sizes'].append(file_size)
                    
                    # Read file content
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError as e:
                        if check_encoding:
                            validation_stats['encoding_issues'].append({
                                'file': str(file_path),
                                'error': str(e)
                            })
                        # Try with different encoding
                        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                            content = f.read()
                    
                    # Check if file is empty
                    if not content.strip():
                        validation_stats['empty_files'].append(str(file_path))
                        progress.update(basic_task, advance=1)
                        continue
                    
                    # Count characters, words, lines
                    char_count = len(content)
                    word_count = len(content.split())
                    line_count = len(content.splitlines())
                    
                    validation_stats['total_characters'] += char_count
                    validation_stats['total_words'] += word_count
                    validation_stats['total_lines'] += line_count
                    
                    validation_stats['character_counts'].append(char_count)
                    validation_stats['word_counts'].append(word_count)
                    validation_stats['line_counts'].append(line_count)
                    
                    progress.update(basic_task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Error validating {file_path}: {e}")
                    progress.update(basic_task, advance=1)
            
            # Duplicate detection if enabled
            if check_duplicates and len(text_files) > 1:
                dup_task = progress.add_task("Checking for duplicates...", total=len(sample_files))
                
                file_hashes = {}
                
                for file_path in sample_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Simple hash-based duplicate detection
                        import hashlib
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        
                        if content_hash in file_hashes:
                            validation_stats['potential_duplicates'].append({
                                'file1': str(file_hashes[content_hash]),
                                'file2': str(file_path),
                                'hash': content_hash
                            })
                        else:
                            file_hashes[content_hash] = file_path
                        
                        progress.update(dup_task, advance=1)
                        
                    except Exception as e:
                        logger.error(f"Error checking duplicates for {file_path}: {e}")
                        progress.update(dup_task, advance=1)
        
        # Calculate statistics
        if validation_stats['character_counts']:
            import statistics
            
            validation_stats['statistics'] = {
                'files': {
                    'total': validation_stats['total_files'],
                    'empty': len(validation_stats['empty_files']),
                    'with_encoding_issues': len(validation_stats['encoding_issues'])
                },
                'characters': {
                    'total': validation_stats['total_characters'],
                    'mean': statistics.mean(validation_stats['character_counts']),
                    'median': statistics.median(validation_stats['character_counts']),
                    'min': min(validation_stats['character_counts']),
                    'max': max(validation_stats['character_counts'])
                },
                'words': {
                    'total': validation_stats['total_words'],
                    'mean': statistics.mean(validation_stats['word_counts']),
                    'median': statistics.median(validation_stats['word_counts']),
                    'min': min(validation_stats['word_counts']),
                    'max': max(validation_stats['word_counts'])
                },
                'lines': {
                    'total': validation_stats['total_lines'],
                    'mean': statistics.mean(validation_stats['line_counts']),
                    'median': statistics.median(validation_stats['line_counts']),
                    'min': min(validation_stats['line_counts']),
                    'max': max(validation_stats['line_counts'])
                }
            }
        
        # Set output file
        if output_file is None:
            output_file = data_path / "validation_report.json"
        
        # Save validation report
        with open(output_file, 'w') as f:
            json.dump(validation_stats, f, indent=2, default=str)
        
        # Display results
        console.print("\n[bold green]Data validation completed![/bold green]")
        
        # Summary table
        table = Table(title="Validation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        stats = validation_stats.get('statistics', {})
        
        if stats:
            table.add_row("Total Files", str(stats['files']['total']))
            table.add_row("Empty Files", str(stats['files']['empty']))
            table.add_row("Encoding Issues", str(stats['files']['with_encoding_issues']))
            table.add_row("Total Characters", f"{stats['characters']['total']:,}")
            table.add_row("Total Words", f"{stats['words']['total']:,}")
            table.add_row("Avg Characters/File", f"{stats['characters']['mean']:.0f}")
            table.add_row("Avg Words/File", f"{stats['words']['mean']:.0f}")
            table.add_row("Potential Duplicates", str(len(validation_stats['potential_duplicates'])))
        
        console.print(table)
        
        # Issues panel
        issues = []
        if validation_stats['empty_files']:
            issues.append(f"• {len(validation_stats['empty_files'])} empty files")
        if validation_stats['encoding_issues']:
            issues.append(f"• {len(validation_stats['encoding_issues'])} files with encoding issues")
        if validation_stats['potential_duplicates']:
            issues.append(f"• {len(validation_stats['potential_duplicates'])} potential duplicates")
        
        if issues:
            console.print(Panel(
                "\n".join(issues),
                title="[yellow]Issues Found[/yellow]",
                border_style="yellow"
            ))
        else:
            console.print(Panel(
                "No significant issues found!",
                title="[green]Validation Results[/green]",
                border_style="green"
            ))
        
        console.print(f"\nDetailed report saved to: {output_file}")
        
    except Exception as e:
        console.print(f"[red]Error during data validation: {e}[/red]")
        logger.error(f"Data validation failed: {e}")
        raise click.ClickException(f"Data validation failed: {e}")


if __name__ == "__main__":
    data()