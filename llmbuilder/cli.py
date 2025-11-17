"""
Command-line interface for LLMBuilder.

This module provides the main CLI entry point for the llmbuilder package.
"""

import click

from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="llmbuilder")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx, verbose):
    """
    🤖 LLMBuilder - A comprehensive toolkit for building, training, and deploying language models.

    Built with ❤️   by  Qub△se  - Empowering developers to create amazing AI applications.

    Use 'llmbuilder COMMAND --help' for more information on a specific command.

    \b
    Quick Start:
      llmbuilder info                    # Show package information
      llmbuilder config create --help    # Create configuration files
      llmbuilder train model --help      # Train a new model
      llmbuilder generate text --help    # Generate text with trained models

    \b
    Visit https://qubase.in for more resources and documentation.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        click.echo("🔧 Verbose mode enabled")


@main.command()
def welcome():
    """🎉 Welcome to LLMBuilder! Get started with guided setup."""
    click.echo("🎉 " + click.style("Welcome to LLMBuilder!", fg="blue", bold=True))
    click.echo("Built with ❤️  by " + click.style("Qub△se", fg="cyan", bold=True))
    click.echo()

    click.echo("🚀 " + click.style("What would you like to do?", fg="green", bold=True))

    options = [
        ("1", "📖 Learn about LLMBuilder", "info"),
        ("2", "⚙️  Create a configuration file", "config create --interactive"),
        ("3", "📁 Load and preprocess data", "data load --interactive"),
        ("4", " англи Train a tokenizer", "data tokenizer --interactive"),
        ("5", "🧠 Train a new model", "train model --interactive"),
        ("6", "🎯 Generate text with a model", "generate text --setup"),
        ("7", "📚 View all available commands", "--help"),
    ]

    for num, desc, _ in options:
        click.echo(f"  {num}. {desc}")

    click.echo()
    choice = click.prompt(
        "Choose an option (1-7)", type=click.Choice([str(i) for i in range(1, 8)])
    )

    _, _, command = options[int(choice) - 1]

    if command == "info":
        # Call the info command directly instead of invoking context
        from .cli import info
        info()
    elif command == "--help":
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
    else:
        click.echo(f"\n🚀 Running: llmbuilder {command}")
        click.echo("💡 You can run this command directly next time!")
        click.echo()
        # Note: In a real implementation, you'd invoke the actual command
        click.echo(f"Command to run: llmbuilder {command}")


@main.command()
@click.argument('project_name')
@click.option('--template', '-t', default='basic', help='Project template to use (basic, advanced, cpu_optimized)')
def init(project_name, template):
    """🚀 Initialize a new LLMBuilder project with the given name."""
    import json
    import os
    from pathlib import Path
    
    # Create project directory
    project_path = Path(project_name)
    project_path.mkdir(exist_ok=True)
    
    click.echo(f"🚀 Creating new LLMBuilder project: {click.style(project_name, fg='blue', bold=True)}")
    
    # Create basic project structure
    directories = [
        "data/raw",
        "data/processed",
        "tokenizer",
        "models/checkpoints",
        "models/final",
        "configs",
        "outputs"
    ]
    
    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)
    
    # Create a basic README with step-by-step instructions
    readme_content = f"""# {project_name}

This is an LLM project created with LLMBuilder.

## Project Structure

- `data/` - Data files
- `tokenizer/` - Tokenizer files
- `models/` - Model checkpoints and final models
- `configs/` - Configuration files
- `outputs/` - Output files

## Step-by-Step Workflow

### 1. Prepare Your Data
Place your text data files in the `data/raw/` directory. Supported formats include:
- Plain text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx)
- Markdown files (.md)

### 2. Process Your Data
Clean and prepare your data for training:

```bash
llmbuilder data load -i data/raw -o data/processed/input.txt --clean
```

### 3. Remove Duplicates (Optional)
Remove duplicate content to improve training quality:

```bash
llmbuilder data deduplicate -i data/processed/input.txt -o data/processed/clean.txt --method both
```

### 4. Train a Tokenizer
Create a custom tokenizer for your data:

```bash
llmbuilder tokenizer train -i data/processed/clean.txt -o tokenizer/ --vocab-size 16000
```

### 5. Train Your Model
Train a language model using your processed data:

```bash
llmbuilder train model -d data/processed/clean.txt -t tokenizer/ -o models/checkpoints
```

### 6. Generate Text
Generate text using your trained model:

```bash
llmbuilder generate text -m models/checkpoints/latest.pt -t tokenizer/ -p "Your prompt here"
```

### 7. Export for Deployment (Optional)
Convert your model to GGUF format for deployment:

```bash
llmbuilder export gguf models/checkpoints/latest.pt -o models/final/model.gguf -q Q8_0
```

## Configuration
A basic configuration file has been created at `configs/config.json`. You can modify this file to adjust:
- Model architecture parameters
- Training hyperparameters
- Data processing settings

## Next Steps
1. Add your data to `data/raw/`
2. Follow the steps above in order
3. Experiment with different configurations
4. Fine-tune your model for specific tasks

For more detailed help with any command, use:
```bash
llmbuilder COMMAND --help
```
"""
    
    with open(project_path / "README.md", "w") as f:
        f.write(readme_content)
    
    # Copy the template config file
    template_path = Path(__file__).parent / "config" / "templates" / f"{template}_config.json"
    config_path = project_path / "configs" / "config.json"
    
    if template_path.exists():
        import shutil
        shutil.copy(template_path, config_path)
        # Update the project name in the config
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        if "project" not in config_data:
            config_data["project"] = {}
        config_data["project"]["name"] = project_name
        config_data["project"]["version"] = "0.1.0"
        
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
    else:
        # Fallback to basic config if template not found
        config_content = f"""{{
  "project": {{
    "name": "{project_name}",
    "version": "0.1.0"
  }},
  "model": {{
    "vocab_size": 16000,
    "embedding_dim": 384,
    "num_layers": 6,
    "num_heads": 6,
    "max_seq_length": 512,
    "dropout": 0.1
  }},
  "training": {{
    "batch_size": 16,
    "learning_rate": 3e-4,
    "num_epochs": 5
  }},
  "data": {{
    "max_length": 512
  }}
}}"""
        
        with open(config_path, "w") as f:
            f.write(config_content)
    
    click.echo(f"✅ Project structure created at {click.style(str(project_path.absolute()), fg='green')}")
    click.echo()
    click.echo("📁 Project structure:")
    click.echo(f"  {project_name}/")
    for directory in directories:
        click.echo(f"    {directory}/")
    click.echo("    README.md")
    click.echo("    configs/config.json")
    click.echo()
    click.echo("💡 Next steps:")
    click.echo(f"  cd {project_name}")
    click.echo("  # Add your data to data/raw/")
    click.echo("  # Process data with: llmbuilder data load -i data/raw -o data/processed/input.txt")


@main.command()
def info():
    """Display package information and credits."""
    click.echo(
        "🤖 "
        + click.style("LLMBuilder", fg="blue", bold=True)
        + f" version {__version__}"
    )
    click.echo()
    click.echo("📖 " + click.style("Description:", fg="green", bold=True))
    click.echo(
        "   A comprehensive toolkit for building, training, and deploying language models."
    )
    click.echo()
    click.echo("🏢 " + click.style("Built by Qub△se:", fg="cyan", bold=True))
    click.echo(
        "   Qub△se is dedicated to empowering developers with cutting-edge AI tools."
    )
    click.echo(
        "   Visit us at: " + click.style("https://Qubase.in", fg="blue", underline=True)
    )
    click.echo()
    click.echo("📦 " + click.style("Available modules:", fg="yellow", bold=True))
    modules = [
        ("data", "Data loading and preprocessing from various formats"),
        ("tokenizer", "Tokenizer training and management with SentencePiece"),
        ("model", "GPT-style transformer model building and management"),
        ("training", "Full training pipeline with checkpointing and metrics"),
        ("finetune", "Fine-tuning with LoRA and domain adaptation"),
        ("inference", "Text generation with various sampling strategies"),
        ("export", "Model export to GGUF, ONNX, and quantized formats"),
        ("config", "Configuration management for different hardware setups"),
        ("utils", "Common utilities for logging and checkpoint management"),
    ]

    for module, description in modules:
        click.echo(f"  • {click.style(module, fg='cyan'):<12} - {description}")

    click.echo()
    click.echo("🚀 " + click.style("Quick Commands:", fg="magenta", bold=True))
    click.echo("   llmbuilder config create --preset cpu_small -o config.json")
    click.echo(
        "   llmbuilder train model --data data.txt --tokenizer tokenizer/ --output model/"
    )
    click.echo(
        "   llmbuilder generate text -m model.pt -t tokenizer/ -p 'Hello world' --interactive"
    )
    click.echo()
    click.echo(
        "💡 "
        + click.style("Need help?", fg="green", bold=True)
        + " Use --help with any command for detailed information."
    )
    click.echo("📚 Documentation: https://github.com/Qubasehq/llmbuilder/wiki")
    click.echo("🐛 Report issues: https://github.com/Qubasehq/llmbuilder/issues")


# Duplicate info command removed - keeping the more detailed version above


# Data processing commands
@main.group()
def data():
    """Data loading and preprocessing commands."""
    pass


@data.command()
@click.option("--input", "-i", help="Input file or directory path")
@click.option("--output", "-o", help="Output file path")
@click.option(
    "--format",
    type=click.Choice(["txt", "pdf", "docx", "all"]),
    default="all",
    help="File format to process",
)
@click.option("--clean", is_flag=True, help="Clean text during processing")
@click.option("--min-length", default=50, help="Minimum text length to keep")
@click.option("--interactive", is_flag=True, help="Interactive mode with prompts")
def load(input, output, format, clean, min_length, interactive):
    """📁 Load and preprocess text data from various formats."""
    from pathlib import Path

    from .data import DataLoader

    # Interactive mode
    if interactive or not input or not output:
        click.echo("🤖 " + click.style("Interactive Data Loading", fg="blue", bold=True))
        click.echo("Let's set up your data loading configuration!\n")

        if not input:
            input = click.prompt("📂 Input file or directory path", type=str)
        if not output:
            output = click.prompt("💾 Output file path", type=str)

        if click.confirm("🧹 Would you like to clean the text during processing?"):
            clean = True

        min_length = click.prompt(
            "📏 Minimum text length to keep", default=min_length, type=int
        )

        click.echo(f"\n🚀 Starting data loading with your settings...")

    click.echo(f"📖 Loading data from {click.style(input, fg='cyan')}...")

    try:
        loader = DataLoader(min_length=min_length, clean_text=clean)
        input_path = Path(input)

        if input_path.is_file():
            with click.progressbar(length=1, label="Processing file") as bar:
                text = loader.load_file(input_path)
                bar.update(1)

            if text:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(text)
                click.echo(
                    f"✅ Processed file saved to {click.style(output, fg='green')}"
                )
                click.echo(f"📊 Text length: {len(text):,} characters")
            else:
                click.echo("❌ Failed to load file")
        elif input_path.is_dir():
            texts = []
            files = list(input_path.rglob("*"))
            supported_files = [
                f
                for f in files
                if f.suffix.lower() in loader.get_supported_extensions()
            ]

            if not supported_files:
                click.echo("❌ No supported files found in directory")
                return

            with click.progressbar(supported_files, label="Processing files") as bar:
                for file_path in bar:
                    try:
                        text = loader.load_file(file_path)
                        if text:
                            texts.append(text)
                            if interactive:
                                click.echo(f"  ✅ {file_path.name}")
                    except Exception as e:
                        if interactive:
                            click.echo(f"  ❌ {file_path.name}: {e}")

            if texts:
                with open(output, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(texts))
                total_chars = sum(len(text) for text in texts)
                click.echo(
                    f"✅ Combined {len(texts)} files saved to {click.style(output, fg='green')}"
                )
                click.echo(f"📊 Total text length: {total_chars:,} characters")
            else:
                click.echo("❌ No files processed successfully")
        else:
            click.echo(f"❌ Input path not found: {input}")

    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if interactive:
            click.echo("💡 Try checking your file paths and permissions.")


@data.command()
@click.option("--input", "-i", required=True, help="Input text file")
@click.option("--output", "-o", required=True, help="Output directory for tokenizer")
@click.option("--vocab-size", default=16000, help="Vocabulary size")
@click.option(
    "--model-type",
    type=click.Choice(["bpe", "unigram", "word", "char"]),
    default="bpe",
    help="Tokenizer model type",
)
def tokenizer(input, output, vocab_size, model_type):
    """Train a tokenizer on text data."""
    from pathlib import Path

    from .config import TokenizerConfig
    from .tokenizer import TokenizerTrainer

    click.echo(f"Training {model_type} tokenizer with vocab size {vocab_size}...")

    config = TokenizerConfig(vocab_size=vocab_size, model_type=model_type)
    trainer = TokenizerTrainer(config=config)

    try:
        results = trainer.train(input_file=input, output_dir=output)
        click.echo(f"✓ Tokenizer training completed!")
        click.echo(f"  Model: {results['model_file']}")
        click.echo(f"  Vocab: {results['vocab_file']}")
        click.echo(f"  Training time: {results['training_time']:.1f}s")
    except Exception as e:
        click.echo(f"✗ Tokenizer training failed: {e}")


@data.command()
@click.option("--input", "-i", required=True, help="Input directory or file path")
@click.option(
    "--output", "-o", required=True, help="Output directory for processed files"
)
@click.option(
    "--formats",
    multiple=True,
    type=click.Choice(["html", "markdown", "epub", "pdf", "all"]),
    default=["all"],
    help="File formats to process",
)
@click.option(
    "--batch-size", default=100, help="Number of files to process in each batch"
)
@click.option("--workers", default=4, help="Number of worker processes")
@click.option("--ocr-fallback", is_flag=True, help="Enable OCR fallback for PDFs")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def ingest(input, output, formats, batch_size, workers, ocr_fallback, verbose):
    """🔄 Ingest and process multi-format documents."""
    from pathlib import Path

    from .data.ingest import IngestionPipeline

    click.echo(
        "🔄 " + click.style("Multi-Format Document Ingestion", fg="blue", bold=True)
    )
    click.echo(
        "Built by "
        + click.style("Qub△se", fg="cyan", bold=True)
        + " - Processing your documents!\n"
    )

    if verbose:
        click.echo(f"📁 Input: {input}")
        click.echo(f"💾 Output: {output}")
        click.echo(f"📄 Formats: {', '.join(formats)}")
        click.echo(f"👥 Workers: {workers}")
        click.echo(f"📦 Batch size: {batch_size}")
        click.echo(f"🔍 OCR fallback: {'enabled' if ocr_fallback else 'disabled'}")
        click.echo()

    try:
        # Initialize ingestion pipeline
        pipeline = IngestionPipeline(
            output_dir=output,
            batch_size=batch_size,
            num_workers=workers,
            enable_ocr=ocr_fallback,
        )

        # Process formats
        if "all" in formats:
            formats = ["html", "markdown", "epub", "pdf"]

        input_path = Path(input)
        if input_path.is_file():
            click.echo(f"📄 Processing single file: {input_path.name}")
            results = pipeline.process_file(input_path)
        else:
            click.echo(f"📁 Processing directory: {input}")
            results = pipeline.process_directory(input_path, file_types=list(formats))

        # Report results
        if results:
            click.echo(
                f"\n✅ " + click.style("Ingestion completed!", fg="green", bold=True)
            )
            click.echo(f"📊 Files processed: {results.get('files_processed', 0)}")
            click.echo(f"✅ Successful: {results.get('successful', 0)}")
            click.echo(f"❌ Failed: {results.get('failed', 0)}")
            click.echo(f"📁 Output directory: {click.style(output, fg='cyan')}")

            if verbose and results.get("errors"):
                click.echo(f"\n❌ " + click.style("Errors encountered:", fg="red"))
                for error in results["errors"][:5]:  # Show first 5 errors
                    click.echo(f"  • {error}")
                if len(results["errors"]) > 5:
                    click.echo(f"  ... and {len(results['errors']) - 5} more")
        else:
            click.echo("❌ " + click.style("Ingestion failed!", fg="red"))

    except Exception as e:
        click.echo(f"❌ Ingestion failed: {e}")
        if verbose:
            import traceback

            click.echo(f"\n🔍 " + click.style("Full error trace:", fg="red"))
            click.echo(traceback.format_exc())


@data.command()
@click.option(
    "--input", "-i", required=True, help="Input file or directory with text data"
)
@click.option("--output", "-o", required=True, help="Output file for deduplicated data")
@click.option(
    "--method",
    type=click.Choice(["exact", "semantic", "both"]),
    default="both",
    help="Deduplication method",
)
@click.option(
    "--similarity-threshold",
    default=0.85,
    help="Similarity threshold for semantic deduplication (0.0-1.0)",
)
@click.option("--batch-size", default=1000, help="Batch size for processing")
@click.option(
    "--embedding-model",
    default="all-MiniLM-L6-v2",
    help="Model for semantic embeddings",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def deduplicate(
    input, output, method, similarity_threshold, batch_size, embedding_model, verbose
):
    """🔍 Remove duplicate content from text data."""
    from pathlib import Path

    from .data.dedup import DeduplicationPipeline

    click.echo("🔍 " + click.style("Text Deduplication", fg="blue", bold=True))
    click.echo(
        "Built by "
        + click.style("Qub△se", fg="cyan", bold=True)
        + " - Cleaning your data!\n"
    )

    if verbose:
        click.echo(f"📁 Input: {input}")
        click.echo(f"💾 Output: {output}")
        click.echo(f"🔧 Method: {method}")
        click.echo(f"📊 Similarity threshold: {similarity_threshold}")
        click.echo(f"📦 Batch size: {batch_size}")
        click.echo(f"🤖 Embedding model: {embedding_model}")
        click.echo()

    try:
        # Initialize deduplication pipeline
        pipeline = DeduplicationPipeline(
            similarity_threshold=similarity_threshold,
            batch_size=batch_size,
            embedding_model=embedding_model,
        )

        # Load input data
        input_path = Path(input)
        if input_path.is_file():
            with open(input_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            # Load from directory
            texts = []
            for file_path in input_path.rglob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.extend([line.strip() for line in f if line.strip()])

        click.echo(f"📄 Loaded {len(texts)} text entries")

        # Perform deduplication
        if method in ["exact", "both"]:
            click.echo("🔍 Performing exact deduplication...")
            texts = pipeline.remove_exact_duplicates(texts)
            click.echo(f"📊 After exact deduplication: {len(texts)} entries")

        if method in ["semantic", "both"]:
            click.echo("🧠 Performing semantic deduplication...")
            with click.progressbar(length=1, label="Computing embeddings") as bar:
                texts = pipeline.remove_semantic_duplicates(texts)
                bar.update(1)
            click.echo(f"📊 After semantic deduplication: {len(texts)} entries")

        # Save results
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")

        click.echo(
            f"\n✅ " + click.style("Deduplication completed!", fg="green", bold=True)
        )
        click.echo(f"📁 Output saved to: {click.style(str(output_path), fg='cyan')}")
        click.echo(f"📊 Final count: {len(texts)} unique entries")

    except Exception as e:
        click.echo(f"❌ Deduplication failed: {e}")
        if verbose:
            import traceback

            click.echo(f"\n🔍 " + click.style("Full error trace:", fg="red"))
            click.echo(traceback.format_exc())


# Tokenizer training commands
@main.group()
def tokenizer():
    """Tokenizer training and management commands."""
    pass


@tokenizer.command()
@click.option("--input", "-i", required=True, help="Input text file or directory")
@click.option("--output", "-o", required=True, help="Output directory for tokenizer")
@click.option("--vocab-size", default=16000, help="Vocabulary size")
@click.option(
    "--algorithm",
    type=click.Choice(["bpe", "unigram", "wordpiece", "sentencepiece"]),
    default="bpe",
    help="Tokenization algorithm",
)
@click.option(
    "--special-tokens", multiple=True, help="Special tokens to add (e.g., <pad>, <unk>)"
)
@click.option("--min-frequency", default=2, help="Minimum token frequency")
@click.option("--coverage", default=0.9995, help="Character coverage for SentencePiece")
@click.option("--validate", is_flag=True, help="Validate trained tokenizer")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def train(
    input,
    output,
    vocab_size,
    algorithm,
    special_tokens,
    min_frequency,
    coverage,
    validate,
    verbose,
):
    """🔤 Train a tokenizer on text data."""
    from pathlib import Path

    from .training.tokenizer import TokenizerConfig, TokenizerTrainer

    click.echo("🔤 " + click.style("Tokenizer Training", fg="blue", bold=True))
    click.echo(
        "Built by "
        + click.style("Qub△se", fg="cyan", bold=True)
        + " - Training your tokenizer!\n"
    )

    if verbose:
        click.echo(f"📁 Input: {input}")
        click.echo(f"💾 Output: {output}")
        click.echo(f"📊 Vocab size: {vocab_size}")
        click.echo(f"🔧 Algorithm: {algorithm}")
        click.echo(
            f"🏷️  Special tokens: {', '.join(special_tokens) if special_tokens else 'default'}"
        )
        click.echo(f"📈 Min frequency: {min_frequency}")
        if algorithm == "sentencepiece":
            click.echo(f"📊 Coverage: {coverage}")
        click.echo()

    try:
        # Create configuration
        config = TokenizerConfig(
            vocab_size=vocab_size,
            algorithm=algorithm,
            special_tokens=list(special_tokens) if special_tokens else None,
            min_frequency=min_frequency,
            character_coverage=coverage,
        )

        # Initialize trainer
        trainer = TokenizerTrainer(config=config)

        # Train tokenizer
        click.echo(f"🚀 Training {algorithm} tokenizer...")
        with click.progressbar(length=1, label="Training") as bar:
            results = trainer.train(input_file=input, output_dir=output)
            bar.update(1)

        click.echo(f"\n✅ " + click.style("Training completed!", fg="green", bold=True))
        click.echo(
            f"📄 Model file: {click.style(results.get('model_file', 'N/A'), fg='cyan')}"
        )
        click.echo(
            f"📚 Vocab file: {click.style(results.get('vocab_file', 'N/A'), fg='cyan')}"
        )
        click.echo(f"⏱️  Training time: {results.get('training_time', 0):.1f}s")
        click.echo(f"📊 Final vocab size: {results.get('final_vocab_size', vocab_size)}")

        # Validation
        if validate:
            click.echo(f"\n🔍 Validating tokenizer...")
            validation_results = trainer.validate_tokenizer(output)
            if validation_results.get("valid", False):
                click.echo(f"✅ " + click.style("Validation passed", fg="green"))
                click.echo(
                    f"📊 Test samples: {validation_results.get('test_samples', 0)}"
                )
            else:
                click.echo(
                    f"⚠️  " + click.style("Validation issues found", fg="yellow")
                )
                for issue in validation_results.get("issues", []):
                    click.echo(f"  • {issue}")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        if verbose:
            import traceback

            click.echo(f"\n🔍 " + click.style("Full error trace:", fg="red"))
            click.echo(traceback.format_exc())


@tokenizer.command()
@click.argument("tokenizer_path")
@click.option("--text", "-t", help="Test text to tokenize")
@click.option("--file", "-f", help="Test file to tokenize")
@click.option("--interactive", "-i", is_flag=True, help="Interactive testing mode")
def test(tokenizer_path, text, file, interactive):
    """🧪 Test a trained tokenizer."""
    from .training.tokenizer import TokenizerTrainer

    click.echo("🧪 " + click.style("Tokenizer Testing", fg="blue", bold=True))

    try:
        trainer = TokenizerTrainer()
        tokenizer = trainer.load_tokenizer(tokenizer_path)

        if interactive:
            click.echo(
                "🎮 "
                + click.style("Interactive mode", fg="green")
                + " - Type 'quit' to exit"
            )
            while True:
                test_text = click.prompt(
                    "Enter text to tokenize", default="", show_default=False
                )
                if test_text.lower() in ["quit", "exit", "q"]:
                    break
                if test_text:
                    tokens = tokenizer.encode(test_text)
                    decoded = tokenizer.decode(tokens)
                    click.echo(f"🔤 Tokens: {tokens}")
                    click.echo(f"📝 Decoded: {decoded}")
                    click.echo(f"📊 Token count: {len(tokens)}")
                    click.echo()

        elif text:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            click.echo(f"📝 Input: {text}")
            click.echo(f"🔤 Tokens: {tokens}")
            click.echo(f"📝 Decoded: {decoded}")
            click.echo(f"📊 Token count: {len(tokens)}")

        elif file:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            tokens = tokenizer.encode(content)
            click.echo(f"📄 File: {file}")
            click.echo(f"📊 Character count: {len(content)}")
            click.echo(f"📊 Token count: {len(tokens)}")
            click.echo(f"📈 Compression ratio: {len(content) / len(tokens):.2f}")

        else:
            click.echo("❌ Please provide --text, --file, or --interactive option")

    except Exception as e:
        click.echo(f"❌ Testing failed: {e}")


# Model training commands
@main.group()
def train():
    """Model training commands."""
    pass


@train.command()
@click.option("--config", "-c", help="Path to training configuration file")
@click.option("--data", "-d", required=True, help="Path to training data")
@click.option("--tokenizer", "-t", required=True, help="Path to tokenizer directory")
@click.option("--output", "-o", required=True, help="Output directory for model")
@click.option("--epochs", default=10, help="Number of training epochs")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--vocab-size", default=16000, help="Vocabulary size")
@click.option("--layers", default=8, help="Number of transformer layers")
@click.option("--heads", default=8, help="Number of attention heads")
@click.option("--dim", default=512, help="Embedding dimension")
def model(
    config,
    data,
    tokenizer,
    output,
    epochs,
    batch_size,
    lr,
    vocab_size,
    layers,
    heads,
    dim,
):
    """Train a new language model from scratch."""
    from pathlib import Path

    from .config import ModelConfig, TrainingConfig
    from .data import TextDataset
    from .model import build_model
    from .training import train_model

    click.echo("Setting up model training...")

    # Create model config
    model_config = ModelConfig(
        vocab_size=vocab_size,
        num_layers=layers,
        num_heads=heads,
        embedding_dim=dim,
        max_seq_length=1024,
        dropout=0.1,
    )

    # Create training config
    training_config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        save_every=1000,
        eval_every=500,
    )

    click.echo(f"Building model: {layers} layers, {heads} heads, {dim} dim...")
    model = build_model(model_config)

    click.echo(f"Loading dataset from {data}...")
    dataset = TextDataset(data, block_size=model_config.max_seq_length)

    click.echo("Starting training...")
    try:
        results = train_model(model, dataset, training_config, checkpoint_dir=output)
        click.echo(f"✓ Training completed!")
        click.echo(f"  Final loss: {results.get('final_loss', 'N/A')}")
        click.echo(f"  Model saved to: {output}")
    except Exception as e:
        click.echo(f"✗ Training failed: {e}")


@train.command()
@click.option(
    "--checkpoint", "-c", required=True, help="Path to checkpoint to resume from"
)
@click.option("--data", "-d", required=True, help="Path to training data")
@click.option("--output", "-o", help="Output directory (defaults to checkpoint dir)")
def resume(checkpoint, data, output):
    """Resume training from a checkpoint."""
    from .data import TextDataset
    from .training import resume_training

    click.echo(f"Resuming training from {checkpoint}...")

    try:
        dataset = TextDataset(data)
        results = resume_training(checkpoint, dataset, output_dir=output)
        click.echo(f"✓ Training resumed and completed!")
        click.echo(f"  Final loss: {results.get('final_loss', 'N/A')}")
    except Exception as e:
        click.echo(f"✗ Resume training failed: {e}")


@main.group()
def finetune():
    """Fine-tuning commands."""
    pass


@finetune.command()
@click.option("--model", "-m", required=True, help="Path to pre-trained model")
@click.option("--dataset", "-d", required=True, help="Path to fine-tuning dataset")
@click.option(
    "--output", "-o", required=True, help="Output directory for fine-tuned model"
)
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--lr", default=5e-5, help="Learning rate")
@click.option("--batch-size", default=4, help="Batch size")
@click.option("--use-lora", is_flag=True, help="Use LoRA fine-tuning")
@click.option("--lora-rank", default=4, help="LoRA rank")
def model(model, dataset, output, epochs, lr, batch_size, use_lora, lora_rank):
    """Fine-tune a pre-trained model."""
    from .data import TextDataset
    from .finetune import FineTuningConfig, finetune_model
    from .model import load_model

    click.echo(f"Loading model from {model}...")
    model_obj = load_model(model)

    click.echo(f"Loading dataset from {dataset}...")
    dataset_obj = TextDataset(dataset)

    config = FineTuningConfig(
        num_epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        use_lora=use_lora,
        lora_rank=lora_rank,
    )

    click.echo("Starting fine-tuning...")
    results = finetune_model(model_obj, dataset_obj, config, checkpoint_dir=output)

    click.echo(f"Fine-tuning completed!")
    click.echo(f"Best validation loss: {results['best_val_loss']:.4f}")
    click.echo(f"Final model saved to: {results.get('final_model_path', output)}")


@main.group()
def generate():
    """Text generation commands.

    Includes an interactive mode. See the 'text' subcommand and use the
    --interactive option to start an interactive prompt for generation.
    """
    pass


@generate.command()
@click.option("--model", "-m", help="Path to trained model checkpoint")
@click.option("--tokenizer", "-t", help="Path to tokenizer directory")
@click.option("--prompt", "-p", help="Text prompt for generation")
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.option("--max-tokens", default=100, help="Maximum tokens to generate")
@click.option("--temperature", default=0.8, help="Sampling temperature (0.1-2.0)")
@click.option("--top-k", default=50, help="Top-k sampling (1-100)")
@click.option("--top-p", default=0.9, help="Top-p sampling (0.1-1.0)")
@click.option("--device", help="Device to use (cpu/cuda)")
@click.option("--setup", is_flag=True, help="Interactive setup mode")
def text(
    model,
    tokenizer,
    prompt,
    interactive,
    max_tokens,
    temperature,
    top_k,
    top_p,
    device,
    setup,
):
    """🎯 Generate text using a trained model with advanced sampling options."""
    from .inference import GenerationConfig, generate_text, interactive_cli

    # Interactive setup mode
    if setup or not model or not tokenizer:
        click.echo(
            "🤖 " + click.style("LLMBuilder Text Generation Setup", fg="blue", bold=True)
        )
        click.echo(
            "Built by "
            + click.style("Qub△se", fg="cyan", bold=True)
            + " - Let's generate some amazing text!\n"
        )

        if not model:
            model = click.prompt("🧠 Path to your trained model checkpoint", type=str)
        if not tokenizer:
            tokenizer = click.prompt("🔤 Path to your tokenizer directory", type=str)

        if not interactive and not prompt:
            mode = click.prompt(
                "🎮 Choose generation mode",
                type=click.Choice(["interactive", "single"]),
                default="interactive",
            )
            interactive = mode == "interactive"

        if not interactive and not prompt:
            prompt = click.prompt("💭 Enter your text prompt", type=str)

        # Advanced settings
        if click.confirm("⚙️  Would you like to customize generation settings?"):
            max_tokens = click.prompt(
                "📏 Maximum tokens to generate", default=max_tokens, type=int
            )
            temperature = click.prompt(
                "🌡️  Temperature (creativity: 0.1=focused, 2.0=creative)",
                default=temperature,
                type=float,
            )
            top_k = click.prompt(
                "🔝 Top-k sampling (diversity)", default=top_k, type=int
            )
            top_p = click.prompt(
                "🎯 Top-p sampling (nucleus)", default=top_p, type=float
            )

        click.echo(f"\n🚀 Starting text generation...")

    config = GenerationConfig(
        max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p
    )

    try:
        if interactive:
            click.echo(
                "🎮 " + click.style("Interactive Text Generation", fg="green", bold=True)
            )
            click.echo("💡 Type your prompts and watch the AI respond!")
            click.echo("🔧 You can adjust settings in real-time during the session.\n")
            interactive_cli(model, tokenizer, device=device, config=config)
        elif prompt:
            click.echo(f"💭 Prompt: {click.style(prompt, fg='yellow')}")
            click.echo("🤔 Generating...")

            with click.progressbar(length=1, label="Thinking") as bar:
                result = generate_text(
                    model, tokenizer, prompt, device=device, **config.__dict__
                )
                bar.update(1)

            click.echo(f"\n🎯 " + click.style("Generated Text:", fg="green", bold=True))
            click.echo("─" * 50)
            click.echo(result)
            click.echo("─" * 50)
            click.echo(
                f"📊 Settings: temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_tokens}"
            )
            click.echo(
                "\n💡 "
                + click.style("Tip:", fg="blue")
                + " Use --interactive for a chat-like experience!"
            )
        else:
            click.echo("❌ Please provide either --prompt or --interactive flag")
            click.echo("💡 Or use --setup for guided configuration")
            click.echo("📚 Use 'llmbuilder generate text --help' for more information")

    except Exception as e:
        click.echo(f"❌ Generation failed: {e}")
        click.echo("💡 " + click.style("Troubleshooting tips:", fg="blue"))
        click.echo("   • Check that your model and tokenizer paths are correct")
        click.echo("   • Ensure your model and tokenizer are compatible")
        click.echo("   • Try reducing max_tokens if you're running out of memory")
        click.echo(
            "   • Visit https://github.com/Qubasehq/llmbuilder/wiki for help"
        )


# Model management commands
@main.group()
def model():
    """Model management commands."""
    pass


@model.command()
@click.option("--config", "-c", help="Path to model configuration file")
@click.option("--vocab-size", default=16000, help="Vocabulary size")
@click.option("--layers", default=8, help="Number of transformer layers")
@click.option("--heads", default=8, help="Number of attention heads")
@click.option("--dim", default=512, help="Embedding dimension")
@click.option("--output", "-o", required=True, help="Output path for model")
def create(config, vocab_size, layers, heads, dim, output):
    """Create a new model with specified architecture."""
    from .config import ModelConfig
    from .model import build_model, save_model

    click.echo("Creating new model...")

    model_config = ModelConfig(
        vocab_size=vocab_size,
        num_layers=layers,
        num_heads=heads,
        embedding_dim=dim,
        max_seq_length=1024,
        dropout=0.1,
    )

    model = build_model(model_config)
    save_model(model, output)

    num_params = sum(p.numel() for p in model.parameters())
    click.echo(f"✓ Model created with {num_params:,} parameters")
    click.echo(f"✓ Model saved to {output}")


@model.command()
@click.argument("model_path")
def info(model_path):
    """Display information about a model."""
    from .model import load_model

    click.echo(f"Loading model from {model_path}...")

    try:
        model = load_model(model_path)
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        click.echo(f"✓ Model loaded successfully")
        click.echo(f"  Total parameters: {num_params:,}")
        click.echo(f"  Trainable parameters: {trainable_params:,}")
        click.echo(f"  Model type: {type(model).__name__}")

        if hasattr(model, "get_metadata"):
            metadata = model.get_metadata()
            click.echo(
                f"  Architecture: {metadata.n_layer} layers, {metadata.n_head} heads"
            )
            click.echo(f"  Embedding dim: {metadata.n_embd}")
            click.echo(f"  Vocab size: {metadata.vocab_size}")
            click.echo(f"  Max sequence length: {metadata.block_size}")
    except Exception as e:
        click.echo(f"✗ Failed to load model: {e}")


@model.command()
@click.argument("model_path")
@click.option("--dataset", "-d", required=True, help="Path to evaluation dataset")
@click.option("--batch-size", default=32, help="Batch size for evaluation")
def evaluate(model_path, dataset, batch_size):
    """Evaluate a model on a dataset."""
    from .data import TextDataset
    from .model import load_model
    from .training import evaluate_model

    click.echo(f"Loading model from {model_path}...")
    model = load_model(model_path)

    click.echo(f"Loading dataset from {dataset}...")
    eval_dataset = TextDataset(dataset)

    click.echo("Evaluating model...")
    try:
        results = evaluate_model(model, eval_dataset, batch_size=batch_size)
        click.echo(f"✓ Evaluation completed!")
        click.echo(f"  Perplexity: {results.get('perplexity', 'N/A')}")
        click.echo(f"  Loss: {results.get('loss', 'N/A')}")
    except Exception as e:
        click.echo(f"✗ Evaluation failed: {e}")


# Export commands
@main.group()
def export():
    """Model export commands."""
    pass


@export.command()
@click.argument("model_path")
@click.option("--output", "-o", required=True, help="Output path for GGUF model")
@click.option(
    "--quantization",
    type=click.Choice(["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "F16", "F32"]),
    default="Q8_0",
    help="Quantization level",
)
@click.option("--validate", is_flag=True, help="Validate conversion result")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def gguf(model_path, output, quantization, validate, verbose):
    """Export model to GGUF format for llama.cpp compatibility."""
    import time

    from .tools.convert_to_gguf import GGUFConverter

    if verbose:
        click.echo(f"🔧 Verbose mode enabled")
        click.echo(f"📁 Input model: {model_path}")
        click.echo(f"💾 Output path: {output}")
        click.echo(f"⚙️  Quantization: {quantization}")
        click.echo(f"✅ Validation: {'enabled' if validate else 'disabled'}")
        click.echo()

    click.echo(f"🚀 Converting model to GGUF format...")

    try:
        converter = GGUFConverter()

        if verbose:
            # Show available conversion scripts
            scripts = converter.conversion_scripts
            click.echo(f"🔍 Available conversion scripts:")
            for script_type, path in scripts.items():
                status = "✅ found" if path else "❌ not found"
                click.echo(f"  • {script_type}: {status}")
                if path and verbose:
                    click.echo(f"    Path: {path}")
            click.echo()

        # Perform conversion
        start_time = time.time()
        result = converter.convert_model(model_path, output, quantization)

        if result.success:
            click.echo(
                f"✅ " + click.style("Conversion successful!", fg="green", bold=True)
            )
            click.echo(f"📄 Output file: {click.style(result.output_path, fg='cyan')}")
            click.echo(f"📊 File size: {result.file_size_bytes / (1024*1024):.1f} MB")
            click.echo(f"⏱️  Conversion time: {result.conversion_time_seconds:.1f}s")
            click.echo(f"🔧 Quantization: {result.quantization_level}")

            if validate or result.validation_passed:
                if result.validation_passed:
                    click.echo(f"✅ " + click.style("Validation passed", fg="green"))
                else:
                    click.echo(f"⚠️  " + click.style("Validation failed", fg="yellow"))

            if verbose:
                # Show additional file information
                file_info = converter.validator.get_file_info(result.output_path)
                click.echo(f"\n📋 " + click.style("File Details:", fg="blue", bold=True))
                for key, value in file_info.items():
                    if key != "error":
                        click.echo(f"  • {key}: {value}")
        else:
            click.echo(f"❌ " + click.style("Conversion failed!", fg="red", bold=True))
            if result.error_message:
                click.echo(f"💥 Error: {result.error_message}")

            # Provide troubleshooting tips
            click.echo(f"\n💡 " + click.style("Troubleshooting tips:", fg="blue"))
            click.echo("   • Ensure llama.cpp or convert_hf_to_gguf.py is available")
            click.echo("   • Check that the input model path is correct")
            click.echo("   • Verify you have sufficient disk space")
            click.echo("   • Try a different quantization level")

    except Exception as e:
        click.echo(f"❌ Conversion failed: {e}")
        if verbose:
            import traceback

            click.echo(f"\n🔍 " + click.style("Full error trace:", fg="red"))
            click.echo(traceback.format_exc())


@export.command()
@click.argument("model_path")
@click.option("--output", "-o", required=True, help="Output path for ONNX model")
@click.option("--opset", default=11, help="ONNX opset version")
def onnx(model_path, output, opset):
    """Export model to ONNX format for mobile/runtime inference."""
    from .export import export_onnx

    click.echo(f"Exporting model to ONNX format...")
    click.echo(f"  Input: {model_path}")
    click.echo(f"  Output: {output}")
    click.echo(f"  Opset version: {opset}")

    try:
        export_onnx(model_path, output, opset_version=opset)
        click.echo(f"✓ Model exported successfully to {output}")
    except Exception as e:
        click.echo(f"✗ Export failed: {e}")


@export.command()
@click.argument("model_path")
@click.option("--output", "-o", required=True, help="Output path for quantized model")
@click.option(
    "--method",
    type=click.Choice(["dynamic", "static", "qat"]),
    default="dynamic",
    help="Quantization method",
)
@click.option("--bits", type=click.Choice([8, 16]), default=8, help="Quantization bits")
def quantize(model_path, output, method, bits):
    """Quantize model for edge deployment."""
    from .export import quantize_model

    click.echo(f"Quantizing model...")
    click.echo(f"  Input: {model_path}")
    click.echo(f"  Output: {output}")
    click.echo(f"  Method: {method}")
    click.echo(f"  Bits: {bits}")

    try:
        quantize_model(model_path, output, method=method, bits=bits)
        click.echo(f"✓ Model quantized successfully to {output}")
    except Exception as e:
        click.echo(f"✗ Quantization failed: {e}")


# GGUF conversion commands
@main.group()
def convert():
    """Model conversion commands."""
    pass


@convert.command()
@click.argument("model_path")
@click.option("--output", "-o", required=True, help="Output path for GGUF model")
@click.option(
    "--quantization",
    "-q",
    type=click.Choice(["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "F16", "F32"]),
    default="Q8_0",
    help="Quantization level",
)
@click.option("--validate", is_flag=True, help="Validate conversion result")
@click.option("--batch", is_flag=True, help="Enable batch conversion mode")
@click.option(
    "--all-quantizations", is_flag=True, help="Convert with all quantization levels"
)
def gguf(model_path, output, quantization, validate, batch, all_quantizations):
    """🔄 Convert model to GGUF format with quantization options."""
    from pathlib import Path

    from .tools.convert_to_gguf import GGUFConverter

    click.echo("🔄 " + click.style("GGUF Model Conversion", fg="blue", bold=True))
    click.echo(
        "Built by "
        + click.style("Qub△se", fg="cyan", bold=True)
        + " - Converting your models for optimal inference!\n"
    )

    try:
        converter = GGUFConverter()

        # Show available quantization options
        options = converter.get_quantization_options()
        click.echo(f"⚙️  Available quantization levels: {', '.join(options)}")

        # Check conversion scripts availability
        scripts = converter.conversion_scripts
        available_scripts = [name for name, path in scripts.items() if path]
        if not available_scripts:
            click.echo(
                "❌ " + click.style("No conversion scripts found!", fg="red", bold=True)
            )
            click.echo(
                "💡 Please install llama.cpp or ensure convert_hf_to_gguf.py is available"
            )
            return

        click.echo(f"🔧 Using conversion scripts: {', '.join(available_scripts)}\n")

        if all_quantizations:
            # Convert with all quantization levels
            quantization_levels = ["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1"]
            click.echo(f"🔄 Converting with all quantization levels...")

            results = []
            for quant_level in quantization_levels:
                output_path = Path(output)
                quant_output = (
                    output_path.parent / f"{output_path.stem}_{quant_level}.gguf"
                )

                click.echo(f"  🔧 Converting with {quant_level}...")
                result = converter.convert_model(
                    model_path, str(quant_output), quant_level
                )
                results.append(result)

                if result.success:
                    click.echo(
                        f"    ✅ {quant_level}: {result.file_size_bytes / (1024*1024):.1f} MB"
                    )
                else:
                    click.echo(f"    ❌ {quant_level}: {result.error_message}")

            # Summary
            successful = [r for r in results if r.success]
            click.echo(
                f"\n📊 " + click.style("Conversion Summary:", fg="green", bold=True)
            )
            click.echo(f"✅ Successful: {len(successful)}/{len(results)}")

            if successful:
                total_size = sum(r.file_size_bytes for r in successful)
                click.echo(f"📁 Total size: {total_size / (1024*1024):.1f} MB")
                click.echo(
                    f"⏱️  Average time: {sum(r.conversion_time_seconds for r in successful) / len(successful):.1f}s"
                )

        else:
            # Single conversion
            click.echo(f"🔧 Converting with {quantization} quantization...")
            result = converter.convert_model(model_path, output, quantization)

            if result.success:
                click.echo(
                    f"✅ " + click.style("Conversion successful!", fg="green", bold=True)
                )
                click.echo(f"📄 Output: {click.style(result.output_path, fg='cyan')}")
                click.echo(f"📊 Size: {result.file_size_bytes / (1024*1024):.1f} MB")
                click.echo(f"⏱️  Time: {result.conversion_time_seconds:.1f}s")

                if validate:
                    click.echo(f"🔍 Validating conversion...")
                    if result.validation_passed:
                        click.echo(f"✅ " + click.style("Validation passed", fg="green"))
                    else:
                        click.echo(
                            f"⚠️  " + click.style("Validation failed", fg="yellow")
                        )
                        click.echo("💡 The file was created but may have issues")
            else:
                click.echo(
                    f"❌ " + click.style("Conversion failed!", fg="red", bold=True)
                )
                if result.error_message:
                    click.echo(f"💥 Error: {result.error_message}")

    except Exception as e:
        click.echo(f"❌ Conversion failed: {e}")


@convert.command()
@click.option(
    "--input-dir", "-i", required=True, help="Directory containing models to convert"
)
@click.option(
    "--output-dir", "-o", required=True, help="Output directory for GGUF models"
)
@click.option(
    "--quantization",
    "-q",
    multiple=True,
    type=click.Choice(["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "F16", "F32"]),
    help="Quantization levels (can specify multiple)",
)
@click.option(
    "--pattern", default="*", help='File pattern to match (e.g., "*.pt", "model_*")'
)
def batch(input_dir, output_dir, quantization, pattern):
    """🔄 Batch convert multiple models to GGUF format."""
    import glob
    from pathlib import Path

    from .tools.convert_to_gguf import GGUFConverter

    click.echo("🔄 " + click.style("Batch GGUF Conversion", fg="blue", bold=True))

    # Default quantization if none specified
    if not quantization:
        quantization = ["Q8_0"]

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find models to convert
    model_files = list(input_path.glob(pattern))
    if not model_files:
        click.echo(f"❌ No models found matching pattern '{pattern}' in {input_dir}")
        return

    click.echo(f"📁 Found {len(model_files)} models to convert")
    click.echo(f"⚙️  Quantization levels: {', '.join(quantization)}")
    click.echo()

    converter = GGUFConverter()

    # Prepare batch conversion data
    models = []
    for model_file in model_files:
        base_name = model_file.stem
        output_base = output_path / f"{base_name}.gguf"
        models.append({"input_path": str(model_file), "output_path": str(output_base)})

    # Perform batch conversion
    results = converter.batch_convert(models, list(quantization))

    # Report results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    click.echo(
        f"\n📊 " + click.style("Batch Conversion Results:", fg="green", bold=True)
    )
    click.echo(f"✅ Successful: {len(successful)}")
    click.echo(f"❌ Failed: {len(failed)}")

    if successful:
        total_size = sum(r.file_size_bytes for r in successful)
        click.echo(f"📁 Total output size: {total_size / (1024*1024):.1f} MB")

    if failed:
        click.echo(f"\n❌ " + click.style("Failed conversions:", fg="red"))
        for result in failed[:5]:  # Show first 5 failures
            click.echo(f"  • {Path(result.output_path).name}: {result.error_message}")
        if len(failed) > 5:
            click.echo(f"  ... and {len(failed) - 5} more")


@main.command()
@click.argument("model_path")
@click.option("--output", "-o", required=True, help="Output path for GGUF model")
@click.option(
    "--quantization",
    "-q",
    type=click.Choice(["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "F16", "F32"]),
    default="Q8_0",
    help="Quantization level",
)
@click.option("--validate", is_flag=True, help="Validate conversion result")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def convert_to_gguf(model_path, output, quantization, validate, verbose):
    """🔄 Convert model to GGUF format (shorthand for convert gguf)."""
    from .tools.convert_to_gguf import GGUFConverter

    click.echo("🔄 " + click.style("GGUF Model Conversion", fg="blue", bold=True))
    click.echo(
        "Built by "
        + click.style("Qub△se", fg="cyan", bold=True)
        + " - Converting your model!\n"
    )

    if verbose:
        click.echo(f"📁 Input model: {model_path}")
        click.echo(f"💾 Output path: {output}")
        click.echo(f"⚙️  Quantization: {quantization}")
        click.echo(f"✅ Validation: {'enabled' if validate else 'disabled'}")
        click.echo()

    try:
        converter = GGUFConverter()

        # Show available scripts
        if verbose:
            scripts = converter.conversion_scripts
            available_scripts = [name for name, path in scripts.items() if path]
            if available_scripts:
                click.echo(
                    f"🔧 Available conversion scripts: {', '.join(available_scripts)}"
                )
            else:
                click.echo(
                    "⚠️  " + click.style("No conversion scripts found!", fg="yellow")
                )
                click.echo(
                    "💡 Please install llama.cpp or ensure convert_hf_to_gguf.py is available"
                )
            click.echo()

        # Perform conversion
        click.echo(f"🚀 Converting with {quantization} quantization...")
        result = converter.convert_model(model_path, output, quantization)

        if result.success:
            click.echo(
                f"✅ " + click.style("Conversion successful!", fg="green", bold=True)
            )
            click.echo(f"📄 Output: {click.style(result.output_path, fg='cyan')}")
            click.echo(f"📊 Size: {result.file_size_bytes / (1024*1024):.1f} MB")
            click.echo(f"⏱️  Time: {result.conversion_time_seconds:.1f}s")

            if validate:
                click.echo(f"🔍 Validating conversion...")
                if result.validation_passed:
                    click.echo(f"✅ " + click.style("Validation passed", fg="green"))
                else:
                    click.echo(f"⚠️  " + click.style("Validation failed", fg="yellow"))
        else:
            click.echo(f"❌ " + click.style("Conversion failed!", fg="red", bold=True))
            if result.error_message:
                click.echo(f"💥 Error: {result.error_message}")

    except Exception as e:
        click.echo(f"❌ Conversion failed: {e}")
        if verbose:
            import traceback

            click.echo(f"\n🔍 " + click.style("Full error trace:", fg="red"))
            click.echo(traceback.format_exc())


# Configuration commands
@main.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
@click.option(
    "--preset",
    type=click.Choice(["cpu_small", "gpu_medium", "gpu_large", "inference"]),
    help="Configuration preset",
)
@click.option(
    "--template",
    type=click.Choice(["basic", "advanced", "cpu_optimized"]),
    help="Configuration template",
)
@click.option("--output", "-o", help="Output path for configuration file")
@click.option("--interactive", is_flag=True, help="Interactive configuration creation")
@click.option(
    "--advanced-features", is_flag=True, help="Enable advanced processing features"
)
def create(preset, template, output, interactive, advanced_features):
    """⚙️  Create a configuration file with guided setup."""
    import json
    from pathlib import Path

    from .config.defaults import Config, DefaultConfigs

    # Interactive mode
    if interactive or (not preset and not template) or not output:
        click.echo(
            "⚙️  "
            + click.style("LLMBuilder Configuration Creator", fg="blue", bold=True)
        )
        click.echo(
            "Built by "
            + click.style("Qub△se", fg="cyan", bold=True)
            + " - Let's configure your setup!\n"
        )

        if not preset and not template:
            click.echo(
                "🖥️  " + click.style("Available presets:", fg="green", bold=True)
            )
            presets = [
                ("cpu_small", "Small model optimized for CPU training"),
                ("gpu_medium", "Medium model optimized for single GPU"),
                ("gpu_large", "Large model for high-end GPU"),
                ("inference", "Configuration optimized for inference"),
            ]

            for name, desc in presets:
                click.echo(f"  • {click.style(name, fg='cyan'):<12} - {desc}")

            click.echo(
                f"\n📄 " + click.style("Available templates:", fg="green", bold=True)
            )
            templates = [
                ("basic", "Basic configuration with standard features"),
                ("advanced", "Advanced configuration with all processing features"),
                (
                    "cpu_optimized",
                    "CPU-optimized configuration for resource-constrained environments",
                ),
            ]

            for name, desc in templates:
                click.echo(f"  • {click.style(name, fg='cyan'):<12} - {desc}")

            click.echo()
            choice_type = click.prompt(
                "Use preset or template?",
                type=click.Choice(["preset", "template"]),
                default="preset",
            )

            if choice_type == "preset":
                preset = click.prompt(
                    "Choose a preset",
                    type=click.Choice(
                        ["cpu_small", "gpu_medium", "gpu_large", "inference"]
                    ),
                    default="cpu_small",
                )
            else:
                template = click.prompt(
                    "Choose a template",
                    type=click.Choice(["basic", "advanced", "cpu_optimized"]),
                    default="basic",
                )

        if not output:
            base_name = preset or template
            output = click.prompt(
                "💾 Output file path", default=f"{base_name}_config.json", type=str
            )

        if not advanced_features and not template:
            advanced_features = click.confirm(
                "🚀 Enable advanced processing features (ingestion, deduplication, etc.)?"
            )

        source = preset or template
        click.echo(f"\n🚀 Creating {click.style(source, fg='yellow')} configuration...")

    try:
        config = None

        if preset:
            config = DefaultConfigs.get_preset(preset)
            click.echo(f"📋 Using {click.style(preset, fg='cyan')} preset")
        elif template:
            # Load from template file
            template_path = Path(f"llmbuilder/config/templates/{template}_config.json")
            # Fix the path to look in the correct location
            if not template_path.exists():
                # Try relative to the package directory
                template_path = Path(__file__).parent / "config" / "templates" / f"{template}_config.json"
            
            if template_path.exists():
                with open(template_path, "r") as f:
                    template_dict = json.load(f)
                config = Config.from_dict(template_dict)
                click.echo(f"📋 Using {click.style(template, fg='cyan')} template")
            else:
                click.echo(f"⚠️  Template file not found, using default configuration")
                config = Config()
        else:
            config = Config()
            click.echo("📋 Using default configuration")

        # Enable advanced features if requested
        if (
            advanced_features and preset
        ):  # Only for presets, templates already have advanced features
            config.data.ingestion.batch_size = 200
            config.data.deduplication.enable_semantic_deduplication = True
            config.tokenizer_training.algorithm = "sentencepiece"
            click.echo("🚀 Advanced processing features enabled")

        # Save configuration
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        click.echo(
            f"✅ Configuration saved to {click.style(str(output_path), fg='green')}"
        )

        # Show key settings
        if interactive:
            click.echo(f"\n📊 " + click.style("Key Settings:", fg="blue", bold=True))
            click.echo(f"  • Model layers: {config.model.num_layers}")
            click.echo(f"  • Embedding dim: {config.model.embedding_dim}")
            click.echo(f"  • Vocab size: {config.model.vocab_size}")
            click.echo(f"  • Batch size: {config.training.batch_size}")
            click.echo(f"  • Learning rate: {config.training.learning_rate}")
            click.echo(f"  • Device: {config.system.device}")

            # Show advanced features
            click.echo(
                f"\n🚀 " + click.style("Advanced Features:", fg="blue", bold=True)
            )
            click.echo(
                f"  • Multi-format ingestion: {config.data.ingestion.supported_formats}"
            )
            click.echo(
                f"  • Deduplication: exact={config.data.deduplication.enable_exact_deduplication}, semantic={config.data.deduplication.enable_semantic_deduplication}"
            )
            click.echo(
                f"  • Tokenizer algorithm: {config.tokenizer_training.algorithm}"
            )
            click.echo(
                f"  • GGUF quantization: {config.gguf_conversion.quantization_level}"
            )

            click.echo(f"\n💡 " + click.style("Next steps:", fg="green"))
            click.echo(f"   llmbuilder config validate {output}")
            click.echo(f"   llmbuilder data ingest --help")
            click.echo(f"   llmbuilder tokenizer train --help")

    except Exception as e:
        click.echo(f"❌ Failed to create configuration: {e}")
        if interactive:
            import traceback

            click.echo(
                "💡 Check that you have write permissions for the output directory"
            )
            click.echo(f"\n🔍 " + click.style("Error details:", fg="red"))
            click.echo(traceback.format_exc())


@config.command()
@click.argument("config_path")
@click.option("--detailed", is_flag=True, help="Show detailed validation results")
def validate(config_path, detailed):
    """Validate a configuration file with detailed reporting."""
    from .config.manager import config_manager

    click.echo(f"🔍 Validating configuration: {click.style(config_path, fg='cyan')}")

    result = config_manager.validate_config_file(config_path)

    if result["valid"]:
        click.echo("✅ " + click.style("Configuration is valid!", fg="green", bold=True))

        if detailed and result["config_summary"]:
            click.echo(
                f"\n📊 " + click.style("Configuration Summary:", fg="blue", bold=True)
            )
            summary = result["config_summary"]

            # Model info
            model = summary["model"]
            click.echo(
                f"🧠 Model: {model['num_layers']} layers, {model['num_heads']} heads, {model['embedding_dim']} dim"
            )
            click.echo(
                f"   Vocab: {model['vocab_size']:,}, Max length: {model['max_seq_length']}"
            )

            # Training info
            training = summary["training"]
            click.echo(
                f"🏋️  Training: batch={training['batch_size']}, lr={training['learning_rate']}, epochs={training['num_epochs']}"
            )

            # Data processing info
            data = summary["data"]
            click.echo(
                f"📁 Data: max_length={data['max_length']}, formats={len(data['ingestion_formats'])}"
            )
            click.echo(
                f"   Deduplication: exact={data['deduplication_enabled']['exact']}, semantic={data['deduplication_enabled']['semantic']}"
            )

            # System info
            system = summary["system"]
            click.echo(
                f"💻 System: {system['device']}, mixed_precision={system['mixed_precision']}, workers={system['num_workers']}"
            )
    else:
        click.echo(
            "❌ " + click.style("Configuration validation failed!", fg="red", bold=True)
        )

        for error in result["errors"]:
            click.echo(f"  💥 {error}")

        for warning in result["warnings"]:
            click.echo(f"  ⚠️  {warning}")


@config.command()
def templates():
    """List available configuration templates."""
    from .config.manager import config_manager, get_available_templates

    click.echo(
        "📋 " + click.style("Available Configuration Templates:", fg="blue", bold=True)
    )

    templates = get_available_templates()

    if not templates:
        click.echo("No templates found.")
        return

    for template_name in sorted(templates):
        try:
            # Load template to get basic info
            template_data = config_manager.load_template(template_name)
            model_info = template_data.get("model", {})
            system_info = template_data.get("system", {})

            # Create description based on template characteristics
            vocab_size = model_info.get("vocab_size", "?")
            device = system_info.get("device", "auto")
            layers = model_info.get("num_layers", "?")

            description = f"Vocab: {vocab_size:,}, Layers: {layers}, Device: {device}"

            click.echo(
                f"  📄 {click.style(template_name, fg='cyan'):<25} - {description}"
            )

        except Exception as e:
            click.echo(
                f"  📄 {click.style(template_name, fg='cyan'):<25} - Error loading: {e}"
            )


@config.command()
@click.argument("template_name")
@click.option("--output", "-o", help="Output file path")
@click.option(
    "--override", "-O", multiple=True, help="Override values (key=value format)"
)
def from_template(template_name, output, override):
    """Create configuration from template with optional overrides."""
    import json

    from .config.manager import config_manager, create_config_from_template

    click.echo(
        f"📋 Creating configuration from template: {click.style(template_name, fg='cyan')}"
    )

    # Parse overrides
    overrides = {}
    for override_str in override:
        if "=" not in override_str:
            click.echo(
                f"❌ Invalid override format: {override_str}. Use key=value format."
            )
            return

        key, value = override_str.split("=", 1)

        # Try to parse value as JSON, fallback to string
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value

        # Support nested keys with dot notation
        keys = key.split(".")
        current = overrides
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = parsed_value

    try:
        config = create_config_from_template(
            template_name, overrides if overrides else None
        )

        if not output:
            output = f"{template_name}_config.json"

        config_manager.save_config_file(config, output)

        click.echo(f"✅ Configuration created: {click.style(output, fg='green')}")

        if overrides:
            click.echo(f"🔧 Applied {len(override)} override(s)")

        # Show summary
        summary = config_manager.get_config_summary(config)
        model = summary["model"]
        click.echo(
            f"📊 Model: {model['vocab_size']:,} vocab, {model['num_layers']} layers, {model['embedding_dim']} dim"
        )

    except Exception as e:
        click.echo(f"❌ Failed to create configuration: {e}")


@config.command()
@click.argument("config_path")
def summary(config_path):
    """Show a summary of configuration file contents."""
    from .config.manager import config_manager

    click.echo(f"📊 " + click.style("Configuration Summary", fg="blue", bold=True))
    click.echo(f"File: {click.style(config_path, fg='cyan')}")

    try:
        config = config_manager.load_config_file(config_path)
        summary = config_manager.get_config_summary(config)

        # Model section
        click.echo(f"\n🧠 " + click.style("Model Architecture:", fg="green", bold=True))
        model = summary["model"]
        click.echo(f"  Vocabulary size: {model['vocab_size']:,}")
        click.echo(f"  Embedding dimension: {model['embedding_dim']}")
        click.echo(f"  Layers: {model['num_layers']}")
        click.echo(f"  Attention heads: {model['num_heads']}")
        click.echo(f"  Max sequence length: {model['max_seq_length']}")

        # Training section
        click.echo(
            f"\n🏋️  " + click.style("Training Configuration:", fg="yellow", bold=True)
        )
        training = summary["training"]
        click.echo(f"  Batch size: {training['batch_size']}")
        click.echo(f"  Learning rate: {training['learning_rate']}")
        click.echo(f"  Epochs: {training['num_epochs']}")

        # Data processing section
        click.echo(f"\n📁 " + click.style("Data Processing:", fg="magenta", bold=True))
        data = summary["data"]
        click.echo(f"  Max length: {data['max_length']}")
        click.echo(f"  Supported formats: {', '.join(data['ingestion_formats'])}")
        dedup = data["deduplication_enabled"]
        click.echo(
            f"  Deduplication: exact={dedup['exact']}, semantic={dedup['semantic']}"
        )

        # Tokenizer training section
        click.echo(f"\n🔤 " + click.style("Tokenizer Training:", fg="cyan", bold=True))
        tokenizer = summary["tokenizer_training"]
        click.echo(f"  Algorithm: {tokenizer['algorithm']}")
        click.echo(f"  Vocabulary size: {tokenizer['vocab_size']:,}")

        # GGUF conversion section
        click.echo(f"\n🔄 " + click.style("GGUF Conversion:", fg="red", bold=True))
        gguf = summary["gguf_conversion"]
        click.echo(f"  Quantization level: {gguf['quantization_level']}")
        click.echo(f"  Preferred script: {gguf['preferred_script']}")

        # System section
        click.echo(f"\n💻 " + click.style("System Configuration:", fg="blue", bold=True))
        system = summary["system"]
        click.echo(f"  Device: {system['device']}")
        click.echo(f"  Mixed precision: {system['mixed_precision']}")
        click.echo(f"  Workers: {system['num_workers']}")

    except Exception as e:
        click.echo(f"❌ Failed to load configuration: {e}")


@config.command()
def list():
    """List available configuration presets."""
    click.echo("Available configuration presets:")
    click.echo("  cpu_small   - Small model optimized for CPU training")
    click.echo("  gpu_medium  - Medium model optimized for single GPU")
    click.echo("  gpu_large   - Large model for high-end GPU")
    click.echo("  inference   - Configuration optimized for inference")


if __name__ == "__main__":
    main()
