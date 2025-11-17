# LLMBuilder - Professional Language Model Toolkit

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://qubasehq.github.io/llmbuilder-package/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for building, training, fine-tuning, and deploying GPT-style language models with advanced data processing capabilities and CPU-friendly defaults.

## About LLMBuilder

LLMBuilder is a production-ready framework for training and fine-tuning Large Language Models (LLMs). Designed for developers, researchers, and AI engineers, LLMBuilder provides a full pipeline to go from raw text data to deployable, optimized LLMs, all running locally on CPUs or GPUs.

## Quick Start

### Installation

```bash
pip install llmbuilder
```

### Initialize a New Project

```bash
# Create a new project with default structure
llmbuilder init my_llm_project

# Navigate to your project directory
cd my_llm_project
```

This creates a structured project with the following directories:
- `data/raw` - Place your raw input files here (.txt, .pdf, .docx)
- `data/processed` - Processed text files
- `tokenizer` - Tokenizer files
- `models/checkpoints` - Training checkpoints
- `models/final` - Final trained models
- `configs` - Configuration files
- `outputs` - Output files

And generates a README.md with quick start instructions:
```
# my_llm_project

This is an LLM project created with LLMBuilder.

## Project Structure

- `data/` - Data files
- `tokenizer/` - Tokenizer files
- `models/` - Model checkpoints and final models
- `configs/` - Configuration files
- `outputs/` - Output files

## Quick Start

1. Prepare your data in `data/raw/`
2. Process data: `llmbuilder data load -i data/raw -o data/processed/input.txt`
3. Train tokenizer: `llmbuilder tokenizer train -i data/processed/input.txt -o tokenizer/`
4. Train model: `llmbuilder train model -d data/processed/input.txt -t tokenizer/ -o models/checkpoints/`
5. Generate text: `llmbuilder generate text -m models/checkpoints/latest.pt -t tokenizer/ -p "Your prompt here"`
```

## Documentation

Complete documentation is available at: [https://qubasehq.github.io/llmbuilder-package/](https://qubasehq.github.io/llmbuilder-package/)

The documentation includes:

- Getting Started Guide - From installation to your first model
- User Guides - Comprehensive guides for all features
- CLI Reference - Complete command-line interface documentation
- Python API - Full API reference with examples
- Examples - Working code examples for common tasks
- FAQ - Answers to frequently asked questions

## CLI Usage

### Getting Started

```bash
# Show help and available commands
llmbuilder --help

# Initialize a new project
llmbuilder init my_project

# Interactive welcome guide for new users
llmbuilder welcome

# Show package information and credits
llmbuilder info
```

### Configuration Management

```bash
# List available configuration templates
llmbuilder config templates

# Create a configuration from a template
llmbuilder config create --preset cpu_small -o configs/my_config.json

# Validate configuration with detailed reporting
llmbuilder config validate configs/my_config.json
```

### Data Processing Pipeline

```bash
# Process raw data files
llmbuilder data load -i data/raw -o data/processed/input.txt --clean

# Remove duplicates from your data
llmbuilder data deduplicate -i data/processed/input.txt -o data/processed/clean.txt --method both

# Train custom tokenizer
llmbuilder tokenizer train -i data/processed/clean.txt -o tokenizer/ --vocab-size 16000
```

### Model Training & Operations

```bash
# Train model
llmbuilder train model -d data/processed/clean.txt -t tokenizer/ -o models/checkpoints

# Interactive text generation setup
llmbuilder generate text --setup

# Generate text with custom parameters
llmbuilder generate text -m models/checkpoints/latest.pt -t tokenizer/ -p "Hello world" --temperature 0.8 --max-tokens 100
```

### Model Export

```bash
# Convert to GGUF format for deployment
llmbuilder export gguf models/checkpoints/latest.pt -o models/final/model.gguf -q Q8_0
```

## Python API

```python
import llmbuilder as lb

# Load a preset config and build a model
cfg = lb.load_config(preset="cpu_small")
model = lb.build_model(cfg.model)

# Train (example; see examples/train_tiny.py for a runnable script)
from llmbuilder.data import TextDataset
dataset = TextDataset("./data/clean.txt", block_size=cfg.model.max_seq_length)
results = lb.train_model(model, dataset, cfg.training)

# Generate text
text = lb.generate_text(
    model_path="./checkpoints/model.pt",
    tokenizer_path="./tokenizers",
    prompt="Hello world",
    max_new_tokens=50,
)
print(text)
```

## Configuration Management

LLMBuilder provides flexible configuration management:

```bash
# List available templates
llmbuilder config templates

# Create a configuration from a template
llmbuilder config create --preset cpu_small -o configs/my_config.json

# Validate your configuration
llmbuilder config validate configs/my_config.json
```

## System Requirements

- Python 3.8 or higher
- For PDF OCR Processing: Tesseract OCR
- For GGUF Model Conversion: llama.cpp or compatible tools

## Troubleshooting

### Installation Issues

**Missing Optional Dependencies**

```bash
# Check what's installed
python -c "import llmbuilder; print('LLMBuilder installed')"

# Install missing dependencies
pip install pymupdf pytesseract ebooklib beautifulsoup4 lxml sentence-transformers

# Verify specific features
python -c "import pytesseract; print('OCR available')"
python -c "import sentence_transformers; print('Semantic deduplication available')"
```

**System Dependencies**

```bash
# Tesseract OCR (for PDF processing)
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Verify Tesseract installation
tesseract --version
python -c "import pytesseract; pytesseract.get_tesseract_version()"
```

### Processing Issues

**PDF Processing Problems**

```bash
# Enable debug logging
export LLMBUILDER_LOG_LEVEL=DEBUG

# Common fixes:
# 1. Install language packs: sudo apt-get install tesseract-ocr-eng
# 2. Set Tesseract path: export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
# 3. Lower OCR threshold: --ocr-threshold 0.3
```

**Memory Issues with Large Datasets**

```bash
# Use configuration to optimize memory usage
llmbuilder config from-template cpu_optimized_config -o memory_config.json \
  --override data.ingestion.batch_size=50 \
  --override data.deduplication.batch_size=500 \
  --override data.deduplication.use_gpu_for_embeddings=false

# Process in smaller chunks
llmbuilder data load -i large_dataset/ -o processed.txt --batch-size 25 --workers 2
```

**Semantic Deduplication Performance**

```bash
# GPU issues - disable GPU acceleration
llmbuilder data deduplicate -i dataset.txt -o clean.txt --method semantic --no-gpu

# Slow processing - increase batch size
llmbuilder data deduplicate -i dataset.txt -o clean.txt --method semantic --batch-size 2000

# Memory issues - reduce embedding cache
llmbuilder config from-template basic_config -o config.json \
  --override data.deduplication.embedding_cache_size=5000
```

### GGUF Conversion Issues

**Missing llama.cpp**

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Add to PATH or specify location
export PATH=$PATH:/path/to/llama.cpp

# Alternative: Use Python package
pip install llama-cpp-python

# Test conversion
llmbuilder convert gguf --help
```

**Conversion Failures**

```bash
# Check available conversion scripts
llmbuilder convert gguf model.pt -o test.gguf --verbose

# Try different quantization levels
llmbuilder convert gguf model.pt -o test.gguf -q F16  # Less compression
llmbuilder convert gguf model.pt -o test.gguf -q Q8_0 # Balanced

# Increase timeout for large models
llmbuilder config from-template basic_config -o config.json \
  --override gguf_conversion.conversion_timeout=7200
```

### Configuration Issues

**Validation Errors**

```bash
# Validate configuration with detailed output
llmbuilder config validate my_config.json --detailed

# Common fixes:
# 1. Vocab size mismatch - ensure model.vocab_size matches tokenizer_training.vocab_size
# 2. Sequence length issues - ensure data.max_length <= model.max_seq_length
# 3. Invalid quantization level - use: F32, F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0
```

**Template Issues**

```bash
# List available templates
llmbuilder config templates

# Create from working template
llmbuilder config from-template basic_config -o working_config.json

# Validate before use
llmbuilder config validate working_config.json
```

## Documentation

Complete documentation is available at: https://qubasehq.github.io/llmbuilder-package/

## License

Apache-2.0