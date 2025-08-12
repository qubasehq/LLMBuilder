# LLMBuilder Usage Guide

This comprehensive guide covers all aspects of using LLMBuilder, from basic setup to advanced features.

## Table of Contents

- [Quick Start](#quick-start)
- [Enhanced Pipeline Stages](#enhanced-pipeline-stages)
- [Document Ingestion](#document-ingestion)
- [Deduplication](#deduplication)
- [Training](#training)
- [GGUF Conversion](#gguf-conversion)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Basic Setup

```bash
# Clone and setup
git clone <repository-url>
cd LLMBuilder
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (see INSTALL_TESSERACT.md for details)
sudo apt-get install tesseract-ocr  # Ubuntu
brew install tesseract              # macOS
```

### 2. Prepare Your Data

Place your documents in `data/raw/`. Supported formats:
- **Text**: `.txt`, `.md`
- **Documents**: `.pdf`, `.docx`
- **Web**: `.html`
- **E-books**: `.epub`

### 3. Run the Complete Pipeline

```bash
# Linux/macOS
./run.sh all

# Windows PowerShell
.\run.ps1 -Stage all

# Windows Command Prompt
run.bat all
```

## Enhanced Pipeline Stages

### Stage Overview

The enhanced pipeline includes these stages:

1. **`ingest`** - Enhanced document ingestion
2. **`dedup`** - Intelligent deduplication
3. **`preprocess`** - Traditional preprocessing (optional)
4. **`tokenizer`** - Tokenizer training
5. **`train`** - Model training
6. **`eval`** - Model evaluation
7. **`gguf`** - GGUF conversion
8. **`test`** - Comprehensive testing

### Running Individual Stages

```bash
# Enhanced document processing
./run.sh ingest

# Remove duplicates
./run.sh dedup

# Train tokenizer
./run.sh tokenizer

# Train model
./run.sh train

# Convert to GGUF
./run.sh gguf
```

## Document Ingestion

### Basic Ingestion

```bash
# Process all files in data/raw
./run.sh ingest

# Or run directly
python scripts/run_ingestion.py --input data/raw --output data/cleaned
```

### Advanced Ingestion Options

```bash
# With OCR language support
python scripts/run_ingestion.py \
  --input data/raw \
  --output data/cleaned \
  --ocr-lang eng fra deu spa \
  --max-size 100 \
  --recursive \
  --verbose
```

**Parameters:**
- `--input`: Input directory with documents
- `--output`: Output directory for cleaned text
- `--ocr-lang`: OCR languages (default: eng)
- `--max-size`: Maximum file size in MB (default: 100)
- `--recursive`: Process subdirectories
- `--verbose`: Enable detailed logging

### Supported Formats

#### PDF Processing
```bash
# Automatic OCR fallback for scanned PDFs
python scripts/run_ingestion.py \
  --input data/pdfs \
  --output data/cleaned \
  --ocr-lang eng
```

#### HTML Processing
```bash
# Clean HTML with BeautifulSoup
python scripts/run_ingestion.py \
  --input data/html \
  --output data/cleaned
```

#### EPUB Processing
```bash
# Extract text from e-books
python scripts/run_ingestion.py \
  --input data/ebooks \
  --output data/cleaned
```

### Example Output

```
INGESTION RESULTS
==================================================
Total files found: 25
Successfully processed: 23
Failed to process: 2
Success rate: 92.0%

Total content extracted:
  Characters: 1,234,567
  Words: 185,432

Output directory: data/cleaned
Supported formats: txt, pdf, docx, html, epub, md
```

## Deduplication

### Basic Deduplication

```bash
# Run both hash and embedding deduplication
./run.sh dedup

# Or run directly
python data/dedup.py \
  --input-dir data/cleaned \
  --output-dir data/deduped
```

### Advanced Deduplication

```bash
# Hash-only deduplication (faster)
python data/dedup.py \
  --input-dir data/cleaned \
  --output-dir data/deduped \
  --hash-only

# Embedding-only deduplication (semantic)
python data/dedup.py \
  --input-dir data/cleaned \
  --output-dir data/deduped \
  --embedding-only \
  --similarity-threshold 0.85

# Custom similarity threshold
python data/dedup.py \
  --input-dir data/cleaned \
  --output-dir data/deduped \
  --similarity-threshold 0.90
```

### Deduplication Methods

#### Hash-based (Exact Duplicates)
- Normalizes whitespace, case, punctuation
- MD5/SHA256 hashing
- Very fast processing
- Finds exact matches

#### Embedding-based (Near Duplicates)
- Uses sentence-transformers
- Semantic similarity detection
- Configurable threshold
- Finds similar content

### Example Output

```
=== Deduplication Results ===
Input files: 150
Output files: 127
Exact duplicates removed: 18
Near duplicates removed: 5
Processing time: 45.2s
Character reduction: 2,456,789 -> 2,123,456
```

## Training

### Basic Training

```bash
# Complete training pipeline
./run.sh train

# Or run individual components
python training/train_tokenizer.py
python training/train.py
```

### Custom Training Configuration

```bash
# Use specific config
python training/train.py --config config_gpu.json

# Override specific parameters
python training/train.py \
  --config config.json \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --num-epochs 5
```

### Training Monitoring

```bash
# Monitor training progress
tail -f logs/training.log

# View training metrics
cat logs/training_history.json | jq '.[-1]'

# Check GPU usage (if available)
nvidia-smi
```

### Training Configurations

#### Small CPU Training
```bash
python training/train.py --config config_cpu_small.json
```

#### GPU Training
```bash
python training/train.py --config config_gpu.json
```

#### Custom Configuration
```json
{
  "model": {
    "vocab_size": 16000,
    "embedding_dim": 384,
    "num_layers": 6,
    "num_heads": 6,
    "max_seq_length": 256
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 0.0002,
    "num_epochs": 10
  }
}
```

## GGUF Conversion

### Basic GGUF Conversion

```bash
# Convert with default quantization levels
./run.sh gguf

# Or run directly
python tools/conversion_pipeline.py \
  exports/checkpoints/best_model.pt \
  exports/gguf
```

### Advanced GGUF Conversion

```bash
# Custom quantization levels
python tools/conversion_pipeline.py \
  exports/checkpoints/best_model.pt \
  exports/gguf \
  --quantization f16 q8_0 q4_0 q4_1 q5_0 \
  --name "MyModel-v1.0" \
  --tokenizer exports/tokenizer

# With validation disabled (faster)
python tools/conversion_pipeline.py \
  exports/checkpoints/best_model.pt \
  exports/gguf \
  --quantization q4_0 \
  --no-validate

# Save conversion report
python tools/conversion_pipeline.py \
  exports/checkpoints/best_model.pt \
  exports/gguf \
  --quantization f16 q8_0 q4_0 \
  --report exports/gguf/conversion_report.json
```

### Quantization Levels

- **f32**: Full precision (largest, highest quality)
- **f16**: Half precision (good balance)
- **q8_0**: 8-bit quantization (smaller, good quality)
- **q4_0**: 4-bit quantization (small, decent quality)
- **q4_1**: 4-bit with improved accuracy
- **q5_0**: 5-bit quantization (balance)
- **q5_1**: 5-bit with improved accuracy

### Example Output

```
CONVERSION PIPELINE SUMMARY
============================================================
Input: exports/checkpoints/best_model.pt
Output: exports/gguf
Total time: 127.3s
Success rate: 100.0% (3/3)

Successful conversions:
  ✅ f16: 245.7MB, 23.4s (2.1x compression)
  ✅ q8_0: 123.2MB, 45.8s (4.2x compression)
  ✅ q4_0: 67.8MB, 58.1s (7.6x compression)
```

## Testing

### Run All Tests

```bash
# Complete test suite
./run.sh test

# Or run directly
python -m pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Document ingestion tests
python -m pytest tests/test_ingestion.py -v

# Deduplication tests
python -m pytest tests/test_deduplication_pipeline.py -v

# GGUF conversion tests
python -m pytest tests/test_conversion_pipeline.py -v

# Tokenizer tests
python -m pytest tests/test_tokenizer_trainer.py -v
```

### Test with Coverage

```bash
# Install coverage
pip install pytest-cov

# Run tests with coverage
python -m pytest tests/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Performance Testing

```bash
# Run performance benchmarks
python -m pytest tests/test_pipeline_basic.py::test_performance -v

# Memory usage testing
python -m pytest tests/test_simple_pipeline.py::test_memory_usage -v
```

## Configuration

### Configuration Files

- **`config.json`**: Balanced configuration for standard CPUs
- **`config_gpu.json`**: Optimized for GPU training
- **`config_cpu_small.json`**: For limited CPU resources

### Custom Configuration

```bash
# Copy existing config
cp config.json my_config.json

# Edit parameters
vim my_config.json

# Use custom config
python training/train.py --config my_config.json
```

### Key Configuration Parameters

#### Model Architecture
```json
{
  "model": {
    "vocab_size": 16000,
    "embedding_dim": 384,
    "num_layers": 6,
    "num_heads": 6,
    "hidden_dim": 1536,
    "max_seq_length": 256,
    "dropout": 0.1
  }
}
```

#### Training Settings
```json
{
  "training": {
    "batch_size": 8,
    "learning_rate": 0.0002,
    "weight_decay": 0.01,
    "num_epochs": 10,
    "warmup_steps": 1000,
    "gradient_clip_norm": 1.0
  }
}
```

#### Device Configuration
```json
{
  "device": {
    "use_cuda": false,
    "cuda_device": 0,
    "cpu_threads": 0,
    "mixed_precision": false
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```bash
# Reduce batch size
python training/train.py --config config.json --batch-size 4

# Use smaller model
python training/train.py --config config_cpu_small.json

# Enable gradient accumulation
# Edit config.json: "gradient_accumulation_steps": 4
```

#### 2. Slow Processing
```bash
# Check system resources
htop  # Linux/macOS
# Task Manager on Windows

# Reduce data size for testing
head -n 1000 data/cleaned/large_file.txt > data/cleaned/test_file.txt

# Use fewer CPU threads
export OMP_NUM_THREADS=4
```

#### 3. OCR Issues
```bash
# Check Tesseract installation
tesseract --version

# Install additional language packs
sudo apt-get install tesseract-ocr-fra tesseract-ocr-deu

# Test OCR manually
tesseract input.pdf output.txt
```

#### 4. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+

# Install missing packages
pip install sentence-transformers ebooklib pytesseract
```

### Debug Mode

```bash
# Enable verbose logging
python scripts/run_ingestion.py --verbose

# Debug training
python training/train.py --config config.json --log-level DEBUG

# Test with minimal data
python training/train.py --config config.json --max-samples 100
```

### Log Analysis

```bash
# Check recent logs
tail -f logs/training.log

# Search for errors
grep -i error logs/training.log

# View ingestion statistics
cat logs/ingestion_stats.json | jq '.'
```

## Advanced Usage

### Batch Processing

```bash
# Process multiple directories
for dir in data/raw/*/; do
  python scripts/run_ingestion.py --input "$dir" --output "data/cleaned/$(basename "$dir")"
done

# Parallel processing
find data/raw -name "*.pdf" | xargs -P 4 -I {} python scripts/process_single.py {}
```

### Custom Preprocessing

```python
# Custom preprocessing script
from data.ingest import DocumentIngester

ingester = DocumentIngester(
    output_dir="data/custom_cleaned",
    ocr_languages=['eng', 'fra'],
    max_file_size_mb=200
)

# Process with custom settings
results = ingester.ingest_directory("data/raw", recursive=True)
print(f"Processed {results['processed_count']} files")
```

### Integration with Other Tools

```bash
# Export to different formats
python tools/export_onnx.py exports/checkpoints/best_model.pt

# Convert to HuggingFace format
python tools/export_hf.py exports/checkpoints/best_model.pt exports/hf_model

# Use with llama.cpp
./llama.cpp/main -m exports/gguf/model_q4_0.gguf -p "Hello, world!"
```

### Performance Optimization

```bash
# Profile memory usage
python -m memory_profiler training/train.py --config config.json

# Profile CPU usage
python -m cProfile -o profile.stats training/train.py --config config.json

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

### Distributed Training (Future)

```bash
# Multi-GPU training (planned feature)
python -m torch.distributed.launch --nproc_per_node=2 training/train.py --config config_gpu.json

# Multi-node training (planned feature)
python -m torch.distributed.launch --nnodes=2 --node_rank=0 training/train.py --config config_distributed.json
```

## Examples

### Complete Workflow Example

```bash
# 1. Setup
git clone <repo>
cd LLMBuilder
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Add your data
cp ~/my_documents/*.pdf data/raw/
cp ~/my_texts/*.txt data/raw/

# 3. Run enhanced pipeline
./run.sh ingest    # Process documents
./run.sh dedup     # Remove duplicates
./run.sh tokenizer # Train tokenizer
./run.sh train     # Train model
./run.sh gguf      # Convert to GGUF

# 4. Test the model
python inference.py --interactive
```

### Custom Pipeline Example

```bash
# Custom ingestion with specific languages
python scripts/run_ingestion.py \
  --input data/multilingual \
  --output data/cleaned \
  --ocr-lang eng fra deu spa \
  --max-size 50

# Aggressive deduplication
python data/dedup.py \
  --input-dir data/cleaned \
  --output-dir data/deduped \
  --similarity-threshold 0.95

# Small model training
python training/train.py --config config_cpu_small.json

# Specific quantization
python tools/conversion_pipeline.py \
  exports/checkpoints/best_model.pt \
  exports/gguf \
  --quantization q4_0 q4_1
```

This guide covers the comprehensive usage of LLMBuilder. For more specific questions, check the individual component documentation or open an issue on the repository.