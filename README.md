# LLMBuilder

Created By Qubase

A comprehensive, production-ready implementation for training and fine-tuning Large Language Models from scratch. This project provides an advanced pipeline with enhanced document ingestion, intelligent deduplication, model training, automated GGUF conversion, and comprehensive testing - all optimized for both CPU and GPU environments.

## Table of Contents

- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Prepare Your Data](#1-prepare-your-data)
  - [2. Run the Pipeline](#2-run-the-pipeline)
  - [3. Run Specific Stages](#3-run-specific-stages)
- [Project Structure](#project-structure)
- [Fine-tuning](#fine-tuning)
- [Text Generation](#text-generation)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
  - [CPU Optimization](#cpu-optimization)
  - [Data Processing](#data-processing)
  - [Training API](#training-api)
- [Monitoring](#monitoring-training)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Model Architecture](#model-architecture)
- [Pre-trained Models](#pre-trained-models)
- [License](#license)
- [Contributing](#contributing)

## Key Features <a name="key-features"></a>

### 🚀 **Enhanced Data Processing**
- **Multi-Format Document Ingestion**: HTML, EPUB, PDF (with OCR), Markdown, DOCX, TXT
- **Intelligent Deduplication**: Hash-based exact + embedding-based semantic duplicate removal
- **OCR Support**: Automatic fallback for scanned PDFs using Tesseract
- **Advanced Text Cleaning**: BeautifulSoup HTML processing, metadata extraction

### 🧠 **Advanced Training Pipeline**
- **End-to-End Workflow**: From raw documents to production-ready models
- **Multiple Tokenizer Options**: HuggingFace Tokenizers + SentencePiece CLI integration
- **CPU/GPU Optimization**: Efficient multi-threaded training with mixed precision
- **Modern GPT Architecture**: Transformer implementation with latest optimizations

### 📦 **Production-Ready Export**
- **Automated GGUF Conversion**: Multiple quantization levels (f16, q8_0, q4_0)
- **Quality Validation**: Comprehensive model validation and quality scoring
- **Batch Processing**: Parallel conversion with error recovery
- **llama.cpp Compatibility**: Direct integration with inference engines

### 🔧 **Developer Experience**
- **Comprehensive Testing**: Automated test suite with pytest integration
- **Robust Error Handling**: Detailed logging and recovery mechanisms
- **Modular Architecture**: Clean, maintainable, extensible codebase
- **Cross-Platform**: Windows PowerShell + Linux/macOS Bash scripts

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB+ recommended for large datasets)
- **Storage**: 5GB+ free disk space
- **OS**: Windows 10+, Linux, or macOS

### Additional Dependencies
- **Tesseract OCR**: For PDF OCR processing (see [INSTALL_TESSERACT.md](INSTALL_TESSERACT.md))
- **Git**: For repository management
- **Optional**: CUDA-compatible GPU for accelerated training

## Installation <a name="installation"></a>

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd LLMBuilder
   ```

2. Create and activate virtual environment:
   ```bash
   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR (for PDF processing):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   
   # Windows - see INSTALL_TESSERACT.md for detailed instructions
   ```

5. Verify installation:
   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__)"
   tesseract --version
   ```

## System Preparation

### System Requirements Check

Before starting, ensure your system meets the requirements:

```bash
# Linux/macOS
free -h      # Check available memory
df -h        # Check disk space
nproc        # Check CPU cores

# Windows
# Use Task Manager → Performance → Memory/Disk
# Check CPU cores in System Information
```

### Recommended Workflow

1. **Start with a small dataset** (100MB) to test the pipeline
2. **Monitor system resources** during initial runs
3. **Use checkpoints** - training progress is saved automatically
4. **Check logs** in `logs/training.log` for any issues

### **🔍 Real-time Monitoring:**
```bash
# Linux/Mac: Monitor system resources
htop
# Windows: Use Task Manager or Resource Monitor
```

## Getting Started

For detailed instructions, see the [**📖 Complete Usage Guide (USAGE.md)**](USAGE.md) which includes:
- **Step-by-step walkthroughs** with example outputs
- **Advanced configuration options** for all components
- **Troubleshooting guide** with common solutions
- **Performance optimization** tips
- **Platform-specific commands** for Windows/Linux/macOS
- **Integration examples** with other tools

## Project Structure <a name="project-structure"></a>

```
LLMBuilder/
├── data/                   # Data directories
│   ├── raw/               # Raw input files (all formats)
│   ├── cleaned/           # Processed text files
│   ├── deduped/           # Deduplicated content
│   ├── tokens/            # Tokenized datasets
│   ├── finetune/          # Fine-tuning datasets
│   ├── ingest.py          # Enhanced document ingester
│   ├── dedup.py           # Deduplication system
│   ├── download_data.py   # Script to download datasets
│   ├── SOURCES.md         # Data sources documentation
│   └── README_INGESTION.md # Ingestion documentation
│
├── debug_scripts/         # Debugging utilities
│   ├── debug_loader.py    # Data loading debugger
│   ├── debug_training.py  # Training process debugger
│   └── debug_timestamps.py # Timing analysis
│
├── eval/                  # Model evaluation
│   └── eval.py           # Evaluation scripts
│
├── exports/               # Output directories
│   ├── checkpoints/      # Training checkpoints
│   ├── gguf/             # GGUF model exports
│   ├── onnx/             # ONNX model exports
│   └── tokenizer/        # Saved tokenizer files
│
├── finetune/             # Fine-tuning scripts
│   ├── finetune.py      # Fine-tuning implementation
│   └── __init__.py      # Package initialization
│
├── logs/                 # Training and evaluation logs
│
├── model/                # Model architecture
│   └── gpt_model.py     # GPT model implementation
│
├── scripts/              # Enhanced processing scripts
│   ├── run_ingestion.py  # Document ingestion CLI
│   ├── enhanced_preprocess.py # Advanced preprocessing
│   ├── train_sentencepiece.py # SentencePiece training
│   └── test_deduplication.py # Deduplication testing
│
├── tests/                # Comprehensive test suite
│   ├── test_ingestion.py # Document ingestion tests
│   ├── test_deduplication.py # Deduplication tests
│   ├── test_conversion_pipeline.py # GGUF conversion tests
│   ├── test_tokenizer_trainer.py # Tokenizer tests
│   └── ... (many more test files)
│
├── tools/                # Utility scripts
│   ├── analyze_data.ps1  # PowerShell data analysis
│   ├── analyze_data.sh   # Bash data analysis
│   ├── download_hf_model.py # HuggingFace model downloader
│   ├── export_gguf.py    # Enhanced GGUF export utility
│   ├── conversion_pipeline.py # Automated GGUF conversion
│   └── quantization_manager.py # Advanced quantization
│
├── training/             # Training pipeline
│   ├── dataset.py       # Dataset handling
│   ├── preprocess.py    # Data preprocessing
│   ├── quantization.py  # Model quantization
│   ├── train.py         # Main training script
│   ├── train_tokenizer.py # Enhanced tokenizer training
│   └── utils.py         # Training utilities
│
├── .gitignore           # Git ignore rules
├── config.json          # Main configuration
├── config_cpu_small.json # Small CPU config
├── config_gpu.json      # GPU configuration
├── inference.py         # Inference script
├── quantize_model.py    # Model quantization
├── README.md           # This file
├── PIPELINE_UPDATES.md  # Recent updates summary
├── INSTALL_TESSERACT.md # OCR installation guide
├── requirements.txt    # Python dependencies
├── run.ps1            # Enhanced PowerShell runner
└── run.sh             # Enhanced Bash runner
```

## Quick Start <a name="quick-start"></a>

### 1. Prepare Your Data <a name="1-prepare-your-data"></a>

#### Enhanced Document Support
Place your documents in `data/raw/`. The system now supports:
- **Text files** (.txt, .md)
- **PDF files** (.pdf) - with automatic OCR for scanned documents
- **Word documents** (.docx)
- **Web content** (.html)
- **E-books** (.epub)
- **Markdown** (.md)

#### Option 1: Download Sample Data
```bash
# Download sample corpus
python data/download_data.py --corpus

# Or download specific topics
python data/download_data.py --topic literature --count 5
python data/download_data.py --topic technology --count 3
```

Available topics: literature, science, technology, business, health, education

#### Option 2: Use Your Own Data
Simply place your documents in `data/raw/` - the enhanced ingestion pipeline will automatically:
- Detect file formats
- Extract text with appropriate methods
- Handle OCR for scanned PDFs
- Clean and normalize content

### 2. Run the Pipeline <a name="2-run-the-pipeline"></a>

#### Linux/macOS:
```bash
chmod +x run.sh
./run.sh
```

#### Windows:
```batch
run.bat
```

Or using PowerShell:
```powershell
.\run.ps1
```

### 3. Run Specific Stages <a name="3-run-specific-stages"></a>

The enhanced pipeline includes new stages for better data processing:

```bash
# NEW: Enhanced document ingestion
./run.sh ingest

# NEW: Intelligent deduplication  
./run.sh dedup

# Traditional preprocessing (optional)
./run.sh preprocess

# Train tokenizer
./run.sh tokenizer

# Train model
./run.sh train

# Evaluate model
./run.sh eval

# Fine-tune existing model
./run.sh finetune

# Interactive text generation
./run.sh inference

# NEW: Convert to GGUF format
./run.sh gguf

# NEW: Run comprehensive tests
./run.sh test
```

#### Windows PowerShell Examples:
```powershell
# Enhanced document processing
.\run.ps1 -Stage ingest

# Run deduplication
.\run.ps1 -Stage dedup

# Complete pipeline
.\run.ps1 -Stage all

# Convert to GGUF
.\run.ps1 -Stage gguf
```

## Enhanced Pipeline Stages

### 🔄 **Document Ingestion** (`ingest`)
Advanced document processing with multi-format support:

```bash
# Process all supported formats with OCR
./run.sh ingest

# With custom options
python scripts/run_ingestion.py \
  --input data/raw \
  --output data/cleaned \
  --ocr-lang eng fra deu \
  --max-size 50 \
  --recursive
```

**Features:**
- **HTML Processing**: BeautifulSoup-based cleaning, removes scripts/styles
- **EPUB Support**: Full e-book text extraction with metadata
- **PDF with OCR**: Automatic fallback to Tesseract for scanned documents
- **Markdown Processing**: Advanced parsing with table/code block support
- **Progress Tracking**: Real-time processing statistics

### 🔍 **Intelligent Deduplication** (`dedup`)
Remove exact and near-duplicate content to improve training quality:

```bash
# Run both hash and embedding deduplication
./run.sh dedup

# Custom similarity threshold
python data/dedup.py \
  --input-dir data/cleaned \
  --output-dir data/deduped \
  --similarity-threshold 0.85
```

**Methods:**
- **Hash-based**: Exact duplicate detection with text normalization
- **Embedding-based**: Semantic similarity using sentence-transformers
- **Quality Preservation**: Keeps highest quality version of duplicates
- **Statistics**: Detailed reporting of removed content

### 📦 **GGUF Conversion** (`gguf`)
Automated conversion to GGUF format for production deployment:

```bash
# Convert with multiple quantization levels
./run.sh gguf

# Custom quantization options
python tools/conversion_pipeline.py \
  exports/checkpoints/best_model.pt \
  exports/gguf \
  --quantization f16 q8_0 q4_0 q4_1 \
  --tokenizer exports/tokenizer
```

**Features:**
- **Multiple Quantization**: f16, q8_0, q4_0, q4_1, q5_0, q5_1
- **Quality Validation**: Automatic validation and quality scoring
- **Batch Processing**: Parallel conversion with error recovery
- **Metadata Preservation**: Complete model metadata in GGUF format

### 🧪 **Comprehensive Testing** (`test`)
Automated test suite for quality assurance:

```bash
# Run all tests
./run.sh test

# Run specific test categories
python -m pytest tests/test_ingestion.py -v
python -m pytest tests/test_deduplication.py -v
python -m pytest tests/test_conversion_pipeline.py -v
```

## Fine-tuning <a name="fine-tuning"></a>

To fine-tune the model on your own data:

1. Place your training files in `data/finetune/`
2. The system will automatically use the latest checkpoint
3. Run the fine-tuning script:
   ```bash
   python finetune/finetune.py \
     --config config.json \
     --pretrained-model exports/checkpoints/latest.pt \
     --train-data data/finetune/ \
     --tokenizer-dir exports/tokenizer/
   ```
4. Fine-tuned models save to `exports/checkpoints/finetuned/`

### Fine-tuning Configuration

You can customize fine-tuning by modifying these parameters:

```yaml
finetune:
  learning_rate: 0.0001    # Lower than training LR
  batch_size: 4           # Adjust based on GPU memory
  num_epochs: 3           # Number of fine-tuning epochs
  warmup_steps: 100       # Learning rate warmup steps
```

### Monitoring Fine-tuning

Monitor the fine-tuning process with:
```bash
tensorboard --logdir=exports/logs/finetune/
```

## Text Generation <a name="text-generation"></a>

Run interactive text generation:

```bash
python inference.py --interactive
```

Options:
- `--temperature`: Controls randomness (0.0-1.0)
- `--top_k`: Limit to top-k predictions
- `--top_p`: Nucleus sampling threshold

## Configuration <a name="configuration"></a>

This project includes multiple configuration files optimized for different hardware setups. Choose the one that best matches your environment:

### Available Configurations

1. **config.json** - Balanced configuration for standard CPUs
   - Moderate model size
   - Good balance between speed and quality
   - Works well on most modern laptops/desktops

2. **config_gpu.json** - Optimized for GPU training
   - Larger model capacity
   - Mixed precision training
   - Gradient accumulation
   - Best for NVIDIA GPUs with 8GB+ VRAM

3. **config_cpu_small.json** - For very limited CPUs
   - Minimal memory footprint
   - Smaller model size
   - Reduced sequence length
   - Ideal for testing or low-resource environments

### Configuration Options

#### Model Architecture
```yaml
model:
  vocab_size: 16000      # Vocabulary size
  embedding_dim: 384     # Size of token embeddings
  num_layers: 6          # Number of transformer layers
  num_heads: 6           # Number of attention heads
  hidden_dim: 1536       # Size of feedforward layers
  max_seq_length: 256    # Maximum sequence length
  dropout: 0.1           # Dropout rate
  use_bias: true         # Use bias in linear layers
  tie_weights: true      # Tie input/output embeddings
```

#### Training Settings
```yaml
training:
  batch_size: 8          # Training batch size
  learning_rate: 0.0002  # Learning rate
  weight_decay: 0.01     # Weight decay for regularization
  num_epochs: 10         # Number of training epochs
  warmup_steps: 1000     # Warmup steps for learning rate
  gradient_clip_norm: 1.0 # Gradient clipping
  save_every: 1000       # Save checkpoint every N steps
  eval_every: 500        # Evaluate every N steps
  log_every: 10          # Log metrics every N steps
  num_workers: 4         # Data loading workers
  pin_memory: true       # Pin memory for faster transfer
  prefetch_factor: 2      # Batches to prefetch
  use_mixed_precision: false # Enable mixed precision
```

#### Device Configuration
```yaml
device:
  use_cuda: false        # Use CUDA if available
  cuda_device: 0         # CUDA device index
  use_mps: false         # Use MPS on Apple Silicon
  cpu_threads: 0         # Number of CPU threads (0 = all)
  enable_mkldnn: true    # Enable MKLDNN acceleration
  mixed_precision: false # Global mixed precision flag
```

### Choosing the Right Configuration

1. **For GPU Training**: Use `config_gpu.json`
   ```bash
   python training/train.py --config config_gpu.json
   ```

2. **For Standard CPU Training**: Use `config.json`
   ```bash
   python training/train.py --config config.json
   ```

3. **For Low-End CPUs**: Use `config_cpu_small.json`
   ```bash
   python training/train.py --config config_cpu_small.json
   ```

### Custom Configuration

1. Copy an existing config file:
   ```bash
   cp config.json my_config.json
   ```

2. Edit the parameters as needed
3. Use your custom config:
   ```bash
   python training/train.py --config my_config.json
   ```

### Important Notes
- Larger `batch_size` and `max_seq_length` require more memory
- `num_workers` should be ≤ number of CPU cores
- Enable `mixed_precision` for GPUs with Tensor Cores (Volta, Turing, Ampere, etc.)
- For small GPUs, reduce `batch_size` and enable `gradient_accumulation_steps`
- For very small CPUs, reduce `num_layers`, `embedding_dim`, and `hidden_dim`

## Debugging <a name="debugging"></a>

The project includes several debugging scripts in the `debug_scripts/` directory to help diagnose issues:

### Available Debug Scripts

1. **debug_loader.py**
   - Tests and profiles the data loading pipeline
   - Helps identify bottlenecks in data loading
   - Usage:
     ```bash
     python debug_scripts/debug_loader.py --config config.json
     ```

2. **debug_training.py**
   - Runs a minimal training loop with extensive logging
   - Verifies model can complete a forward/backward pass
   - Usage:
     ```bash
     python debug_scripts/debug_training.py --config config.json --max-steps 10
     ```

3. **debug_timestamps.py**
   - Profiles different components of the training loop
   - Helps identify slow operations
   - Usage:
     ```bash
     python debug_scripts/debug_timestamps.py --config config.json
     ```

### Debugging Tips

1. **Reduced Test Case**
   - Use a tiny dataset with `--max-samples 10`
   - Set `num_workers=0` to simplify data loading
   - Reduce `batch_size` and `max_seq_length`

2. **Common Issues**
   - **CUDA Out of Memory**: Reduce `batch_size` or model dimensions
   - **Slow Training**: Check data loading with `debug_loader.py`
   - **NaN/Inf Losses**: Try gradient clipping and lower learning rate

3. **Verbose Logging**
   ```bash
   python training/train.py --config config.json --log-level DEBUG
   ```

4. **Memory Profiling**
   ```bash
   python -m memory_profiler training/train.py --config config.json
   ```

## Advanced Usage <a name="advanced-usage"></a>

### CPU Optimization <a name="cpu-optimization"></a>

Optimize for CPU training with:
- Multi-threading
- Memory efficiency
- Gradient accumulation
- MKLDNN acceleration

### Data Processing <a name="data-processing"></a>

Example custom preprocessing:

```python
from training.preprocess import DataPreprocessor

processor = DataPreprocessor(
    min_length=100,       # Min text length
    max_length=500000,    # Max text length
    remove_urls=True,     # Clean URLs
    remove_emails=True,   # Clean emails
    normalize_whitespace=True
)
```

### Training API <a name="training-api"></a>

```python
from training.train import Trainer

# Initialize trainer with JSON config
trainer = Trainer(config_path="config.json")

# Start training
trainer.train()

# Example with custom settings
custom_trainer = Trainer(
    config_path="config.json",
    train_data_dir="data/processed/train",
    val_data_dir="data/processed/val",
    output_dir="exports/models/custom_run"
)
custom_trainer.train()
```

**Configuration Options**:
- `config_path`: Path to JSON config file (e.g., `config.json`)
- `train_data_dir`: Directory containing training data (overrides config)
- `val_data_dir`: Directory containing validation data (overrides config)
- `output_dir`: Directory to save checkpoints and logs (overrides config)

## Training Monitoring <a name="monitoring-training"></a>

### Logs
- Console: Real-time progress
- File: `logs/training.log`
- Metrics: `logs/training_history.json`

### Checkpoints
- `checkpoint_epoch_N.pt`: Regular saves
- `best_model.pt`: Best validation score
- `latest.pt`: Most recent checkpoint

## Performance Optimization <a name="performance-optimization"></a>

### CPU Training
- Batch size: 8-32 (adjust for RAM)
- Use all CPU cores
- Enable gradient accumulation
- Try mixed precision if available

### Memory Management
- Reduce `block_size` (128-256)
- Decrease `batch_size`
- Use smaller model dimensions

### Speed Improvements
- Increase `batch_size` (if RAM allows)
- Use larger `block_size` for context
- Multiple data files improve shuffling

## Troubleshooting <a name="troubleshooting"></a>

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size` in config.yaml
   - Decrease `block_size` or model size
   - Close other applications

2. **No Training Data**
   - Check `data/raw/` directory
   - Supported formats: .txt, .pdf, .docx
   - Verify file permissions

3. **Slow Training**
   - Optimize CPU thread count
   - Reduce model size
   - Monitor system resources

4. **Import Errors**
   ```bash
   pip install -r requirements.txt
   python --version  # Requires 3.8+
   ```

Check `logs/` for detailed error messages.

## Model Architecture <a name="model-architecture"></a>

GPT-style transformer with:
- Multi-head self-attention
- GELU activation
- Pre-norm layer normalization
- Learned positional embeddings
- Weight-tied embeddings

### Default Specs
- Parameters: ~50M
- Layers: 12
- Heads: 12
- Embedding: 768D
- Context: 512 tokens
- Vocabulary: 16K BPE

## Recent Updates <a name="recent-updates"></a>

### ✨ **Latest Features** (See [PIPELINE_UPDATES.md](PIPELINE_UPDATES.md))
- **Enhanced Document Ingestion**: Multi-format support with OCR
- **Intelligent Deduplication**: Hash + embedding-based duplicate removal
- **Automated GGUF Conversion**: Production-ready model export
- **Comprehensive Testing**: Full test suite with pytest
- **Cross-platform Scripts**: Enhanced PowerShell and Bash runners

### 🚀 **Future Enhancements**
- **Distributed Training**: Multi-GPU and multi-node support
- **Web Interface**: Real-time monitoring dashboard
- **More Architectures**: LLaMA, BERT, and custom models
- **Cloud Integration**: AWS/GCP/Azure deployment
- **Advanced Optimizations**: Dynamic quantization, pruning

## Pre-trained Models <a name="pre-trained-models"></a>

Download models from HuggingFace:

```bash
python tools/download_hf_model.py \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --output-dir ./models/Qwen2.5-Coder-0.5B
```

## License <a name="license"></a>

MIT Licensed. See [LICENSE](LICENSE) for details.

## Contributing <a name="contributing"></a>

Contributions welcome! Please submit PRs or open issues.

## Quick Reference

### 🚀 **One-Command Setup**
```bash
# Complete pipeline with enhanced features
./run.sh all          # Linux/macOS
.\run.ps1 -Stage all  # Windows PowerShell
```

### 📋 **Essential Commands**
```bash
# Enhanced document processing
./run.sh ingest       # Process HTML, PDF, EPUB, etc.
./run.sh dedup        # Remove duplicates intelligently
./run.sh train        # Train your model
./run.sh gguf         # Convert to GGUF format
./run.sh test         # Run comprehensive tests
```

### 📚 **Documentation**
- **[USAGE.md](USAGE.md)** - Complete usage guide with examples
- **[PIPELINE_UPDATES.md](PIPELINE_UPDATES.md)** - Recent feature updates
- **[INSTALL_TESSERACT.md](INSTALL_TESSERACT.md)** - OCR setup guide
- **[data/README_INGESTION.md](data/README_INGESTION.md)** - Document ingestion details

### 🆘 **Need Help?**
1. Check the [Usage Guide](USAGE.md) for detailed examples
2. Review logs in `logs/` directory
3. Run tests: `./run.sh test`
4. Open an issue on the repository

---

**Get started** by adding your documents to `data/raw/` and running:

```bash
./run.sh all  # Complete enhanced pipeline
``` 
