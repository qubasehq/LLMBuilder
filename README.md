# LLMBuilder

A production-ready implementation for training and fine-tuning Large Language Models from scratch. This project provides a complete pipeline for data preprocessing, tokenizer training, model training, and evaluation, with optimizations for both CPU and GPU environments.

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

- **End-to-End Training Pipeline**: Complete workflow from raw data to trained model
- **CPU Optimization**: Efficient multi-threaded training on CPU
- **Robust Error Handling**: Comprehensive error handling and logging
- **Modular Architecture**: Clean, maintainable code structure
- **Multiple Data Formats**: Support for PDF, DOCX, and TXT files
- **Advanced Tokenization**: SentencePiece-based tokenizer with BPE
- **GPT Architecture**: Modern transformer implementation
- **Model Evaluation**: Includes perplexity and text generation metrics
- **Checkpointing**: Automatic save/restore of training progress
- **Detailed Logging**: Comprehensive training logs and metrics

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB+ recommended)
- 2GB+ free disk space
- Windows, Linux, or macOS

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

For detailed instructions, see the [Complete Usage Guide](USAGE.md) which includes:
- Step-by-step walkthroughs
- Example terminal outputs
- Common issues and solutions
- Platform-specific commands
- Troubleshooting guide

## Project Structure <a name="project-structure"></a>

```
LLMBuilder/
├── data/                  # Data directories
│   ├── raw/              # Raw input files (.txt, .pdf, .docx)
│   ├── cleaned/          # Processed text files
│   └── tokens/           # Tokenized datasets
├── model/                # Model architecture
│   └── gpt_model.py      # GPT model implementation
├── training/             # Training pipeline
│   ├── config.yaml       # Configuration
│   ├── preprocess.py     # Data preprocessing
│   ├── train_tokenizer.py # Tokenizer training
│   ├── train.py          # Model training
│   ├── quantization.py   # Model quantization
│   ├── dataset.py        # Dataset handling
│   └── utils.py          # Utility functions
├── eval/                 # Evaluation scripts
│   └── eval.py
├── tools/                # Additional tools
│   ├── download_hf_model.py
│   └── export_gguf.py
├── exports/              # Output files
│   ├── checkpoints/     # Training checkpoints
│   └── models/          # Exported models
├── run.sh               # Linux/Mac script
├── run.bat              # Windows batch script
├── run.ps1              # PowerShell script
├── quantize_model.py    # Model quantization
└── requirements.txt     # Dependencies
```

## Quick Start <a name="quick-start"></a>

### 1. Prepare Your Data <a name="1-prepare-your-data"></a>

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
Place your documents in `data/raw/`:
- Text files (.txt)
- PDF files (.pdf)
- Word documents (.docx)

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

Run individual pipeline stages as needed:

```bash
# Preprocess data
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
```

On Windows, use `run.bat` or `run.ps1` with the same stage names.

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

trainer = Trainer(config_path="training/config.yaml")
trainer.train()
```

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

## Future Enhancements <a name="future-enhancements"></a>

Planned features:
- GPU acceleration
- Distributed training
- Web monitoring interface
- More model architectures
- Additional optimizations

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

---

**Get started** by adding your data to `data/raw/` and running:

```bash
./run.sh  # Linux/macOS
run.bat   # Windows
``` 
