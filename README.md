# LLMBuilder

A production-ready implementation for training and fine-tuning Large Language Models from scratch. This project provides a complete pipeline for data preprocessing, tokenizer training, model training, and evaluation, with optimizations for both CPU and GPU environments.

## Key Features

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

## Installation

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

## Project Structure

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

## Quick Start

### 1. Prepare Your Data

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

### 2. Run the Pipeline

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

### 3. Run Specific Stages

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

## Fine-tuning

To fine-tune the model on your own data:

1. Place your training files in `data/finetune/`
2. The system will automatically use the latest checkpoint
3. Fine-tuned models save to `exports/checkpoints/finetuned/`

## Text Generation

Run interactive text generation:

```bash
python inference.py --interactive
```

Options:
- `--temperature`: Controls randomness (0.0-1.0)
- `--top_k`: Limit to top-k predictions
- `--top_p`: Nucleus sampling threshold

## Configuration

Edit `training/config.yaml` to customize model and training settings:

```yaml
# Model architecture
model:
  vocab_size: 16000    # Vocabulary size
  n_layer: 6          # Transformer layers
  n_head: 6           # Attention heads
  n_embd: 512         # Embedding dimension
  block_size: 256     # Context tokens
  dropout: 0.1        # Dropout rate

# Training settings
train:
  learning_rate: 0.0003
  batch_size: 16
  max_iters: 100000
  eval_interval: 1000
  log_interval: 100
  device: cpu         # cpu or cuda
```

## Advanced Usage

### CPU Optimization

Optimize for CPU training with:
- Multi-threading
- Memory efficiency
- Gradient accumulation
- MKLDNN acceleration

### Data Processing

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

### Training API

```python
from training.train import Trainer

trainer = Trainer(config_path="training/config.yaml")
trainer.train()
```

## Training Monitoring

### Logs
- Console: Real-time progress
- File: `logs/training.log`
- Metrics: `logs/training_history.json`

### Checkpoints
- `checkpoint_epoch_N.pt`: Regular saves
- `best_model.pt`: Best validation score
- `latest.pt`: Most recent checkpoint

## Performance Optimization

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

## Troubleshooting

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

## Model Architecture

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

## Future Enhancements

Planned features:
- GPU acceleration
- Distributed training
- Web monitoring interface
- More model architectures
- Additional optimizations

## Pre-trained Models

Download models from HuggingFace:

```bash
python tools/download_hf_model.py \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --output-dir ./models/Qwen2.5-Coder-0.5B
```

## License

MIT Licensed. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please submit PRs or open issues.

---

**Get started** by adding your data to `data/raw/` and running:

```bash
./run.sh  # Linux/macOS
run.bat   # Windows
``` 
