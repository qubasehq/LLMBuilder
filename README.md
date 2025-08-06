# LLM From Scratch

A robust, production-ready implementation for training Large Language Models from scratch. This project provides a complete pipeline for data preprocessing, tokenizer training, model training, and evaluation, optimized for CPU execution with optional GPU support.

## 🚀 Features

- **Complete Training Pipeline**: End-to-end automation from raw data to trained model
- **CPU-Optimized**: Efficient training on CPU with multi-threading support
- **Robust Error Handling**: Comprehensive error handling and logging throughout
- **Modular Design**: Clean, maintainable code with clear separation of concerns
- **Multiple Data Formats**: Support for PDF, DOCX, and TXT files
- **Advanced Tokenization**: SentencePiece-based tokenizer with BPE encoding
- **GPT Architecture**: Modern transformer architecture with attention mechanisms
- **Comprehensive Evaluation**: Perplexity, text generation, and performance benchmarks
- **Checkpointing**: Automatic model checkpointing and recovery
- **Detailed Logging**: Structured logging with rotation and retention

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- 2GB+ disk space for models and data
- Windows/Linux/macOS

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 0day
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
├── data/
│   ├── raw/           # Place your training data here (.txt, .pdf, .docx)
│   ├── cleaned/       # Preprocessed text files
│   └── tokens/        # Tokenized datasets
├── model/
│   └── gpt_model.py   # GPT model architecture
├── training/
│   ├── config.yaml    # Training configuration
│   ├── preprocess.py  # Data preprocessing
│   ├── train_tokenizer.py  # Tokenizer training
│   ├── train.py       # Model training
│   ├── dataset.py     # Dataset utilities
│   └── utils.py       # Common utilities
├── eval/
│   └── eval.py        # Model evaluation
├── exports/
│   ├── checkpoints/   # Model checkpoints
│   ├── gguf/         # GGUF exports (future)
│   └── onnx/         # ONNX exports (future)
├── tokenizer/         # Trained tokenizer files
├── logs/             # Training and evaluation logs
└── run.sh            # Main pipeline script
```

## 🚀 Quick Start

### 1. Prepare Your Data

Place your training documents in the `data/raw/` directory:
- **Text files** (.txt): Plain text documents
- **PDF files** (.pdf): Will be automatically extracted
- **Word documents** (.docx): Will be automatically extracted

### 2. Run the Complete Pipeline

```bash
# Make the script executable (Linux/macOS)
chmod +x run.sh

# Run the complete pipeline
./run.sh

# Or run with Python on Windows
python -c "exec(open('run.sh').read())"
```

### 3. Individual Stages

You can also run individual stages:

```bash
# Data preprocessing only
./run.sh preprocess

# Tokenizer training only
./run.sh tokenizer

# Model training only
./run.sh train

# Model evaluation only
./run.sh eval

# Fine-tune a pre-trained model (add your data to data/finetune/)
./run.sh finetune

# Launch interactive inference (text generation)
./run.sh inference
```

**On Windows:**

```bat
REM Use run.bat (Batch) or run.ps1 (PowerShell)
run.bat finetune
run.ps1 -Stage finetune
run.bat inference
run.ps1 -Stage inference
```

### Fine-tuning
- Place your fine-tuning data (txt/pdf/docx) in `data/finetune/`.
- The script will use the latest checkpoint for fine-tuning.
- Output will be saved to `exports/checkpoints/finetuned/`.

### Inference (Text Generation)
- The `inference` stage launches an interactive prompt for text generation.
- Uses the latest model checkpoint and tokenizer.
- Supports temperature, top-k, and top-p sampling.

## ⚙️ Configuration

Edit `training/config.yaml` to customize your training:

```yaml
model:
  vocab_size: 16000      # Vocabulary size
  n_layer: 6            # Number of transformer layers
  n_head: 6             # Number of attention heads
  n_embd: 512           # Embedding dimension
  block_size: 256       # Context window size
  dropout: 0.1          # Dropout rate

train:
  learning_rate: 0.0003 # Learning rate
  batch_size: 16        # Batch size
  max_iters: 100000     # Maximum training iterations
  eval_interval: 1000   # Evaluation frequency
  log_interval: 100     # Logging frequency
  device: cpu           # Training device (cpu/cuda)
```

## 🔧 Advanced Usage

### CPU Optimization

The system is optimized for CPU training:
- Multi-threading support
- Memory-efficient operations
- Gradient accumulation for larger effective batch sizes
- MKLDNN optimizations

### Custom Data Processing

```python
from training.preprocess import DataPreprocessor

# Custom preprocessing
processor = DataPreprocessor(
    min_length=100,        # Minimum text length
    max_length=500000,     # Maximum text length
    remove_urls=True,      # Remove URLs
    remove_emails=True,    # Remove email addresses
    normalize_whitespace=True  # Normalize whitespace
)

results = processor.process_all()
```

### Custom Training

```python
from training.train import Trainer

# Initialize trainer
trainer = Trainer(config_path="training/config.yaml")

# Start training
trainer.train()
```

### Model Evaluation

```python
from eval.eval import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(
    model_path="exports/checkpoints/best_model.pt",
    tokenizer_path="tokenizer/tokenizer.model"
)

# Generate text
generated = evaluator.generate_text(
    prompt="The future of AI is",
    max_new_tokens=100,
    temperature=0.8
)
```

## 📊 Monitoring Training

### Logs

All training progress is logged to:
- **Console**: Real-time progress with colored output
- **Files**: Detailed logs in `logs/training.log`
- **Metrics**: Training history in `logs/training_history.json`

### Checkpoints

Models are automatically saved to:
- `exports/checkpoints/checkpoint_epoch_N.pt` - Regular checkpoints
- `exports/checkpoints/best_model.pt` - Best model based on validation loss
- `exports/checkpoints/latest.pt` - Latest checkpoint

## 🎯 Performance Tips

### For CPU Training
- Use batch sizes of 8-32 depending on your RAM
- Enable all CPU cores with `torch.set_num_threads()`
- Consider gradient accumulation for larger effective batch sizes
- Use mixed precision if available

### For Memory Optimization
- Reduce `block_size` in config (e.g., 128 or 256)
- Decrease `batch_size`
- Use smaller model dimensions (`n_embd`, `n_layer`)

### For Faster Training
- Increase `batch_size` if you have enough RAM
- Use larger `block_size` for better context
- Consider using multiple data files for better shuffling

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch_size in config.yaml
   - Reduce block_size or model dimensions
   - Close other applications

2. **No Training Data**:
   - Ensure files are in `data/raw/`
   - Check supported formats: .txt, .pdf, .docx
   - Verify file permissions

3. **Slow Training**:
   - Check CPU usage and optimize thread count
   - Consider reducing model size for faster iteration
   - Monitor memory usage

4. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)
   - Verify virtual environment activation

### Getting Help

Check the logs in `logs/` directory for detailed error messages and training progress.

## 📈 Model Architecture

The model implements a GPT-style transformer architecture:

- **Multi-head self-attention** with causal masking
- **Feed-forward networks** with GELU activation
- **Layer normalization** (pre-norm architecture)
- **Positional embeddings** for sequence understanding
- **Weight tying** between input and output embeddings

### Default Configuration

- **Parameters**: ~50M (configurable)
- **Layers**: 12 transformer blocks
- **Attention Heads**: 12
- **Embedding Dimension**: 768
- **Context Window**: 512 tokens
- **Vocabulary**: 16,000 tokens (BPE)

## 🔮 Future Enhancements

- [ ] GPU acceleration support
- [ ] Model export to GGUF format
- [ ] ONNX export for inference
- [ ] Distributed training
- [ ] Web interface for training monitoring
- [x] Pre-trained model downloads
- [x] Fine-tuning capabilities
- [ ] Quantization support

## 📦 Downloading Pre-trained Models

You can automatically download any pre-trained model from HuggingFace using the provided script:

```bash
python tools/download_hf_model.py --model Qwen/Qwen2.5-Coder-0.5B --output-dir ./models/Qwen2.5-Coder-0.5B
```

- The script will fetch all files for the specified model and save them to the output directory.
- You can use these files for further fine-tuning or inference with this pipeline.

## 📖 **Complete Step-by-Step Guide**

**New to LLM training?** Check out our **[Complete Usage Guide](USAGE.md)** for:
- 🎯 **Beginner-friendly walkthrough** of every stage
- 📊 **Real terminal outputs** you'll see
- 🚨 **Common problems** and their solutions
- ☕ **Platform-specific commands** (Windows/Mac/Linux)
- 🎉 **Interactive examples** and troubleshooting

Perfect for first-time users! No technical background required.

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

---

**Ready to train your own LLM?** Start by adding your data to `data/raw/` and running `./run.sh`! 
