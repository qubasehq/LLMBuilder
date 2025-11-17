# LLMBuilder Documentation

<div align="center">
  <h1>🤖 LLMBuilder</h1>
  <p><strong>A toolkit for building, training, and deploying language models</strong></p>
</div>

[![PyPI version](https://badge.fury.io/py/llmbuilder.svg)](https://badge.fury.io/py/llmbuilder)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is LLMBuilder?

**LLMBuilder** is a framework for training and fine-tuning Large Language Models (LLMs). It provides a complete pipeline to go from raw documents to deployable models, with support for both CPU and GPU training.

### Key Features

- **Easy to Use**: Simple commands to train and deploy models
- **Multi-Format Support**: Process HTML, Markdown, EPUB, PDF, TXT files
- **Complete Pipeline**: From data processing to model deployment
- **Flexible**: Works on both CPU and GPU

## Quick Start

```bash
# Install LLMBuilder
pip install llmbuilder

# Create a new project
llmbuilder init my_project

# Navigate to your project
cd my_project

# Follow the step-by-step instructions in README.md
```

## Simple Example

```python
import llmbuilder as lb

# Load configuration
cfg = lb.load_config(preset="cpu_small")

# Build model
model = lb.build_model(cfg.model)

# Prepare data
from llmbuilder.data import TextDataset
dataset = TextDataset("data.txt", block_size=cfg.model.max_seq_length)

# Train model
results = lb.train_model(model, dataset, cfg.training)

# Generate text
text = lb.generate_text(
    model_path="./checkpoints/model.pt",
    tokenizer_path="./tokenizers",
    prompt="The future of AI is",
    max_new_tokens=50
)
print(text)
```

## Getting Started

1. **[Installation](getting-started/installation.md)** - Install LLMBuilder
2. **[Quick Start](getting-started/quickstart.md)** - Train your first model
3. **[User Guide](user-guide/configuration.md)** - Learn all features

## Community & Support

- **GitHub**: [Qubasehq/llmbuilder](https://github.com/Qubasehq/llmbuilder)
- **Issues**: [Report bugs](https://github.com/Qubasehq/llmbuilder/issues)

<div align="center">
  <p>Built by <strong>Qub△se</strong></p>
</div>