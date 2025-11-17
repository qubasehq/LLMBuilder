# Quick Start

Get up and running with LLMBuilder quickly! This guide will walk you through training your first language model.

## Simple Setup

### Step 1: Install LLMBuilder

```bash
pip install llmbuilder
```

### Step 2: Create a New Project

```bash
# Create a new project
llmbuilder init my_project

# Navigate to your project
cd my_project
```

### Step 3: Prepare Your Data

Add your text data files to the `data/raw/` directory.

### Step 4: Process Your Data

```bash
llmbuilder data load -i data/raw -o data/processed/input.txt --clean
```

### Step 5: Train a Tokenizer

```bash
llmbuilder tokenizer train -i data/processed/input.txt -o tokenizer/ --vocab-size 16000
```

### Step 6: Train Your Model

```bash
llmbuilder train model -d data/processed/input.txt -t tokenizer/ -o models/checkpoints
```

### Step 7: Generate Text

```bash
llmbuilder generate text -m models/checkpoints/latest.pt -t tokenizer/ -p "Artificial intelligence" --max-tokens 50
```

🎉 **Congratulations!** You've just trained and used your first language model with LLMBuilder!

## Python API Quick Start

Prefer Python code? Here's the same workflow using the Python API:

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
    prompt="Artificial intelligence",
    max_new_tokens=50
)
print(text)
```

## What Just Happened?

1. **Project Creation**: Set up the directory structure
2. **Data Processing**: Loaded and cleaned your text data
3. **Tokenization**: Created a vocabulary and tokenized the text
4. **Model Training**: Trained a transformer model
5. **Text Generation**: Used the model to generate new text

## Next Steps

- **[Installation Guide](installation.md)** - Detailed installation instructions
- **[User Guide](../user-guide/configuration.md)** - Learn about configuration options
- **[CLI Reference](../cli/overview.md)** - Complete CLI documentation

<div align="center">
  <p>You now have a working LLMBuilder setup! Try experimenting with different data and parameters.</p>
</div>