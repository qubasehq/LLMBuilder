# Configuration

LLMBuilder uses a configuration system that makes it easy to customize your model training.

## Configuration Overview

LLMBuilder configurations are organized into sections:
- Model Config - Architecture settings
- Training Config - Training parameters
- Data Config - Data processing settings

## Configuration Methods

### Method 1: Using Templates (Recommended)

LLMBuilder provides pre-configured templates:

```bash
# List available templates
llmbuilder config templates

# Create configuration from template
llmbuilder config from-template basic_config --output my_config.json
```

Available templates:
- `basic_config` - General purpose
- `cpu_optimized_config` - CPU training
- `advanced_processing_config` - Full features

### Method 2: Using Code Presets

```python
from llmbuilder.config.defaults import DefaultConfigs

# Load a preset configuration
config = DefaultConfigs.get_preset("cpu_small")
```

### Method 3: From Configuration File

```python
from llmbuilder.config.manager import load_config

# Load from JSON file
config = load_config("my_config.json")
```

## Model Configuration

### Core Settings

```json
{
  "model": {
    "vocab_size": 16000,
    "num_layers": 12,
    "num_heads": 12,
    "embedding_dim": 768,
    "max_seq_length": 1024,
    "dropout": 0.1
  }
}
```

**`vocab_size`**: Size of the vocabulary
**`num_layers`**: Number of transformer layers
**`num_heads`**: Number of attention heads
**`embedding_dim`**: Dimension of token embeddings
**`max_seq_length`**: Maximum sequence length
**`dropout`**: Dropout rate for regularization

## Training Configuration

### Basic Settings

```json
{
  "training": {
    "batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 3e-4,
    "weight_decay": 0.01
  }
}
```

**`batch_size`**: Number of samples per training step
**`learning_rate`**: Step size for parameter updates
**`num_epochs`**: Number of training epochs

## Next Steps

- **[Training Guide](training.md)** - Learn about training models
- **[Data Processing](data-processing.md)** - Process your data
- **[Tokenization](tokenization.md)** - Train tokenizers

<div align="center">
  <p>Start with the basic templates and modify as needed.</p>
</div>