# Model Training

This guide covers training language models with LLMBuilder.

## Quick Start Training

### Basic Training Command

```bash
llmbuilder train model \
  --data training_data.txt \
  --tokenizer ./tokenizer \
  --output ./model \
  --epochs 10 \
  --batch-size 16
```

### Python API Training

```python
import llmbuilder as lb

# Load configuration
config = lb.load_config(preset="cpu_small")

# Build model
model = lb.build_model(config.model)

# Prepare dataset
from llmbuilder.data import TextDataset
dataset = TextDataset("training_data.txt", block_size=config.model.max_seq_length)

# Train model
results = lb.train_model(model, dataset, config.training)
```

## Training Configuration

### Core Training Parameters

```python
from llmbuilder.config import TrainingConfig

config = TrainingConfig(
    # Basic settings
    batch_size=16,              # Samples per training step
    num_epochs=10,              # Number of training epochs
    learning_rate=3e-4,         # Learning rate

    # Optimization
    optimizer="adamw",          # Optimizer type
    weight_decay=0.01,          # Weight decay for regularization
    max_grad_norm=1.0,          # Gradient clipping

    # Scheduling
    warmup_steps=1000,          # Learning rate warmup steps

    # Checkpointing
    save_every=1000,            # Save checkpoint every N steps
    eval_every=500,             # Evaluate every N steps
)
```

## Monitoring Training

### Progress Tracking

```bash
# Monitor training progress
tail -f ./model/training.log

# Or use the built-in progress display
llmbuilder train model --data data.txt --output model/ --verbose
```

## Common Training Issues

### Out of Memory

```bash
# Reduce batch size
llmbuilder train model --data data.txt --output model/ --batch-size 1

# Use CPU-only mode
llmbuilder train model --data data.txt --output model/ --device cpu
```

### Slow Training

```bash
# Use GPU if available
llmbuilder train model --data data.txt --output model/ --device cuda

# Reduce model size
llmbuilder train model --data data.txt --output model/ --layers 4 --dim 256
```

## Next Steps

- **[Fine-tuning Guide](fine-tuning.md)** - Fine-tune models
- **[Generation Guide](generation.md)** - Generate text
- **[Export Guide](export.md)** - Export models

<div align="center">
  <p>Start with small models and datasets to learn the workflow.</p>
</div>