# LLMBuilder Documentation

LLMBuilder is a comprehensive toolkit for training, fine-tuning, and deploying Large Language Models. This documentation provides detailed information on all features and capabilities.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Command Reference](#command-reference)
5. [Workflows and Pipelines](#workflows-and-pipelines)
6. [Configuration](#configuration)
7. [Examples](#examples)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Installation

```bash
# Install from PyPI
pip install llmbuilder

# Or install with GPU support
pip install llmbuilder[gpu]

# Or install for development
pip install llmbuilder[dev]
```

### 2. Create a New Project

```bash
# Create a new project
llmbuilder init my-llm-project

# Navigate to project directory
cd my-llm-project

# Check project structure
llmbuilder config list
```

### 3. Prepare Your Data

```bash
# Prepare training data
llmbuilder data prepare --input ./raw_data --output ./processed_data

# Split data for training
llmbuilder data split --input ./processed_data --ratios 0.8,0.1,0.1
```

### 4. Select a Base Model

```bash
# List available models
llmbuilder model list

# Select a model from Hugging Face
llmbuilder model select microsoft/DialoGPT-medium

# Or use a local GGUF model
llmbuilder model select --local ./models/my-model.gguf
```

### 5. Start Training

```bash
# Start training with default settings
llmbuilder train start

# Or with custom parameters
llmbuilder train start --epochs 5 --batch-size 8 --learning-rate 2e-5
```

### 6. Evaluate Your Model

```bash
# Run evaluation
llmbuilder eval run --model ./checkpoints/final --data ./data/test

# Run benchmarks
llmbuilder eval benchmark --model ./checkpoints/final
```

### 7. Deploy Your Model

```bash
# Start API server
llmbuilder deploy start --model ./checkpoints/final --port 8000

# Or create a deployment package
llmbuilder deploy package --model ./checkpoints/final --output ./deployment
```

## Installation

### System Requirements

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

### Installation Options

#### Standard Installation

```bash
pip install llmbuilder
```

#### GPU Support

```bash
pip install llmbuilder[gpu]
```

This includes:
- CUDA-enabled PyTorch
- bitsandbytes for quantization
- accelerate for distributed training

#### Development Installation

```bash
pip install llmbuilder[dev]
```

This includes all development tools:
- pytest for testing
- black for code formatting
- mypy for type checking
- Documentation tools

#### From Source

```bash
git clone https://github.com/qubase/llmbuilder.git
cd llmbuilder
pip install -e .
```

## Project Structure

When you create a new project with `llmbuilder init`, the following structure is created:

```
my-project/
├── .llmbuilder/           # Project metadata and workflows
│   ├── config.json        # Project configuration
│   ├── workflows/         # Saved workflows
│   └── logs/             # Project logs
├── data/                 # Training data
│   ├── raw/              # Original data files
│   ├── processed/        # Cleaned and processed data
│   ├── train/            # Training split
│   ├── val/              # Validation split
│   └── test/             # Test split
├── models/               # Model files
│   ├── base/             # Base models
│   └── checkpoints/      # Training checkpoints
├── configs/              # Configuration files
│   ├── training.json     # Training configuration
│   ├── model.json        # Model configuration
│   └── deployment.json   # Deployment configuration
├── scripts/              # Custom scripts
├── docs/                 # Project documentation
└── README.md            # Project readme
```

## Command Reference

### Core Commands

#### `llmbuilder init`
Initialize a new LLM project.

```bash
llmbuilder init [PROJECT_NAME] [OPTIONS]

Options:
  --template TEXT     Project template (default, research, production, fine-tuning)
  --model TEXT        Base model to use
  --description TEXT  Project description
  --force            Overwrite existing project
```

#### `llmbuilder config`
Manage project configuration.

```bash
# Set configuration values
llmbuilder config set training.epochs 10
llmbuilder config set model.max_length 512

# Get configuration values
llmbuilder config get training.epochs
llmbuilder config get --all

# List all configurations
llmbuilder config list

# Reset to defaults
llmbuilder config reset
```

### Data Management

#### `llmbuilder data prepare`
Prepare and clean training data.

```bash
llmbuilder data prepare [OPTIONS]

Options:
  --input PATH        Input directory or file
  --output PATH       Output directory
  --formats TEXT      File formats to process (pdf,txt,json,csv)
  --clean            Enable advanced cleaning
  --deduplicate      Enable deduplication
  --min-length INT   Minimum text length
  --max-length INT   Maximum text length
```

#### `llmbuilder data split`
Split data into train/validation/test sets.

```bash
llmbuilder data split [OPTIONS]

Options:
  --input PATH           Input data directory
  --output PATH          Output directory
  --ratios TEXT          Split ratios (default: 0.8,0.1,0.1)
  --stratify            Use stratified splitting
  --seed INT            Random seed for reproducibility
```

#### `llmbuilder data validate`
Validate data quality and format.

```bash
llmbuilder data validate [OPTIONS]

Options:
  --input PATH          Data directory to validate
  --report PATH         Output validation report
  --fix                Automatically fix issues
```

### Model Management

#### `llmbuilder model select`
Select and download models.

```bash
llmbuilder model select [MODEL_NAME] [OPTIONS]

Options:
  --local PATH          Use local model file
  --output PATH         Download directory
  --format TEXT         Model format (huggingface, gguf)
  --revision TEXT       Model revision/branch
```

#### `llmbuilder model list`
List available models.

```bash
llmbuilder model list [OPTIONS]

Options:
  --source TEXT         Model source (huggingface, local, all)
  --filter TEXT         Filter by name or tag
  --details            Show detailed information
```

#### `llmbuilder model info`
Show model information.

```bash
llmbuilder model info [MODEL_NAME] [OPTIONS]

Options:
  --local PATH          Local model path
  --detailed           Show detailed architecture info
```

### Training

#### `llmbuilder train start`
Start model training.

```bash
llmbuilder train start [OPTIONS]

Options:
  --data PATH           Training data directory
  --model PATH          Base model path
  --output PATH         Output directory for checkpoints
  --config PATH         Training configuration file
  --epochs INT          Number of training epochs
  --batch-size INT      Training batch size
  --learning-rate FLOAT Learning rate
  --method TEXT         Training method (full, lora, qlora)
  --resume PATH         Resume from checkpoint
```

#### `llmbuilder train resume`
Resume training from checkpoint.

```bash
llmbuilder train resume [CHECKPOINT_PATH] [OPTIONS]

Options:
  --epochs INT          Additional epochs to train
  --learning-rate FLOAT New learning rate
```

#### `llmbuilder train stop`
Stop running training.

```bash
llmbuilder train stop [SESSION_ID]
```

### Evaluation

#### `llmbuilder eval run`
Run model evaluation.

```bash
llmbuilder eval run [OPTIONS]

Options:
  --model PATH          Model to evaluate
  --data PATH           Test data directory
  --metrics TEXT        Metrics to compute (perplexity,bleu,rouge)
  --output PATH         Output report path
  --batch-size INT      Evaluation batch size
```

#### `llmbuilder eval benchmark`
Run standardized benchmarks.

```bash
llmbuilder eval benchmark [OPTIONS]

Options:
  --model PATH          Model to benchmark
  --suite TEXT          Benchmark suite (glue, superglue, custom)
  --output PATH         Results output path
```

#### `llmbuilder eval compare`
Compare multiple models.

```bash
llmbuilder eval compare [MODEL1] [MODEL2] [OPTIONS]

Options:
  --data PATH           Test data for comparison
  --metrics TEXT        Metrics to compare
  --output PATH         Comparison report path
```

### Optimization

#### `llmbuilder optimize quantize`
Quantize models for deployment.

```bash
llmbuilder optimize quantize [OPTIONS]

Options:
  --model PATH          Model to quantize
  --format TEXT         Output format (gguf, int8, q4, q5)
  --output PATH         Output path
  --calibration PATH    Calibration data path
```

#### `llmbuilder optimize prune`
Prune model weights.

```bash
llmbuilder optimize prune [OPTIONS]

Options:
  --model PATH          Model to prune
  --ratio FLOAT         Pruning ratio (0.0-1.0)
  --method TEXT         Pruning method (magnitude, structured)
  --output PATH         Output path
```

### Deployment

#### `llmbuilder deploy start`
Start model serving.

```bash
llmbuilder deploy start [OPTIONS]

Options:
  --model PATH          Model to serve
  --port INT            Server port (default: 8000)
  --host TEXT           Server host (default: localhost)
  --workers INT         Number of worker processes
  --api-key TEXT        API key for authentication
```

#### `llmbuilder deploy package`
Create deployment package.

```bash
llmbuilder deploy package [OPTIONS]

Options:
  --model PATH          Model to package
  --output PATH         Output package path
  --format TEXT         Package format (docker, zip, tar)
  --include TEXT        Additional files to include
```

#### `llmbuilder deploy export-mobile`
Export for mobile deployment.

```bash
llmbuilder deploy export-mobile [OPTIONS]

Options:
  --model PATH          Model to export
  --platform TEXT       Target platform (android, ios, both)
  --output PATH         Output directory
  --optimize           Apply mobile optimizations
```

### Pipeline Execution

#### `llmbuilder pipeline train`
Create and execute training pipeline.

```bash
llmbuilder pipeline train [NAME] [OPTIONS]

Options:
  --data-path PATH      Training data path
  --model TEXT          Base model name
  --output-dir PATH     Output directory
  --epochs INT          Training epochs
  --batch-size INT      Batch size
  --dry-run            Show steps without executing
```

#### `llmbuilder pipeline deploy`
Create and execute deployment pipeline.

```bash
llmbuilder pipeline deploy [NAME] [OPTIONS]

Options:
  --model-path PATH     Trained model path
  --type TEXT           Deployment type (api, mobile)
  --optimize           Optimize before deployment
  --dry-run            Show steps without executing
```

#### `llmbuilder pipeline run`
Execute a workflow.

```bash
llmbuilder pipeline run [WORKFLOW_ID] [OPTIONS]

Options:
  --step INT            Start from specific step
  --continue-on-error   Continue on step failure
```

#### `llmbuilder pipeline list`
List all workflows.

```bash
llmbuilder pipeline list [OPTIONS]

Options:
  --status TEXT         Filter by status (all, running, completed, failed)
```

### Monitoring

#### `llmbuilder monitor dashboard`
Launch monitoring dashboard.

```bash
llmbuilder monitor dashboard [OPTIONS]

Options:
  --port INT            Dashboard port (default: 8080)
  --host TEXT           Dashboard host
  --data-dir PATH       Monitoring data directory
```

#### `llmbuilder monitor logs`
View and search logs.

```bash
llmbuilder monitor logs [OPTIONS]

Options:
  --level TEXT          Log level filter
  --since TEXT          Show logs since timestamp
  --follow             Follow log output
  --search TEXT        Search pattern
```

### Tools and Extensions

#### `llmbuilder tools register`
Register custom tools.

```bash
llmbuilder tools register [TOOL_PATH] [OPTIONS]

Options:
  --name TEXT           Tool name
  --description TEXT    Tool description
  --schema PATH         JSON schema file
```

#### `llmbuilder tools list`
List registered tools.

```bash
llmbuilder tools list [OPTIONS]

Options:
  --category TEXT       Filter by category
  --status TEXT         Filter by status
```

### Help and Maintenance

#### `llmbuilder help`
Show interactive help.

```bash
llmbuilder help [COMMAND]
```

#### `llmbuilder upgrade`
Update LLMBuilder.

```bash
llmbuilder upgrade [OPTIONS]

Options:
  --check              Check for updates only
  --pre-release        Include pre-release versions
```

## Workflows and Pipelines

LLMBuilder supports creating and executing complex workflows that chain multiple commands together.

### Creating Workflows

#### Training Pipeline

```bash
# Create a complete training pipeline
llmbuilder pipeline train my-training \
  --data-path ./raw_data \
  --model microsoft/DialoGPT-medium \
  --output-dir ./output \
  --epochs 5 \
  --batch-size 8

# Execute the pipeline
llmbuilder pipeline run my-training_<timestamp>
```

#### Deployment Pipeline

```bash
# Create a deployment pipeline
llmbuilder pipeline deploy my-deployment \
  --model-path ./checkpoints/final \
  --type api \
  --optimize

# Execute the pipeline
llmbuilder pipeline run my-deployment_<timestamp>
```

### Custom Workflows

You can create custom workflows by defining steps in JSON:

```json
{
  "name": "custom-workflow",
  "steps": [
    {
      "command": "data prepare",
      "args": {
        "input": "./raw_data",
        "output": "./processed_data",
        "clean": true
      }
    },
    {
      "command": "model select",
      "args": {
        "model": "gpt2",
        "output": "./models"
      }
    },
    {
      "command": "train start",
      "args": {
        "data": "./processed_data",
        "model": "./models",
        "epochs": 3
      }
    }
  ]
}
```

## Configuration

LLMBuilder uses a hierarchical configuration system:

1. Command-line arguments (highest priority)
2. Project configuration (`.llmbuilder/config.json`)
3. User configuration (`~/.llmbuilder/config.json`)
4. Default configuration (lowest priority)

### Configuration Structure

```json
{
  "project": {
    "name": "my-project",
    "version": "1.0.0",
    "description": "My LLM project"
  },
  "model": {
    "architecture": "gpt",
    "vocab_size": 50257,
    "max_length": 1024,
    "embedding_dim": 768,
    "num_layers": 12,
    "num_heads": 12
  },
  "training": {
    "batch_size": 4,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "optimizer": "adamw",
    "scheduler": "linear",
    "mixed_precision": true,
    "gradient_checkpointing": false
  },
  "data": {
    "input_formats": ["txt", "json", "csv"],
    "max_length": 512,
    "min_length": 10,
    "preprocessing": {
      "clean": true,
      "deduplicate": true,
      "normalize": true
    }
  },
  "deployment": {
    "api_framework": "fastapi",
    "host": "localhost",
    "port": 8000,
    "workers": 1,
    "cors_enabled": true
  }
}
```

### Environment Variables

You can also use environment variables for configuration:

```bash
export LLMBUILDER_MODEL_MAX_LENGTH=1024
export LLMBUILDER_TRAINING_BATCH_SIZE=8
export LLMBUILDER_API_PORT=8080
```

## Examples

### Example 1: Fine-tuning GPT-2 on Custom Data

```bash
# 1. Create project
llmbuilder init gpt2-finetune --template fine-tuning

# 2. Prepare data
llmbuilder data prepare \
  --input ./my_texts \
  --output ./data/processed \
  --formats txt,json \
  --clean \
  --deduplicate

# 3. Split data
llmbuilder data split \
  --input ./data/processed \
  --ratios 0.8,0.1,0.1

# 4. Select base model
llmbuilder model select gpt2

# 5. Configure training
llmbuilder config set training.epochs 5
llmbuilder config set training.batch_size 4
llmbuilder config set training.learning_rate 5e-5

# 6. Start training
llmbuilder train start \
  --method lora \
  --data ./data/train \
  --output ./checkpoints

# 7. Evaluate
llmbuilder eval run \
  --model ./checkpoints/final \
  --data ./data/test

# 8. Deploy
llmbuilder deploy start \
  --model ./checkpoints/final \
  --port 8000
```

### Example 2: Using Pipelines for Automation

```bash
# Create and execute a complete pipeline
llmbuilder pipeline train automated-training \
  --data-path ./datasets/my_data \
  --model microsoft/DialoGPT-medium \
  --output-dir ./experiments/exp1 \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 2e-5

# Monitor pipeline execution
llmbuilder pipeline list
llmbuilder pipeline status automated-training_<timestamp>

# Deploy the trained model
llmbuilder pipeline deploy automated-deployment \
  --model-path ./experiments/exp1/checkpoints/final \
  --type api \
  --optimize
```

### Example 3: Model Optimization and Deployment

```bash
# Quantize model for deployment
llmbuilder optimize quantize \
  --model ./checkpoints/final \
  --format gguf \
  --output ./optimized/model.gguf

# Create deployment package
llmbuilder deploy package \
  --model ./optimized/model.gguf \
  --format docker \
  --output ./deployment

# Export for mobile
llmbuilder deploy export-mobile \
  --model ./optimized/model.gguf \
  --platform android \
  --output ./mobile
```

## Advanced Usage

### Custom Training Configurations

Create custom training configurations for different scenarios:

```json
{
  "lora_config": {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1
  },
  "qlora_config": {
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4"
  }
}
```

### Custom Data Processing

Extend data processing with custom modules:

```python
from llmbuilder.core.data import DataProcessor

class CustomProcessor(DataProcessor):
    def process_text(self, text: str) -> str:
        # Custom processing logic
        return processed_text

# Register custom processor
llmbuilder.register_processor("custom", CustomProcessor)
```

### Tool Integration

Register custom tools for extended functionality:

```python
def my_custom_tool(input_data: str) -> str:
    """Custom tool for processing data."""
    # Tool implementation
    return processed_data

# Register tool
llmbuilder tools register ./my_tool.py --name custom-processor
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce batch size
llmbuilder config set training.batch_size 2

# Enable gradient checkpointing
llmbuilder config set training.gradient_checkpointing true

# Use mixed precision
llmbuilder config set training.mixed_precision true
```

#### Model Loading Issues

```bash
# Check model compatibility
llmbuilder model info <model_name>

# Verify model files
llmbuilder data validate --input ./models

# Clear model cache
llmbuilder config reset model
```

#### Training Convergence Issues

```bash
# Adjust learning rate
llmbuilder config set training.learning_rate 1e-5

# Change optimizer
llmbuilder config set training.optimizer adamw

# Enable learning rate scheduling
llmbuilder config set training.scheduler cosine
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
llmbuilder --verbose <command>
llmbuilder monitor debug
```

### Getting Help

- Use `llmbuilder help <command>` for command-specific help
- Check logs with `llmbuilder monitor logs`
- Run diagnostics with `llmbuilder monitor debug`
- Visit the documentation at https://llmbuilder.readthedocs.io
- Report issues at https://github.com/qubase/llmbuilder/issues

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for information on contributing to LLMBuilder.

## License

LLMBuilder is released under the MIT License. See [LICENSE](../LICENSE) for details.