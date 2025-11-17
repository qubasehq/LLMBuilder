# Installation

This guide will help you install LLMBuilder and set up your environment.

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB+ recommended)
- **Storage**: 2GB free space
- **OS**: Windows, macOS, or Linux

## Installation

### Simple Installation

The easiest way to install LLMBuilder is via PyPI:

```bash
pip install llmbuilder
```

### Verify Installation

Test your installation by running:

```bash
# Check if LLMBuilder is installed
python -c "import llmbuilder; print(f'LLMBuilder {llmbuilder.__version__} installed successfully!')"

# Test the CLI
llmbuilder --version
```

You should see output showing the version number.

## Optional Dependencies

### For GPU Training

If you have an NVIDIA GPU:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For Document Processing

For processing PDF, DOCX, and other document formats:

```bash
pip install llmbuilder[data]
```

## Environment Setup

### Virtual Environment (Recommended)

Create a dedicated virtual environment:

```bash
# Create virtual environment
python -m venv llmbuilder-env

# Activate it
# On Windows:
llmbuilder-env\Scripts\activate
# On macOS/Linux:
source llmbuilder-env/bin/activate

# Install LLMBuilder
pip install llmbuilder
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'torch'

**Solution**: Install PyTorch first:

```bash
pip install torch
```

#### Permission denied errors

**Solution**: Use `--user` flag or virtual environment:

```bash
pip install --user llmbuilder
```

## Next Steps

Once you have LLMBuilder installed:

1. **[Quick Start](quickstart.md)** - Train your first model
2. **[User Guide](../user-guide/configuration.md)** - Learn about features

<div align="center">
  <p>For the best experience, we recommend using a virtual environment.</p>
</div>