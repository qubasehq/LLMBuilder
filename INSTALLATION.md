# LLMBuilder Installation Guide

## Quick Installation

### From PyPI (Recommended)

```bash
pip install llmbuilder
```

### With Optional Dependencies

```bash
# For GPU support
pip install llmbuilder[gpu]

# For development
pip install llmbuilder[dev]

# For documentation building
pip install llmbuilder[docs]

# For monitoring and tracking
pip install llmbuilder[monitoring]

# All optional dependencies
pip install llmbuilder[all]
```

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: At least 2GB free space

## Optional System Dependencies

### For OCR Support (Optional)

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

#### macOS
```bash
brew install tesseract
```

#### Windows
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### For GPU Support (Optional)

Ensure you have CUDA-compatible GPU drivers installed:
- NVIDIA CUDA 11.8 or higher
- cuDNN 8.0 or higher

## Installation Methods

### 1. Standard Installation

```bash
pip install llmbuilder
```

This installs the core package with essential dependencies.

### 2. Development Installation

For contributors and developers:

```bash
git clone https://github.com/qubase/llmbuilder.git
cd llmbuilder
pip install -e .[dev]
```

### 3. Custom Installation Script

Use the provided installation script for automatic dependency detection:

```bash
python scripts/install.py --auto
```

Options:
- `--auto`: Auto-detect and install recommended dependencies
- `--gpu`: Include GPU dependencies
- `--dev`: Include development dependencies
- `--all`: Install all optional dependencies

## Verification

After installation, verify that LLMBuilder is working:

```bash
# Check version
llmbuilder --version

# Run basic help
llmbuilder --help

# Initialize a test project
llmbuilder init test-project
```

## Troubleshooting

### Common Issues

#### Import Errors
If you encounter import errors, ensure you're using the correct Python environment:

```bash
python -c "import llmbuilder; print(llmbuilder.__version__)"
```

#### Permission Errors
On some systems, you might need to use `--user` flag:

```bash
pip install --user llmbuilder
```

#### GPU Dependencies
If GPU dependencies fail to install:

```bash
# Install CPU version first
pip install llmbuilder

# Then add GPU support separately
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install llmbuilder[gpu] --no-deps
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/qubase/llmbuilder/issues)
2. Run the verification script: `python scripts/verify_package.py`
3. Check system compatibility: `python scripts/install.py --auto`

## Uninstallation

To remove LLMBuilder:

```bash
pip uninstall llmbuilder
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade llmbuilder
```

Or use the built-in upgrade command:

```bash
llmbuilder upgrade
```
