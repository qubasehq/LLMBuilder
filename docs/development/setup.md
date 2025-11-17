# Development Setup

This guide will help you set up a development environment for contributing to LLMBuilder.

## Quick Setup

The fastest way to get started is using our automated setup script:

```bash
# Clone the repository
git clone https://github.com/Qubasehq/llmbuilder.git
cd llmbuilder

# Run the automated setup script
python scripts/setup_dev.py --full
```

This script will:

- Check your Python version
- Set up a virtual environment (if needed)
- Install LLMBuilder in development mode
- Install all development dependencies
- Install optional dependencies
- Set up pre-commit hooks
- Run initial tests and checks

## Manual Setup

If you prefer to set up manually or need more control:

### 1. Prerequisites

**Python Requirements:**

- Python 3.8 or higher
- pip (latest version recommended)
- git

**System Dependencies (Optional):**

- **Tesseract OCR**: For PDF processing with OCR
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
  - macOS: `brew install tesseract`
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

- **llama.cpp**: For GGUF model conversion

  ```bash
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && make
  export PATH=$PATH:$(pwd)
  ```

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Qubasehq/llmbuilder.git
cd llmbuilder

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install LLMBuilder in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install all optional dependencies (recommended for full testing)
pip install -e ".[all]"
```

### 4. Set Up Development Tools

```bash
# Install pre-commit hooks
pre-commit install

# Verify installation
pre-commit run --all-files
```

## Development Dependencies

LLMBuilder uses several dependency groups:

### Core Dependencies

```bash
pip install -e "."
```

Installs the basic LLMBuilder package with core functionality.

### Development Dependencies

```bash
pip install -e ".[dev]"
```

Includes:

- Testing: pytest, pytest-cov, pytest-xdist, pytest-mock
- Code quality: black, isort, flake8, mypy
- Pre-commit: pre-commit, bandit, pydocstyle
- Documentation: sphinx, mkdocs, mkdocs-material

### Optional Feature Dependencies

```bash
# PDF processing with OCR
pip install -e ".[pdf]"

# EPUB processing
pip install -e ".[epub]"

# HTML processing
pip install -e ".[html]"

# Semantic deduplication
pip install -e ".[semantic]"

# GGUF model conversion
pip install -e ".[conversion]"

# All optional features
pip install -e ".[all]"
```

## Verification

After setup, verify everything works:

```bash
# Run tests
python -m pytest tests/ -v

# Check code style
black --check llmbuilder/
isort --check-only llmbuilder/
flake8 llmbuilder/

# Run type checking
mypy llmbuilder/

# Test CLI
llmbuilder --help
llmbuilder info
```

## IDE Configuration

### VS Code

Recommended extensions:

- Python
- Pylance
- Black Formatter
- isort
- GitLens
- Markdown All in One

Settings (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm

1. Open the project directory
2. Configure Python interpreter to use the virtual environment
3. Enable pytest as the test runner
4. Install plugins: Black, IdeaVim (optional)
5. Configure code style to use Black formatting

## Environment Variables

Set these environment variables for development:

```bash
# Enable debug logging
export LLMBUILDER_LOG_LEVEL=DEBUG

# Enable slow tests (optional)
export RUN_SLOW=1

# Enable performance tests (optional)
export RUN_PERF=1

# Tesseract path (if not in system PATH)
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

## Common Issues

### Python Version Issues

```bash
# Check Python version
python --version

# If using pyenv
pyenv install 3.9.16
pyenv local 3.9.16
```

### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Dependency Conflicts

```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip uninstall llmbuilder
pip install -e ".[dev,all]"
```

### Pre-commit Issues

```bash
# Reinstall pre-commit hooks
pre-commit uninstall
pre-commit install

# Update pre-commit hooks
pre-commit autoupdate
```

### Test Failures

```bash
# Run tests with verbose output
python -m pytest tests/ -v -s

# Run specific test
python -m pytest tests/test_specific.py::test_function -v

# Run tests without optional dependencies
python -m pytest tests/ -m "not optional_deps"
```

## Docker Development (Optional)

For consistent development environments:

```dockerfile
# Dockerfile.dev
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install -e ".[dev,all]"

# Copy source code
COPY . .

# Install in development mode
RUN pip install -e .

# Set up pre-commit
RUN pre-commit install

CMD ["bash"]
```

```bash
# Build and run development container
docker build -f Dockerfile.dev -t llmbuilder-dev .
docker run -it -v $(pwd):/app llmbuilder-dev
```

## Performance Profiling

For performance development:

```bash
# Install profiling tools
pip install py-spy memory-profiler line-profiler

# Profile CPU usage
py-spy record -o profile.svg -- python your_script.py

# Profile memory usage
mprof run your_script.py
mprof plot

# Line-by-line profiling
kernprof -l -v your_script.py
```

## Documentation Development

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally
cd docs/
mkdocs serve

# Build documentation
mkdocs build

# Check for broken links
mkdocs build --strict
```

## Next Steps

After setting up your development environment:

1. **Read the contributing guide**: [Contributing Guide](../contributing/testing-guidelines.md)
2. **Explore the codebase**: Start with `llmbuilder/__init__.py`
3. **Run the examples**: Try `examples/basic_training.py`
4. **Check existing issues**: Look for "good first issue" labels
5. **Join discussions**: Participate in GitHub Discussions

## Getting Help

If you encounter issues during setup:

1. **Check this guide** for common solutions
2. **Search existing issues** on GitHub
3. **Ask in GitHub Discussions** for setup help
4. **Create an issue** if you find a bug in the setup process

---

Happy coding! 🚀
