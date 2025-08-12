# Contributing to LLMBuilder

Thank you for your interest in contributing to LLMBuilder! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be collaborative**: Work together to improve the project
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone has different experience levels

## Getting Started

### Prerequisites

- **Python 3.8+**: Required for development
- **Git**: For version control
- **Tesseract OCR**: For PDF processing features
- **Virtual Environment**: Recommended for isolation

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/qubasehq/LLMBuilder.git
   cd LLMBuilder
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Install System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-eng
   
   # macOS
   brew install tesseract
   
   # Windows - see INSTALL_TESSERACT.md
   ```

4. **Verify Installation**
   ```bash
   python run_tests.py --deps  # Check dependencies
   python run_tests.py --unit  # Run unit tests
   ```

5. **Set Up Pre-commit Hooks** (Optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Branch Strategy

- **main**: Stable release branch
- **develop**: Development integration branch
- **feature/**: Feature development branches
- **bugfix/**: Bug fix branches
- **hotfix/**: Critical fixes for production

### Workflow Steps

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our coding standards
   - Add tests for new functionality
   - Update documentation as needed
   - Run tests locally

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add enhanced document ingestion"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

### Commit Message Format

We use conventional commits for clear history:

<type>(<scdescription>

[optional body]

[optional fo```

**Types:**
- `feat`: New feature
- `fix`g fix
- `docs`: Documentatioges
 Code le changes
- `rer`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maie tasks

**Examples:**
```
feat(ingestion): add EPUB sut with metadata extraction
fix(dedup): resolve memory leak in embedding deduplication
docs(readme): update installation instructions
test(integration): add end-to-end pipeline tests
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line Length**: 100 characters (not 79)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Use type hints for all public functions

### Code Formatting

We use automated formatting tools:

```bash
# Install formatting tools
pip install black isort flake8 mypy

# Format code
black .
isort .

# Check style
flake8 .
mypy .
```

### Example Code Style

```python
#!/usr/bin/env python3
"""
Module docstring describing the purpose.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np

from data.ingest import DocumentIngester


class ExampleClass:
    """Example class following our coding standards.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Attributes:
        attribute1: Description of attribute 1
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        self.param1 = param1
        self.param2 = param2 or 0
        self.attribute1: List[str] = []
    
    def example_method(self, input_data: Dict[str, Any]) -> List[str]:
        """Example method with proper documentation.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            List of processed strings
            
        Raises:
            ValueError: If input_data is invalid
        """
        if not input_data:
            raise ValueError("input_data cannot be empty")
        
        results = []
        for key, value in input_data.items():
            processed = self._process_item(key, value)
            results.append(processed)
        
        return results
    
    def _process_item(self, key: str, value: Any) -> str:
        """Private method for processing individual items."""
        return f"{key}: {value}"
```

### File Organization

```
project/
├── module/
│   ├── __init__.py          # Module initialization
│   ├── core.py              # Core functionality
│   ├── utils.py             # Utility functions
│   └── exceptions.py        # Custom exceptions
├── tests/
│   ├── test_module.py       # Module tests
│   └── conftest.py          # Test configuration
└── docs/
    └── module.md            # Module documentation
```

## Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Test speed and memory usage
4. **End-to-End Tests**: Test complete workflows

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from module.core import ExampleClass


class TestExampleClass:
    """Test suite for ExampleClass."""
    
    @pytest.fixture
    def example_instance(self):
        """Create test instance."""
        return ExampleClass("test_param")
    
    def test_initialization(self, example_instance):
        """Test proper initialization."""
        assert example_instance.param1 == "test_param"
        assert example_instance.param2 == 0
        assert example_instance.attribute1 == []
    
    def test_example_method_success(self, example_instance):
        """Test successful method execution."""
        input_data = {"key1": "value1", "key2": "value2"}
        result = example_instance.example_method(input_data)
        
        assert len(result) == 2
        assert "key1: value1" in result
        assert "key2: value2" in result
    
    def test_example_method_empty_input(self, example_instance):
        """Test method with empty input."""
        with pytest.raises(ValueError, match="input_data cannot be empty"):
            example_instance.example_method({})
    
    @pytest.mark.slow
    def test_performance_benchmark(self, example_instance):
        """Test performance with large input."""
        large_input = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        import time
        start_time = time.time()
        result = example_instance.example_method(large_input)
        duration = time.time() - start_time
        
        assert len(result) == 1000
        assert duration < 1.0  # Should complete within 1 second
```

### Test Execution

```bash
# Run all tests
python run_tests.py --all

# Run specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --performance

# Run tests with coverage
python run_tests.py --coverage

# Run specific test file
pytest tests/test_example.py -v

# Run tests matching pattern
pytest -k "test_example" -v
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.slow          # Slow-running tests
@pytest.mark.integration   # Integration tests
@pytest.mark.performance   # Performance tests
@pytest.mark.gpu          # GPU-required tests
@pytest.mark.ocr          # OCR-dependent tests
```

## Documentation Guidelines

### Documentation Types

1. **Code Documentation**: Docstrings and comments
2. **User Documentation**: README, USAGE guides
3. **Developer Documentation**: CONTRIBUTING, architecture docs
4. **API Documentation**: Auto-generated from docstrings

### Docstring Standards

Use Google-style docstrings:

```python
def process_documents(input_dir: str, output_dir: str, 
                     recursive: bool = False) -> Dict[str, Any]:
    """Process documents from input directory.
    
    This function processes all supported document formats in the input
    directory and saves cleaned text to the output directory.
    
    Args:
        input_dir: Path to directory containing input documents
        output_dir: Path to directory for output files
        recursive: Whether to process subdirectories recursively
        
    Returns:
        Dictionary containing processing statistics:
            - total_files: Number of files found
            - processed_count: Number of successfully processed files
            - failed_count: Number of failed files
            - total_characters: Total characters extracted
            
    Raises:
        FileNotFoundError: If input_dir does not exist
        PermissionError: If unable to write to output_dir
        
    Example:
        >>> results = process_documents("data/raw", "data/cleaned")
        >>> print(f"Processed {results['processed_count']} files")
        Processed 25 files
        
    Note:
        Supported formats include PDF, DOCX, HTML, EPUB, and TXT files.
        OCR is automatically applied to scanned PDFs.
    """
```

### Documentation Updates

When making changes:

1. **Update docstrings** for modified functions
2. **Update README.md** for new features
3. **Update USAGE.md** for new usage patterns
4. **Add examples** for new functionality
5. **Update configuration docs** for new settings

## Pull Request Process

### Before Submitting

1. **Run Tests**
   ```bash
   python run_tests.py --all
   ```

2. **Check Code Style**
   ```bash
   black --check .
   isort --check-only .
   flake8 .
   ```

3. **Update Documentation**
   - Add/update docstrings
   - Update relevant markdown files
   - Add usage examples

4. **Write Good Commit Messages**
   - Use conventional commit format
   - Include clear descriptions
   - Reference issues when applicable

### PR Template

When creating a pull request, include:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Docstrings updated
- [ ] README.md updated (if needed)
- [ ] USAGE.md updated (if needed)
- [ ] Examples added (if needed)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] No new warnings introduced
- [ ] Changes are backward compatible (or breaking changes documented)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code quality
3. **Testing**: Reviewers test functionality
4. **Documentation Review**: Check documentation updates
5. **Approval**: At least one maintainer approval required

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- LLMBuilder Version: [e.g. 1.2.0]

**Additional Context**
Any other context about the problem.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the feature you'd like to see.

**Use Case**
Describe the use case and why this feature would be valuable.

**Proposed Solution**
Describe how you envision this feature working.

**Alternatives Considered**
Describe any alternative solutions you've considered.

**Additional Context**
Any other context or screenshots about the feature request.
```

## Release Process

### Version Numbering

We use semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update Version Numbers**
2. **Update CHANGELOG.md**
3. **Create Release Branch**
4. **Run Full Test Suite**
5. **Create GitHub Release**
6. **Update Documentation**

## Development Tools

### Recommended IDE Setup

**VS Code Extensions:**
- Python
- Pylance
- Black Formatter
- isort
- GitLens
- Markdown All in One

**PyCharm Configuration:**
- Enable PEP 8 inspections
- Configure Black as external tool
- Set up pytest as test runner

### Useful Commands

```bash
# Development workflow
git checkout develop
git pull origin develop
git checkout -b feature/new-feature

# Code quality
black .
isort .
flake8 .
mypy .

# Testing
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --coverage

# Documentation
python -m pydoc -w module_name
```

## Getting Help

- **Documentation**: Check README.md and USAGE.md
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask for help in pull requests

## Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **GitHub**: Contributor statistics and graphs

Thank you for contributing to LLMBuilder! Your contributions help make this project better for everyone.