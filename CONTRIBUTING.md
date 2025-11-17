# Contributing to LLMBuilder

Thank you for your interest in contributing to LLMBuilder! This guide will help you get started with contributing to our comprehensive toolkit for building, training, and deploying language models.

## 🎯 Overview

LLMBuilder is a production-ready framework that provides:

- Advanced data processing (multi-format ingestion, deduplication)
- Flexible tokenizer training (BPE, SentencePiece, Unigram, WordPiece)
- Model training and fine-tuning capabilities
- GGUF conversion and quantization
- Comprehensive CLI and configuration management

We welcome contributions in all areas, from bug fixes to new features, documentation improvements, and performance optimizations.

## 🚀 Quick Start for Contributors

### 1. Development Setup

```bash
# Clone the repository
git clone https://github.com/Qubasehq/llmbuilder.git
cd llmbuilder

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Verify Installation

```bash
# Run basic tests
python -m pytest tests/ -v

# Test CLI functionality
llmbuilder --help
llmbuilder info

# Test advanced features (requires optional dependencies)
python -c "import llmbuilder; print('✅ LLMBuilder installed')"
python -c "import sentence_transformers; print('✅ Semantic deduplication available')"
```

### 3. Development Dependencies

```bash
# Core development dependencies
pip install pytest pytest-cov black isort flake8 mypy pre-commit

# Documentation dependencies
pip install mkdocs mkdocs-material mkdocs-mermaid2-plugin

# Optional feature dependencies for testing
pip install pymupdf pytesseract ebooklib beautifulsoup4 lxml sentence-transformers
```

## 📋 Development Guidelines

### Code Style

We use automated code formatting and linting:

```bash
# Format code
black llmbuilder/ tests/ examples/
isort llmbuilder/ tests/ examples/

# Lint code
flake8 llmbuilder/ tests/ examples/
mypy llmbuilder/

# Run all checks
pre-commit run --all-files
```

**Code Style Rules:**

- **Line length**: 88 characters (Black default)
- **Import sorting**: isort with Black compatibility
- **Type hints**: Required for public APIs
- **Docstrings**: Google style for all public functions/classes
- **Variable naming**: snake_case for functions/variables, PascalCase for classes

### Testing

We maintain comprehensive test coverage:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=llmbuilder --cov-report=html

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/integration/   # Integration tests
python -m pytest tests/performance/   # Performance tests

# Run slow tests (requires RUN_SLOW=1)
RUN_SLOW=1 python -m pytest tests/

# Run performance tests (requires RUN_PERF=1)
RUN_PERF=1 python -m pytest tests/performance/
```

**Testing Guidelines:**

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark critical operations
- **Mock external dependencies**: Use mocks for file I/O, network calls
- **Test data**: Use small, synthetic data for fast tests
- **Parametrized tests**: Test multiple scenarios efficiently

### Documentation

Documentation is built with MkDocs:

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocs-mermaid2-plugin

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

**Documentation Guidelines:**

- **API documentation**: Comprehensive docstrings for all public APIs
- **Examples**: Working code examples for all major features
- **Tutorials**: Step-by-step guides for common workflows
- **Configuration**: Document all configuration options
- **Troubleshooting**: Include common issues and solutions

## 🔧 Development Workflow

### 1. Issue-Based Development

Before starting work:

1. **Check existing issues**: Look for related issues or discussions
2. **Create an issue**: Describe the problem or feature request
3. **Get feedback**: Discuss the approach with maintainers
4. **Assign yourself**: Avoid duplicate work

### 2. Branch Naming

Use descriptive branch names:

```bash
# Feature branches
git checkout -b feature/semantic-deduplication
git checkout -b feature/gguf-quantization

# Bug fix branches
git checkout -b fix/tokenizer-encoding-issue
git checkout -b fix/memory-leak-deduplication

# Documentation branches
git checkout -b docs/advanced-examples
git checkout -b docs/api-reference
```

### 3. Commit Messages

Follow conventional commit format:

```bash
# Feature commits
git commit -m "feat(data): add semantic deduplication with sentence-transformers"
git commit -m "feat(cli): add batch GGUF conversion command"

# Bug fix commits
git commit -m "fix(tokenizer): handle unicode encoding errors in SentencePiece"
git commit -m "fix(config): validate vocab_size consistency across components"

# Documentation commits
git commit -m "docs(examples): add multi-format ingestion tutorial"
git commit -m "docs(api): update configuration reference"

# Test commits
git commit -m "test(dedup): add integration tests for semantic deduplication"
git commit -m "test(gguf): add performance benchmarks for conversion"
```

### 4. Pull Request Process

1. **Create feature branch** from `main`
2. **Implement changes** with tests and documentation
3. **Run full test suite** and ensure all checks pass
4. **Update documentation** if needed
5. **Create pull request** using the template
6. **Address review feedback** promptly
7. **Squash and merge** after approval

## 🎯 Contribution Areas

### High-Priority Areas

1. **Performance Optimization**
   - Memory usage reduction for large datasets
   - GPU acceleration for semantic operations
   - Parallel processing improvements
   - Caching and streaming optimizations

2. **New Data Formats**
   - Additional document format support
   - Improved OCR integration
   - Better metadata extraction
   - Format-specific optimizations

3. **Advanced Features**
   - New tokenization algorithms
   - Additional quantization methods
   - Model architecture improvements
   - Export format support

4. **Testing and Quality**
   - Increase test coverage
   - Performance benchmarks
   - Integration test improvements
   - Error handling robustness

### Feature Requests

When implementing new features:

1. **Start with an issue**: Describe the feature and get feedback
2. **Design document**: For complex features, create a design document
3. **Incremental implementation**: Break large features into smaller PRs
4. **Comprehensive testing**: Include unit, integration, and performance tests
5. **Documentation**: Update all relevant documentation
6. **Examples**: Provide working examples for new features

### Bug Reports

When fixing bugs:

1. **Reproduce the issue**: Create a minimal reproduction case
2. **Write a failing test**: Test that demonstrates the bug
3. **Fix the issue**: Implement the fix
4. **Verify the fix**: Ensure the test now passes
5. **Check for regressions**: Run full test suite

## 📚 Architecture Overview

Understanding LLMBuilder's architecture helps with contributions:

```
llmbuilder/
├── data/                   # Data processing modules
│   ├── ingest.py          # Multi-format document ingestion
│   ├── dedup.py           # Deduplication (exact + semantic)
│   ├── pdf_processor.py   # PDF processing with OCR
│   └── __init__.py        # Data processing API
├── tokenizer/             # Tokenizer training and management
│   ├── trainer.py         # Tokenizer training implementations
│   ├── sentencepiece.py   # SentencePiece integration
│   └── huggingface.py     # Hugging Face tokenizers
├── model/                 # Model architecture and training
│   ├── gpt.py            # GPT-style transformer implementation
│   ├── training.py       # Training loop and optimization
│   └── __init__.py       # Model API
├── tools/                 # Conversion and export tools
│   ├── convert_to_gguf.py # GGUF conversion pipeline
│   └── __init__.py       # Tools API
├── config/                # Configuration management
│   ├── defaults.py       # Configuration classes and presets
│   ├── manager.py        # Configuration loading and validation
│   └── templates/        # Configuration templates
├── cli.py                 # Command-line interface
└── __init__.py           # Main package API
```

### Key Design Principles

1. **Modularity**: Each component is independently testable
2. **Configuration-driven**: Behavior controlled through configuration
3. **Extensibility**: Easy to add new formats, algorithms, etc.
4. **Performance**: Optimized for large-scale data processing
5. **Usability**: Simple APIs with sensible defaults

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/                  # Unit tests (fast, isolated)
│   ├── test_data/        # Data processing unit tests
│   ├── test_tokenizer/   # Tokenizer unit tests
│   ├── test_config/      # Configuration unit tests
│   └── test_cli/         # CLI unit tests
├── integration/           # Integration tests (slower, realistic)
│   ├── test_data_processing.py
│   ├── test_tokenizer_training_integration.py
│   └── test_pipeline_workflow.py
├── performance/           # Performance benchmarks
│   ├── test_deduplication_performance.py
│   └── test_ingestion_performance.py
└── fixtures/             # Test data and utilities
    ├── sample_data/      # Sample documents for testing
    └── conftest.py       # Pytest configuration
```

### Writing Tests

**Unit Test Example:**

```python
import pytest
from llmbuilder.data.dedup import ExactDuplicateDetector
from llmbuilder.config.defaults import DeduplicationConfig

class TestExactDuplicateDetector:
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DeduplicationConfig(
            enable_exact_deduplication=True,
            normalize_text=True
        )
        self.detector = ExactDuplicateDetector(self.config)

    def test_exact_duplicate_detection(self):
        """Test basic exact duplicate detection."""
        texts = [
            "Hello world",
            "Hello world",  # Exact duplicate
            "Hello World",  # Case difference
            "Different text"
        ]

        unique_texts, duplicates = self.detector.deduplicate(texts)

        assert len(unique_texts) == 3  # Should remove one duplicate
        assert "Hello world" in unique_texts
        assert "Different text" in unique_texts

    def test_normalization(self):
        """Test text normalization during deduplication."""
        texts = [
            "  Hello world  ",
            "Hello world",
            "HELLO WORLD"
        ]

        unique_texts, duplicates = self.detector.deduplicate(texts)

        # With normalization, all should be considered duplicates
        assert len(unique_texts) == 1
```

**Integration Test Example:**

```python
import tempfile
from pathlib import Path
from llmbuilder.data.ingest import IngestionPipeline
from llmbuilder.config.defaults import IngestionConfig

class TestIngestionPipeline:
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = IngestionConfig(
            supported_formats=["txt", "html"],
            batch_size=10
        )
        self.pipeline = IngestionPipeline(self.config)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_directory_processing(self):
        """Test processing a directory of mixed files."""
        # Create test files
        (self.temp_dir / "test.txt").write_text("Plain text content")
        (self.temp_dir / "test.html").write_text("<p>HTML content</p>")

        # Process directory
        results = self.pipeline.process_directory(
            str(self.temp_dir),
            str(self.temp_dir / "output")
        )

        assert results['files_processed'] == 2
        assert results['total_characters'] > 0
        assert 'txt' in results['format_stats']
        assert 'html' in results['format_stats']
```

### Performance Testing

```python
import pytest
import time
from llmbuilder.data.dedup import SemanticDuplicateDetector

@pytest.mark.performance
class TestDeduplicationPerformance:
    def test_semantic_deduplication_performance(self):
        """Test semantic deduplication performance with large dataset."""
        # Create large test dataset
        texts = [f"This is test text number {i}" for i in range(1000)]

        detector = SemanticDuplicateDetector(config)

        start_time = time.time()
        unique_texts, _ = detector.deduplicate(texts)
        processing_time = time.time() - start_time

        # Performance assertions
        assert processing_time < 30.0  # Should complete in under 30 seconds
        assert len(unique_texts) > 0

        # Log performance metrics
        print(f"Processed {len(texts)} texts in {processing_time:.2f}s")
        print(f"Rate: {len(texts) / processing_time:.1f} texts/second")
```

## 📖 Documentation Standards

### API Documentation

Use Google-style docstrings:

```python
def process_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
    """Process a single file through the ingestion pipeline.

    Args:
        input_file: Path to the input file to process.
        output_file: Path where processed content will be saved.

    Returns:
        Dictionary containing processing results with keys:
        - 'success': Boolean indicating if processing succeeded
        - 'characters': Number of characters extracted
        - 'processing_time': Time taken in seconds
        - 'metadata': Optional metadata dictionary

    Raises:
        FileNotFoundError: If input file doesn't exist.
        PermissionError: If unable to write to output file.
        ValueError: If file format is not supported.

    Example:
        >>> pipeline = IngestionPipeline(config)
        >>> result = pipeline.process_file("document.pdf", "output.txt")
        >>> print(f"Extracted {result['characters']} characters")
    """
```

### Configuration Documentation

Document all configuration options:

```python
@dataclass
class DeduplicationConfig:
    """Configuration for text deduplication.

    Attributes:
        enable_exact_deduplication: Whether to remove exact duplicates.
            Default: True
        enable_semantic_deduplication: Whether to remove semantic duplicates.
            Default: True
        similarity_threshold: Similarity threshold for semantic deduplication.
            Range: 0.0-1.0, where 1.0 means identical.
            Default: 0.85
        embedding_model: Model name for semantic similarity computation.
            Options: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'
            Default: 'all-MiniLM-L6-v2'
        batch_size: Number of texts to process in each batch.
            Larger values use more memory but may be faster.
            Default: 1000
    """
```

## 🐛 Issue Templates

### Bug Report Template

```markdown
**Bug Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear and concise description of what you expected to happen.

**Actual Behavior**
What actually happened instead.

**Environment**
- OS: [e.g. Windows 10, Ubuntu 20.04, macOS 12.0]
- Python version: [e.g. 3.9.7]
- LLMBuilder version: [e.g. 1.2.3]
- Optional dependencies installed: [e.g. sentence-transformers, pytesseract]

**Code to Reproduce**
```python
# Minimal code example that reproduces the issue
```

**Error Messages**

```
Full error traceback here
```

**Additional Context**
Add any other context about the problem here.

```

### Feature Request Template

```markdown
**Feature Description**
A clear and concise description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve or the workflow it would improve.

**Proposed Solution**
Describe how you envision this feature working.

**Alternative Solutions**
Describe any alternative solutions or features you've considered.

**Implementation Ideas**
If you have ideas about how this could be implemented, share them here.

**Additional Context**
Add any other context, mockups, or examples about the feature request here.
```

## 🚀 Release Process

### Version Numbering

We follow semantic versioning (SemVer):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

1. **Update version** in `setup.py` and `__init__.py`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** including performance tests
4. **Update documentation** if needed
5. **Create release PR** and get approval
6. **Tag release** and push to GitHub
7. **Build and publish** to PyPI
8. **Update documentation** site

## 💬 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone has different experience levels
- **Be collaborative**: Work together to improve the project

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Pull Requests**: Code contributions and reviews
- **Documentation**: Comprehensive guides and examples

### Getting Help

If you need help:

1. **Check documentation**: Look for existing guides and examples
2. **Search issues**: See if your question has been asked before
3. **Create an issue**: Ask specific questions with context
4. **Join discussions**: Participate in community discussions

## 🎉 Recognition

We appreciate all contributions! Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Major contributions highlighted
- **Documentation**: Examples and guides credited to authors
- **GitHub**: Contributor statistics and recognition

## 📞 Contact

For questions about contributing:

- **GitHub Issues**: Technical questions and bug reports
- **GitHub Discussions**: General questions and ideas
- **Email**: [maintainer email] for sensitive issues

---

Thank you for contributing to LLMBuilder! Your contributions help make language model development more accessible and efficient for everyone. 🚀r feature requests
5. **Join our community** for real-time help

---

Thank you for contributing to LLMBuilder! Your contributions help make language model development more accessible to everyone. 🚀
