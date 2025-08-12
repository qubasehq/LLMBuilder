#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for LLMBuilder tests.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any
import torch
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set environment variables for testing
    os.environ['PYTHONPATH'] = str(PROJECT_ROOT)
    os.environ['TESTING'] = '1'
    
    # Disable CUDA for consistent testing
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set torch to use CPU only
    torch.set_default_tensor_type('torch.FloatTensor')


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to tests that take longer
        if "performance" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker
        if "test_performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Add GPU marker for GPU-specific tests
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Add OCR marker for OCR-dependent tests
        if "ocr" in item.name.lower() or "pdf" in item.name.lower():
            item.add_marker(pytest.mark.ocr)
        
        # Add embedding marker for embedding-dependent tests
        if "embedding" in item.name.lower() or "semantic" in item.name.lower():
            item.add_marker(pytest.mark.embedding)


@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory."""
    return PROJECT_ROOT / "tests" / "data"


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create standard directory structure
        directories = [
            "data/raw",
            "data/cleaned", 
            "data/deduped",
            "data/tokens",
            "data/finetune",
            "exports/checkpoints",
            "exports/gguf",
            "exports/tokenizer",
            "logs"
        ]
        
        for directory in directories:
            (workspace / directory).mkdir(parents=True, exist_ok=True)
        
        yield workspace


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "vocab_size": 1000,
            "embedding_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 128,
            "max_seq_length": 64,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "num_epochs": 1,
            "max_steps": 5,
            "save_every": 2,
            "eval_every": 2,
            "log_every": 1
        },
        "tokenizer": {
            "vocab_size": 1000,
            "model_type": "bpe",
            "special_tokens": ["<pad>", "<unk>", "<s>", "</s>"]
        },
        "device": {
            "use_cuda": False,
            "cpu_threads": 1
        }
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return {
        "simple.txt": "This is a simple text document for testing.",
        "longer.txt": "This is a longer document with multiple sentences. It contains more content for testing purposes. The content should be processed correctly by the ingestion pipeline.",
        "duplicate.txt": "This is a simple text document for testing.",  # Exact duplicate
        "similar.txt": "This is a basic text file for testing purposes.",  # Similar content
        "html_sample.html": """
        <html>
            <head><title>Test</title></head>
            <body>
                <h1>Test HTML</h1>
                <p>This is test HTML content.</p>
                <script>alert('should be removed');</script>
            </body>
        </html>
        """,
        "markdown_sample.md": """
        # Test Markdown
        
        This is **test** markdown content with:
        - Lists
        - **Bold text**
        - `Code`
        
        ## Section 2
        More content here.
        """
    }


@pytest.fixture
def mock_model_checkpoint():
    """Mock model checkpoint for testing."""
    return {
        'model': {
            'embedding.weight': torch.randn(100, 32),
            'transformer.layers.0.attention.weight': torch.randn(32, 32),
            'transformer.layers.0.mlp.weight': torch.randn(32, 64),
            'lm_head.weight': torch.randn(100, 32)
        },
        'config': {
            'vocab_size': 100,
            'embedding_dim': 32,
            'num_layers': 1,
            'num_heads': 2,
            'hidden_dim': 64,
            'max_seq_length': 64
        },
        'training_stats': {
            'step': 100,
            'loss': 2.5,
            'training_time': 300.0
        }
    }


@pytest.fixture
def create_test_files():
    """Factory fixture to create test files."""
    def _create_files(directory: Path, files: Dict[str, str]):
        """Create test files in the specified directory."""
        created_files = []
        for filename, content in files.items():
            file_path = directory / filename
            file_path.write_text(content, encoding='utf-8')
            created_files.append(file_path)
        return created_files
    
    return _create_files


@pytest.fixture
def skip_if_no_tesseract():
    """Skip test if Tesseract OCR is not available."""
    import shutil
    if not shutil.which('tesseract'):
        pytest.skip("Tesseract OCR not available")


@pytest.fixture
def skip_if_no_sentence_transformers():
    """Skip test if sentence-transformers is not available."""
    try:
        import sentence_transformers
    except ImportError:
        pytest.skip("sentence-transformers not available")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    
    # Cleanup any temporary files that might have been created
    import gc
    gc.collect()
    
    # Clear torch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def test_requirements():
    """Check test requirements and skip if not met."""
    requirements = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'pytest': 'Pytest'
    }
    
    missing = []
    for module, name in requirements.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        pytest.skip(f"Missing required packages: {', '.join(missing)}")


# Custom markers for test organization
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Set consistent random seeds for reproducible tests
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


def pytest_runtest_teardown(item):
    """Teardown after each test run."""
    # Clear any cached data
    import gc
    gc.collect()


# Test data constants
TEST_CONSTANTS = {
    'SMALL_FILE_SIZE': 1000,      # ~1KB
    'MEDIUM_FILE_SIZE': 10000,    # ~10KB  
    'LARGE_FILE_SIZE': 100000,    # ~100KB
    'MAX_TEST_DURATION': 300,     # 5 minutes
    'MAX_MEMORY_INCREASE_MB': 500, # 500MB
    'MIN_THROUGHPUT_FILES_SEC': 0.5,
    'MIN_THROUGHPUT_CHARS_SEC': 1000
}


@pytest.fixture
def test_constants():
    """Test constants for consistent testing."""
    return TEST_CONSTANTS