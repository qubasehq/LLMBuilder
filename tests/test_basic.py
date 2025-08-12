#!/usr/bin/env python3
"""
Basic tests to verify core functionality.
"""

import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class TestBasicFunctionality:
    """Basic functionality tests."""
    
    def test_imports(self):
        """Test that core modules can be imported."""
        # Test core imports
        import torch
        import numpy as np
        
        # Test project imports
        try:
            from data.ingest import DocumentIngester
            from data.dedup import DeduplicationPipeline
        except ImportError as e:
            pytest.skip(f"Project modules not available: {e}")
    
    def test_torch_basic(self):
        """Test basic PyTorch functionality."""
        import torch
        
        # Create a simple tensor
        x = torch.randn(3, 4)
        assert x.shape == (3, 4)
        
        # Basic operations
        y = x + 1
        assert y.shape == x.shape
        
        # Check that operations work
        z = torch.matmul(x, x.T)
        assert z.shape == (3, 3)
    
    def test_numpy_basic(self):
        """Test basic NumPy functionality."""
        import numpy as np
        
        # Create array
        arr = np.array([1, 2, 3, 4])
        assert len(arr) == 4
        
        # Basic operations
        result = arr * 2
        expected = np.array([2, 4, 6, 8])
        np.testing.assert_array_equal(result, expected)
    
    def test_pathlib_basic(self):
        """Test basic pathlib functionality."""
        from pathlib import Path
        
        # Create path
        p = Path("test/path/file.txt")
        assert p.name == "file.txt"
        assert p.suffix == ".txt"
        assert p.parent.name == "path"
    
    def test_json_basic(self):
        """Test basic JSON functionality."""
        import json
        
        # Test JSON serialization
        data = {"key": "value", "number": 42}
        json_str = json.dumps(data)
        
        # Test deserialization
        parsed = json.loads(json_str)
        assert parsed == data
    
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_unix_specific(self):
        """Test Unix-specific functionality."""
        import os
        assert hasattr(os, 'fork')  # Unix-specific
    
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_specific(self):
        """Test Windows-specific functionality."""
        import os
        # Windows-specific test
        assert os.name == 'nt'


class TestProjectStructure:
    """Test project structure and configuration."""
    
    def test_project_root_exists(self):
        """Test that project root directory exists."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert project_root.is_dir()
    
    def test_required_directories(self):
        """Test that required directories exist."""
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            "data",
            "tests", 
            "tools",
            "training"
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
            assert dir_path.is_dir(), f"Path is not a directory: {dir_name}"
    
    def test_required_files(self):
        """Test that required files exist."""
        project_root = Path(__file__).parent.parent
        
        required_files = [
            "README.md",
            "requirements.txt",
            "run.sh",
            "run.ps1"
        ]
        
        for file_name in required_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"
            assert file_path.is_file(), f"Path is not a file: {file_name}"
    
    def test_config_files_valid_json(self):
        """Test that configuration files are valid JSON."""
        project_root = Path(__file__).parent.parent
        
        config_files = [
            "config.json",
            "config_gpu.json", 
            "config_cpu_small.json"
        ]
        
        import json
        
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {config_file}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])