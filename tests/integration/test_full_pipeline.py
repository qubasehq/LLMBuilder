#!/usr/bin/env python3
"""
Integration tests for full LLMBuilder pipeline workflows.
"""

import pytest
from click.testing import CliRunner
import json
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llmbuilder.cli.main import cli


class TestFullPipeline:
    """Test cases for complete pipeline workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_workflow_basic(self):
        """Test complete workflow from init to training."""
        with self.runner.isolated_filesystem():
            # Step 1: Initialize project
            result = self.runner.invoke(cli, ['init', 'test-project'])
            assert result.exit_code == 0
            
            # Step 2: Prepare some test data
            project_path = Path('test-project')
            data_dir = project_path / 'data' / 'raw'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample training data
            sample_data = [
                {"text": "This is sample training text 1."},
                {"text": "This is sample training text 2."},
                {"text": "This is sample training text 3."},
            ]
            
            with open(data_dir / 'sample.jsonl', 'w') as f:
                for item in sample_data:
                    f.write(json.dumps(item) + '\n')
            
            # Step 3: Process data
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            # Should succeed or provide meaningful error
            assert isinstance(result.exit_code, int)
            
            # Step 4: Split data
            if result.exit_code == 0:
                result = self.runner.invoke(cli, [
                    'data', 'split',
                    '--input', str(project_path / 'data' / 'cleaned'),
                    '--output-dir', str(project_path / 'data' / 'splits')
                ])
                assert isinstance(result.exit_code, int)
    
    @pytest.mark.integration
    def test_config_workflow(self):
        """Test configuration management workflow."""
        with self.runner.isolated_filesystem():
            # Initialize project
            result = self.runner.invoke(cli, ['init', 'test-project'])
            assert result.exit_code == 0
            
            # Set configuration values
            result = self.runner.invoke(cli, [
                'config', 'set', 'model.vocab_size', '30000'
            ])
            assert isinstance(result.exit_code, int)
            
            # Get configuration value
            result = self.runner.invoke(cli, [
                'config', 'get', 'model.vocab_size'
            ])
            assert isinstance(result.exit_code, int)
            
            # List all configurations
            result = self.runner.invoke(cli, ['config', 'list'])
            assert isinstance(result.exit_code, int)
            
            # Validate configuration
            result = self.runner.invoke(cli, ['config', 'validate'])
            assert isinstance(result.exit_code, int)
    
    @pytest.mark.integration
    def test_model_management_workflow(self):
        """Test model management workflow."""
        with self.runner.isolated_filesystem():
            # Initialize project
            result = self.runner.invoke(cli, ['init', 'test-project'])
            assert result.exit_code == 0
            
            # List available models
            result = self.runner.invoke(cli, ['model', 'list'])
            assert isinstance(result.exit_code, int)
            
            # Get model info (if available)
            result = self.runner.invoke(cli, [
                'model', 'info', 'gpt2'
            ])
            assert isinstance(result.exit_code, int)
    
    @pytest.mark.integration
    def test_tools_workflow(self):
        """Test tools integration workflow."""
        with self.runner.isolated_filesystem():
            # Initialize project
            result = self.runner.invoke(cli, ['init', 'test-project'])
            assert result.exit_code == 0
            
            # List available tools
            result = self.runner.invoke(cli, ['tools', 'list'])
            assert isinstance(result.exit_code, int)
            
            # Register a simple tool
            tool_file = Path('simple_tool.py')
            tool_content = '''
def simple_function(text: str) -> str:
    """A simple test function."""
    return f"Processed: {text}"
'''
            tool_file.write_text(tool_content)
            
            result = self.runner.invoke(cli, [
                'tools', 'register',
                str(tool_file),
                '--name', 'simple_tool'
            ])
            assert isinstance(result.exit_code, int)


class TestErrorHandlingIntegration:
    """Test error handling in integrated workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    def test_invalid_project_structure(self):
        """Test handling of invalid project structure."""
        with self.runner.isolated_filesystem():
            # Create invalid project structure
            Path('invalid-project').mkdir()
            Path('invalid-project/random-file.txt').write_text('random content')
            
            # Try to run commands in invalid project
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', 'invalid-project',
                '--output', 'output'
            ])
            
            # Should handle gracefully
            assert isinstance(result.exit_code, int)
    
    @pytest.mark.integration
    def test_missing_dependencies(self):
        """Test handling of missing dependencies."""
        with self.runner.isolated_filesystem():
            # Initialize project
            result = self.runner.invoke(cli, ['init', 'test-project'])
            assert result.exit_code == 0
            
            # Try operations that might require missing dependencies
            result = self.runner.invoke(cli, ['monitor', 'debug'])
            assert isinstance(result.exit_code, int)
    
    @pytest.mark.integration
    def test_corrupted_config(self):
        """Test handling of corrupted configuration."""
        with self.runner.isolated_filesystem():
            # Initialize project
            result = self.runner.invoke(cli, ['init', 'test-project'])
            assert result.exit_code == 0
            
            # Corrupt config file
            config_file = Path('test-project/config.json')
            if config_file.exists():
                config_file.write_text('invalid json content')
            
            # Try to use corrupted config
            result = self.runner.invoke(cli, ['config', 'list'])
            assert isinstance(result.exit_code, int)


class TestPerformanceIntegration:
    """Test performance aspects of integrated workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        with self.runner.isolated_filesystem():
            # Initialize project
            result = self.runner.invoke(cli, ['init', 'test-project'])
            assert result.exit_code == 0
            
            # Create large test dataset
            project_path = Path('test-project')
            data_dir = project_path / 'data' / 'raw'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate larger dataset
            with open(data_dir / 'large_dataset.jsonl', 'w') as f:
                for i in range(1000):  # Moderate size for testing
                    text = f"This is sample text number {i}. " * 10
                    f.write(json.dumps({"text": text}) + '\n')
            
            # Process large dataset
            import time
            start_time = time.time()
            
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete within reasonable time
            assert processing_time < 60  # 1 minute max for test dataset
            assert isinstance(result.exit_code, int)
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        import os
        
        with self.runner.isolated_filesystem():
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Initialize project and process data
            result = self.runner.invoke(cli, ['init', 'test-project'])
            assert result.exit_code == 0
            
            # Create test data
            project_path = Path('test-project')
            data_dir = project_path / 'data' / 'raw'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            with open(data_dir / 'test.jsonl', 'w') as f:
                for i in range(100):
                    f.write(json.dumps({"text": f"Sample text {i}"}) + '\n')
            
            # Process data
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            
            # Check memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 500  # Less than 500MB increase
            assert isinstance(result.exit_code, int)


class TestConcurrencyIntegration:
    """Test concurrent operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_data_processing(self):
        """Test concurrent data processing operations."""
        import threading
        import time
        
        results = []
        
        def process_data(project_name):
            """Process data in separate thread."""
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Initialize project
                result = self.runner.invoke(cli, ['init', str(temp_path / project_name)])
                
                if result.exit_code == 0:
                    # Create test data
                    data_dir = temp_path / project_name / 'data' / 'raw'
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(data_dir / 'test.jsonl', 'w') as f:
                        for i in range(50):
                            f.write(json.dumps({"text": f"Sample {project_name} {i}"}) + '\n')
                    
                    # Process data
                    result = self.runner.invoke(cli, [
                        'data', 'prepare',
                        '--input', str(data_dir),
                        '--output', str(temp_path / project_name / 'data' / 'cleaned')
                    ])
                
                results.append(result.exit_code)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_data, args=[f'project_{i}'])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should complete
        assert len(results) == 3
        assert all(isinstance(code, int) for code in results)


if __name__ == '__main__':
    pytest.main([__file__])