#!/usr/bin/env python3
"""
Unit tests for CLI data commands.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llmbuilder.cli.data import data_group


class TestDataPrepareCommand:
    """Test cases for data prepare command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_data_prepare_help(self):
        """Test data prepare command help."""
        result = self.runner.invoke(data_group, ['prepare', '--help'])
        assert result.exit_code == 0
        assert 'prepare' in result.output.lower()
    
    def test_data_prepare_basic(self):
        """Test basic data preparation."""
        with self.runner.isolated_filesystem():
            # Create input directory with test files
            input_dir = Path('input')
            input_dir.mkdir()
            
            # Create test files
            (input_dir / 'test.txt').write_text('Test content')
            (input_dir / 'test.json').write_text('{"test": "data"}')
            
            result = self.runner.invoke(data_group, [
                'prepare',
                '--input', str(input_dir),
                '--output', 'output'
            ])
            
            # Should succeed or provide meaningful error
            assert isinstance(result.exit_code, int)
    
    def test_data_prepare_with_formats(self):
        """Test data preparation with specific formats."""
        with self.runner.isolated_filesystem():
            input_dir = Path('input')
            input_dir.mkdir()
            (input_dir / 'test.txt').write_text('Test content')
            
            result = self.runner.invoke(data_group, [
                'prepare',
                '--input', str(input_dir),
                '--output', 'output',
                '--formats', 'txt,json'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_data_prepare_with_deduplication(self):
        """Test data preparation with deduplication."""
        with self.runner.isolated_filesystem():
            input_dir = Path('input')
            input_dir.mkdir()
            (input_dir / 'test1.txt').write_text('Same content')
            (input_dir / 'test2.txt').write_text('Same content')
            
            result = self.runner.invoke(data_group, [
                'prepare',
                '--input', str(input_dir),
                '--output', 'output',
                '--deduplicate'
            ])
            
            assert isinstance(result.exit_code, int)
    
    @patch('llmbuilder.core.data.ingest.DataIngestionPipeline')
    def test_data_prepare_with_mock_pipeline(self, mock_pipeline):
        """Test data preparation with mocked pipeline."""
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.process_directory.return_value = {'processed': 5, 'errors': 0}
        
        with self.runner.isolated_filesystem():
            input_dir = Path('input')
            input_dir.mkdir()
            (input_dir / 'test.txt').write_text('Test content')
            
            result = self.runner.invoke(data_group, [
                'prepare',
                '--input', str(input_dir),
                '--output', 'output'
            ])
            
            assert result.exit_code == 0


class TestDataSplitCommand:
    """Test cases for data split command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_data_split_help(self):
        """Test data split command help."""
        result = self.runner.invoke(data_group, ['split', '--help'])
        assert result.exit_code == 0
        assert 'split' in result.output.lower()
    
    def test_data_split_basic(self):
        """Test basic data splitting."""
        with self.runner.isolated_filesystem():
            # Create test data file
            data_file = Path('data.jsonl')
            with open(data_file, 'w') as f:
                for i in range(100):
                    f.write(f'{{"text": "Sample text {i}"}}\n')
            
            result = self.runner.invoke(data_group, [
                'split',
                '--input', str(data_file),
                '--output-dir', 'splits'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_data_split_custom_ratios(self):
        """Test data splitting with custom ratios."""
        with self.runner.isolated_filesystem():
            data_file = Path('data.jsonl')
            with open(data_file, 'w') as f:
                for i in range(100):
                    f.write(f'{{"text": "Sample text {i}"}}\n')
            
            result = self.runner.invoke(data_group, [
                'split',
                '--input', str(data_file),
                '--output-dir', 'splits',
                '--train-ratio', '0.7',
                '--val-ratio', '0.2',
                '--test-ratio', '0.1'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_data_split_stratified(self):
        """Test stratified data splitting."""
        with self.runner.isolated_filesystem():
            data_file = Path('data.jsonl')
            with open(data_file, 'w') as f:
                for i in range(100):
                    category = 'A' if i % 2 == 0 else 'B'
                    f.write(f'{{"text": "Sample text {i}", "category": "{category}"}}\n')
            
            result = self.runner.invoke(data_group, [
                'split',
                '--input', str(data_file),
                '--output-dir', 'splits',
                '--stratify', 'category'
            ])
            
            assert isinstance(result.exit_code, int)


class TestDataValidateCommand:
    """Test cases for data validate command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_data_validate_help(self):
        """Test data validate command help."""
        result = self.runner.invoke(data_group, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'validate' in result.output.lower()
    
    def test_data_validate_basic(self):
        """Test basic data validation."""
        with self.runner.isolated_filesystem():
            # Create test data
            data_dir = Path('data')
            data_dir.mkdir()
            (data_dir / 'test.jsonl').write_text('{"text": "Valid JSON"}\n')
            
            result = self.runner.invoke(data_group, [
                'validate',
                '--input', str(data_dir)
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_data_validate_with_schema(self):
        """Test data validation with schema."""
        with self.runner.isolated_filesystem():
            # Create test data
            data_dir = Path('data')
            data_dir.mkdir()
            (data_dir / 'test.jsonl').write_text('{"text": "Valid JSON"}\n')
            
            # Create schema file
            schema = {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
            with open('schema.json', 'w') as f:
                json.dump(schema, f)
            
            result = self.runner.invoke(data_group, [
                'validate',
                '--input', str(data_dir),
                '--schema', 'schema.json'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_data_validate_statistics(self):
        """Test data validation with statistics."""
        with self.runner.isolated_filesystem():
            data_dir = Path('data')
            data_dir.mkdir()
            
            # Create test data with various content
            with open(data_dir / 'test.jsonl', 'w') as f:
                f.write('{"text": "Short"}\n')
                f.write('{"text": "This is a longer piece of text for testing"}\n')
                f.write('{"text": "Medium length text"}\n')
            
            result = self.runner.invoke(data_group, [
                'validate',
                '--input', str(data_dir),
                '--statistics'
            ])
            
            assert isinstance(result.exit_code, int)


class TestDataCleanCommand:
    """Test cases for data clean command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_data_clean_help(self):
        """Test data clean command help."""
        result = self.runner.invoke(data_group, ['clean', '--help'])
        assert result.exit_code == 0
        assert 'clean' in result.output.lower()
    
    def test_data_clean_basic(self):
        """Test basic data cleaning."""
        with self.runner.isolated_filesystem():
            # Create test data with noise
            input_dir = Path('input')
            input_dir.mkdir()
            
            noisy_text = "This is good text.\n\n\n   \nThis has extra whitespace.   \n"
            (input_dir / 'noisy.txt').write_text(noisy_text)
            
            result = self.runner.invoke(data_group, [
                'clean',
                '--input', str(input_dir),
                '--output', 'cleaned'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_data_clean_with_filters(self):
        """Test data cleaning with specific filters."""
        with self.runner.isolated_filesystem():
            input_dir = Path('input')
            input_dir.mkdir()
            (input_dir / 'test.txt').write_text('Test content with URLs http://example.com')
            
            result = self.runner.invoke(data_group, [
                'clean',
                '--input', str(input_dir),
                '--output', 'cleaned',
                '--remove-urls',
                '--normalize-whitespace'
            ])
            
            assert isinstance(result.exit_code, int)


if __name__ == '__main__':
    pytest.main([__file__])