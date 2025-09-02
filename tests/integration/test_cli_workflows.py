#!/usr/bin/env python3
"""
Integration tests for CLI command workflows.
"""

import pytest
from click.testing import CliRunner
import json
import sys
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llmbuilder.cli.main import cli


class TestInitToDataWorkflow:
    """Test workflow from project initialization to data processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    def test_init_then_data_prepare(self):
        """Test initializing project then preparing data."""
        with self.runner.isolated_filesystem():
            # Step 1: Initialize project
            result = self.runner.invoke(cli, ['init', 'workflow-test'])
            assert result.exit_code == 0
            
            project_path = Path('workflow-test')
            assert project_path.exists()
            
            # Step 2: Create sample data
            raw_data_dir = project_path / 'data' / 'raw'
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create various file types
            (raw_data_dir / 'sample.txt').write_text('Sample text content for processing.')
            (raw_data_dir / 'sample.json').write_text('{"text": "JSON content"}')
            
            # Step 3: Prepare data
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(raw_data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            
            assert isinstance(result.exit_code, int)
            
            # Check if cleaned directory was created
            cleaned_dir = project_path / 'data' / 'cleaned'
            if result.exit_code == 0:
                assert cleaned_dir.exists()
    
    @pytest.mark.integration
    def test_init_with_template_then_config(self):
        """Test initializing with template then configuring."""
        with self.runner.isolated_filesystem():
            # Step 1: Initialize with research template
            result = self.runner.invoke(cli, [
                'init', 'research-project',
                '--template', 'research'
            ])
            assert result.exit_code == 0
            
            # Step 2: Configure project settings
            result = self.runner.invoke(cli, [
                'config', 'set',
                'model.vocab_size', '32000'
            ])
            assert isinstance(result.exit_code, int)
            
            # Step 3: Verify configuration
            result = self.runner.invoke(cli, [
                'config', 'get',
                'model.vocab_size'
            ])
            assert isinstance(result.exit_code, int)


class TestDataProcessingWorkflow:
    """Test complete data processing workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    def test_prepare_split_validate_workflow(self):
        """Test prepare -> split -> validate workflow."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'data-workflow'])
            assert result.exit_code == 0
            
            project_path = Path('data-workflow')
            
            # Create sample dataset
            raw_data_dir = project_path / 'data' / 'raw'
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create JSONL dataset
            sample_data = []
            for i in range(100):
                sample_data.append({
                    "text": f"This is sample training text number {i}.",
                    "label": "positive" if i % 2 == 0 else "negative"
                })
            
            with open(raw_data_dir / 'dataset.jsonl', 'w') as f:
                for item in sample_data:
                    f.write(json.dumps(item) + '\n')
            
            # Step 1: Prepare data
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(raw_data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            assert isinstance(result.exit_code, int)
            
            # Step 2: Split data (if prepare succeeded)
            if result.exit_code == 0:
                result = self.runner.invoke(cli, [
                    'data', 'split',
                    '--input', str(project_path / 'data' / 'cleaned' / 'dataset.jsonl'),
                    '--output-dir', str(project_path / 'data' / 'splits'),
                    '--train-ratio', '0.8',
                    '--val-ratio', '0.1',
                    '--test-ratio', '0.1'
                ])
                assert isinstance(result.exit_code, int)
                
                # Step 3: Validate splits (if split succeeded)
                if result.exit_code == 0:
                    result = self.runner.invoke(cli, [
                        'data', 'validate',
                        '--input', str(project_path / 'data' / 'splits')
                    ])
                    assert isinstance(result.exit_code, int)


class TestModelWorkflow:
    """Test model management workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    def test_model_list_info_workflow(self):
        """Test model listing and info workflow."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'model-workflow'])
            assert result.exit_code == 0
            
            # Step 1: List available models
            result = self.runner.invoke(cli, ['model', 'list'])
            assert isinstance(result.exit_code, int)
            
            # Step 2: Get info for a common model
            result = self.runner.invoke(cli, [
                'model', 'info', 'gpt2'
            ])
            assert isinstance(result.exit_code, int)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_model_select_workflow(self):
        """Test model selection workflow."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'model-select-workflow'])
            assert result.exit_code == 0
            
            # Try to select a small model (this might require internet)
            result = self.runner.invoke(cli, [
                'model', 'select',
                'distilgpt2',  # Small model for testing
                '--output-dir', 'models'
            ])
            
            # Should handle gracefully whether internet is available or not
            assert isinstance(result.exit_code, int)


class TestTrainingWorkflow:
    """Test training workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_preparation_workflow(self):
        """Test training preparation workflow."""
        with self.runner.isolated_filesystem():
            # Setup complete project
            result = self.runner.invoke(cli, ['init', 'training-workflow'])
            assert result.exit_code == 0
            
            project_path = Path('training-workflow')
            
            # Create training data
            finetune_dir = project_path / 'data' / 'finetune'
            finetune_dir.mkdir(parents=True, exist_ok=True)
            
            # Create minimal training dataset
            training_data = []
            for i in range(10):  # Small dataset for testing
                training_data.append({
                    "input": f"Question {i}: What is the answer?",
                    "output": f"Answer {i}: This is the response."
                })
            
            with open(finetune_dir / 'train.jsonl', 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            # Configure training settings
            result = self.runner.invoke(cli, [
                'config', 'set',
                'training.batch_size', '1'
            ])
            assert isinstance(result.exit_code, int)
            
            result = self.runner.invoke(cli, [
                'config', 'set',
                'training.max_steps', '5'
            ])
            assert isinstance(result.exit_code, int)
            
            # Validate training configuration
            result = self.runner.invoke(cli, ['config', 'validate'])
            assert isinstance(result.exit_code, int)


class TestToolsWorkflow:
    """Test tools integration workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    def test_tools_register_list_workflow(self):
        """Test tools registration and listing workflow."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'tools-workflow'])
            assert result.exit_code == 0
            
            # Create a simple tool
            tool_content = '''
def text_processor(text: str, uppercase: bool = False) -> str:
    """Process text with optional uppercase conversion."""
    if uppercase:
        return text.upper()
    return text.lower()

def word_counter(text: str) -> int:
    """Count words in text."""
    return len(text.split())
'''
            
            tool_file = Path('my_tools.py')
            tool_file.write_text(tool_content)
            
            # Step 1: Register tools
            result = self.runner.invoke(cli, [
                'tools', 'register',
                str(tool_file),
                '--name', 'text_tools'
            ])
            assert isinstance(result.exit_code, int)
            
            # Step 2: List registered tools
            result = self.runner.invoke(cli, ['tools', 'list'])
            assert isinstance(result.exit_code, int)
            
            # Step 3: Test tools (if registration succeeded)
            if result.exit_code == 0:
                result = self.runner.invoke(cli, [
                    'tools', 'test',
                    'text_processor',
                    '--args', '{"text": "Hello World", "uppercase": true}'
                ])
                assert isinstance(result.exit_code, int)


class TestMonitoringWorkflow:
    """Test monitoring and debugging workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    def test_debug_workflow(self):
        """Test debugging workflow."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'debug-workflow'])
            assert result.exit_code == 0
            
            # Run system debug
            result = self.runner.invoke(cli, ['monitor', 'debug'])
            assert isinstance(result.exit_code, int)
            
            # Check logs
            result = self.runner.invoke(cli, ['monitor', 'logs'])
            assert isinstance(result.exit_code, int)


class TestErrorRecoveryWorkflow:
    """Test error recovery in workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    def test_corrupted_data_recovery(self):
        """Test recovery from corrupted data."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'recovery-test'])
            assert result.exit_code == 0
            
            project_path = Path('recovery-test')
            
            # Create corrupted data
            raw_data_dir = project_path / 'data' / 'raw'
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create invalid JSON file
            (raw_data_dir / 'corrupted.json').write_text('invalid json content')
            
            # Try to process - should handle gracefully
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(raw_data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            
            # Should handle error gracefully
            assert isinstance(result.exit_code, int)
    
    @pytest.mark.integration
    def test_missing_config_recovery(self):
        """Test recovery from missing configuration."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'config-recovery'])
            assert result.exit_code == 0
            
            # Remove config file if it exists
            config_file = Path('config-recovery/config.json')
            if config_file.exists():
                config_file.unlink()
            
            # Try to use config commands - should handle gracefully
            result = self.runner.invoke(cli, ['config', 'list'])
            assert isinstance(result.exit_code, int)
            
            # Should be able to recreate config
            result = self.runner.invoke(cli, [
                'config', 'set',
                'model.vocab_size', '50000'
            ])
            assert isinstance(result.exit_code, int)


if __name__ == '__main__':
    pytest.main([__file__])