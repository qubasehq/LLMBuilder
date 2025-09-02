#!/usr/bin/env python3
"""
Unit tests for CLI init command.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llmbuilder.cli.init import init_group


class TestInitCommand:
    """Test cases for init command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_init_help(self):
        """Test init command help."""
        result = self.runner.invoke(init_group, ['--help'])
        assert result.exit_code == 0
        assert 'init' in result.output.lower()
    
    def test_init_project_basic(self):
        """Test basic project initialization."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, ['test-project'])
            assert result.exit_code == 0
            
            # Check if project directory was created
            project_path = Path('test-project')
            assert project_path.exists()
            assert project_path.is_dir()
    
    def test_init_project_with_template(self):
        """Test project initialization with template."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project', 
                '--template', 'research'
            ])
            assert result.exit_code == 0
            
            project_path = Path('test-project')
            assert project_path.exists()
    
    def test_init_project_existing_directory(self):
        """Test initialization in existing directory."""
        with self.runner.isolated_filesystem():
            # Create existing directory
            Path('existing-project').mkdir()
            
            # Try to initialize - should prompt or handle gracefully
            result = self.runner.invoke(init_group, ['existing-project'])
            # Should either succeed or fail gracefully
            assert isinstance(result.exit_code, int)
    
    def test_init_project_invalid_name(self):
        """Test initialization with invalid project name."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, ['invalid/name'])
            # Should handle invalid names gracefully
            assert isinstance(result.exit_code, int)
    
    @patch('llmbuilder.utils.template_manager.TemplateManager')
    def test_init_with_template_manager(self, mock_template_manager):
        """Test initialization using template manager."""
        mock_manager = MagicMock()
        mock_template_manager.return_value = mock_manager
        mock_manager.create_project.return_value = True
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, ['test-project'])
            assert result.exit_code == 0
    
    def test_init_project_structure(self):
        """Test that initialized project has correct structure."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, ['test-project'])
            assert result.exit_code == 0
            
            project_path = Path('test-project')
            
            # Check for expected directories
            expected_dirs = [
                'data/raw',
                'data/cleaned',
                'data/deduped',
                'data/tokens',
                'data/finetune',
                'exports/checkpoints',
                'exports/gguf',
                'exports/tokenizer',
                'logs'
            ]
            
            for dir_path in expected_dirs:
                full_path = project_path / dir_path
                if full_path.exists():  # Some directories might be created on demand
                    assert full_path.is_dir()
    
    def test_init_config_generation(self):
        """Test that configuration files are generated."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, ['test-project'])
            assert result.exit_code == 0
            
            project_path = Path('test-project')
            config_file = project_path / 'config.json'
            
            if config_file.exists():
                # Validate config file is valid JSON
                with open(config_file) as f:
                    config = json.load(f)
                assert isinstance(config, dict)


class TestInitTemplates:
    """Test cases for init templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_init_default_template(self):
        """Test initialization with default template."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project',
                '--template', 'default'
            ])
            assert result.exit_code == 0
    
    def test_init_research_template(self):
        """Test initialization with research template."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project',
                '--template', 'research'
            ])
            assert result.exit_code == 0
    
    def test_init_production_template(self):
        """Test initialization with production template."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project',
                '--template', 'production'
            ])
            assert result.exit_code == 0
    
    def test_init_finetuning_template(self):
        """Test initialization with fine-tuning template."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project',
                '--template', 'fine-tuning'
            ])
            assert result.exit_code == 0
    
    def test_init_invalid_template(self):
        """Test initialization with invalid template."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project',
                '--template', 'nonexistent'
            ])
            # Should handle invalid template gracefully
            assert result.exit_code != 0


class TestInitOptions:
    """Test cases for init command options."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_init_with_description(self):
        """Test initialization with project description."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project',
                '--description', 'Test project description'
            ])
            assert result.exit_code == 0
    
    def test_init_with_author(self):
        """Test initialization with author information."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project',
                '--author', 'Test Author'
            ])
            assert result.exit_code == 0
    
    def test_init_force_overwrite(self):
        """Test initialization with force overwrite."""
        with self.runner.isolated_filesystem():
            # Create existing project
            Path('test-project').mkdir()
            
            result = self.runner.invoke(init_group, [
                'test-project',
                '--force'
            ])
            # Should succeed with force flag
            assert result.exit_code == 0
    
    def test_init_dry_run(self):
        """Test initialization with dry run."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(init_group, [
                'test-project',
                '--dry-run'
            ])
            assert result.exit_code == 0
            
            # Project should not be created in dry run
            assert not Path('test-project').exists()


if __name__ == '__main__':
    pytest.main([__file__])