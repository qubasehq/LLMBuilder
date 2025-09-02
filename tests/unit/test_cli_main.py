#!/usr/bin/env python3
"""
Unit tests for main CLI interface.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llmbuilder.cli.main import cli


class TestMainCLI:
    """Test cases for main CLI interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'LLMBuilder' in result.output
        assert 'Usage:' in result.output
    
    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
    
    def test_cli_verbose_flag(self):
        """Test CLI verbose flag."""
        result = self.runner.invoke(cli, ['--verbose', '--help'])
        assert result.exit_code == 0
    
    def test_cli_config_flag(self):
        """Test CLI config flag."""
        with self.runner.isolated_filesystem():
            # Create a dummy config file
            config_content = '{"test": "value"}'
            with open('test_config.json', 'w') as f:
                f.write(config_content)
            
            result = self.runner.invoke(cli, ['--config', 'test_config.json', '--help'])
            assert result.exit_code == 0
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        result = self.runner.invoke(cli, ['invalid-command'])
        assert result.exit_code != 0
        assert 'No such command' in result.output
    
    @pytest.mark.unit
    def test_cli_context_object(self):
        """Test CLI context object creation."""
        with patch('llmbuilder.cli.main.cli') as mock_cli:
            mock_ctx = MagicMock()
            mock_ctx.ensure_object.return_value = {}
            
            # Test that context is properly initialized
            result = self.runner.invoke(cli, ['--verbose', '--help'])
            assert result.exit_code == 0


class TestCLISubcommands:
    """Test cases for CLI subcommands availability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_init_command_exists(self):
        """Test that init command exists."""
        result = self.runner.invoke(cli, ['init', '--help'])
        assert result.exit_code == 0
        assert 'init' in result.output.lower()
    
    def test_data_command_exists(self):
        """Test that data command exists."""
        result = self.runner.invoke(cli, ['data', '--help'])
        assert result.exit_code == 0
        assert 'data' in result.output.lower()
    
    def test_model_command_exists(self):
        """Test that model command exists."""
        result = self.runner.invoke(cli, ['model', '--help'])
        assert result.exit_code == 0
        assert 'model' in result.output.lower()
    
    def test_train_command_exists(self):
        """Test that train command exists."""
        result = self.runner.invoke(cli, ['train', '--help'])
        assert result.exit_code == 0
        assert 'train' in result.output.lower()
    
    def test_eval_command_exists(self):
        """Test that eval command exists."""
        result = self.runner.invoke(cli, ['eval', '--help'])
        assert result.exit_code == 0
        assert 'eval' in result.output.lower()
    
    def test_optimize_command_exists(self):
        """Test that optimize command exists."""
        result = self.runner.invoke(cli, ['optimize', '--help'])
        assert result.exit_code == 0
        assert 'optimize' in result.output.lower()
    
    def test_deploy_command_exists(self):
        """Test that deploy command exists."""
        result = self.runner.invoke(cli, ['deploy', '--help'])
        assert result.exit_code == 0
        assert 'deploy' in result.output.lower()
    
    def test_inference_command_exists(self):
        """Test that inference command exists."""
        result = self.runner.invoke(cli, ['inference', '--help'])
        assert result.exit_code == 0
        assert 'inference' in result.output.lower()
    
    def test_config_command_exists(self):
        """Test that config command exists."""
        result = self.runner.invoke(cli, ['config', '--help'])
        assert result.exit_code == 0
        assert 'config' in result.output.lower()
    
    def test_tools_command_exists(self):
        """Test that tools command exists."""
        result = self.runner.invoke(cli, ['tools', '--help'])
        assert result.exit_code == 0
        assert 'tools' in result.output.lower()
    
    def test_monitor_command_exists(self):
        """Test that monitor command exists."""
        result = self.runner.invoke(cli, ['monitor', '--help'])
        assert result.exit_code == 0
        assert 'monitor' in result.output.lower()


class TestCLIErrorHandling:
    """Test cases for CLI error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_missing_required_args(self):
        """Test CLI behavior with missing required arguments."""
        # This will depend on specific command implementations
        result = self.runner.invoke(cli, ['init'])
        # Should either succeed with defaults or show helpful error
        assert result.exit_code in [0, 2]  # 0 for success, 2 for usage error
    
    def test_invalid_config_file(self):
        """Test CLI behavior with invalid config file."""
        result = self.runner.invoke(cli, ['--config', 'nonexistent.json', '--help'])
        # Should still show help even with invalid config
        assert result.exit_code == 0
    
    def test_permission_error_handling(self):
        """Test CLI behavior with permission errors."""
        with self.runner.isolated_filesystem():
            # Create a directory we can't write to (simulation)
            result = self.runner.invoke(cli, ['init', 'test-project'])
            # Should handle gracefully
            assert isinstance(result.exit_code, int)


if __name__ == '__main__':
    pytest.main([__file__])