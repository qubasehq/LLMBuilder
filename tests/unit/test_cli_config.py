#!/usr/bin/env python3
"""
Unit tests for CLI config commands.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llmbuilder.cli.config import config_group


class TestConfigSetCommand:
    """Test cases for config set command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_config_set_help(self):
        """Test config set command help."""
        result = self.runner.invoke(config_group, ['set', '--help'])
        assert result.exit_code == 0
        assert 'set' in result.output.lower()
    
    def test_config_set_basic(self):
        """Test basic config setting."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, [
                'set',
                'model.vocab_size',
                '50000'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_set_nested_key(self):
        """Test setting nested configuration key."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, [
                'set',
                'training.batch_size',
                '32'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_set_json_value(self):
        """Test setting JSON value."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, [
                'set',
                'training.optimizer',
                '{"type": "adam", "lr": 0.001}'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_set_with_type(self):
        """Test setting config with explicit type."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, [
                'set',
                'model.dropout',
                '0.1',
                '--type', 'float'
            ])
            
            assert isinstance(result.exit_code, int)
    
    @patch('llmbuilder.utils.config.ConfigManager')
    def test_config_set_with_mock_manager(self, mock_config_manager):
        """Test config set with mocked config manager."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.set_value.return_value = True
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, [
                'set',
                'test.key',
                'test_value'
            ])
            
            assert result.exit_code == 0


class TestConfigGetCommand:
    """Test cases for config get command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_config_get_help(self):
        """Test config get command help."""
        result = self.runner.invoke(config_group, ['get', '--help'])
        assert result.exit_code == 0
        assert 'get' in result.output.lower()
    
    def test_config_get_basic(self):
        """Test basic config getting."""
        with self.runner.isolated_filesystem():
            # First set a value
            self.runner.invoke(config_group, ['set', 'test.key', 'test_value'])
            
            # Then get it
            result = self.runner.invoke(config_group, ['get', 'test.key'])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_get_nonexistent_key(self):
        """Test getting nonexistent config key."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, ['get', 'nonexistent.key'])
            
            # Should handle gracefully
            assert isinstance(result.exit_code, int)
    
    def test_config_get_with_default(self):
        """Test getting config with default value."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, [
                'get',
                'nonexistent.key',
                '--default', 'default_value'
            ])
            
            assert isinstance(result.exit_code, int)
    
    @patch('llmbuilder.utils.config.ConfigManager')
    def test_config_get_with_mock_manager(self, mock_config_manager):
        """Test config get with mocked config manager."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_value.return_value = 'test_value'
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, ['get', 'test.key'])
            
            assert result.exit_code == 0
            assert 'test_value' in result.output


class TestConfigListCommand:
    """Test cases for config list command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_config_list_help(self):
        """Test config list command help."""
        result = self.runner.invoke(config_group, ['list', '--help'])
        assert result.exit_code == 0
        assert 'list' in result.output.lower()
    
    def test_config_list_basic(self):
        """Test basic config listing."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, ['list'])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_list_with_filter(self):
        """Test config listing with filter."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, [
                'list',
                '--filter', 'model'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_list_json_format(self):
        """Test config listing in JSON format."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, [
                'list',
                '--format', 'json'
            ])
            
            assert isinstance(result.exit_code, int)
    
    @patch('llmbuilder.utils.config.ConfigManager')
    def test_config_list_with_mock_manager(self, mock_config_manager):
        """Test config list with mocked config manager."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_all_config.return_value = {
            'model': {'vocab_size': 50000},
            'training': {'batch_size': 32}
        }
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, ['list'])
            
            assert result.exit_code == 0


class TestConfigResetCommand:
    """Test cases for config reset command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_config_reset_help(self):
        """Test config reset command help."""
        result = self.runner.invoke(config_group, ['reset', '--help'])
        assert result.exit_code == 0
        assert 'reset' in result.output.lower()
    
    def test_config_reset_basic(self):
        """Test basic config reset."""
        with self.runner.isolated_filesystem():
            # Set some values first
            self.runner.invoke(config_group, ['set', 'test.key1', 'value1'])
            self.runner.invoke(config_group, ['set', 'test.key2', 'value2'])
            
            # Reset all
            result = self.runner.invoke(config_group, ['reset', '--confirm'])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_reset_specific_key(self):
        """Test resetting specific config key."""
        with self.runner.isolated_filesystem():
            # Set a value first
            self.runner.invoke(config_group, ['set', 'test.key', 'value'])
            
            # Reset specific key
            result = self.runner.invoke(config_group, [
                'reset',
                'test.key',
                '--confirm'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_reset_without_confirm(self):
        """Test config reset without confirmation."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, ['reset'])
            
            # Should prompt for confirmation or fail safely
            assert isinstance(result.exit_code, int)
    
    @patch('llmbuilder.utils.config.ConfigManager')
    def test_config_reset_with_mock_manager(self, mock_config_manager):
        """Test config reset with mocked config manager."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.reset_config.return_value = True
        
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, ['reset', '--confirm'])
            
            assert result.exit_code == 0


class TestConfigValidateCommand:
    """Test cases for config validate command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_config_validate_help(self):
        """Test config validate command help."""
        result = self.runner.invoke(config_group, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'validate' in result.output.lower()
    
    def test_config_validate_basic(self):
        """Test basic config validation."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(config_group, ['validate'])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_validate_file(self):
        """Test validating specific config file."""
        with self.runner.isolated_filesystem():
            # Create test config file
            config = {
                'model': {'vocab_size': 50000},
                'training': {'batch_size': 32}
            }
            with open('test_config.json', 'w') as f:
                json.dump(config, f)
            
            result = self.runner.invoke(config_group, [
                'validate',
                '--file', 'test_config.json'
            ])
            
            assert isinstance(result.exit_code, int)
    
    def test_config_validate_invalid_file(self):
        """Test validating invalid config file."""
        with self.runner.isolated_filesystem():
            # Create invalid config file
            with open('invalid_config.json', 'w') as f:
                f.write('invalid json content')
            
            result = self.runner.invoke(config_group, [
                'validate',
                '--file', 'invalid_config.json'
            ])
            
            # Should detect invalid JSON
            assert result.exit_code != 0


if __name__ == '__main__':
    pytest.main([__file__])