"""
Simplified integration tests for CLI functionality.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path

from llmbuilder.cli.main import cli
from llmbuilder.utils.workflow import WorkflowManager


class TestCLIIntegration:
    """Test CLI integration and basic functionality."""
    
    def test_cli_help_system(self):
        """Test that help system works correctly."""
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "LLMBuilder" in result.output
        
        # Test version
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
    
    def test_command_registration(self):
        """Test that all commands are properly registered."""
        runner = CliRunner()
        
        # Test that main command groups exist
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        # Check for key commands
        expected_commands = ['init', 'config', 'data', 'model', 'train', 'pipeline']
        for command in expected_commands:
            assert command in result.output
    
    def test_workflow_manager_basic(self, temp_workspace):
        """Test basic workflow manager functionality."""
        workflow_manager = WorkflowManager(temp_workspace)
        
        # Create a simple workflow
        steps = [
            {"command": "test command", "args": {"param": "value"}}
        ]
        
        workflow_id = workflow_manager.create_workflow("test", steps)
        assert workflow_id is not None
        
        # Load the workflow
        workflow_data = workflow_manager.load_workflow(workflow_id)
        assert workflow_data["name"] == "test"
        assert len(workflow_data["steps"]) == 1
        
        # Test shared data
        workflow_manager.set_shared_data(workflow_id, "key", "value")
        value = workflow_manager.get_shared_data(workflow_id, "key")
        assert value == "value"
    
    def test_configuration_system(self):
        """Test basic configuration system functionality."""
        from llmbuilder.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        default_config = config_manager.get_default_config()
        
        # Verify default config structure
        assert "model" in default_config
        assert "training" in default_config
        assert "data" in default_config
    
    def test_error_handling(self):
        """Test basic error handling."""
        runner = CliRunner()
        
        # Test invalid command
        result = runner.invoke(cli, ['nonexistent-command'])
        assert result.exit_code != 0
        
        # Test invalid subcommand
        result = runner.invoke(cli, ['config', 'invalid-subcommand'])
        assert result.exit_code != 0
    
    def test_pipeline_commands_exist(self):
        """Test that pipeline commands are available."""
        runner = CliRunner()
        
        # Test pipeline help
        result = runner.invoke(cli, ['pipeline', '--help'])
        assert result.exit_code == 0
        assert "pipeline" in result.output.lower()
    
    def test_global_options(self):
        """Test global CLI options."""
        runner = CliRunner()
        
        # Test verbose option
        result = runner.invoke(cli, ['--verbose', '--help'])
        assert result.exit_code == 0
        
        # Test quiet option
        result = runner.invoke(cli, ['--quiet', '--help'])
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])