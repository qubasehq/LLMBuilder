"""
Tests for the enhanced help and documentation system.
"""

import pytest
import subprocess
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from llmbuilder.cli.help import help, HelpSystem
from llmbuilder.cli.upgrade import upgrade, UpgradeManager
from llmbuilder.utils.usage_analytics import UsageAnalytics
from llmbuilder.utils.error_handler import ErrorHandler, LLMBuilderError


class TestHelpSystem:
    """Test cases for the help system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.help_system = HelpSystem()
    
    def test_help_command_exists(self):
        """Test that help command is available."""
        result = self.runner.invoke(help, ['--help'])
        assert result.exit_code == 0
        assert "Enhanced help and documentation system" in result.output
    
    def test_interactive_help_command(self):
        """Test interactive help command."""
        # Test with exit option
        result = self.runner.invoke(help, ['interactive'], input='0\n')
        assert result.exit_code == 0
        assert "LLMBuilder Interactive Help" in result.output
    
    def test_discover_command(self):
        """Test command discovery."""
        result = self.runner.invoke(help, ['discover'])
        assert result.exit_code == 0
        assert "Command Discovery" in result.output
        assert "Project Setup" in result.output
        assert "Data Management" in result.output
    
    def test_examples_command(self):
        """Test examples command."""
        result = self.runner.invoke(help, ['examples'])
        assert result.exit_code == 0
        assert "Usage Examples" in result.output
    
    def test_examples_with_category(self):
        """Test examples command with specific category."""
        result = self.runner.invoke(help, ['examples', '--category', 'init'])
        assert result.exit_code == 0
    
    def test_workflows_command(self):
        """Test workflows command."""
        result = self.runner.invoke(help, ['workflows'])
        assert result.exit_code == 0
        assert "Complete Workflows" in result.output
    
    def test_troubleshooting_command(self):
        """Test troubleshooting command."""
        result = self.runner.invoke(help, ['troubleshooting'])
        assert result.exit_code == 0
        assert "Troubleshooting" in result.output
    
    def test_docs_command_no_topic(self):
        """Test docs command without topic."""
        result = self.runner.invoke(help, ['docs'])
        assert result.exit_code == 0
        assert "Available Documentation Topics" in result.output
    
    def test_docs_command_with_topic(self):
        """Test docs command with specific topic."""
        result = self.runner.invoke(help, ['docs', 'getting-started'])
        assert result.exit_code == 0
        assert "Getting Started" in result.output
    
    def test_docs_command_invalid_topic(self):
        """Test docs command with invalid topic."""
        result = self.runner.invoke(help, ['docs', 'invalid-topic'])
        assert result.exit_code == 0
        assert "Unknown topic" in result.output
    
    def test_help_system_examples_loading(self):
        """Test that examples are loaded correctly."""
        examples = self.help_system.examples
        assert isinstance(examples, dict)
        assert "init" in examples
        assert "data" in examples
        assert "train" in examples
    
    def test_help_system_workflows_loading(self):
        """Test that workflows are loaded correctly."""
        workflows = self.help_system.workflows
        assert isinstance(workflows, dict)
        assert len(workflows) > 0
        
        # Check workflow structure
        for workflow_name, workflow in workflows.items():
            assert "description" in workflow
            assert "steps" in workflow
            assert isinstance(workflow["steps"], list)


class TestUpgradeManager:
    """Test cases for the upgrade manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.upgrade_manager = UpgradeManager()
    
    def test_upgrade_command_exists(self):
        """Test that upgrade command is available."""
        result = self.runner.invoke(upgrade, ['--help'])
        assert result.exit_code == 0
        assert "Package upgrade and update system" in result.output
    
    def test_check_command(self):
        """Test check for updates command."""
        with patch('requests.get') as mock_get:
            # Mock PyPI response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "info": {
                    "version": "2.0.0",
                    "summary": "Test package"
                },
                "releases": {
                    "2.0.0": [{"upload_time": "2023-01-01T00:00:00"}]
                }
            }
            mock_get.return_value = mock_response
            
            result = self.runner.invoke(upgrade, ['check'])
            assert result.exit_code == 0
    
    def test_package_check_only(self):
        """Test package upgrade with check-only flag."""
        result = self.runner.invoke(upgrade, ['package', '--check-only'])
        assert result.exit_code == 0
    
    def test_templates_command(self):
        """Test templates update command."""
        result = self.runner.invoke(upgrade, ['templates'])
        assert result.exit_code == 0
    
    def test_changelog_command(self):
        """Test changelog command."""
        result = self.runner.invoke(upgrade, ['changelog'])
        assert result.exit_code == 0
    
    def test_version_comparison(self):
        """Test version comparison logic."""
        assert self.upgrade_manager._is_newer_version("2.0.0", "1.9.0") is True
        assert self.upgrade_manager._is_newer_version("1.9.0", "2.0.0") is False
        assert self.upgrade_manager._is_newer_version("1.0.0", "1.0.0") is False
    
    @patch('subprocess.run')
    def test_package_upgrade_success(self, mock_run):
        """Test successful package upgrade."""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = self.upgrade_manager.upgrade_package("2.0.0")
        assert result is True
    
    @patch('subprocess.run')
    def test_package_upgrade_failure(self, mock_run):
        """Test failed package upgrade."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "pip")
        
        result = self.upgrade_manager.upgrade_package("2.0.0")
        assert result is False


class TestUsageAnalytics:
    """Test cases for usage analytics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analytics = UsageAnalytics()
    
    def test_record_command(self):
        """Test recording command usage."""
        self.analytics.record_command("test", ["--help"], success=True, execution_time=1.0)
        
        # Check that command was recorded
        commands = self.analytics.usage_data.get("commands", [])
        assert len(commands) > 0
        
        last_command = commands[-1]
        assert last_command["command"] == "test"
        assert last_command["args"] == ["--help"]
        assert last_command["success"] is True
        assert last_command["execution_time"] == 1.0
    
    def test_get_command_suggestions(self):
        """Test getting command suggestions."""
        # Record some commands to build patterns
        self.analytics.record_command("init", [], success=True)
        
        suggestions = self.analytics.get_command_suggestions("init")
        assert isinstance(suggestions, list)
    
    def test_usage_statistics(self):
        """Test usage statistics calculation."""
        # Record some test commands
        self.analytics.record_command("test1", [], success=True, execution_time=1.0)
        self.analytics.record_command("test2", [], success=False, execution_time=2.0)
        
        # Test that statistics are updated
        stats = self.analytics.usage_data["statistics"]
        assert stats["total_commands"] >= 2
        assert stats["successful_commands"] >= 1
        assert stats["failed_commands"] >= 1


class TestErrorHandler:
    """Test cases for error handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_llmbuilder_error_creation(self):
        """Test creating LLMBuilder errors."""
        error = LLMBuilderError(
            "Test error",
            suggestions=["Try this", "Or this"],
            recovery_commands=["llmbuilder help"]
        )
        
        assert str(error) == "Test error"
        assert error.suggestions == ["Try this", "Or this"]
        assert error.recovery_commands == ["llmbuilder help"]
    
    def test_error_pattern_matching(self):
        """Test error pattern matching."""
        # Test CUDA out of memory error
        suggestions = self.error_handler._match_error_patterns(
            "CUDA out of memory", "RuntimeError"
        )
        
        assert len(suggestions) > 0
        assert any("batch size" in s["message"].lower() for s in suggestions)
    
    def test_module_name_extraction(self):
        """Test extracting module name from import errors."""
        error_msg = "No module named 'torch'"
        module_name = self.error_handler._extract_module_name(error_msg)
        assert module_name == "torch"
        
        error_msg = "No module named \"transformers\""
        module_name = self.error_handler._extract_module_name(error_msg)
        assert module_name == "transformers"
    
    def test_error_patterns_loading(self):
        """Test that error patterns are loaded correctly."""
        patterns = self.error_handler.error_patterns
        assert isinstance(patterns, dict)
        assert "cuda_oom" in patterns
        assert "file_not_found" in patterns
        assert "dependency_missing" in patterns


if __name__ == "__main__":
    pytest.main([__file__])