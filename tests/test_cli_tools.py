"""
Tests for the CLI tools commands.
"""

import pytest
import tempfile
import json
from pathlib import Path
from click.testing import CliRunner

from llmbuilder.cli.tools import tools
from llmbuilder.core.tools.registry import ToolRegistry


class TestCLITools:
    """Test cases for CLI tools commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create a test tool file
        self.test_tool_file = self.temp_dir / "test_tool.py"
        tool_content = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of the two numbers
    """
    return a + b


def greet_person(name: str, greeting: str = "Hello") -> str:
    """Greet a person with a custom greeting.
    
    Args:
        name: Person's name
        greeting: Greeting message
        
    Returns:
        Formatted greeting
    """
    return f"{greeting}, {name}!"
'''
        
        with open(self.test_tool_file, 'w') as f:
            f.write(tool_content)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_register_tool_command(self):
        """Test the register command."""
        result = self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--name', 'sum_tool',
            '--description', 'Tool for calculating sums',
            '--category', 'custom',
            '--author', 'test_author',
            '--no-validate',
            '--no-test'
        ])
        
        assert result.exit_code == 0
        assert "registered successfully" in result.output.lower()
    
    def test_register_tool_with_validation(self):
        """Test register command with validation enabled."""
        result = self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--validate'
        ])
        
        assert result.exit_code == 0
        # Should show validation results
        assert "validating" in result.output.lower() or "validation" in result.output.lower()
    
    def test_register_nonexistent_file(self):
        """Test register command with non-existent file."""
        result = self.runner.invoke(tools, [
            'register',
            'nonexistent.py',
            'some_function'
        ])
        
        assert result.exit_code != 0
    
    def test_register_nonexistent_function(self):
        """Test register command with non-existent function."""
        result = self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'nonexistent_function'
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
    
    def test_list_tools_empty(self):
        """Test list command with no tools."""
        result = self.runner.invoke(tools, ['list'])
        
        assert result.exit_code == 0
        assert "no tools found" in result.output.lower()
    
    def test_list_tools_with_tools(self):
        """Test list command with registered tools."""
        # First register a tool
        self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--no-validate',
            '--no-test'
        ])
        
        # Then list tools
        result = self.runner.invoke(tools, ['list'])
        
        assert result.exit_code == 0
        assert "calculate_sum" in result.output
    
    def test_list_tools_by_category(self):
        """Test list command with category filter."""
        # Register tools in different categories
        self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--category', 'custom',
            '--no-validate',
            '--no-test'
        ])
        
        self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'greet_person',
            '--category', 'messaging',
            '--no-validate',
            '--no-test'
        ])
        
        # List only custom tools
        result = self.runner.invoke(tools, ['list', '--category', 'custom'])
        
        assert result.exit_code == 0
        assert "calculate_sum" in result.output
        assert "greet_person" not in result.output
    
    def test_info_command(self):
        """Test the info command."""
        # Register a tool first
        self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--name', 'sum_tool',
            '--description', 'Tool for calculating sums',
            '--no-validate',
            '--no-test'
        ])
        
        # Get tool info
        result = self.runner.invoke(tools, ['info', 'sum_tool'])
        
        assert result.exit_code == 0
        assert "sum_tool" in result.output
        assert "Tool for calculating sums" in result.output
        assert "Function Signature" in result.output
        assert "Tool Schema" in result.output
    
    def test_info_nonexistent_tool(self):
        """Test info command with non-existent tool."""
        result = self.runner.invoke(tools, ['info', 'nonexistent_tool'])
        
        assert result.exit_code == 0
        assert "not found" in result.output.lower()
    
    def test_test_command(self):
        """Test the test command."""
        # Register a tool first
        self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--no-validate',
            '--no-test'
        ])
        
        # Test the tool
        test_args = json.dumps({"a": 5, "b": 3})
        result = self.runner.invoke(tools, [
            'test', 
            'calculate_sum',
            '--test-args', test_args
        ])
        
        assert result.exit_code == 0
        assert "test passed" in result.output.lower() or "✓" in result.output
    
    def test_test_command_invalid_args(self):
        """Test test command with invalid JSON arguments."""
        # Register a tool first
        self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--no-validate',
            '--no-test'
        ])
        
        # Test with invalid JSON
        result = self.runner.invoke(tools, [
            'test',
            'calculate_sum',
            '--test-args', 'invalid json'
        ])
        
        assert result.exit_code == 0
        assert "invalid json" in result.output.lower()
    
    def test_unregister_command(self):
        """Test the unregister command."""
        # Register a tool first
        self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--no-validate',
            '--no-test'
        ])
        
        # Unregister the tool (with confirmation)
        result = self.runner.invoke(tools, [
            'unregister',
            'calculate_sum'
        ], input='y\n')
        
        assert result.exit_code == 0
        assert "unregistered successfully" in result.output.lower()
    
    def test_enable_disable_commands(self):
        """Test enable and disable commands."""
        # Register a tool first
        self.runner.invoke(tools, [
            'register',
            str(self.test_tool_file),
            'calculate_sum',
            '--no-validate',
            '--no-test'
        ])
        
        # Disable the tool
        result = self.runner.invoke(tools, ['disable', 'calculate_sum'])
        assert result.exit_code == 0
        assert "disabled" in result.output.lower()
        
        # Enable the tool
        result = self.runner.invoke(tools, ['enable', 'calculate_sum'])
        assert result.exit_code == 0
        assert "enabled" in result.output.lower()
    
    def test_search_command(self):
        """Test the search command."""
        result = self.runner.invoke(tools, ['search', '--query', 'example'])
        
        # Should not crash even if marketplace is not available
        assert result.exit_code == 0
    
    def test_template_command(self):
        """Test the template command."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(tools, [
                'template',
                '--category', 'alarm'
            ])
            
            assert result.exit_code == 0
            assert "template created" in result.output.lower()
            
            # Check that template file was created
            assert Path("my_alarm_tool.py").exists()
    
    def test_template_command_interactive(self):
        """Test template command with interactive category selection."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(tools, ['template'], input='alarm\n')
            
            assert result.exit_code == 0
            assert Path("my_alarm_tool.py").exists()
    
    def test_template_overwrite_confirmation(self):
        """Test template command with overwrite confirmation."""
        with self.runner.isolated_filesystem():
            # Create existing file
            Path("my_alarm_tool.py").touch()
            
            # Try to create template (decline overwrite)
            result = self.runner.invoke(tools, [
                'template',
                '--category', 'alarm'
            ], input='n\n')
            
            assert result.exit_code == 0
    
    def test_install_command_not_found(self):
        """Test install command with non-existent tool."""
        result = self.runner.invoke(tools, ['install', 'nonexistent_tool'])
        
        # Should handle gracefully even if marketplace is not available
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])