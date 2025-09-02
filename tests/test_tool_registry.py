"""
Tests for the tool registry system.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from llmbuilder.core.tools.registry import ToolRegistry, ToolMetadata


def example_function(name: str, age: int = 25) -> str:
    """Example function for testing.
    
    Args:
        name: Person's name
        age: Person's age
        
    Returns:
        Greeting message
    """
    return f"Hello {name}, you are {age} years old!"


class TestToolRegistry:
    """Test cases for ToolRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry_path = self.temp_dir / "test_registry.json"
        self.registry = ToolRegistry(self.registry_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_register_function(self):
        """Test registering a function as a tool."""
        tool_name = self.registry.register_function(
            func=example_function,
            name="test_tool",
            description="Test tool for greeting",
            category="custom",
            version="1.0.0",
            author="test_author"
        )
        
        assert tool_name == "test_tool"
        assert "test_tool" in self.registry.tools
        
        tool = self.registry.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.description == "Test tool for greeting"
        assert tool.category == "custom"
        assert tool.version == "1.0.0"
        assert tool.author == "test_author"
        assert tool.enabled is True
    
    def test_function_signature_generation(self):
        """Test function signature generation."""
        self.registry.register_function(example_function)
        tool = self.registry.get_tool("example_function")
        
        assert tool is not None
        sig = tool.function_signature
        
        assert sig['name'] == "example_function"
        assert 'name' in sig['parameters']
        assert 'age' in sig['parameters']
        
        # Check parameter details
        name_param = sig['parameters']['name']
        assert name_param['annotation'] == "<class 'str'>"
        assert name_param['default'] == "None"
        
        age_param = sig['parameters']['age']
        assert age_param['annotation'] == "<class 'int'>"
        assert age_param['default'] == "25"
    
    def test_tool_schema_generation(self):
        """Test OpenAI-compatible tool schema generation."""
        self.registry.register_function(example_function)
        tool = self.registry.get_tool("example_function")
        
        assert tool is not None
        schema = tool.schema
        
        assert schema['type'] == "function"
        assert 'function' in schema
        
        func_schema = schema['function']
        assert func_schema['name'] == "example_function"
        assert 'description' in func_schema
        assert 'parameters' in func_schema
        
        params = func_schema['parameters']
        assert params['type'] == "object"
        assert 'properties' in params
        assert 'required' in params
        
        # Check properties
        properties = params['properties']
        assert 'name' in properties
        assert 'age' in properties
        
        # Check required parameters (those without defaults)
        required = params['required']
        assert 'name' in required
        assert 'age' not in required  # Has default value
    
    def test_list_tools(self):
        """Test listing tools with filters."""
        # Register multiple tools
        self.registry.register_function(example_function, category="custom")
        
        def another_function():
            return "test"
        
        self.registry.register_function(another_function, category="alarm")
        
        # Test listing all tools
        all_tools = self.registry.list_tools()
        assert len(all_tools) == 2
        
        # Test filtering by category
        custom_tools = self.registry.list_tools(category="custom")
        assert len(custom_tools) == 1
        assert custom_tools[0].name == "example_function"
        
        alarm_tools = self.registry.list_tools(category="alarm")
        assert len(alarm_tools) == 1
        assert alarm_tools[0].name == "another_function"
    
    def test_enable_disable_tool(self):
        """Test enabling and disabling tools."""
        self.registry.register_function(example_function)
        
        # Tool should be enabled by default
        tool = self.registry.get_tool("example_function")
        assert tool.enabled is True
        
        # Disable tool
        assert self.registry.disable_tool("example_function") is True
        tool = self.registry.get_tool("example_function")
        assert tool.enabled is False
        
        # Enable tool
        assert self.registry.enable_tool("example_function") is True
        tool = self.registry.get_tool("example_function")
        assert tool.enabled is True
        
        # Test with non-existent tool
        assert self.registry.disable_tool("nonexistent") is False
        assert self.registry.enable_tool("nonexistent") is False
    
    def test_unregister_tool(self):
        """Test unregistering tools."""
        self.registry.register_function(example_function)
        
        # Tool should exist
        assert self.registry.get_tool("example_function") is not None
        
        # Unregister tool
        assert self.registry.unregister_tool("example_function") is True
        assert self.registry.get_tool("example_function") is None
        
        # Test unregistering non-existent tool
        assert self.registry.unregister_tool("nonexistent") is False
    
    def test_registry_persistence(self):
        """Test that registry persists to disk."""
        # Register a tool
        self.registry.register_function(example_function, name="persistent_tool")
        
        # Create new registry instance with same path
        new_registry = ToolRegistry(self.registry_path)
        
        # Tool should be loaded from disk
        tool = new_registry.get_tool("persistent_tool")
        assert tool is not None
        assert tool.name == "persistent_tool"
    
    def test_register_from_file(self):
        """Test registering a tool from a Python file."""
        # Create a temporary Python file
        tool_file = self.temp_dir / "test_tool.py"
        tool_content = '''
def greet_user(name: str, greeting: str = "Hello") -> str:
    """Greet a user with a custom greeting.
    
    Args:
        name: User's name
        greeting: Greeting message
        
    Returns:
        Formatted greeting
    """
    return f"{greeting}, {name}!"
'''
        
        with open(tool_file, 'w') as f:
            f.write(tool_content)
        
        # Register function from file
        tool_name = self.registry.register_from_file(
            file_path=tool_file,
            function_name="greet_user",
            name="file_tool",
            description="Tool from file"
        )
        
        assert tool_name == "file_tool"
        
        tool = self.registry.get_tool("file_tool")
        assert tool is not None
        assert tool.file_path == str(tool_file)
        assert tool.function_signature['name'] == "greet_user"
    
    def test_invalid_file_registration(self):
        """Test error handling for invalid file registration."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            self.registry.register_from_file(
                file_path=Path("nonexistent.py"),
                function_name="test_func"
            )
        
        # Test file without the specified function
        tool_file = self.temp_dir / "empty_tool.py"
        with open(tool_file, 'w') as f:
            f.write("# Empty file\n")
        
        with pytest.raises(AttributeError):
            self.registry.register_from_file(
                file_path=tool_file,
                function_name="nonexistent_func"
            )


if __name__ == "__main__":
    pytest.main([__file__])