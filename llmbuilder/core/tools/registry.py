"""
Tool Registry for managing custom tools and functions.
"""

import json
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    category: str  # 'alarm', 'messaging', 'data_processing', 'custom'
    version: str
    author: str
    created_at: datetime
    updated_at: datetime
    function_signature: Dict[str, Any]
    schema: Dict[str, Any]
    file_path: Optional[str] = None
    module_name: Optional[str] = None
    enabled: bool = True


class ToolRegistry:
    """Registry for managing custom tools and functions."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize the tool registry.
        
        Args:
            registry_path: Path to the registry file. Defaults to ~/.llmbuilder/tools/registry.json
        """
        if registry_path is None:
            registry_path = Path.home() / ".llmbuilder" / "tools" / "registry.json"
        
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.tools: Dict[str, ToolMetadata] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load tools from the registry file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                
                for tool_name, tool_data in data.items():
                    # Convert datetime strings back to datetime objects
                    tool_data['created_at'] = datetime.fromisoformat(tool_data['created_at'])
                    tool_data['updated_at'] = datetime.fromisoformat(tool_data['updated_at'])
                    self.tools[tool_name] = ToolMetadata(**tool_data)
                    
            except Exception as e:
                logger.error(f"Failed to load tool registry: {e}")
                self.tools = {}
    
    def _save_registry(self) -> None:
        """Save tools to the registry file."""
        try:
            data = {}
            for tool_name, tool_metadata in self.tools.items():
                tool_dict = asdict(tool_metadata)
                # Convert datetime objects to strings for JSON serialization
                tool_dict['created_at'] = tool_metadata.created_at.isoformat()
                tool_dict['updated_at'] = tool_metadata.updated_at.isoformat()
                data[tool_name] = tool_dict
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save tool registry: {e}")
            raise
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "custom",
        version: str = "1.0.0",
        author: str = "unknown"
    ) -> str:
        """Register a function as a tool.
        
        Args:
            func: The function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            category: Tool category
            version: Tool version
            author: Tool author
            
        Returns:
            The registered tool name
        """
        if name is None:
            name = func.__name__
        
        if description is None:
            description = func.__doc__ or f"Tool: {name}"
        
        # Generate function signature and schema
        signature = self._generate_function_signature(func)
        schema = self._generate_tool_schema(func, name, description)
        
        # Create tool metadata
        now = datetime.now()
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            version=version,
            author=author,
            created_at=now,
            updated_at=now,
            function_signature=signature,
            schema=schema,
            module_name=func.__module__
        )
        
        self.tools[name] = metadata
        self._save_registry()
        
        logger.info(f"Registered tool: {name}")
        return name
    
    def register_from_file(
        self,
        file_path: Path,
        function_name: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "custom",
        version: str = "1.0.0",
        author: str = "unknown"
    ) -> str:
        """Register a function from a Python file.
        
        Args:
            file_path: Path to the Python file
            function_name: Name of the function to register
            name: Tool name (defaults to function name)
            description: Tool description
            category: Tool category
            version: Tool version
            author: Tool author
            
        Returns:
            The registered tool name
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("tool_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        if not hasattr(module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in {file_path}")
        
        func = getattr(module, function_name)
        
        # Register the function
        tool_name = self.register_function(
            func=func,
            name=name,
            description=description,
            category=category,
            version=version,
            author=author
        )
        
        # Update with file path
        self.tools[tool_name].file_path = str(file_path)
        self._save_registry()
        
        return tool_name
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self.tools:
            del self.tools[name]
            self._save_registry()
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def list_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[ToolMetadata]:
        """List registered tools.
        
        Args:
            category: Filter by category (optional)
            enabled_only: Only return enabled tools
            
        Returns:
            List of tool metadata
        """
        tools = []
        for tool in self.tools.values():
            if enabled_only and not tool.enabled:
                continue
            if category and tool.category != category:
                continue
            tools.append(tool)
        
        return sorted(tools, key=lambda t: t.name)
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool metadata or None if not found
        """
        return self.tools.get(name)
    
    def enable_tool(self, name: str) -> bool:
        """Enable a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was enabled, False if not found
        """
        if name in self.tools:
            self.tools[name].enabled = True
            self.tools[name].updated_at = datetime.now()
            self._save_registry()
            return True
        return False
    
    def disable_tool(self, name: str) -> bool:
        """Disable a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was disabled, False if not found
        """
        if name in self.tools:
            self.tools[name].enabled = False
            self.tools[name].updated_at = datetime.now()
            self._save_registry()
            return True
        return False
    
    def _generate_function_signature(self, func: Callable) -> Dict[str, Any]:
        """Generate function signature information.
        
        Args:
            func: The function
            
        Returns:
            Function signature information
        """
        sig = inspect.signature(func)
        signature_info = {
            'name': func.__name__,
            'parameters': {},
            'return_annotation': str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None
        }
        
        for param_name, param in sig.parameters.items():
            param_info = {
                'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                'kind': str(param.kind)
            }
            signature_info['parameters'][param_name] = param_info
        
        return signature_info
    
    def _generate_tool_schema(self, func: Callable, name: str, description: str) -> Dict[str, Any]:
        """Generate OpenAI-compatible tool schema.
        
        Args:
            func: The function
            name: Tool name
            description: Tool description
            
        Returns:
            Tool schema
        """
        sig = inspect.signature(func)
        
        # Build parameters schema
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_schema = {"type": "string"}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if annotation == int:
                    param_schema["type"] = "integer"
                elif annotation == float:
                    param_schema["type"] = "number"
                elif annotation == bool:
                    param_schema["type"] = "boolean"
                elif annotation == list:
                    param_schema["type"] = "array"
                elif annotation == dict:
                    param_schema["type"] = "object"
            
            # Add description from docstring if available
            if func.__doc__:
                # Simple docstring parsing - could be enhanced
                param_schema["description"] = f"Parameter: {param_name}"
            
            properties[param_name] = param_schema
            
            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        
        return schema


# Global registry instance
_registry = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry