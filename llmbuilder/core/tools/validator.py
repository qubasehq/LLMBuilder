"""
Tool validation and testing framework.
"""

import inspect
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of tool validation."""
    level: ValidationLevel
    message: str
    details: Optional[str] = None


@dataclass
class TestResult:
    """Result of tool testing."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0


class ToolValidator:
    """Validator for custom tools and functions."""
    
    def __init__(self):
        self.validation_rules = [
            self._validate_function_signature,
            self._validate_docstring,
            self._validate_type_hints,
            self._validate_parameter_names,
            self._validate_return_type,
        ]
    
    def validate_function(self, func: Callable) -> List[ValidationResult]:
        """Validate a function for tool registration.
        
        Args:
            func: The function to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        for rule in self.validation_rules:
            try:
                rule_results = rule(func)
                results.extend(rule_results)
            except Exception as e:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Validation rule failed: {rule.__name__}",
                    details=str(e)
                ))
        
        return results
    
    def _validate_function_signature(self, func: Callable) -> List[ValidationResult]:
        """Validate function signature."""
        results = []
        
        try:
            sig = inspect.signature(func)
        except Exception as e:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="Cannot inspect function signature",
                details=str(e)
            ))
            return results
        
        # Check for *args or **kwargs
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message="Function uses *args which may complicate tool calling",
                    details=f"Parameter: {param.name}"
                ))
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message="Function uses **kwargs which may complicate tool calling",
                    details=f"Parameter: {param.name}"
                ))
        
        # Check parameter count
        param_count = len(sig.parameters)
        if param_count > 10:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Function has many parameters ({param_count}), consider simplifying",
                details="Tools with many parameters can be difficult to use"
            ))
        
        return results
    
    def _validate_docstring(self, func: Callable) -> List[ValidationResult]:
        """Validate function docstring."""
        results = []
        
        if not func.__doc__:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="Function lacks docstring",
                details="Docstrings help users understand tool functionality"
            ))
        elif len(func.__doc__.strip()) < 10:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="Function docstring is very short",
                details="Consider providing more detailed description"
            ))
        
        return results
    
    def _validate_type_hints(self, func: Callable) -> List[ValidationResult]:
        """Validate type hints."""
        results = []
        
        sig = inspect.signature(func)
        
        # Check parameter type hints
        missing_hints = []
        for param_name, param in sig.parameters.items():
            if param.annotation == inspect.Parameter.empty:
                missing_hints.append(param_name)
        
        if missing_hints:
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                message="Some parameters lack type hints",
                details=f"Parameters without hints: {', '.join(missing_hints)}"
            ))
        
        # Check return type hint
        if sig.return_annotation == inspect.Signature.empty:
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                message="Function lacks return type hint",
                details="Return type hints help with tool schema generation"
            ))
        
        return results
    
    def _validate_parameter_names(self, func: Callable) -> List[ValidationResult]:
        """Validate parameter names."""
        results = []
        
        sig = inspect.signature(func)
        
        for param_name in sig.parameters.keys():
            # Check for reserved names
            if param_name in ['self', 'cls']:
                continue  # These are fine for methods
            
            # Check for Python keywords
            import keyword
            if keyword.iskeyword(param_name):
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Parameter name '{param_name}' is a Python keyword",
                    details="Use a different parameter name"
                ))
            
            # Check naming convention
            if not param_name.islower() or '_' not in param_name and len(param_name) > 8:
                results.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    message=f"Parameter '{param_name}' doesn't follow snake_case convention",
                    details="Consider using snake_case for parameter names"
                ))
        
        return results
    
    def _validate_return_type(self, func: Callable) -> List[ValidationResult]:
        """Validate return type."""
        results = []
        
        sig = inspect.signature(func)
        
        # Check if return type is serializable
        if sig.return_annotation != inspect.Signature.empty:
            annotation = sig.return_annotation
            
            # Check for complex types that might not be JSON serializable
            if hasattr(annotation, '__module__') and annotation.__module__ not in ['builtins', 'typing']:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message="Return type may not be JSON serializable",
                    details=f"Return type: {annotation}. Consider using basic types or implementing serialization."
                ))
        
        return results
    
    def test_function(
        self,
        func: Callable,
        test_args: Optional[List[Any]] = None,
        test_kwargs: Optional[Dict[str, Any]] = None
    ) -> TestResult:
        """Test a function with provided arguments.
        
        Args:
            func: The function to test
            test_args: Positional arguments for testing
            test_kwargs: Keyword arguments for testing
            
        Returns:
            Test result
        """
        import time
        
        if test_args is None:
            test_args = []
        if test_kwargs is None:
            test_kwargs = {}
        
        start_time = time.time()
        
        try:
            result = func(*test_args, **test_kwargs)
            execution_time = time.time() - start_time
            
            return TestResult(
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time=execution_time
            )
    
    def generate_test_cases(self, func: Callable) -> List[Tuple[List[Any], Dict[str, Any]]]:
        """Generate basic test cases for a function.
        
        Args:
            func: The function to generate test cases for
            
        Returns:
            List of (args, kwargs) tuples for testing
        """
        test_cases = []
        sig = inspect.signature(func)
        
        # Generate a test case with default values where possible
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                continue  # Use default value
            
            # Generate basic test values based on type hints
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if annotation == int:
                    kwargs[param_name] = 42
                elif annotation == float:
                    kwargs[param_name] = 3.14
                elif annotation == str:
                    kwargs[param_name] = "test_string"
                elif annotation == bool:
                    kwargs[param_name] = True
                elif annotation == list:
                    kwargs[param_name] = ["test", "list"]
                elif annotation == dict:
                    kwargs[param_name] = {"test": "dict"}
                else:
                    kwargs[param_name] = None
            else:
                # No type hint, use string as default
                kwargs[param_name] = "test_value"
        
        test_cases.append(([], kwargs))
        
        return test_cases
    
    def validate_and_test(
        self,
        func: Callable,
        test_args: Optional[List[Any]] = None,
        test_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[ValidationResult], TestResult]:
        """Validate and test a function.
        
        Args:
            func: The function to validate and test
            test_args: Test arguments
            test_kwargs: Test keyword arguments
            
        Returns:
            Tuple of (validation results, test result)
        """
        validation_results = self.validate_function(func)
        
        # If no test arguments provided, try to generate them
        if test_args is None and test_kwargs is None:
            try:
                test_cases = self.generate_test_cases(func)
                if test_cases:
                    test_args, test_kwargs = test_cases[0]
            except Exception as e:
                logger.warning(f"Failed to generate test cases: {e}")
        
        test_result = self.test_function(func, test_args, test_kwargs)
        
        return validation_results, test_result


def validate_tool(func: Callable) -> List[ValidationResult]:
    """Convenience function to validate a tool function.
    
    Args:
        func: The function to validate
        
    Returns:
        List of validation results
    """
    validator = ToolValidator()
    return validator.validate_function(func)


def test_tool(
    func: Callable,
    test_args: Optional[List[Any]] = None,
    test_kwargs: Optional[Dict[str, Any]] = None
) -> TestResult:
    """Convenience function to test a tool function.
    
    Args:
        func: The function to test
        test_args: Test arguments
        test_kwargs: Test keyword arguments
        
    Returns:
        Test result
    """
    validator = ToolValidator()
    return validator.test_function(func, test_args, test_kwargs)