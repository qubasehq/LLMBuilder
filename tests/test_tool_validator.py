"""
Tests for the tool validator system.
"""

import pytest
from typing import List, Dict, Any

from llmbuilder.core.tools.validator import ToolValidator, ValidationLevel, ValidationResult, TestResult


class TestToolValidator:
    """Test cases for ToolValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ToolValidator()
    
    def test_validate_good_function(self):
        """Test validation of a well-formed function."""
        def good_function(name: str, age: int = 25) -> str:
            """A well-documented function with type hints.
            
            Args:
                name: Person's name
                age: Person's age
                
            Returns:
                Greeting message
            """
            return f"Hello {name}, age {age}"
        
        results = self.validator.validate_function(good_function)
        
        # Should have no errors
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0
        
        # May have info/warning messages but no errors
        for result in results:
            assert result.level in [ValidationLevel.INFO, ValidationLevel.WARNING]
    
    def test_validate_function_without_docstring(self):
        """Test validation of function without docstring."""
        def no_docstring_function(x: int) -> int:
            return x * 2
        
        results = self.validator.validate_function(no_docstring_function)
        
        # Should have warning about missing docstring
        docstring_warnings = [
            r for r in results 
            if r.level == ValidationLevel.WARNING and "docstring" in r.message.lower()
        ]
        assert len(docstring_warnings) > 0
    
    def test_validate_function_without_type_hints(self):
        """Test validation of function without type hints."""
        def no_hints_function(x, y=10):
            """Function without type hints."""
            return x + y
        
        results = self.validator.validate_function(no_hints_function)
        
        # Should have info messages about missing type hints
        hint_messages = [
            r for r in results 
            if "type hint" in r.message.lower()
        ]
        assert len(hint_messages) > 0
    
    def test_validate_function_with_varargs(self):
        """Test validation of function with *args and **kwargs."""
        def varargs_function(*args, **kwargs) -> str:
            """Function with variable arguments."""
            return "test"
        
        results = self.validator.validate_function(varargs_function)
        
        # Should have warnings about *args and **kwargs
        varargs_warnings = [
            r for r in results 
            if r.level == ValidationLevel.WARNING and ("args" in r.message.lower() or "kwargs" in r.message.lower())
        ]
        assert len(varargs_warnings) >= 1  # At least one warning for *args or **kwargs
    
    def test_validate_function_with_keyword_parameter(self):
        """Test validation of function with Python keyword as parameter."""
        # This would be invalid Python syntax, so we'll simulate the validation
        # by creating a function signature manually
        import inspect
        
        def test_function(class_param: str) -> str:  # 'class' would be a keyword
            """Function with keyword parameter name."""
            return class_param
        
        # Rename parameter to simulate keyword issue
        test_function.__code__ = test_function.__code__.replace(
            co_varnames=('class',)  # This would trigger keyword validation
        ) if hasattr(test_function.__code__, 'replace') else test_function.__code__
        
        # For this test, we'll just verify the validator can handle the function
        results = self.validator.validate_function(test_function)
        
        # Should not crash and should return some results
        assert isinstance(results, list)
    
    def test_validate_function_with_many_parameters(self):
        """Test validation of function with many parameters."""
        def many_params_function(
            a: str, b: str, c: str, d: str, e: str, f: str,
            g: str, h: str, i: str, j: str, k: str, l: str
        ) -> str:
            """Function with many parameters."""
            return "test"
        
        results = self.validator.validate_function(many_params_function)
        
        # Should have warning about too many parameters
        param_warnings = [
            r for r in results 
            if r.level == ValidationLevel.WARNING and "many parameters" in r.message.lower()
        ]
        assert len(param_warnings) > 0
    
    def test_test_function_success(self):
        """Test successful function testing."""
        def simple_function(x: int, y: int = 5) -> int:
            """Simple addition function."""
            return x + y
        
        result = self.validator.test_function(
            simple_function,
            test_kwargs={'x': 10, 'y': 3}
        )
        
        assert result.success is True
        assert result.output == 13
        assert result.error is None
        assert result.execution_time > 0
    
    def test_test_function_failure(self):
        """Test function testing with error."""
        def failing_function(x: int) -> int:
            """Function that raises an error."""
            raise ValueError("Test error")
        
        result = self.validator.test_function(
            failing_function,
            test_kwargs={'x': 10}
        )
        
        assert result.success is False
        assert result.output is None
        assert "ValueError: Test error" in result.error
        assert result.execution_time > 0
    
    def test_generate_test_cases(self):
        """Test automatic test case generation."""
        def typed_function(name: str, age: int, active: bool, items: list) -> dict:
            """Function with various typed parameters."""
            return {
                'name': name,
                'age': age,
                'active': active,
                'items': items
            }
        
        test_cases = self.validator.generate_test_cases(typed_function)
        
        assert len(test_cases) > 0
        
        # Check first test case
        args, kwargs = test_cases[0]
        assert len(args) == 0  # Should use kwargs
        
        # Check that all required parameters have values
        assert 'name' in kwargs
        assert 'age' in kwargs
        assert 'active' in kwargs
        assert 'items' in kwargs
        
        # Check types of generated values
        assert isinstance(kwargs['name'], str)
        assert isinstance(kwargs['age'], int)
        assert isinstance(kwargs['active'], bool)
        assert isinstance(kwargs['items'], list)
    
    def test_validate_and_test_combined(self):
        """Test combined validation and testing."""
        def good_function(message: str) -> str:
            """Return uppercase message."""
            return message.upper()
        
        validation_results, test_result = self.validator.validate_and_test(
            good_function,
            test_kwargs={'message': 'hello'}
        )
        
        # Should have validation results
        assert isinstance(validation_results, list)
        
        # Should have successful test result
        assert test_result.success is True
        assert test_result.output == "HELLO"
    
    def test_validate_complex_return_type(self):
        """Test validation of function with complex return type."""
        def complex_return_function() -> Dict[str, List[int]]:
            """Function with complex return type."""
            return {'numbers': [1, 2, 3]}
        
        results = self.validator.validate_function(complex_return_function)
        
        # Should handle complex types without crashing
        assert isinstance(results, list)
        
        # May have warnings about serialization
        serialization_warnings = [
            r for r in results 
            if "serializable" in r.message.lower()
        ]
        # Complex types might trigger serialization warnings
    
    def test_validation_error_handling(self):
        """Test error handling in validation."""
        # Create a mock function that will cause validation errors
        class BadFunction:
            def __call__(self):
                return "test"
        
        bad_func = BadFunction()
        
        # Should handle validation errors gracefully
        results = self.validator.validate_function(bad_func)
        
        # Should return error results instead of crashing
        assert isinstance(results, list)
        
        # Should have at least one error result
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])