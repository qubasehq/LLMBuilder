"""
Base interface for all cleaning modules in the Advanced Cybersecurity Dataset Cleaning system.

This module defines the abstract base class that all cleaning modules must implement,
ensuring consistent interfaces and behavior across the cleaning pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from .data_models import CleaningResult, CleaningOperation, CleaningOperationType
import time
from loguru import logger


class CleaningModule(ABC):
    """
    Abstract base class for all text cleaning modules.
    
    All cleaning modules must inherit from this class and implement the required methods.
    This ensures consistent behavior and interfaces across the cleaning pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cleaning module.
        
        Args:
            config: Configuration dictionary for the module
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.module_name = self.__class__.__name__
        self._initialize_module()
    
    @abstractmethod
    def _initialize_module(self) -> None:
        """
        Initialize module-specific components.
        
        This method should be implemented by each module to set up any
        required resources, compile patterns, load models, etc.
        """
        pass
    
    @abstractmethod
    def clean_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, CleaningOperation]:
        """
        Clean the provided text according to the module's functionality.
        
        Args:
            text: The text to be cleaned
            metadata: Optional metadata about the text (file path, source, etc.)
        
        Returns:
            Tuple of (cleaned_text, cleaning_operation)
        """
        pass
    
    @abstractmethod
    def get_operation_type(self) -> CleaningOperationType:
        """
        Get the type of cleaning operation this module performs.
        
        Returns:
            The CleaningOperationType enum value for this module
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Check if the module is enabled.
        
        Returns:
            True if the module is enabled, False otherwise
        """
        return self.enabled
    
    def enable(self) -> None:
        """Enable the module."""
        self.enabled = True
        logger.info(f"Enabled cleaning module: {self.module_name}")
    
    def disable(self) -> None:
        """Disable the module."""
        self.enabled = False
        logger.info(f"Disabled cleaning module: {self.module_name}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration for the module.
        
        Returns:
            Dictionary containing the module's configuration
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the module's configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.enabled = self.config.get('enabled', self.enabled)
        logger.info(f"Updated configuration for {self.module_name}")
        
        # Re-initialize with new config
        try:
            self._initialize_module()
        except Exception as e:
            logger.error(f"Failed to re-initialize {self.module_name} with new config: {e}")
            raise
    
    def validate_config(self) -> List[str]:
        """
        Validate the module's configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Basic validation - subclasses can override for specific validation
        if not isinstance(self.config, dict):
            errors.append(f"{self.module_name}: config must be a dictionary")
        
        if 'enabled' in self.config and not isinstance(self.config['enabled'], bool):
            errors.append(f"{self.module_name}: 'enabled' must be a boolean")
        
        return errors
    
    def _create_operation(self, 
                         description: str,
                         original_length: int,
                         final_length: int,
                         items_removed: int = 0,
                         items_modified: int = 0,
                         processing_time: float = 0.0,
                         success: bool = True,
                         error_message: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> CleaningOperation:
        """
        Create a CleaningOperation instance for this module.
        
        Args:
            description: Description of what was done
            original_length: Length of text before cleaning
            final_length: Length of text after cleaning
            items_removed: Number of items removed
            items_modified: Number of items modified
            processing_time: Time taken for processing
            success: Whether the operation was successful
            error_message: Error message if operation failed
            metadata: Additional metadata about the operation
        
        Returns:
            CleaningOperation instance
        """
        return CleaningOperation(
            operation_type=self.get_operation_type(),
            module_name=self.module_name,
            description=description,
            original_length=original_length,
            final_length=final_length,
            items_removed=items_removed,
            items_modified=items_modified,
            processing_time=processing_time,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
    
    def process_with_timing(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, CleaningOperation]:
        """
        Process text with automatic timing measurement.
        
        Args:
            text: The text to be cleaned
            metadata: Optional metadata about the text
        
        Returns:
            Tuple of (cleaned_text, cleaning_operation)
        """
        if not self.is_enabled():
            # Return original text with no-op operation
            operation = self._create_operation(
                description=f"{self.module_name} is disabled",
                original_length=len(text),
                final_length=len(text),
                success=True,
                metadata={"disabled": True}
            )
            return text, operation
        
        start_time = time.time()
        
        try:
            cleaned_text, operation = self.clean_text(text, metadata)
            processing_time = time.time() - start_time
            
            # Update the operation with actual processing time
            operation.processing_time = processing_time
            
            logger.debug(f"{self.module_name} processed {len(text)} -> {len(cleaned_text)} chars in {processing_time:.3f}s")
            
            return cleaned_text, operation
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = f"Error in {self.module_name}: {str(e)}"
            
            logger.error(error_message)
            
            # Create error operation
            operation = self._create_operation(
                description=f"Failed to process text: {str(e)}",
                original_length=len(text),
                final_length=len(text),
                processing_time=processing_time,
                success=False,
                error_message=error_message
            )
            
            # Return original text on error
            return text, operation
    
    def get_module_info(self) -> Dict[str, Any]:
        """
        Get information about the module.
        
        Returns:
            Dictionary containing module information
        """
        return {
            "name": self.module_name,
            "enabled": self.enabled,
            "operation_type": self.get_operation_type().value,
            "config": self.get_config(),
            "description": self.__doc__ or "No description available"
        }
    
    def __str__(self) -> str:
        """String representation of the module."""
        status = "enabled" if self.enabled else "disabled"
        return f"{self.module_name} ({status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the module."""
        return f"{self.__class__.__name__}(enabled={self.enabled}, config={self.config})"