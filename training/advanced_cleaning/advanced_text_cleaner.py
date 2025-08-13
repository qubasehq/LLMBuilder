"""
Main orchestrator for the Advanced Cybersecurity Dataset Cleaning system.

This module provides the AdvancedTextCleaner class that coordinates all cleaning
modules and manages the overall cleaning pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import time
from loguru import logger

from .data_models import CleaningResult, CleaningStats, CleaningOperation
from .base_module import CleaningModule
from .config_manager import ConfigManager, AdvancedCleaningConfig


class AdvancedTextCleaner:
    """
    Main orchestrator for advanced text cleaning operations.
    
    This class coordinates all cleaning modules and manages the overall
    cleaning pipeline according to the configured processing order.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], AdvancedCleaningConfig, str, Path]] = None):
        """
        Initialize the advanced text cleaner.
        
        Args:
            config: Configuration as dict, AdvancedCleaningConfig, or path to config file
        """
        self.config_manager = ConfigManager()
        self.modules: Dict[str, CleaningModule] = {}
        self.processing_order: List[str] = []
        
        # Load configuration
        if isinstance(config, (str, Path)):
            self.config = self.config_manager.load_config(config)
        elif isinstance(config, AdvancedCleaningConfig):
            self.config = config
        elif isinstance(config, dict):
            # Parse dictionary config
            self.config = self.config_manager._parse_config(config)
        else:
            self.config = self.config_manager.load_config()
        
        # Initialize modules
        self._initialize_modules()
        
        logger.info(f"AdvancedTextCleaner initialized with {len(self.modules)} modules")
    
    def _initialize_modules(self) -> None:
        """Initialize all cleaning modules based on configuration."""
        # Note: Individual modules will be implemented in subsequent tasks
        # For now, we'll create placeholders that can be replaced later
        
        self.processing_order = self.config.processing_order.copy()
        
        # Initialize module registry (modules will be added as they're implemented)
        self.modules = {}
        
        logger.info(f"Module initialization complete. Processing order: {self.processing_order}")
    
    def register_module(self, name: str, module: CleaningModule) -> None:
        """
        Register a cleaning module.
        
        Args:
            name: Name of the module
            module: CleaningModule instance
        """
        self.modules[name] = module
        logger.info(f"Registered cleaning module: {name}")
    
    def unregister_module(self, name: str) -> None:
        """
        Unregister a cleaning module.
        
        Args:
            name: Name of the module to unregister
        """
        if name in self.modules:
            del self.modules[name]
            logger.info(f"Unregistered cleaning module: {name}")
        else:
            logger.warning(f"Module not found for unregistration: {name}")
    
    def get_available_modules(self) -> List[str]:
        """Get list of available module names."""
        return list(self.modules.keys())
    
    def get_enabled_modules(self) -> List[str]:
        """Get list of enabled module names."""
        return [name for name, module in self.modules.items() if module.is_enabled()]
    
    def enable_module(self, name: str) -> None:
        """
        Enable a specific module.
        
        Args:
            name: Name of the module to enable
        """
        if name in self.modules:
            self.modules[name].enable()
        else:
            logger.warning(f"Module not found: {name}")
    
    def disable_module(self, name: str) -> None:
        """
        Disable a specific module.
        
        Args:
            name: Name of the module to disable
        """
        if name in self.modules:
            self.modules[name].disable()
        else:
            logger.warning(f"Module not found: {name}")
    
    def clean_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> CleaningResult:
        """
        Clean text using all enabled modules in the configured order.
        
        Args:
            text: The text to be cleaned
            metadata: Optional metadata about the text
            
        Returns:
            CleaningResult containing the cleaned text and operation details
        """
        if not self.config.enabled:
            logger.info("Advanced cleaning is disabled, returning original text")
            return CleaningResult(
                original_text=text,
                cleaned_text=text,
                success=True,
                metadata={"advanced_cleaning_disabled": True}
            )
        
        start_time = time.time()
        current_text = text
        operations: List[CleaningOperation] = []
        warnings: List[str] = []
        
        logger.debug(f"Starting text cleaning pipeline with {len(text)} characters")
        
        # Process through each module in order (only if registered)
        for module_name in self.processing_order:
            if module_name not in self.modules:
                # Skip unregistered modules silently during normal operation
                logger.debug(f"Module '{module_name}' not registered, skipping")
                continue
            
            module = self.modules[module_name]
            
            if not module.is_enabled():
                logger.debug(f"Skipping disabled module: {module_name}")
                continue
            
            try:
                logger.debug(f"Processing with module: {module_name}")
                current_text, operation = module.process_with_timing(current_text, metadata)
                operations.append(operation)
                
                if not operation.success:
                    warning = f"Module '{module_name}' failed: {operation.error_message}"
                    warnings.append(warning)
                    logger.warning(warning)
                
            except Exception as e:
                error_msg = f"Unexpected error in module '{module_name}': {str(e)}"
                warnings.append(error_msg)
                logger.error(error_msg)
                
                # Create error operation
                error_operation = CleaningOperation(
                    operation_type=module.get_operation_type(),
                    module_name=module_name,
                    description=f"Failed with error: {str(e)}",
                    original_length=len(current_text),
                    final_length=len(current_text),
                    success=False,
                    error_message=error_msg
                )
                operations.append(error_operation)
        
        total_time = time.time() - start_time
        
        # Create comprehensive result
        result = CleaningResult(
            original_text=text,
            cleaned_text=current_text,
            cleaning_operations=operations,
            processing_time=total_time,
            success=len([op for op in operations if not op.success]) == 0,
            warnings=warnings,
            metadata=metadata or {}
        )
        
        logger.info(f"Text cleaning complete: {len(text)} -> {len(current_text)} chars in {total_time:.3f}s")
        
        return result
    
    def clean_file(self, file_path: Union[str, Path]) -> CleaningResult:
        """
        Clean text from a file.
        
        Args:
            file_path: Path to the file to clean
            
        Returns:
            CleaningResult containing the cleaned text and operation details
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return CleaningResult(
                original_text="",
                cleaned_text="",
                success=False,
                error_message=error_msg
            )
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Prepare metadata
            metadata = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix
            }
            
            # Clean the text
            result = self.clean_text(text, metadata)
            
            logger.info(f"Cleaned file: {file_path}")
            return result
            
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {str(e)}"
            logger.error(error_msg)
            return CleaningResult(
                original_text="",
                cleaned_text="",
                success=False,
                error_message=error_msg,
                metadata={"file_path": str(file_path)}
            )
    
    def clean_files(self, file_paths: List[Union[str, Path]]) -> List[CleaningResult]:
        """
        Clean multiple files.
        
        Args:
            file_paths: List of file paths to clean
            
        Returns:
            List of CleaningResult objects
        """
        results = []
        
        logger.info(f"Starting batch cleaning of {len(file_paths)} files")
        
        for file_path in file_paths:
            result = self.clean_file(file_path)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch cleaning complete: {successful}/{len(file_paths)} files processed successfully")
        
        return results
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the cleaning system.
        
        Returns:
            Dictionary containing system statistics
        """
        stats = {
            "system_info": {
                "enabled": self.config.enabled,
                "total_modules": len(self.modules),
                "enabled_modules": len(self.get_enabled_modules()),
                "processing_order": self.processing_order
            },
            "modules": {}
        }
        
        # Get module information
        for name, module in self.modules.items():
            stats["modules"][name] = module.get_module_info()
        
        return stats
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate processing order
        for module_name in self.processing_order:
            if module_name not in self.modules:
                errors.append(f"Module '{module_name}' in processing order is not registered")
        
        # Validate individual modules
        for name, module in self.modules.items():
            module_errors = module.validate_config()
            errors.extend([f"{name}: {error}" for error in module_errors])
        
        return errors
    
    def update_config(self, new_config: Union[Dict[str, Any], AdvancedCleaningConfig]) -> None:
        """
        Update the configuration and reinitialize modules.
        
        Args:
            new_config: New configuration
        """
        if isinstance(new_config, dict):
            self.config = self.config_manager._parse_config(new_config)
        else:
            self.config = new_config
        
        # Update module configurations
        for name, module in self.modules.items():
            module_config = getattr(self.config, name, None)
            if module_config:
                module.update_config(module_config.to_dict())
        
        # Update processing order
        self.processing_order = self.config.processing_order.copy()
        
        logger.info("Configuration updated successfully")
    
    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Module information dictionary or None if not found
        """
        if module_name in self.modules:
            return self.modules[module_name].get_module_info()
        return None
    
    def __str__(self) -> str:
        """String representation of the cleaner."""
        enabled_count = len(self.get_enabled_modules())
        return f"AdvancedTextCleaner({enabled_count}/{len(self.modules)} modules enabled)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the cleaner."""
        return f"AdvancedTextCleaner(modules={list(self.modules.keys())}, enabled={self.config.enabled})"