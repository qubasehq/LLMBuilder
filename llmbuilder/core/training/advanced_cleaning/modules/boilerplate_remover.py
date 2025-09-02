"""
Boilerplate Removal Module for Advanced Cybersecurity Dataset Cleaning.

This module implements detection and removal of boilerplate content including
copyright notices, legal disclaimers, quiz patterns, and multiple choice sequences.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger

from ..base_module import CleaningModule
from ..data_models import CleaningOperation, CleaningOperationType


class BoilerplateRemover(CleaningModule):
    """
    Removes boilerplate content from text including copyright notices,
    legal disclaimers, quiz patterns, and multiple choice sequences.
    
    This module is specifically designed for cybersecurity datasets where
    educational materials often contain quiz questions and legal boilerplate
    that should be removed from training data.
    """
    
    def _initialize_module(self) -> None:
        """Initialize the boilerplate removal patterns and configuration."""
        # Default boilerplate patterns
        self.default_patterns = [
            r'(?i)(copyright|all rights reserved|quiz|correct answer|©)',
            r'(?i)\b[a-d]\.\s*[^.]{1,100}[.?!]\s*(?:\b[a-d]\.\s*[^.]{1,100}[.?!]\s*)+',  # Multiple choice
            r'(?i)©\s*\d{4}',  # Copyright with year
            r'(?i)\b(disclaimer|terms of use|privacy policy)\b',
            r'(?i)\b(all rights reserved|proprietary and confidential)\b',
            r'(?i)\b(question \d+|answer:|correct answer is)\b',
            r'(?i)\b(choose the correct|select all that apply|which of the following)\b'
        ]
        
        # Get custom patterns from configuration
        custom_patterns = self.config.get('custom_patterns', [])
        
        # Combine default and custom patterns
        all_patterns = self.default_patterns + custom_patterns
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = []
        for pattern in all_patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                self.compiled_patterns.append(compiled)
                logger.debug(f"Compiled boilerplate pattern: {pattern}")
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        # Configuration options
        self.remove_headers = self.config.get('remove_headers', True)
        self.remove_footers = self.config.get('remove_footers', True)
        self.min_line_length = self.config.get('min_line_length', 10)
        self.max_consecutive_choices = self.config.get('max_consecutive_choices', 2)
        
        # Multiple choice detection patterns
        self.choice_patterns = [
            re.compile(r'^\s*[a-d]\.\s*', re.IGNORECASE),  # a. b. c. d.
            re.compile(r'^\s*[a-d]\)\s*', re.IGNORECASE),  # a) b) c) d)
            re.compile(r'^\s*\([a-d]\)\s*', re.IGNORECASE),  # (a) (b) (c) (d)
            re.compile(r'^\s*[ivx]+\.\s*', re.IGNORECASE),  # i. ii. iii. iv.
            re.compile(r'^\s*\d+\.\s*', re.IGNORECASE)  # 1. 2. 3. 4.
        ]
        
        # Header/footer detection patterns
        self.header_footer_patterns = [
            re.compile(r'(?i)^(page \d+|chapter \d+|\d+\s*of\s*\d+)'),
            re.compile(r'(?i)(continued on next page|end of chapter|references?)$'),
            re.compile(r'^[-=_]{3,}$'),  # Separator lines
            re.compile(r'^\s*\d+\s*$'),  # Page numbers alone
        ]
        
        logger.info(f"BoilerplateRemover initialized with {len(self.compiled_patterns)} patterns")
    
    def get_operation_type(self) -> CleaningOperationType:
        """Return the operation type for this module."""
        return CleaningOperationType.BOILERPLATE_REMOVAL
    
    def clean_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, CleaningOperation]:
        """
        Remove boilerplate content from the provided text.
        
        Args:
            text: The text to clean
            metadata: Optional metadata about the text
            
        Returns:
            Tuple of (cleaned_text, cleaning_operation)
        """
        if not text or not text.strip():
            operation = self._create_operation(
                description="No text to process",
                original_length=len(text),
                final_length=len(text),
                success=True
            )
            return text, operation
        
        original_length = len(text)
        lines = text.split('\n')
        original_line_count = len(lines)
        
        # Track what was removed
        removed_lines = 0
        removed_patterns = []
        
        # Process lines for boilerplate removal
        cleaned_lines = []
        consecutive_choices = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines (preserve them for now)
            if not line_stripped:
                cleaned_lines.append(line)
                consecutive_choices = 0
                continue
            
            # Check if line is too short
            if len(line_stripped) < self.min_line_length:
                logger.debug(f"Removing short line: {line_stripped[:50]}...")
                removed_lines += 1
                consecutive_choices = 0
                continue
            
            # Check for header/footer patterns
            if self._is_header_footer(line_stripped, i, len(lines)):
                logger.debug(f"Removing header/footer: {line_stripped[:50]}...")
                removed_lines += 1
                consecutive_choices = 0
                continue
            
            # Check for boilerplate patterns
            if self._contains_boilerplate(line_stripped):
                pattern_found = self._get_matching_pattern(line_stripped)
                if pattern_found:
                    removed_patterns.append(pattern_found)
                logger.debug(f"Removing boilerplate: {line_stripped[:50]}...")
                removed_lines += 1
                consecutive_choices = 0
                continue
            
            # Check for multiple choice patterns
            if self._is_choice_line(line_stripped):
                consecutive_choices += 1
                if consecutive_choices >= self.max_consecutive_choices:
                    # Remove this line and mark previous choice lines for removal
                    logger.debug(f"Removing multiple choice sequence: {line_stripped[:50]}...")
                    removed_lines += 1
                    
                    # Remove previous consecutive choice lines
                    while (cleaned_lines and 
                           self._is_choice_line(cleaned_lines[-1].strip())):
                        removed_line = cleaned_lines.pop()
                        removed_lines += 1
                        logger.debug(f"Removing previous choice: {removed_line.strip()[:50]}...")
                    
                    continue
                else:
                    # Keep this choice line for now
                    cleaned_lines.append(line)
            else:
                consecutive_choices = 0
                cleaned_lines.append(line)
        
        # Join cleaned lines
        cleaned_text = '\n'.join(cleaned_lines)
        final_length = len(cleaned_text)
        
        # Create operation record
        description_parts = []
        if removed_lines > 0:
            description_parts.append(f"removed {removed_lines} boilerplate lines")
        if removed_patterns:
            unique_patterns = list(set(removed_patterns))
            description_parts.append(f"matched patterns: {', '.join(unique_patterns[:3])}")
        
        description = "; ".join(description_parts) if description_parts else "no boilerplate detected"
        
        operation = self._create_operation(
            description=f"Boilerplate removal: {description}",
            original_length=original_length,
            final_length=final_length,
            items_removed=removed_lines,
            success=True,
            metadata={
                "original_lines": original_line_count,
                "final_lines": len(cleaned_lines),
                "removed_patterns": removed_patterns,
                "patterns_used": len(self.compiled_patterns)
            }
        )
        
        return cleaned_text, operation
    
    def _is_header_footer(self, line: str, line_index: int, total_lines: int) -> bool:
        """
        Check if a line appears to be a header or footer.
        
        Args:
            line: The line to check
            line_index: Index of the line in the document
            total_lines: Total number of lines in the document
            
        Returns:
            True if the line appears to be a header or footer
        """
        if not (self.remove_headers or self.remove_footers):
            return False
        
        # Check if it's in header/footer position
        is_header_position = self.remove_headers and line_index < 3
        is_footer_position = self.remove_footers and line_index >= total_lines - 3
        
        if not (is_header_position or is_footer_position):
            return False
        
        # Check against header/footer patterns
        for pattern in self.header_footer_patterns:
            if pattern.search(line):
                return True
        
        return False
    
    def _contains_boilerplate(self, line: str) -> bool:
        """
        Check if a line contains boilerplate content.
        
        Args:
            line: The line to check
            
        Returns:
            True if the line contains boilerplate content
        """
        for pattern in self.compiled_patterns:
            if pattern.search(line):
                return True
        return False
    
    def _get_matching_pattern(self, line: str) -> Optional[str]:
        """
        Get the first matching boilerplate pattern for a line.
        
        Args:
            line: The line to check
            
        Returns:
            String representation of the matching pattern, or None
        """
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(line):
                # Return a simplified pattern description
                if i < len(self.default_patterns):
                    return f"default_pattern_{i}"
                else:
                    return f"custom_pattern_{i - len(self.default_patterns)}"
        return None
    
    def _is_choice_line(self, line: str) -> bool:
        """
        Check if a line appears to be a multiple choice option.
        
        Args:
            line: The line to check
            
        Returns:
            True if the line appears to be a multiple choice option
        """
        for pattern in self.choice_patterns:
            if pattern.match(line):
                return True
        return False
    
    def validate_config(self) -> List[str]:
        """
        Validate the module's configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = super().validate_config()
        
        # Validate custom patterns
        custom_patterns = self.config.get('custom_patterns', [])
        if not isinstance(custom_patterns, list):
            errors.append("custom_patterns must be a list")
        else:
            for i, pattern in enumerate(custom_patterns):
                if not isinstance(pattern, str):
                    errors.append(f"custom_patterns[{i}] must be a string")
                else:
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        errors.append(f"custom_patterns[{i}] is invalid regex: {e}")
        
        # Validate numeric parameters
        numeric_params = {
            'min_line_length': (1, 1000),
            'max_consecutive_choices': (1, 10)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            if param in self.config:
                value = self.config[param]
                if not isinstance(value, int) or not (min_val <= value <= max_val):
                    errors.append(f"{param} must be an integer between {min_val} and {max_val}")
        
        # Validate boolean parameters
        boolean_params = ['remove_headers', 'remove_footers']
        for param in boolean_params:
            if param in self.config and not isinstance(self.config[param], bool):
                errors.append(f"{param} must be a boolean")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the boilerplate removal configuration.
        
        Returns:
            Dictionary containing module statistics
        """
        return {
            "total_patterns": len(self.compiled_patterns),
            "default_patterns": len(self.default_patterns),
            "custom_patterns": len(self.config.get('custom_patterns', [])),
            "choice_patterns": len(self.choice_patterns),
            "header_footer_patterns": len(self.header_footer_patterns),
            "remove_headers": self.remove_headers,
            "remove_footers": self.remove_footers,
            "min_line_length": self.min_line_length,
            "max_consecutive_choices": self.max_consecutive_choices
        }