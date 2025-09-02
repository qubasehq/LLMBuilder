"""
Prompt template management for LLMBuilder.

This module provides functionality for creating, managing, and applying
prompt templates with associated generation parameters.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


class PromptTemplateManager:
    """
    Manager for prompt templates and their associated parameters.
    
    Templates allow users to save prompt formats and generation settings
    for reuse across different inference sessions.
    """
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory to store templates (default: .llmbuilder/templates)
        """
        if templates_dir is None:
            templates_dir = Path.cwd() / '.llmbuilder' / 'templates'
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize with default templates
        self._create_default_templates()
        
        logger.debug(f"Template manager initialized with directory: {self.templates_dir}")
    
    def _create_default_templates(self):
        """Create default templates if they don't exist."""
        default_templates = {
            'creative': {
                'description': 'Creative writing with high temperature',
                'format': '{prompt}',
                'parameters': {
                    'temperature': 1.0,
                    'top_p': 0.9,
                    'top_k': 40,
                    'max_new_tokens': 200
                }
            },
            'technical': {
                'description': 'Technical explanations with low temperature',
                'format': 'Explain the following in technical terms: {prompt}',
                'parameters': {
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 20,
                    'max_new_tokens': 150
                }
            },
            'conversational': {
                'description': 'Natural conversation style',
                'format': '{prompt}',
                'parameters': {
                    'temperature': 0.8,
                    'top_p': 0.9,
                    'top_k': 50,
                    'max_new_tokens': 100
                }
            },
            'code': {
                'description': 'Code generation and explanation',
                'format': 'Write code for: {prompt}',
                'parameters': {
                    'temperature': 0.2,
                    'top_p': 0.7,
                    'top_k': 10,
                    'max_new_tokens': 300
                }
            },
            'story': {
                'description': 'Story and narrative generation',
                'format': 'Write a story about: {prompt}',
                'parameters': {
                    'temperature': 1.1,
                    'top_p': 0.95,
                    'top_k': 60,
                    'max_new_tokens': 500
                }
            }
        }
        
        for name, template in default_templates.items():
            template_path = self.templates_dir / f"{name}.json"
            if not template_path.exists():
                self.save_template(name, template)
    
    def save_template(self, name: str, template_config: Dict[str, Any]) -> None:
        """
        Save a prompt template.
        
        Args:
            name: Template name
            template_config: Template configuration dictionary
        """
        # Validate template configuration
        self._validate_template_config(template_config)
        
        # Add metadata
        template_config['created_at'] = datetime.now().isoformat()
        template_config['name'] = name
        
        # Save to file
        template_path = self.templates_dir / f"{name}.json"
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved template '{name}' to {template_path}")
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a prompt template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template configuration or None if not found
        """
        template_path = self.templates_dir / f"{name}.json"
        
        if not template_path.exists():
            logger.warning(f"Template '{name}' not found at {template_path}")
            return None
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = json.load(f)
            
            logger.debug(f"Loaded template '{name}'")
            return template
            
        except Exception as e:
            logger.error(f"Failed to load template '{name}': {e}")
            return None
    
    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            Dictionary mapping template names to their configurations
        """
        templates = {}
        
        for template_file in self.templates_dir.glob("*.json"):
            name = template_file.stem
            template = self.get_template(name)
            if template:
                templates[name] = template
        
        logger.debug(f"Found {len(templates)} templates")
        return templates
    
    def delete_template(self, name: str) -> bool:
        """
        Delete a prompt template.
        
        Args:
            name: Template name
            
        Returns:
            True if deleted successfully, False if not found
        """
        template_path = self.templates_dir / f"{name}.json"
        
        if not template_path.exists():
            logger.warning(f"Template '{name}' not found for deletion")
            return False
        
        try:
            template_path.unlink()
            logger.info(f"Deleted template '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete template '{name}': {e}")
            return False
    
    def apply_template(self, template_name: str, prompt: str) -> tuple[str, Dict[str, Any]]:
        """
        Apply a template to a prompt.
        
        Args:
            template_name: Name of the template to apply
            prompt: Original prompt text
            
        Returns:
            Tuple of (formatted_prompt, generation_parameters)
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Format the prompt
        format_string = template.get('format', '{prompt}')
        try:
            formatted_prompt = format_string.format(prompt=prompt)
        except KeyError as e:
            logger.warning(f"Template format error: {e}, using original prompt")
            formatted_prompt = prompt
        
        # Get generation parameters
        parameters = template.get('parameters', {})
        
        logger.debug(f"Applied template '{template_name}' to prompt")
        return formatted_prompt, parameters
    
    def _validate_template_config(self, config: Dict[str, Any]) -> None:
        """
        Validate template configuration.
        
        Args:
            config: Template configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        if 'description' not in config:
            logger.warning("Template missing description")
        
        # Validate format string if present
        if 'format' in config:
            format_string = config['format']
            if not isinstance(format_string, str):
                raise ValueError("Template format must be a string")
            
            # Check if format string is valid
            try:
                format_string.format(prompt="test")
            except KeyError as e:
                raise ValueError(f"Invalid format string: missing placeholder {e}")
            except Exception as e:
                raise ValueError(f"Invalid format string: {e}")
        
        # Validate parameters if present
        if 'parameters' in config:
            params = config['parameters']
            if not isinstance(params, dict):
                raise ValueError("Template parameters must be a dictionary")
            
            # Validate individual parameters
            for key, value in params.items():
                if key == 'temperature':
                    if not isinstance(value, (int, float)) or value < 0:
                        raise ValueError("temperature must be a non-negative number")
                elif key == 'top_k':
                    if not isinstance(value, int) or value < 0:
                        raise ValueError("top_k must be a non-negative integer")
                elif key == 'top_p':
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        raise ValueError("top_p must be between 0 and 1")
                elif key == 'max_new_tokens':
                    if not isinstance(value, int) or value < 1:
                        raise ValueError("max_new_tokens must be a positive integer")
    
    def export_templates(self, export_path: Path) -> None:
        """
        Export all templates to a single file.
        
        Args:
            export_path: Path to export file
        """
        templates = self.list_templates()
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'templates': templates
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(templates)} templates to {export_path}")
    
    def import_templates(self, import_path: Path, overwrite: bool = False) -> int:
        """
        Import templates from a file.
        
        Args:
            import_path: Path to import file
            overwrite: Whether to overwrite existing templates
            
        Returns:
            Number of templates imported
        """
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        templates = import_data.get('templates', {})
        imported_count = 0
        
        for name, template_config in templates.items():
            # Remove metadata that shouldn't be imported
            template_config.pop('created_at', None)
            template_config.pop('name', None)
            
            # Check if template already exists
            if not overwrite and self.get_template(name):
                logger.warning(f"Template '{name}' already exists, skipping")
                continue
            
            try:
                self.save_template(name, template_config)
                imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import template '{name}': {e}")
        
        logger.info(f"Imported {imported_count} templates from {import_path}")
        return imported_count
    
    def get_template_stats(self) -> Dict[str, Any]:
        """
        Get statistics about templates.
        
        Returns:
            Dictionary with template statistics
        """
        templates = self.list_templates()
        
        stats = {
            'total_templates': len(templates),
            'templates_by_type': {},
            'parameter_usage': {}
        }
        
        # Analyze templates
        for name, template in templates.items():
            # Count parameter usage
            params = template.get('parameters', {})
            for param_name in params.keys():
                stats['parameter_usage'][param_name] = stats['parameter_usage'].get(param_name, 0) + 1
        
        return stats