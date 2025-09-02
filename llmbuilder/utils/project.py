"""
Project management utilities for LLMBuilder.

This module provides functionality for creating, validating, and managing
LLM training projects with proper directory structure and configuration.
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from llmbuilder.utils.logging import get_logger
from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.template_manager import get_template_manager, get_project_validator

logger = get_logger(__name__)


class ProjectManager:
    """Manages LLM project creation and structure."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.template_manager = get_template_manager()
        self.validator = get_project_validator()
    
    def create_project(
        self, 
        name: str, 
        template: str = 'default',
        description: Optional[str] = None,
        force: bool = False,
        initialize_git: bool = True,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Create a new LLM project with proper structure.
        
        Args:
            name: Project name and directory name
            template: Project template to use
            description: Project description
            force: Overwrite existing directory
            initialize_git: Initialize git repository
            custom_config: Custom configuration overrides
            
        Returns:
            Path to created project directory
        """
        project_path = Path.cwd() / name
        
        # Check if directory exists
        if project_path.exists():
            if not force:
                raise ValueError(f"Directory '{name}' already exists. Use --force to overwrite.")
            shutil.rmtree(project_path)
        
        # Create project directory
        project_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating project directory: {project_path}")
        
        # Create project from template
        self.template_manager.create_project_from_template(
            project_path=project_path,
            template_name=template,
            project_name=name,
            description=description,
            custom_config=custom_config
        )
        
        # Create additional documentation
        self._create_documentation(project_path, name, description, template)
        
        # Initialize git repository
        if initialize_git:
            self._initialize_git(project_path)
        
        logger.info(f"Project '{name}' created successfully")
        return project_path
    
    def validate_project(self, path: Path) -> Dict[str, Any]:
        """
        Validate project structure and configuration.
        
        Args:
            path: Project directory path
            
        Returns:
            Validation results dictionary
        """
        return self.validator.validate_project(path)
    
    def health_check(self, path: Path) -> Dict[str, Any]:
        """
        Perform comprehensive project health check.
        
        Args:
            path: Project directory path
            
        Returns:
            Health check results dictionary
        """
        return self.validator.health_check(path)
    
    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """List all available project templates."""
        return self.template_manager.list_templates()
    
    def create_custom_template(
        self,
        name: str,
        description: str,
        base_template: str = "default",
        customizations: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Create a custom project template."""
        return self.template_manager.create_custom_template(
            name=name,
            description=description,
            base_template=base_template,
            customizations=customizations
        )
    

    
    def _create_documentation(self, project_path: Path, name: str, description: Optional[str], template: str) -> None:
        """Create project documentation."""
        template_info = self.template_manager.get_template(template)
        template_desc = template_info.get("description", "LLM training project") if template_info else "LLM training project"
        
        readme_content = f"""# {name}

{description or f'{template_desc} created with LLMBuilder'}

## Template: {template}

{template_desc}

## Quick Start

1. **Add training data**:
   ```bash
   # Copy your documents to data/raw/
   cp /path/to/your/documents/* data/raw/
   ```

2. **Prepare data**:
   ```bash
   llmbuilder data prepare
   ```

3. **Train model**:
   ```bash
   llmbuilder train start
   ```

## Project Structure

- `data/raw/` - Raw training documents
- `data/cleaned/` - Processed text files
- `data/tokens/` - Tokenized training data
- `exports/` - Model checkpoints and exports
- `logs/` - Training logs

## Configuration

Edit `llmbuilder.json` to customize training parameters.

## Validation

Check project health:
```bash
llmbuilder init validate
```

## Help

For more help:
```bash
llmbuilder help
llmbuilder help docs getting-started
```

Created with LLMBuilder v1.0.0 using template: {template}
"""
        
        readme_path = project_path / 'README.md'
        if not readme_path.exists():  # Don't overwrite if template already created it
            readme_path.write_text(readme_content)
    
    def _initialize_git(self, project_path: Path) -> None:
        """Initialize git repository."""
        try:
            subprocess.run(['git', 'init'], cwd=project_path, check=True, capture_output=True)
            logger.info("Git repository initialized")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Failed to initialize git repository")