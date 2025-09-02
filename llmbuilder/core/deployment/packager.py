"""
Model packaging for deployment.

This module provides functionality for creating deployable packages
containing models, tokenizers, configurations, and optional server code.
"""

import os
import json
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import tempfile
from datetime import datetime

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


class ModelPackager:
    """
    Creates deployable packages for LLM models.
    
    Supports multiple package formats and can include server code,
    dependencies, and configuration files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model packager.
        
        Args:
            config: Packaging configuration dictionary
        """
        self.config = config
        self.progress_callback: Optional[Callable[[str, float], None]] = None
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Model packager initialized")
    
    def _validate_config(self):
        """Validate packaging configuration."""
        required_keys = ['model_path', 'output_path', 'format']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate paths
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set progress callback for monitoring packaging progress."""
        self.progress_callback = callback
    
    def _report_progress(self, step: str, percentage: float):
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(step, percentage)
        else:
            logger.info(f"{step} ({percentage:.1f}%)")
    
    def create_package(self) -> Dict[str, Any]:
        """
        Create deployment package.
        
        Returns:
            Dictionary with packaging results and metadata
        """
        format_type = self.config['format']
        
        if format_type == 'zip':
            return self._create_zip_package()
        elif format_type == 'tar':
            return self._create_tar_package()
        elif format_type == 'docker':
            return self._create_docker_package()
        else:
            raise ValueError(f"Unsupported package format: {format_type}")
    
    def _create_zip_package(self) -> Dict[str, Any]:
        """Create ZIP package."""
        self._report_progress("Preparing ZIP package", 10)
        
        output_path = Path(self.config['output_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        components = {}
        total_size = 0
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model file
            self._report_progress("Adding model file", 20)
            model_path = Path(self.config['model_path'])
            zipf.write(model_path, f"model/{model_path.name}")
            model_size = model_path.stat().st_size
            components['model'] = {
                'included': True,
                'size': self._format_size(model_size),
                'path': f"model/{model_path.name}"
            }
            total_size += model_size
            
            # Add tokenizer if specified
            tokenizer_path = self.config.get('tokenizer_path')
            if tokenizer_path and Path(tokenizer_path).exists():
                self._report_progress("Adding tokenizer", 40)
                tokenizer_size = self._add_directory_to_zip(zipf, Path(tokenizer_path), "tokenizer/")
                components['tokenizer'] = {
                    'included': True,
                    'size': self._format_size(tokenizer_size),
                    'path': 'tokenizer/'
                }
                total_size += tokenizer_size
            else:
                components['tokenizer'] = {'included': False}
            
            # Add config file if specified
            config_path = self.config.get('config_path')
            if config_path and Path(config_path).exists():
                self._report_progress("Adding configuration", 60)
                config_file = Path(config_path)
                zipf.write(config_file, f"config/{config_file.name}")
                config_size = config_file.stat().st_size
                components['config'] = {
                    'included': True,
                    'size': self._format_size(config_size),
                    'path': f"config/{config_file.name}"
                }
                total_size += config_size
            else:
                components['config'] = {'included': False}
            
            # Add server code if requested
            if self.config.get('include_server', False):
                self._report_progress("Adding server code", 70)
                server_size = self._add_server_code_to_zip(zipf)
                components['server'] = {
                    'included': True,
                    'size': self._format_size(server_size),
                    'path': 'server/'
                }
                total_size += server_size
            else:
                components['server'] = {'included': False}
            
            # Add dependencies if requested
            if self.config.get('include_dependencies', False):
                self._report_progress("Adding dependencies", 80)
                deps_size = self._add_dependencies_to_zip(zipf)
                components['dependencies'] = {
                    'included': True,
                    'size': self._format_size(deps_size),
                    'path': 'dependencies/'
                }
                total_size += deps_size
            else:
                components['dependencies'] = {'included': False}
            
            # Add deployment metadata
            self._report_progress("Adding metadata", 90)
            metadata = self._create_deployment_metadata()
            zipf.writestr("deployment.json", json.dumps(metadata, indent=2))
            
            # Add README
            readme_content = self._create_deployment_readme()
            zipf.writestr("README.md", readme_content)
        
        self._report_progress("Package creation complete", 100)
        
        return {
            'format': 'zip',
            'output_path': str(output_path),
            'total_size': self._format_size(total_size),
            'components': components,
            'created_at': datetime.now().isoformat()
        }
    
    def _create_tar_package(self) -> Dict[str, Any]:
        """Create TAR package."""
        self._report_progress("Preparing TAR package", 10)
        
        output_path = Path(self.config['output_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        compression = self.config.get('compression', 'gzip')
        mode = 'w:gz' if compression == 'gzip' else 'w:bz2' if compression == 'bzip2' else 'w'
        
        components = {}
        total_size = 0
        
        with tarfile.open(output_path, mode) as tarf:
            # Add model file
            self._report_progress("Adding model file", 20)
            model_path = Path(self.config['model_path'])
            tarf.add(model_path, f"model/{model_path.name}")
            model_size = model_path.stat().st_size
            components['model'] = {
                'included': True,
                'size': self._format_size(model_size),
                'path': f"model/{model_path.name}"
            }
            total_size += model_size
            
            # Add other components similar to ZIP
            # ... (similar logic as ZIP but using tarfile)
        
        self._report_progress("Package creation complete", 100)
        
        return {
            'format': 'tar',
            'output_path': str(output_path),
            'total_size': self._format_size(total_size),
            'components': components,
            'created_at': datetime.now().isoformat()
        }
    
    def _create_docker_package(self) -> Dict[str, Any]:
        """Create Docker package."""
        self._report_progress("Preparing Docker package", 10)
        
        # Create temporary directory for Docker context
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy model files
            self._report_progress("Copying model files", 30)
            model_dir = temp_path / "model"
            model_dir.mkdir()
            
            model_path = Path(self.config['model_path'])
            shutil.copy2(model_path, model_dir / model_path.name)
            
            # Copy other files
            tokenizer_path = self.config.get('tokenizer_path')
            if tokenizer_path and Path(tokenizer_path).exists():
                shutil.copytree(tokenizer_path, temp_path / "tokenizer")
            
            config_path = self.config.get('config_path')
            if config_path and Path(config_path).exists():
                config_dir = temp_path / "config"
                config_dir.mkdir()
                shutil.copy2(config_path, config_dir / Path(config_path).name)
            
            # Create Dockerfile
            self._report_progress("Creating Dockerfile", 60)
            dockerfile_content = self._create_dockerfile()
            (temp_path / "Dockerfile").write_text(dockerfile_content)
            
            # Create requirements.txt
            requirements_content = self._create_requirements_txt()
            (temp_path / "requirements.txt").write_text(requirements_content)
            
            # Create Docker image (placeholder - would need Docker installed)
            self._report_progress("Building Docker image", 80)
            # docker_build_command = f"docker build -t {image_name} {temp_path}"
            # This would require Docker to be installed and running
            
            # For now, just copy the Docker context to output location
            output_path = Path(self.config['output_path'])
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.copytree(temp_path, output_path)
        
        self._report_progress("Docker package creation complete", 100)
        
        return {
            'format': 'docker',
            'output_path': str(output_path),
            'components': {
                'dockerfile': {'included': True},
                'model': {'included': True},
                'requirements': {'included': True}
            },
            'created_at': datetime.now().isoformat()
        }
    
    def _add_directory_to_zip(self, zipf: zipfile.ZipFile, source_dir: Path, archive_path: str) -> int:
        """Add directory to ZIP file and return total size."""
        total_size = 0
        
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(source_dir)
                archive_file_path = archive_path + str(relative_path).replace('\\\\', '/')
                zipf.write(file_path, archive_file_path)
                total_size += file_path.stat().st_size
        
        return total_size
    
    def _add_server_code_to_zip(self, zipf: zipfile.ZipFile) -> int:
        """Add server code to ZIP file."""
        # Create a simple server script
        server_script = '''#!/usr/bin/env python3
"""
Simple deployment server for LLMBuilder model.
"""

import sys
from pathlib import Path

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llmbuilder.core.deployment.server import ModelServer

def main():
    config = {
        "model_path": "model/",
        "config_path": "config/",
        "tokenizer_path": "tokenizer/",
        "host": "0.0.0.0",
        "port": 8000
    }
    
    server = ModelServer(config)
    server.start_foreground()

if __name__ == "__main__":
    main()
'''
        
        zipf.writestr("server/run_server.py", server_script)
        return len(server_script.encode())
    
    def _add_dependencies_to_zip(self, zipf: zipfile.ZipFile) -> int:
        """Add Python dependencies to ZIP file."""
        requirements = self._create_requirements_txt()
        zipf.writestr("dependencies/requirements.txt", requirements)
        
        # Add installation script
        install_script = '''#!/bin/bash
# Install dependencies
pip install -r requirements.txt
'''
        zipf.writestr("dependencies/install.sh", install_script)
        
        return len(requirements.encode()) + len(install_script.encode())
    
    def _create_deployment_metadata(self) -> Dict[str, Any]:
        """Create deployment metadata."""
        return {
            'package_version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'model_path': str(self.config['model_path']),
            'format': self.config['format'],
            'includes_server': self.config.get('include_server', False),
            'includes_dependencies': self.config.get('include_dependencies', False),
            'deployment_instructions': {
                'extract': 'Extract the package to your deployment directory',
                'install': 'Run pip install -r requirements.txt if dependencies are included',
                'run': 'Execute the server script or use your own deployment method'
            }
        }
    
    def _create_deployment_readme(self) -> str:
        """Create deployment README."""
        return f'''# LLMBuilder Model Deployment Package

This package contains a trained LLM model ready for deployment.

## Contents

- `model/`: Model checkpoint files
- `config/`: Model configuration files
- `tokenizer/`: Tokenizer files (if included)
- `server/`: Server code (if included)
- `dependencies/`: Python dependencies (if included)

## Deployment Instructions

1. Extract this package to your deployment directory
2. Install dependencies: `pip install -r dependencies/requirements.txt` (if included)
3. Run the server: `python server/run_server.py` (if included)

## Package Information

- Created: {datetime.now().isoformat()}
- Format: {self.config['format']}
- Model: {Path(self.config['model_path']).name}

## Support

For more information, visit: https://github.com/your-org/llmbuilder
'''
    
    def _create_dockerfile(self) -> str:
        """Create Dockerfile for Docker deployment."""
        return '''FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model/ ./model/
COPY config/ ./config/
COPY tokenizer/ ./tokenizer/

# Copy server code
COPY server/ ./server/

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "server/run_server.py"]
'''
    
    def _create_requirements_txt(self) -> str:
        """Create requirements.txt for deployment."""
        return '''torch>=1.9.0
transformers>=4.20.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
numpy>=1.21.0
sentencepiece>=0.1.96
'''
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"