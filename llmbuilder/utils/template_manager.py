"""
Template management system for LLMBuilder projects.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from llmbuilder.utils.logging import get_logger
from llmbuilder.utils.config import ConfigManager

logger = get_logger(__name__)


class TemplateManager:
    """Manages project templates and customization."""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent.parent / "templates" / "projects"
        self.user_templates_dir = Path.home() / ".llmbuilder" / "templates"
        self.user_templates_dir.mkdir(parents=True, exist_ok=True)
        self.config_manager = ConfigManager()
    
    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """List all available templates."""
        templates = {}
        
        # Load built-in templates
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                templates[template_data["name"]] = {
                    "name": template_data["name"],
                    "description": template_data["description"],
                    "version": template_data["version"],
                    "type": "built-in",
                    "path": str(template_file)
                }
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")
        
        # Load user templates
        for template_file in self.user_templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                templates[template_data["name"]] = {
                    "name": template_data["name"],
                    "description": template_data["description"],
                    "version": template_data.get("version", "1.0.0"),
                    "type": "user",
                    "path": str(template_file)
                }
            except Exception as e:
                logger.warning(f"Failed to load user template {template_file}: {e}")
        
        return templates
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get template data by name."""
        templates = self.list_templates()
        
        if name not in templates:
            return None
        
        template_info = templates[name]
        
        try:
            with open(template_info["path"], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load template {name}: {e}")
            return None
    
    def create_project_from_template(
        self,
        project_path: Path,
        template_name: str,
        project_name: str,
        description: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create project from template."""
        template = self.get_template(template_name)
        
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        logger.info(f"Creating project from template: {template_name}")
        
        # Create directories
        self._create_directories(project_path, template.get("directories", []))
        
        # Create files
        self._create_files(project_path, template.get("files", {}))
        
        # Create configuration
        self._create_configuration(
            project_path, 
            template, 
            project_name, 
            description,
            custom_config
        )
        
        # Create sample data if enabled
        if template.get("sample_data", {}).get("enabled", False):
            self._create_sample_data(project_path, template["sample_data"])
        
        logger.info(f"Project created successfully from template: {template_name}")
    
    def create_custom_template(
        self,
        name: str,
        description: str,
        base_template: str = "default",
        customizations: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Create a custom template based on existing template."""
        base = self.get_template(base_template)
        
        if not base:
            raise ValueError(f"Base template '{base_template}' not found")
        
        # Create custom template
        custom_template = base.copy()
        custom_template["name"] = name
        custom_template["description"] = description
        custom_template["version"] = "1.0.0"
        custom_template["created_at"] = datetime.now().isoformat()
        custom_template["base_template"] = base_template
        
        # Apply customizations
        if customizations:
            custom_template.update(customizations)
        
        # Save custom template
        template_file = self.user_templates_dir / f"{name}.json"
        
        with open(template_file, 'w') as f:
            json.dump(custom_template, f, indent=2)
        
        logger.info(f"Custom template created: {name}")
        return template_file
    
    def validate_template(self, template_data: Dict[str, Any]) -> List[str]:
        """Validate template structure."""
        errors = []
        
        # Required fields
        required_fields = ["name", "description", "version"]
        for field in required_fields:
            if field not in template_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate directories
        if "directories" in template_data:
            if not isinstance(template_data["directories"], list):
                errors.append("'directories' must be a list")
        
        # Validate files
        if "files" in template_data:
            if not isinstance(template_data["files"], dict):
                errors.append("'files' must be a dictionary")
        
        # Validate config overrides
        if "config_overrides" in template_data:
            if not isinstance(template_data["config_overrides"], dict):
                errors.append("'config_overrides' must be a dictionary")
        
        return errors
    
    def _create_directories(self, project_path: Path, directories: List[str]) -> None:
        """Create project directories."""
        for dir_path in directories:
            full_path = project_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            if not any(full_path.iterdir()):
                (full_path / ".gitkeep").touch()
    
    def _create_files(self, project_path: Path, files: Dict[str, str]) -> None:
        """Create project files from template."""
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle executable files
            if file_path.endswith(('.py', '.sh')):
                full_path.write_text(content)
                full_path.chmod(0o755)  # Make executable
            else:
                full_path.write_text(content)
    
    def _create_configuration(
        self,
        project_path: Path,
        template: Dict[str, Any],
        project_name: str,
        description: Optional[str],
        custom_config: Optional[Dict[str, Any]]
    ) -> None:
        """Create project configuration."""
        # Start with default config
        config = self.config_manager.get_default_config()
        
        # Apply template overrides
        if "config_overrides" in template:
            config = self._deep_merge(config, template["config_overrides"])
        
        # Apply custom config
        if custom_config:
            config = self._deep_merge(config, custom_config)
        
        # Set project metadata
        config["project"] = {
            "name": project_name,
            "version": "1.0.0",
            "description": description or f"LLM project: {project_name}",
            "template": template["name"],
            "created_at": datetime.now().isoformat()
        }
        
        # Save configuration
        config_path = project_path / "llmbuilder.json"
        self.config_manager.save_config(config, config_path)
    
    def _create_sample_data(self, project_path: Path, sample_data: Dict[str, Any]) -> None:
        """Create sample data files."""
        data_dir = project_path / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for file_info in sample_data.get("files", []):
            file_path = data_dir / file_info["name"]
            file_path.write_text(file_info["content"])
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class ProjectValidator:
    """Validates project structure and health."""
    
    def __init__(self):
        self.template_manager = TemplateManager()
    
    def validate_project(self, project_path: Path) -> Dict[str, Any]:
        """Validate project structure and configuration."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check if it's a valid LLMBuilder project
        config_file = project_path / "llmbuilder.json"
        if not config_file.exists():
            results["errors"].append("Missing llmbuilder.json configuration file")
            results["valid"] = False
            return results
        
        # Load and validate configuration
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            results["errors"].append(f"Invalid configuration file: {e}")
            results["valid"] = False
            return results
        
        # Check required directories
        required_dirs = [
            "data/raw",
            "exports/checkpoints",
            "logs"
        ]
        
        for dir_path in required_dirs:
            full_path = project_path / dir_path
            if not full_path.exists():
                results["warnings"].append(f"Missing directory: {dir_path}")
                results["suggestions"].append(f"Create directory: mkdir -p {dir_path}")
        
        # Check data availability
        data_dir = project_path / "data" / "raw"
        if data_dir.exists():
            data_files = list(data_dir.glob("*"))
            if not data_files:
                results["warnings"].append("No training data found in data/raw/")
                results["suggestions"].append("Add training documents to data/raw/ directory")
        
        # Check for common issues
        self._check_common_issues(project_path, results)
        
        return results
    
    def health_check(self, project_path: Path) -> Dict[str, Any]:
        """Perform comprehensive project health check."""
        health = {
            "overall_health": "good",
            "checks": {},
            "recommendations": []
        }
        
        # Configuration check
        health["checks"]["configuration"] = self._check_configuration(project_path)
        
        # Data check
        health["checks"]["data"] = self._check_data(project_path)
        
        # Dependencies check
        health["checks"]["dependencies"] = self._check_dependencies(project_path)
        
        # Disk space check
        health["checks"]["disk_space"] = self._check_disk_space(project_path)
        
        # Determine overall health
        failed_checks = [name for name, check in health["checks"].items() if not check["status"]]
        
        if failed_checks:
            if len(failed_checks) > 2:
                health["overall_health"] = "poor"
            else:
                health["overall_health"] = "fair"
        
        return health
    
    def _check_common_issues(self, project_path: Path, results: Dict[str, Any]) -> None:
        """Check for common project issues."""
        # Check for large files in git
        gitignore_path = project_path / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            
            if "*.pt" not in gitignore_content and "*.pth" not in gitignore_content:
                results["suggestions"].append("Add model files (*.pt, *.pth) to .gitignore")
        
        # Check for README
        readme_path = project_path / "README.md"
        if not readme_path.exists():
            results["warnings"].append("Missing README.md file")
            results["suggestions"].append("Create README.md with project documentation")
    
    def _check_configuration(self, project_path: Path) -> Dict[str, Any]:
        """Check configuration validity."""
        try:
            config_file = project_path / "llmbuilder.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            return {
                "status": True,
                "message": "Configuration is valid",
                "details": f"Project: {config.get('project', {}).get('name', 'Unknown')}"
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"Configuration error: {e}",
                "details": "Check llmbuilder.json syntax and structure"
            }
    
    def _check_data(self, project_path: Path) -> Dict[str, Any]:
        """Check data availability and quality."""
        data_dir = project_path / "data" / "raw"
        
        if not data_dir.exists():
            return {
                "status": False,
                "message": "No data directory found",
                "details": "Create data/raw/ directory and add training documents"
            }
        
        data_files = list(data_dir.glob("*"))
        if not data_files:
            return {
                "status": False,
                "message": "No training data found",
                "details": "Add documents to data/raw/ directory"
            }
        
        total_size = sum(f.stat().st_size for f in data_files if f.is_file())
        
        return {
            "status": True,
            "message": f"Found {len(data_files)} data files",
            "details": f"Total size: {total_size / (1024*1024):.1f} MB"
        }
    
    def _check_dependencies(self, project_path: Path) -> Dict[str, Any]:
        """Check if dependencies are installed."""
        requirements_file = project_path / "requirements.txt"
        
        if not requirements_file.exists():
            return {
                "status": False,
                "message": "No requirements.txt found",
                "details": "Create requirements.txt with project dependencies"
            }
        
        return {
            "status": True,
            "message": "Requirements file found",
            "details": "Run 'pip install -r requirements.txt' to install dependencies"
        }
    
    def _check_disk_space(self, project_path: Path) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(project_path)
            
            free_gb = free / (1024**3)
            
            if free_gb < 1:
                status = False
                message = f"Low disk space: {free_gb:.1f} GB free"
            elif free_gb < 5:
                status = True
                message = f"Limited disk space: {free_gb:.1f} GB free"
            else:
                status = True
                message = f"Sufficient disk space: {free_gb:.1f} GB free"
            
            return {
                "status": status,
                "message": message,
                "details": f"Total: {total/(1024**3):.1f} GB, Used: {used/(1024**3):.1f} GB"
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"Cannot check disk space: {e}",
                "details": "Manual disk space check recommended"
            }


# Global instances
_template_manager = None
_project_validator = None


def get_template_manager() -> TemplateManager:
    """Get the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager


def get_project_validator() -> ProjectValidator:
    """Get the global project validator instance."""
    global _project_validator
    if _project_validator is None:
        _project_validator = ProjectValidator()
    return _project_validator