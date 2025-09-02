"""
Configuration management for LLMBuilder.

This module provides a hierarchical configuration system that supports:
- Command-line arguments (highest priority)
- Environment variables
- Project-specific config files
- User-level config files
- Package defaults (lowest priority)
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import shutil

try:
    from jsonschema import validate, ValidationError, Draft7Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Manages configuration loading, validation, and merging."""
    
    DEFAULT_CONFIG_NAMES = ["llmbuilder.json", "config.json"]
    USER_CONFIG_DIR = Path.home() / ".llmbuilder"
    ENV_PREFIX = "LLMBUILDER_"
    
    def __init__(self):
        self._schema = self._load_config_schema()
        self._cli_overrides = {}
    
    def load_config(self, path: Optional[Path] = None, cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration with hierarchical precedence.
        
        Precedence order (highest to lowest):
        1. CLI arguments/overrides
        2. Environment variables
        3. Project-specific config files
        4. User-level config files
        5. Package defaults
        
        Args:
            path: Specific config file path (optional)
            cli_overrides: CLI argument overrides (optional)
            
        Returns:
            Merged configuration dictionary
        """
        configs = []
        
        # 1. Load package defaults (lowest priority)
        default_config = self.get_default_config()
        configs.append(default_config)
        
        # 2. Load user-level config
        user_config = self._load_user_config()
        if user_config:
            configs.append(user_config)
        
        # 3. Load project-specific config
        if path:
            project_config = self._load_config_file(path)
        else:
            project_config = self._find_and_load_project_config()
        
        if project_config:
            configs.append(project_config)
        
        # 4. Load environment variables
        env_config = self._load_env_config()
        if env_config:
            configs.append(env_config)
        
        # 5. Apply CLI overrides (highest priority)
        if cli_overrides:
            self._cli_overrides = cli_overrides
            configs.append(cli_overrides)
        elif self._cli_overrides:
            configs.append(self._cli_overrides)
        
        # Merge all configurations
        merged_config = self.merge_configs(*configs)
        
        # Validate merged configuration
        validation_errors = self.validate_config(merged_config)
        if validation_errors:
            logger.warning(f"Configuration validation warnings: {validation_errors}")
        
        return merged_config
    
    def save_config(self, config: Dict[str, Any], path: Path) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        Later configs override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        result = {}
        
        for config in configs:
            if config:
                result = self._deep_merge(result, config)
        
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema and business rules.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # JSON Schema validation
        if JSONSCHEMA_AVAILABLE and self._schema:
            try:
                validate(instance=config, schema=self._schema)
            except ValidationError as e:
                errors.append(f"Schema validation: {e.message}")
            except Exception as e:
                errors.append(f"Schema validation error: {e}")
        
        # Business rule validation
        errors.extend(self._validate_business_rules(config))
        
        return errors
    
    def set_cli_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Set CLI argument overrides.
        
        Args:
            overrides: Dictionary of CLI overrides
        """
        self._cli_overrides = overrides
    
    def backup_config(self, config_path: Path) -> Path:
        """
        Create a backup of the configuration file.
        
        Args:
            config_path: Path to configuration file to backup
            
        Returns:
            Path to backup file
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f".backup_{timestamp}{config_path.suffix}")
        
        shutil.copy2(config_path, backup_path)
        logger.info(f"Configuration backed up to {backup_path}")
        
        return backup_path
    
    def migrate_config(self, config_path: Path, target_version: str = "latest") -> Dict[str, Any]:
        """
        Migrate configuration file to newer format.
        
        Args:
            config_path: Path to configuration file to migrate
            target_version: Target configuration version
            
        Returns:
            Migrated configuration dictionary
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load current config
        current_config = self._load_config_file(config_path)
        if not current_config:
            raise ValueError(f"Could not load configuration from {config_path}")
        
        # Determine current version
        current_version = current_config.get("_version", "1.0.0")
        
        # Create backup before migration
        backup_path = self.backup_config(config_path)
        
        try:
            # Apply migrations
            migrated_config = self._apply_migrations(current_config, current_version, target_version)
            
            # Save migrated config
            self.save_config(migrated_config, config_path)
            
            logger.info(f"Configuration migrated from {current_version} to {target_version}")
            logger.info(f"Backup saved to {backup_path}")
            
            return migrated_config
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            # Restore backup
            shutil.copy2(backup_path, config_path)
            logger.info(f"Configuration restored from backup")
            raise
    
    def get_config_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available configuration templates.
        
        Returns:
            Dictionary of template name to configuration
        """
        templates = {
            "default": self.get_default_config(),
            "research": self._get_research_template(),
            "production": self._get_production_template(),
            "minimal": self._get_minimal_template(),
            "gpu": self._get_gpu_template(),
            "cpu": self._get_cpu_template()
        }
        
        return templates
    
    def create_config_from_template(self, template_name: str, output_path: Path, 
                                  overrides: Optional[Dict[str, Any]] = None) -> None:
        """
        Create configuration file from template.
        
        Args:
            template_name: Name of template to use
            output_path: Path to save configuration file
            overrides: Optional overrides to apply to template
        """
        templates = self.get_config_templates()
        
        if template_name not in templates:
            available = ", ".join(templates.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
        
        config = templates[template_name].copy()
        
        # Apply overrides
        if overrides:
            config = self.merge_configs(config, overrides)
        
        # Add metadata
        config["_version"] = "2.0.0"
        config["_template"] = template_name
        config["_created"] = datetime.now().isoformat()
        
        # Save configuration
        self.save_config(config, output_path)
        logger.info(f"Configuration created from '{template_name}' template: {output_path}")
    
    def get_project_config(self) -> Dict[str, Any]:
        """
        Get project-specific configuration with fallbacks.
        
        Returns:
            Project configuration dictionary
        """
        return self.load_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "project": {
                "name": "llm-project",
                "version": "1.0.0",
                "description": "LLM training project"
            },
            "model": {
                "architecture": "gpt",
                "vocab_size": 16000,
                "embedding_dim": 384,
                "num_layers": 6,
                "num_heads": 6,
                "hidden_dim": 1536,
                "max_seq_length": 256,
                "dropout": 0.1,
                "use_bias": True,
                "tie_weights": True
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.0002,
                "weight_decay": 0.01,
                "num_epochs": 5,
                "warmup_steps": 1000,
                "gradient_clip_norm": 1.0,
                "save_every": 500,
                "num_workers": 4,
                "pin_memory": True,
                "prefetch_factor": 2,
                "use_mixed_precision": False,
                "log_every": 10,
                "eval_every": 500
            },
            "data": {
                "input_formats": ["txt", "pdf", "docx", "html", "epub", "md"],
                "max_file_size_mb": 100,
                "preprocessing": {
                    "max_samples": 1000,
                    "min_length": 10,
                    "max_length": 10000,
                    "remove_duplicates": True,
                    "normalize_whitespace": True,
                    "remove_empty_lines": True,
                    "encoding": "utf-8"
                },
                "deduplication": {
                    "enabled": True,
                    "similarity_threshold": 0.85
                }
            },
            "paths": {
                "data_dir": "data",
                "raw_data_dir": "data/raw",
                "cleaned_data_dir": "data/cleaned",
                "deduped_data_dir": "data/deduped",
                "tokenized_data_dir": "data/tokens",
                "finetune_data_dir": "data/finetune",
                "tokenizer_dir": "exports/tokenizer",
                "checkpoint_dir": "exports/checkpoints",
                "gguf_dir": "exports/gguf",
                "log_dir": "logs",
                "export_dir": "exports"
            },
            "logging": {
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
                "rotation": "10 MB",
                "retention": "7 days",
                "compression": "gz",
                "colorize": True
            },
            "device": {
                "use_cuda": True,
                "cuda_device": 0,
                "use_mps": False,
                "cpu_threads": 8,
                "enable_mkldnn": True,
                "mixed_precision": True,
                "fallback_to_cpu": True,
                "auto_detect_device": True
            },
            "deployment": {
                "api_framework": "fastapi",
                "host": "localhost",
                "port": 8000,
                "workers": 1,
                "timeout": 60,
                "cors_enabled": True,
                "cors_origins": ["*"],
                "max_request_size": "10MB",
                "rate_limit": "100/minute",
                "authentication": {
                    "enabled": False,
                    "api_key": None,
                    "jwt_secret": None
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_endpoint": "/metrics",
                    "health_endpoint": "/health"
                },
                "model_serving": {
                    "max_batch_size": 8,
                    "max_sequence_length": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.1
                }
            }
        }
    
    def _load_config_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration from a specific file."""
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return None
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {e}")
            return None
    
    def _load_user_config(self) -> Optional[Dict[str, Any]]:
        """Load user-level configuration."""
        for config_name in self.DEFAULT_CONFIG_NAMES:
            config_path = self.USER_CONFIG_DIR / config_name
            if config_path.exists():
                return self._load_config_file(config_path)
        return None
    
    def _find_and_load_project_config(self) -> Optional[Dict[str, Any]]:
        """Find and load project-specific configuration."""
        current_dir = Path.cwd()
        
        # Search up the directory tree for config files
        for parent in [current_dir] + list(current_dir.parents):
            for config_name in self.DEFAULT_CONFIG_NAMES:
                config_path = parent / config_name
                if config_path.exists():
                    return self._load_config_file(config_path)
        
        return None
    
    def _load_env_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with LLMBUILDER_ and use
        double underscores to separate nested keys.
        
        Examples:
            LLMBUILDER_MODEL__VOCAB_SIZE=32000
            LLMBUILDER_TRAINING__BATCH_SIZE=8
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                # Remove prefix and convert to nested dict
                config_key = key[len(self.ENV_PREFIX):].lower()
                keys = config_key.split('__')
                
                # Parse value
                parsed_value = self._parse_env_value(value)
                
                # Set nested value
                current = env_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = parsed_value
        
        return env_config
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _validate_business_rules(self, config: Dict[str, Any]) -> List[str]:
        """Validate business rules for configuration."""
        errors = []
        
        # Model validation
        model = config.get('model', {})
        if model.get('vocab_size', 0) <= 0:
            errors.append("model.vocab_size must be positive")
        
        if model.get('num_layers', 0) <= 0:
            errors.append("model.num_layers must be positive")
        
        if model.get('embedding_dim', 0) <= 0:
            errors.append("model.embedding_dim must be positive")
        
        # Training validation
        training = config.get('training', {})
        if training.get('batch_size', 0) <= 0:
            errors.append("training.batch_size must be positive")
        
        if training.get('learning_rate', 0) <= 0:
            errors.append("training.learning_rate must be positive")
        
        if training.get('num_epochs', 0) <= 0:
            errors.append("training.num_epochs must be positive")
        
        # Path validation
        paths = config.get('paths', {})
        for path_key, path_value in paths.items():
            if path_value and not isinstance(path_value, str):
                errors.append(f"paths.{path_key} must be a string")
        
        return errors
    
    def _apply_migrations(self, config: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Apply configuration migrations."""
        migrated = config.copy()
        
        # Normalize target version
        if to_version == "latest":
            to_version = "2.0.0"
        
        # Migration from 1.x to 2.x
        if from_version.startswith("1.") and (to_version.startswith("2.") or to_version == "latest"):
            # Add version info
            migrated["_version"] = "2.0.0"
            
            # Migrate old field names
            if "vocab_size" in migrated:
                if "model" not in migrated:
                    migrated["model"] = {}
                migrated["model"]["vocab_size"] = migrated.pop("vocab_size")
            
            if "batch_size" in migrated:
                if "training" not in migrated:
                    migrated["training"] = {}
                migrated["training"]["batch_size"] = migrated.pop("batch_size")
            
            if "learning_rate" in migrated:
                if "training" not in migrated:
                    migrated["training"] = {}
                migrated["training"]["learning_rate"] = migrated.pop("learning_rate")
            
            # Add missing default sections
            default_config = self.get_default_config()
            for section in ["model", "training", "data", "paths", "device", "logging"]:
                if section not in migrated:
                    migrated[section] = default_config[section]
                else:
                    # Merge with defaults to add missing keys
                    migrated[section] = self._deep_merge(default_config[section], migrated[section])
        
        return migrated
    
    def _get_research_template(self) -> Dict[str, Any]:
        """Get research configuration template."""
        config = self.get_default_config()
        
        # Optimize for research/experimentation
        config["model"].update({
            "num_layers": 4,
            "embedding_dim": 256,
            "hidden_dim": 1024,
            "max_seq_length": 128
        })
        
        config["training"].update({
            "batch_size": 2,
            "num_epochs": 3,
            "learning_rate": 0.001,
            "log_every": 5,
            "eval_every": 100
        })
        
        config["data"]["preprocessing"]["max_samples"] = 500
        
        return config
    
    def _get_production_template(self) -> Dict[str, Any]:
        """Get production configuration template."""
        config = self.get_default_config()
        
        # Optimize for production
        config["model"].update({
            "num_layers": 12,
            "embedding_dim": 768,
            "hidden_dim": 3072,
            "max_seq_length": 512
        })
        
        config["training"].update({
            "batch_size": 8,
            "num_epochs": 10,
            "learning_rate": 0.0001,
            "use_mixed_precision": True,
            "gradient_clip_norm": 1.0
        })
        
        config["device"].update({
            "use_cuda": True,
            "mixed_precision": True
        })
        
        return config
    
    def _get_minimal_template(self) -> Dict[str, Any]:
        """Get minimal configuration template."""
        return {
            "project": {
                "name": "minimal-project"
            },
            "model": {
                "architecture": "gpt",
                "vocab_size": 8000,
                "embedding_dim": 128,
                "num_layers": 2,
                "num_heads": 2,
                "max_seq_length": 64
            },
            "training": {
                "batch_size": 1,
                "learning_rate": 0.001,
                "num_epochs": 1
            },
            "paths": {
                "data_dir": "data",
                "checkpoint_dir": "checkpoints"
            }
        }
    
    def _get_gpu_template(self) -> Dict[str, Any]:
        """Get GPU-optimized configuration template."""
        config = self.get_default_config()
        
        config["device"].update({
            "use_cuda": True,
            "mixed_precision": True,
            "enable_mkldnn": False
        })
        
        config["training"].update({
            "batch_size": 16,
            "use_mixed_precision": True,
            "num_workers": 8
        })
        
        return config
    
    def _get_cpu_template(self) -> Dict[str, Any]:
        """Get CPU-optimized configuration template."""
        config = self.get_default_config()
        
        config["device"].update({
            "use_cuda": False,
            "use_mps": False,
            "cpu_threads": 8,
            "enable_mkldnn": True
        })
        
        config["training"].update({
            "batch_size": 2,
            "use_mixed_precision": False,
            "num_workers": 4
        })
        
        config["model"].update({
            "num_layers": 4,
            "embedding_dim": 256
        })
        
        return config
    
    def _load_config_schema(self) -> Optional[Dict[str, Any]]:
        """Load configuration schema for validation."""
        if not JSONSCHEMA_AVAILABLE:
            return None
        
        # Define JSON schema for configuration validation
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "project": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name"]
                },
                "model": {
                    "type": "object",
                    "properties": {
                        "architecture": {"type": "string"},
                        "vocab_size": {"type": "integer", "minimum": 1},
                        "embedding_dim": {"type": "integer", "minimum": 1},
                        "num_layers": {"type": "integer", "minimum": 1},
                        "num_heads": {"type": "integer", "minimum": 1},
                        "max_seq_length": {"type": "integer", "minimum": 1}
                    },
                    "required": ["architecture", "vocab_size"]
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "batch_size": {"type": "integer", "minimum": 1},
                        "learning_rate": {"type": "number", "minimum": 0},
                        "num_epochs": {"type": "integer", "minimum": 1}
                    },
                    "required": ["batch_size", "learning_rate"]
                }
            },
            "required": ["project", "model", "training"]
        }
        
        return schema
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result