"""
Configuration management for the Advanced Cybersecurity Dataset Cleaning system.

This module handles loading, validation, and management of configuration settings
for all cleaning modules and the overall cleaning pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from loguru import logger
import jsonschema
from jsonschema import validate, ValidationError


@dataclass
class ModuleConfig:
    """Configuration for a single cleaning module."""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"enabled": self.enabled, **self.config}


@dataclass
class BoilerplateConfig(ModuleConfig):
    """Configuration for boilerplate removal module."""
    custom_patterns: List[str] = field(default_factory=lambda: [
        r"(copyright|all rights reserved|quiz|correct answer|©)",
        r"(\b[a-d]\.\s*){2,}"
    ])
    remove_headers: bool = True
    remove_footers: bool = True
    min_line_length: int = 10


@dataclass
class LanguageFilterConfig(ModuleConfig):
    """Configuration for language filtering module."""
    target_languages: List[str] = field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.8
    detection_method: str = "fasttext"
    fallback_enabled: bool = True


@dataclass
class DomainFilterConfig(ModuleConfig):
    """Configuration for domain filtering module."""
    method: str = "keyword"
    threshold: float = 0.85
    cybersecurity_keywords: List[str] = field(default_factory=lambda: [
        "firewall", "malware", "vulnerability", "encryption", "security",
        "attack", "threat", "exploit", "penetration", "authentication",
        "authorization", "cryptography", "phishing", "ransomware", "botnet"
    ])
    ml_model_path: Optional[str] = None


@dataclass
class QualityAssessmentConfig(ModuleConfig):
    """Configuration for quality assessment module."""
    grammar_tool: str = "languagetool"
    min_quality_score: float = 0.6
    check_completeness: bool = True
    min_sentence_length: int = 5
    max_typo_ratio: float = 0.1
    min_entropy: float = 3.0


@dataclass
class EntityPreservationConfig(ModuleConfig):
    """Configuration for entity preservation module."""
    entity_types: List[str] = field(default_factory=lambda: ["CVE", "PROTOCOL", "TOOL"])
    ner_model: str = "en_core_web_sm"
    confidence_threshold: float = 0.7
    custom_patterns: Dict[str, str] = field(default_factory=dict)


@dataclass
class RepetitionHandlingConfig(ModuleConfig):
    """Configuration for repetition handling module."""
    max_word_repetitions: int = 3
    ngram_frequency_cap: int = 5
    adaptive_penalty: bool = True
    similarity_threshold: float = 0.9
    min_ngram_length: int = 3


@dataclass
class AdvancedCleaningConfig:
    """Complete configuration for advanced cleaning system."""
    enabled: bool = True
    boilerplate_removal: BoilerplateConfig = field(default_factory=BoilerplateConfig)
    language_filtering: LanguageFilterConfig = field(default_factory=LanguageFilterConfig)
    domain_filtering: DomainFilterConfig = field(default_factory=DomainFilterConfig)
    quality_assessment: QualityAssessmentConfig = field(default_factory=QualityAssessmentConfig)
    entity_preservation: EntityPreservationConfig = field(default_factory=EntityPreservationConfig)
    repetition_handling: RepetitionHandlingConfig = field(default_factory=RepetitionHandlingConfig)
    processing_order: List[str] = field(default_factory=lambda: [
        "boilerplate_removal",
        "language_filtering",
        "domain_filtering", 
        "quality_assessment",
        "entity_preservation",
        "repetition_handling"
    ])
    parallel_processing: bool = False
    max_workers: int = 4
    batch_size: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def get_module_config(self, module_name: str) -> Optional[ModuleConfig]:
        """Get configuration for a specific module."""
        return getattr(self, module_name, None)


class ConfigManager:
    """Manages configuration loading, validation, and updates."""
    
    # JSON Schema for configuration validation
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "preprocessing": {
                "type": "object",
                "properties": {
                    "advanced_cleaning": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "modules": {
                                "type": "object",
                                "properties": {
                                    "boilerplate_removal": {
                                        "type": "object",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "custom_patterns": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "remove_headers": {"type": "boolean"},
                                            "remove_footers": {"type": "boolean"},
                                            "min_line_length": {"type": "integer", "minimum": 1}
                                        }
                                    },
                                    "language_filtering": {
                                        "type": "object",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "target_languages": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "confidence_threshold": {
                                                "type": "number",
                                                "minimum": 0,
                                                "maximum": 1
                                            },
                                            "detection_method": {
                                                "type": "string",
                                                "enum": ["fasttext", "langdetect"]
                                            }
                                        }
                                    },
                                    "domain_filtering": {
                                        "type": "object",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "method": {
                                                "type": "string",
                                                "enum": ["keyword", "ml", "embedding"]
                                            },
                                            "threshold": {
                                                "type": "number",
                                                "minimum": 0,
                                                "maximum": 1
                                            },
                                            "cybersecurity_keywords": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            }
                                        }
                                    },
                                    "quality_assessment": {
                                        "type": "object",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "grammar_tool": {
                                                "type": "string",
                                                "enum": ["languagetool", "spacy"]
                                            },
                                            "min_quality_score": {
                                                "type": "number",
                                                "minimum": 0,
                                                "maximum": 1
                                            }
                                        }
                                    },
                                    "entity_preservation": {
                                        "type": "object",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "entity_types": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "ner_model": {"type": "string"}
                                        }
                                    },
                                    "repetition_handling": {
                                        "type": "object",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "max_word_repetitions": {
                                                "type": "integer",
                                                "minimum": 1
                                            },
                                            "ngram_frequency_cap": {
                                                "type": "integer",
                                                "minimum": 1
                                            }
                                        }
                                    }
                                }
                            },
                            "processing_order": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[AdvancedCleaningConfig] = None
        self._raw_config: Dict[str, Any] = {}
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> AdvancedCleaningConfig:
        """
        Load configuration from file or create default.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            AdvancedCleaningConfig instance
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._raw_config = json.load(f)
                
                # Validate configuration
                self.validate_config(self._raw_config)
                
                # Extract advanced cleaning config
                advanced_config = self._raw_config.get('preprocessing', {}).get('advanced_cleaning', {})
                self._config = self._parse_config(advanced_config)
                
                logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load configuration from {self.config_path}: {e}")
                logger.info("Using default configuration")
                self._config = AdvancedCleaningConfig()
        else:
            logger.info("No configuration file found, using defaults")
            self._config = AdvancedCleaningConfig()
        
        return self._config
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> AdvancedCleaningConfig:
        """
        Parse configuration dictionary into AdvancedCleaningConfig.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            AdvancedCleaningConfig instance
        """
        # Start with defaults
        config = AdvancedCleaningConfig()
        
        # Update with provided values
        if 'enabled' in config_dict:
            config.enabled = config_dict['enabled']
        
        modules = config_dict.get('modules', {})
        
        # Parse module configurations
        if 'boilerplate_removal' in modules:
            module_config = modules['boilerplate_removal']
            config.boilerplate_removal = BoilerplateConfig(
                enabled=module_config.get('enabled', True),
                custom_patterns=module_config.get('custom_patterns', config.boilerplate_removal.custom_patterns),
                remove_headers=module_config.get('remove_headers', True),
                remove_footers=module_config.get('remove_footers', True),
                min_line_length=module_config.get('min_line_length', 10)
            )
        
        if 'language_filtering' in modules:
            module_config = modules['language_filtering']
            config.language_filtering = LanguageFilterConfig(
                enabled=module_config.get('enabled', True),
                target_languages=module_config.get('target_languages', ["en"]),
                confidence_threshold=module_config.get('confidence_threshold', 0.8),
                detection_method=module_config.get('detection_method', "fasttext"),
                fallback_enabled=module_config.get('fallback_enabled', True)
            )
        
        if 'domain_filtering' in modules:
            module_config = modules['domain_filtering']
            config.domain_filtering = DomainFilterConfig(
                enabled=module_config.get('enabled', True),
                method=module_config.get('method', "keyword"),
                threshold=module_config.get('threshold', 0.85),
                cybersecurity_keywords=module_config.get('cybersecurity_keywords', config.domain_filtering.cybersecurity_keywords),
                ml_model_path=module_config.get('ml_model_path')
            )
        
        if 'quality_assessment' in modules:
            module_config = modules['quality_assessment']
            config.quality_assessment = QualityAssessmentConfig(
                enabled=module_config.get('enabled', True),
                grammar_tool=module_config.get('grammar_tool', "languagetool"),
                min_quality_score=module_config.get('min_quality_score', 0.6),
                check_completeness=module_config.get('check_completeness', True),
                min_sentence_length=module_config.get('min_sentence_length', 5),
                max_typo_ratio=module_config.get('max_typo_ratio', 0.1),
                min_entropy=module_config.get('min_entropy', 3.0)
            )
        
        if 'entity_preservation' in modules:
            module_config = modules['entity_preservation']
            config.entity_preservation = EntityPreservationConfig(
                enabled=module_config.get('enabled', True),
                entity_types=module_config.get('entity_types', ["CVE", "PROTOCOL", "TOOL"]),
                ner_model=module_config.get('ner_model', "en_core_web_sm"),
                confidence_threshold=module_config.get('confidence_threshold', 0.7),
                custom_patterns=module_config.get('custom_patterns', {})
            )
        
        if 'repetition_handling' in modules:
            module_config = modules['repetition_handling']
            config.repetition_handling = RepetitionHandlingConfig(
                enabled=module_config.get('enabled', True),
                max_word_repetitions=module_config.get('max_word_repetitions', 3),
                ngram_frequency_cap=module_config.get('ngram_frequency_cap', 5),
                adaptive_penalty=module_config.get('adaptive_penalty', True),
                similarity_threshold=module_config.get('similarity_threshold', 0.9),
                min_ngram_length=module_config.get('min_ngram_length', 3)
            )
        
        # Parse processing order
        if 'processing_order' in config_dict:
            config.processing_order = config_dict['processing_order']
        
        # Parse other settings
        config.parallel_processing = config_dict.get('parallel_processing', False)
        config.max_workers = config_dict.get('max_workers', 4)
        config.batch_size = config_dict.get('batch_size', 100)
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            validate(instance=config, schema=self.CONFIG_SCHEMA)
        except ValidationError as e:
            errors.append(f"Configuration validation error: {e.message}")
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
        
        return errors
    
    def save_config(self, config: AdvancedCleaningConfig, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration file
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path:
            raise ValueError("No configuration path specified")
        
        # Update the raw config with advanced cleaning settings
        if not self._raw_config:
            self._raw_config = {}
        
        if 'preprocessing' not in self._raw_config:
            self._raw_config['preprocessing'] = {}
        
        self._raw_config['preprocessing']['advanced_cleaning'] = config.to_dict()
        
        # Save to file
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._raw_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {self.config_path}: {e}")
            raise
    
    def get_config(self) -> Optional[AdvancedCleaningConfig]:
        """Get the current configuration."""
        return self._config
    
    def update_module_config(self, module_name: str, module_config: Dict[str, Any]) -> None:
        """
        Update configuration for a specific module.
        
        Args:
            module_name: Name of the module to update
            module_config: New configuration for the module
        """
        if not self._config:
            self._config = AdvancedCleaningConfig()
        
        current_module = getattr(self._config, module_name, None)
        if current_module:
            # Update the module configuration
            for key, value in module_config.items():
                if hasattr(current_module, key):
                    setattr(current_module, key, value)
            
            logger.info(f"Updated configuration for module: {module_name}")
        else:
            logger.warning(f"Unknown module: {module_name}")
    
    def create_default_config_file(self, config_path: Union[str, Path]) -> None:
        """
        Create a default configuration file with all options documented.
        
        Args:
            config_path: Path where to create the configuration file
        """
        config_path = Path(config_path)
        
        default_config = {
            "preprocessing": {
                "advanced_cleaning": {
                    "enabled": True,
                    "modules": {
                        "boilerplate_removal": {
                            "enabled": True,
                            "custom_patterns": [
                                "(copyright|all rights reserved|quiz|correct answer|©)",
                                "\\b[a-d]\\.\\s*){2,}"
                            ],
                            "remove_headers": True,
                            "remove_footers": True,
                            "min_line_length": 10
                        },
                        "language_filtering": {
                            "enabled": True,
                            "target_languages": ["en"],
                            "confidence_threshold": 0.8,
                            "detection_method": "fasttext",
                            "fallback_enabled": True
                        },
                        "domain_filtering": {
                            "enabled": True,
                            "method": "keyword",
                            "threshold": 0.85,
                            "cybersecurity_keywords": [
                                "firewall", "malware", "vulnerability", "encryption", "security",
                                "attack", "threat", "exploit", "penetration", "authentication"
                            ]
                        },
                        "quality_assessment": {
                            "enabled": True,
                            "grammar_tool": "languagetool",
                            "min_quality_score": 0.6,
                            "check_completeness": True,
                            "min_sentence_length": 5,
                            "max_typo_ratio": 0.1
                        },
                        "entity_preservation": {
                            "enabled": True,
                            "entity_types": ["CVE", "PROTOCOL", "TOOL"],
                            "ner_model": "en_core_web_sm",
                            "confidence_threshold": 0.7
                        },
                        "repetition_handling": {
                            "enabled": True,
                            "max_word_repetitions": 3,
                            "ngram_frequency_cap": 5,
                            "adaptive_penalty": True
                        }
                    },
                    "processing_order": [
                        "boilerplate_removal",
                        "language_filtering",
                        "domain_filtering",
                        "quality_assessment",
                        "entity_preservation",
                        "repetition_handling"
                    ],
                    "parallel_processing": False,
                    "max_workers": 4,
                    "batch_size": 100
                }
            }
        }
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created default configuration file: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to create default configuration file: {e}")
            raise