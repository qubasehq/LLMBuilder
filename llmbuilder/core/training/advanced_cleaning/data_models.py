"""
Data models for the Advanced Cybersecurity Dataset Cleaning system.

This module defines the core data structures used throughout the cleaning pipeline,
including results, statistics, entities, and operations tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class EntityType(Enum):
    """Enumeration of cybersecurity-specific entity types."""
    CVE = "CVE"
    PROTOCOL = "PROTOCOL"
    TOOL = "TOOL"
    VULNERABILITY = "VULNERABILITY"
    MALWARE = "MALWARE"
    HASH = "HASH"
    IP_ADDRESS = "IP_ADDRESS"
    DOMAIN = "DOMAIN"
    FILE_PATH = "FILE_PATH"
    REGISTRY_KEY = "REGISTRY_KEY"
    OTHER = "OTHER"


class CleaningOperationType(Enum):
    """Types of cleaning operations that can be performed."""
    BOILERPLATE_REMOVAL = "boilerplate_removal"
    LANGUAGE_FILTERING = "language_filtering"
    DOMAIN_FILTERING = "domain_filtering"
    QUALITY_ASSESSMENT = "quality_assessment"
    ENTITY_PRESERVATION = "entity_preservation"
    REPETITION_HANDLING = "repetition_handling"
    NORMALIZATION = "normalization"


@dataclass
class Entity:
    """Represents a named entity found in text."""
    text: str
    label: str
    start_pos: int
    end_pos: int
    confidence: float
    entity_type: EntityType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity data after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.start_pos < 0 or self.end_pos < 0:
            raise ValueError("Position values must be non-negative")
        if self.start_pos >= self.end_pos:
            raise ValueError("start_pos must be less than end_pos")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "text": self.text,
            "label": self.label,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "entity_type": self.entity_type.value,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary representation."""
        return cls(
            text=data["text"],
            label=data["label"],
            start_pos=data["start_pos"],
            end_pos=data["end_pos"],
            confidence=data["confidence"],
            entity_type=EntityType(data["entity_type"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class CleaningOperation:
    """Represents a single cleaning operation performed on text."""
    operation_type: CleaningOperationType
    module_name: str
    description: str
    original_length: int
    final_length: int
    items_removed: int = 0
    items_modified: int = 0
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def characters_removed(self) -> int:
        """Calculate number of characters removed."""
        return max(0, self.original_length - self.final_length)
    
    @property
    def reduction_percentage(self) -> float:
        """Calculate percentage of text reduced."""
        if self.original_length == 0:
            return 0.0
        return (self.characters_removed / self.original_length) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operation to dictionary representation."""
        return {
            "operation_type": self.operation_type.value,
            "module_name": self.module_name,
            "description": self.description,
            "original_length": self.original_length,
            "final_length": self.final_length,
            "items_removed": self.items_removed,
            "items_modified": self.items_modified,
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "characters_removed": self.characters_removed,
            "reduction_percentage": self.reduction_percentage
        }


@dataclass
class CleaningStats:
    """Comprehensive statistics for cleaning operations."""
    original_length: int
    final_length: int
    boilerplate_removed: int = 0
    language_filtered: bool = False
    domain_relevant: bool = True
    quality_score: float = 1.0
    entities_count: int = 0
    repetitions_removed: int = 0
    processing_modules: List[str] = field(default_factory=list)
    total_processing_time: float = 0.0
    operations_performed: int = 0
    errors_encountered: int = 0
    warnings_generated: int = 0
    
    @property
    def characters_removed(self) -> int:
        """Calculate total characters removed."""
        return max(0, self.original_length - self.final_length)
    
    @property
    def reduction_percentage(self) -> float:
        """Calculate percentage of text reduced."""
        if self.original_length == 0:
            return 0.0
        return (self.characters_removed / self.original_length) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of operations."""
        if self.operations_performed == 0:
            return 1.0
        return (self.operations_performed - self.errors_encountered) / self.operations_performed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary representation."""
        return {
            "original_length": self.original_length,
            "final_length": self.final_length,
            "boilerplate_removed": self.boilerplate_removed,
            "language_filtered": self.language_filtered,
            "domain_relevant": self.domain_relevant,
            "quality_score": self.quality_score,
            "entities_count": self.entities_count,
            "repetitions_removed": self.repetitions_removed,
            "processing_modules": self.processing_modules,
            "total_processing_time": self.total_processing_time,
            "operations_performed": self.operations_performed,
            "errors_encountered": self.errors_encountered,
            "warnings_generated": self.warnings_generated,
            "characters_removed": self.characters_removed,
            "reduction_percentage": self.reduction_percentage,
            "success_rate": self.success_rate
        }


@dataclass
class CleaningResult:
    """Complete result of text cleaning operations."""
    original_text: str
    cleaned_text: str
    cleaning_operations: List[CleaningOperation] = field(default_factory=list)
    quality_score: float = 1.0
    entities_preserved: List[Entity] = field(default_factory=list)
    statistics: Optional[CleaningStats] = None
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize statistics if not provided."""
        if self.statistics is None:
            self.statistics = CleaningStats(
                original_length=len(self.original_text),
                final_length=len(self.cleaned_text),
                quality_score=self.quality_score,
                entities_count=len(self.entities_preserved),
                processing_modules=[op.module_name for op in self.cleaning_operations],
                total_processing_time=self.processing_time,
                operations_performed=len(self.cleaning_operations),
                errors_encountered=sum(1 for op in self.cleaning_operations if not op.success),
                warnings_generated=len(self.warnings)
            )
    
    @property
    def characters_removed(self) -> int:
        """Calculate number of characters removed."""
        return max(0, len(self.original_text) - len(self.cleaned_text))
    
    @property
    def reduction_percentage(self) -> float:
        """Calculate percentage of text reduced."""
        if len(self.original_text) == 0:
            return 0.0
        return (self.characters_removed / len(self.original_text)) * 100
    
    def add_operation(self, operation: CleaningOperation) -> None:
        """Add a cleaning operation to the result."""
        self.cleaning_operations.append(operation)
        if self.statistics:
            self.statistics.operations_performed += 1
            self.statistics.total_processing_time += operation.processing_time
            if not operation.success:
                self.statistics.errors_encountered += 1
            if operation.module_name not in self.statistics.processing_modules:
                self.statistics.processing_modules.append(operation.module_name)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message to the result."""
        self.warnings.append(warning)
        if self.statistics:
            self.statistics.warnings_generated += 1
    
    def add_entity(self, entity: Entity) -> None:
        """Add a preserved entity to the result."""
        self.entities_preserved.append(entity)
        if self.statistics:
            self.statistics.entities_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "cleaning_operations": [op.to_dict() for op in self.cleaning_operations],
            "quality_score": self.quality_score,
            "entities_preserved": [entity.to_dict() for entity in self.entities_preserved],
            "statistics": self.statistics.to_dict() if self.statistics else None,
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "characters_removed": self.characters_removed,
            "reduction_percentage": self.reduction_percentage
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the cleaning result."""
        summary_lines = [
            f"Cleaning Summary:",
            f"  Original length: {len(self.original_text):,} characters",
            f"  Final length: {len(self.cleaned_text):,} characters",
            f"  Reduction: {self.reduction_percentage:.1f}%",
            f"  Quality score: {self.quality_score:.2f}",
            f"  Entities preserved: {len(self.entities_preserved)}",
            f"  Operations performed: {len(self.cleaning_operations)}",
            f"  Processing time: {self.processing_time:.2f}s",
            f"  Success: {self.success}"
        ]
        
        if self.warnings:
            summary_lines.append(f"  Warnings: {len(self.warnings)}")
        
        if self.error_message:
            summary_lines.append(f"  Error: {self.error_message}")
        
        return "\n".join(summary_lines)


# Type aliases for convenience
CleaningResultDict = Dict[str, Any]
CleaningStatsDict = Dict[str, Any]
EntityDict = Dict[str, Any]
CleaningOperationDict = Dict[str, Any]