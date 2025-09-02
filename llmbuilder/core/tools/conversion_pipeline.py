#!/usr/bin/env python3
"""
ConversionPipeline for automated GGUF conversion workflow.

This module provides a comprehensive pipeline for converting PyTorch models
to GGUF format with multiple quantization levels, batch processing, and
error handling.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from .export_gguf import GGUFConverter
from .quantization_manager import QuantizationManager, create_quantization_config


@dataclass
class ConversionConfig:
    """Configuration for conversion pipeline."""
    input_checkpoint: Path
    output_dir: Path
    quantization_levels: List[str]
    model_name: Optional[str] = None
    tokenizer_path: Optional[Path] = None
    validate_output: bool = True
    cleanup_intermediate: bool = False
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.input_checkpoint = Path(self.input_checkpoint)
        self.output_dir = Path(self.output_dir)
        
        if self.tokenizer_path:
            self.tokenizer_path = Path(self.tokenizer_path)
        
        # Validate quantization levels
        valid_levels = {"f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"}
        invalid_levels = set(self.quantization_levels) - valid_levels
        if invalid_levels:
            raise ValueError(f"Invalid quantization levels: {invalid_levels}")


@dataclass
class ConversionResult:
    """Result of a single conversion."""
    quantization_level: str
    output_path: Path
    success: bool
    file_size_mb: float
    conversion_time: float
    compression_ratio: Optional[float] = None
    quality_score: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of the entire conversion pipeline."""
    input_checkpoint: Path
    output_dir: Path
    total_time: float
    successful_conversions: int
    failed_conversions: int
    results: List[ConversionResult]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = len(self.results)
        return (self.successful_conversions / total * 100) if total > 0 else 0.0
    
    def get_result(self, quantization_level: str) -> Optional[ConversionResult]:
        """Get result for specific quantization level."""
        for result in self.results:
            if result.quantization_level == quantization_level:
                return result
        return None


class ConversionPipeline:
    """
    Automated pipeline for converting PyTorch models to GGUF format
    with multiple quantization levels.
    """
    
    def __init__(self, config: ConversionConfig):
        """
        Initialize conversion pipeline.
        
        Args:
            config: Configuration for the conversion pipeline
        """
        self.config = config
        self.results: List[ConversionResult] = []
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ConversionPipeline")
        logger.info(f"Input: {self.config.input_checkpoint}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info(f"Quantization levels: {self.config.quantization_levels}")
    
    def convert_all(self) -> PipelineResult:
        """
        Convert model to all specified quantization levels.
        
        Returns:
            PipelineResult with conversion results and statistics
        """
        logger.info("🚀 Starting conversion pipeline...")
        start_time = time.time()
        
        self.results = []
        successful = 0
        failed = 0
        
        for level in self.config.quantization_levels:
            logger.info(f"Converting with quantization level: {level}")
            
            result = self._convert_single(level)
            self.results.append(result)
            
            if result.success:
                successful += 1
                logger.info(f"✅ {level}: {result.file_size_mb:.1f}MB in {result.conversion_time:.1f}s")
            else:
                failed += 1
                logger.error(f"❌ {level}: {result.error_message}")
        
        total_time = time.time() - start_time
        
        pipeline_result = PipelineResult(
            input_checkpoint=self.config.input_checkpoint,
            output_dir=self.config.output_dir,
            total_time=total_time,
            successful_conversions=successful,
            failed_conversions=failed,
            results=self.results
        )
        
        self._log_summary(pipeline_result)
        return pipeline_result
    
    def _convert_single(self, quantization_level: str) -> ConversionResult:
        """
        Convert model with a single quantization level.
        
        Args:
            quantization_level: Quantization level to use
            
        Returns:
            ConversionResult with conversion details
        """
        output_filename = self._get_output_filename(quantization_level)
        output_path = self.config.output_dir / output_filename
        
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                # Use the GGUF converter
                from .export_gguf import GGUFConverter, ModelMetadata
                
                # Prepare metadata
                metadata = {}
                if self.config.model_name:
                    metadata['name'] = self.config.model_name
                
                # Create model metadata
                model_metadata = ModelMetadata(
                    name=metadata.get('name', 'Unknown'),
                    architecture="gpt2",  # Default architecture
                    version="1.0"
                )
                
                # Export with quantization
                converter = GGUFConverter(
                    model_path=str(self.config.input_checkpoint),
                    output_path=str(output_path),
                    metadata=model_metadata,
                    quantization_type=quantization_level
                )
                
                success = converter.export_to_gguf(
                    tokenizer_path=str(self.config.tokenizer_path) if self.config.tokenizer_path else None,
                    validate=self.config.validate_output
                )
                
                if not success:
                    raise ValueError("GGUF export failed")
                
                # Calculate file size
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                conversion_time = time.time() - start_time
                
                # Calculate compression ratio if possible
                compression_ratio = None
                quality_score = None
                
                if self.config.input_checkpoint.exists():
                    input_size_mb = self.config.input_checkpoint.stat().st_size / (1024 * 1024)
                    if input_size_mb > 0:
                        compression_ratio = input_size_mb / file_size_mb
                
                # Validate output if requested
                if self.config.validate_output:
                    if not self.validate_conversion(output_path):
                        raise ValueError("Output validation failed")
                
                return ConversionResult(
                    quantization_level=quantization_level,
                    output_path=output_path,
                    success=True,
                    file_size_mb=file_size_mb,
                    conversion_time=conversion_time,
                    compression_ratio=compression_ratio,
                    quality_score=quality_score
                )
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.config.max_retries}: {str(e)}"
                logger.warning(error_msg)
                
                if attempt == self.config.max_retries - 1:
                    # Final attempt failed
                    conversion_time = time.time() - start_time
                    return ConversionResult(
                        quantization_level=quantization_level,
                        output_path=output_path,
                        success=False,
                        file_size_mb=0.0,
                        conversion_time=conversion_time,
                        error_message=str(e)
                    )
                
                # Wait before retry
                time.sleep(1.0)
        
        # Should not reach here
        return ConversionResult(
            quantization_level=quantization_level,
            output_path=output_path,
            success=False,
            file_size_mb=0.0,
            conversion_time=0.0,
            error_message="Unknown error"
        )
    
    def validate_conversion(self, gguf_path: Path) -> bool:
        """
        Validate converted GGUF model.
        
        Args:
            gguf_path: Path to GGUF file to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check file exists and has reasonable size
            if not gguf_path.exists():
                logger.error(f"Output file does not exist: {gguf_path}")
                return False
            
            file_size = gguf_path.stat().st_size
            if file_size < 1024:  # Less than 1KB is suspicious
                logger.error(f"Output file too small: {file_size} bytes")
                return False
            
            # Basic validation - check if file looks like GGUF
            with open(gguf_path, 'rb') as f:
                header = f.read(4)
                if header != b'GGUF':
                    logger.error(f"Invalid GGUF header: {header}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def _get_output_filename(self, quantization_level: str) -> str:
        """Generate output filename for quantization level."""
        base_name = self.config.input_checkpoint.stem
        if self.config.model_name:
            base_name = self.config.model_name.replace(" ", "_").lower()
        
        return f"{base_name}_{quantization_level}.gguf"
    
    def _log_summary(self, result: PipelineResult) -> None:
        """Log conversion pipeline summary."""
        logger.info("=" * 60)
        logger.info("CONVERSION PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Input: {result.input_checkpoint}")
        logger.info(f"Output: {result.output_dir}")
        logger.info(f"Total time: {result.total_time:.1f}s")
        logger.info(f"Success rate: {result.success_rate:.1f}% ({result.successful_conversions}/{len(result.results)})")
        
        if result.successful_conversions > 0:
            logger.info("\nSuccessful conversions:")
            for conv_result in result.results:
                if conv_result.success:
                    size_info = f"{conv_result.file_size_mb:.1f}MB"
                    time_info = f"{conv_result.conversion_time:.1f}s"
                    
                    extra_info = []
                    if conv_result.compression_ratio:
                        extra_info.append(f"{conv_result.compression_ratio:.1f}x compression")
                    if conv_result.quality_score:
                        extra_info.append(f"{conv_result.quality_score:.3f} quality")
                    
                    extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
                    logger.info(f"  ✅ {conv_result.quantization_level}: {size_info}, {time_info}{extra_str}")
        
        if result.failed_conversions > 0:
            logger.info("\nFailed conversions:")
            for conv_result in result.results:
                if not conv_result.success:
                    logger.info(f"  ❌ {conv_result.quantization_level}: {conv_result.error_message}")
    
    def save_report(self, report_path: Path) -> None:
        """
        Save conversion report to JSON file.
        
        Args:
            report_path: Path to save the report
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Create pipeline result
        pipeline_result = PipelineResult(
            input_checkpoint=self.config.input_checkpoint,
            output_dir=self.config.output_dir,
            total_time=sum(r.conversion_time for r in self.results),
            successful_conversions=sum(1 for r in self.results if r.success),
            failed_conversions=sum(1 for r in self.results if not r.success),
            results=self.results
        )
        
        # Convert to serializable format
        report_data = {
            "pipeline_config": {
                "input_checkpoint": str(self.config.input_checkpoint),
                "output_dir": str(self.config.output_dir),
                "quantization_levels": self.config.quantization_levels,
                "model_name": self.config.model_name,
                "tokenizer_path": str(self.config.tokenizer_path) if self.config.tokenizer_path else None,
            },
            "pipeline_result": {
                "input_checkpoint": str(pipeline_result.input_checkpoint),
                "output_dir": str(pipeline_result.output_dir),
                "total_time": pipeline_result.total_time,
                "successful_conversions": pipeline_result.successful_conversions,
                "failed_conversions": pipeline_result.failed_conversions,
                "success_rate": pipeline_result.success_rate,
            },
            "conversion_results": [
                {
                    "quantization_level": r.quantization_level,
                    "output_path": str(r.output_path),
                    "success": r.success,
                    "file_size_mb": r.file_size_mb,
                    "conversion_time": r.conversion_time,
                    "compression_ratio": r.compression_ratio,
                    "quality_score": r.quality_score,
                    "error_message": r.error_message,
                }
                for r in self.results
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Conversion report saved to {report_path}")


def create_conversion_pipeline(
    input_checkpoint: Path,
    output_dir: Path,
    quantization_levels: List[str] = None,
    **kwargs
) -> ConversionPipeline:
    """
    Factory function to create a ConversionPipeline with sensible defaults.
    
    Args:
        input_checkpoint: Path to input PyTorch checkpoint
        output_dir: Directory to save converted models
        quantization_levels: List of quantization levels to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured ConversionPipeline instance
    """
    if quantization_levels is None:
        quantization_levels = ["f16", "q8_0", "q4_0"]
    
    config = ConversionConfig(
        input_checkpoint=input_checkpoint,
        output_dir=output_dir,
        quantization_levels=quantization_levels,
        **kwargs
    )
    
    return ConversionPipeline(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch model to GGUF with multiple quantization levels")
    parser.add_argument("input", type=Path, help="Input PyTorch checkpoint")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--quantization", nargs="+", default=["f16", "q8_0", "q4_0"],
                       help="Quantization levels to use")
    parser.add_argument("--name", help="Model name")
    parser.add_argument("--tokenizer", type=Path, help="Tokenizer path")
    parser.add_argument("--no-validate", action="store_true", help="Skip output validation")
    parser.add_argument("--report", type=Path, help="Save conversion report to file")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = create_conversion_pipeline(
        input_checkpoint=args.input,
        output_dir=args.output,
        quantization_levels=args.quantization,
        model_name=args.name,
        tokenizer_path=args.tokenizer,
        validate_output=not args.no_validate
    )
    
    # Run conversion
    result = pipeline.convert_all()
    
    # Save report if requested
    if args.report:
        pipeline.save_report(args.report)
    
    # Exit with error code if any conversions failed
    if result.failed_conversions > 0:
        exit(1)