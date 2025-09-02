"""
Logging configuration for LLMBuilder.

This module provides centralized logging setup with support for:
- Multiple log levels
- File and console output
- Structured logging
- Log rotation
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: str = "llmbuilder.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
    colorize: bool = True
) -> None:
    """
    Setup logging configuration for LLMBuilder.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (optional)
        log_file: Log file name
        rotation: Log rotation size
        retention: Log retention period
        colorize: Enable colored console output
    """
    # Remove default logger
    logger.remove()
    
    # Console handler
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=colorize,
        backtrace=True,
        diagnose=True
    )
    
    # File handler (if log_dir is specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
        
        logger.add(
            log_dir / log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            backtrace=True,
            diagnose=True
        )


def get_logger(name: str):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)