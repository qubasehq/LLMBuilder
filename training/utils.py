"""
Common training utilities for LLM training pipeline.
Provides configuration loading, checkpointing, logging, and device management.
"""

import os
import json
import yaml
import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger
import time


class ConfigManager:
    """Manages configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['model', 'train']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section '{section}' in config")
            
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            raise


class CheckpointManager:
    """Manages model checkpointing and loading."""
    
    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       config: Dict[str, Any],
                       metrics: Optional[Dict[str, float]] = None) -> Path:
        """Save model checkpoint with metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            epoch: Current epoch number
            loss: Current loss value
            config: Training configuration
            metrics: Optional evaluation metrics
            
        Returns:
            Path to saved checkpoint
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config,
                'metrics': metrics or {},
                'timestamp': time.time()
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save latest checkpoint reference
            latest_path = self.checkpoint_dir / "latest.pt"
            torch.save(checkpoint, latest_path)
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest.
            
        Returns:
            Dictionary containing checkpoint data
        """
        try:
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / "latest.pt"
            else:
                checkpoint_path = Path(checkpoint_path)
                
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        return sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))


class DeviceManager:
    """Manages device selection and optimization for CPU/GPU."""
    
    @staticmethod
    def get_device(prefer_cpu: bool = False) -> torch.device:
        """Get optimal device for training.
        
        Args:
            prefer_cpu: Force CPU usage even if GPU is available
            
        Returns:
            torch.device object
        """
        if prefer_cpu:
            device = torch.device('cpu')
            logger.info("Using CPU (forced)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU (GPU not available)")
            
        return device
    
    @staticmethod
    def optimize_for_cpu():
        """Apply CPU-specific optimizations."""
        # Set number of threads for CPU training
        torch.set_num_threads(os.cpu_count())
        
        # Enable CPU optimizations
        torch.backends.mkldnn.enabled = True
        
        logger.info(f"CPU optimizations applied. Using {torch.get_num_threads()} threads")


class MetricsTracker:
    """Tracks and logs training metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, **kwargs):
        """Update current metrics."""
        self.metrics.update(kwargs)
    
    def log_epoch(self, epoch: int):
        """Log metrics for current epoch."""
        epoch_metrics = {'epoch': epoch, **self.metrics}
        self.history.append(epoch_metrics)
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        logger.info(f"Epoch {epoch} - {metrics_str}")
    
    def save_history(self, filepath: Union[str, Path]):
        """Save metrics history to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Metrics history saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics history: {e}")


def setup_logging(log_dir: Union[str, Path] = "logs", level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level
    )
    
    # Add file logger
    logger.add(
        log_dir / "training.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=level,
        rotation="10 MB",
        retention="7 days"
    )
    
    logger.info("Logging setup complete")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.0f}s"

