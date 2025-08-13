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
from typing import Dict, Any, Optional, Union, List
from loguru import logger
import time


class ConfigManager:
    """Manages configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is malformed or missing required sections
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            # Convert JSON format to expected YAML format if needed
            if 'training' in config and 'train' not in config:
                # Map JSON training section to train section
                training_config = config['training']
                config['train'] = {
                    'learning_rate': training_config.get('learning_rate', 0.0003),
                    'batch_size': training_config.get('batch_size', 16),
                    'max_iters': training_config.get('num_epochs', 10) * 1000,  # Rough conversion
                    'eval_interval': training_config.get('eval_every', 1000),
                    'eval_iters': training_config.get('eval_every', 100),
                    'log_interval': training_config.get('log_every', 100),
                    'device': config.get('device', {}).get('use_cuda', False) and 'cuda' or 'cpu'
                }
            
            # Ensure model section has required fields with defaults
            if 'model' in config:
                model_config = config['model']
                # Map JSON model fields to expected YAML fields
                if 'num_layers' in model_config and 'n_layer' not in model_config:
                    model_config['n_layer'] = model_config['num_layers']
                if 'num_heads' in model_config and 'n_head' not in model_config:
                    model_config['n_head'] = model_config['num_heads']
                if 'embedding_dim' in model_config and 'n_embd' not in model_config:
                    model_config['n_embd'] = model_config['embedding_dim']
                if 'max_seq_length' in model_config and 'block_size' not in model_config:
                    model_config['block_size'] = model_config['max_seq_length']
            
            # Validate required sections
            required_sections = ['model', 'train']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section '{section}' in config")
            
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON config: {e}")
            raise
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
    import sys
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger with proper Unicode handling
    def safe_console_write(msg):
        """Safely write to console with Unicode support."""
        try:
            # Try to print normally first
            print(msg, end="")
        except UnicodeEncodeError:
            # If that fails, encode to ASCII with replacement characters
            safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
            print(safe_msg, end="")
    
    logger.add(
        safe_console_write,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level
    )
    
    # Add file logger (files can handle UTF-8 properly)
    logger.add(
        log_dir / "training.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8"
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
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_checkpoint_vocab_size(checkpoint_path: Union[str, Path]) -> int:
    """
    Get vocabulary size from checkpoint file.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Vocabulary size from checkpoint
        
    Raises:
        ValueError: If checkpoint format is invalid
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
        else:
            # Direct model format
            vocab_size = checkpoint['token_embedding.weight'].shape[0]
            
        return vocab_size
        
    except Exception as e:
        raise ValueError(f"Failed to get vocab size from checkpoint: {e}")


def get_tokenizer_vocab_size(tokenizer_paths: List[Union[str, Path]]) -> Optional[int]:
    """
    Get vocabulary size from tokenizer files.
    
    Args:
        tokenizer_paths: List of possible tokenizer file paths
        
    Returns:
        Vocabulary size if found, None otherwise
    """
    for tokenizer_path in tokenizer_paths:
        tokenizer_path = Path(tokenizer_path)
        if tokenizer_path.exists():
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.load(str(tokenizer_path))
                return sp.vocab_size()
            except ImportError:
                # Fallback for HuggingFace tokenizers
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path.parent))
                    return tokenizer.vocab_size
                except:
                    continue
            except:
                continue
    
    return None


def synchronize_vocab_size(config: Dict[str, Any], checkpoint_path: Optional[Union[str, Path]] = None) -> int:
    """
    Ensure vocabulary size consistency across all stages.
    
    Args:
        config: Model configuration dictionary
        checkpoint_path: Optional checkpoint path to get vocab size from
        
    Returns:
        Synchronized vocabulary size
    """
    try:
        actual_vocab_size = None
        
        # Priority 1: Get from checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                actual_vocab_size = get_checkpoint_vocab_size(checkpoint_path)
                logger.info(f"Got vocab size from checkpoint: {actual_vocab_size}")
            except Exception as e:
                logger.warning(f"Could not get vocab size from checkpoint: {e}")
        
        # Priority 2: Get from tokenizer files
        if actual_vocab_size is None:
            tokenizer_paths = []
            
            # Build possible tokenizer paths
            if 'paths' in config:
                paths_config = config['paths']
                tokenizer_dir = paths_config.get('tokenizer_dir', 'exports/tokenizer')
                tokenizer_paths.extend([
                    Path(tokenizer_dir) / 'tokenizer.model',
                    Path('exports/tokenizer') / 'tokenizer.model',
                    Path('tokenizer') / 'tokenizer.model'
                ])
            
            tokenizer_paths.extend([
                Path('exports/tokenizer') / 'tokenizer.model',
                Path('tokenizer') / 'tokenizer.model'
            ])
            
            actual_vocab_size = get_tokenizer_vocab_size(tokenizer_paths)
            if actual_vocab_size:
                logger.info(f"Got vocab size from tokenizer: {actual_vocab_size}")
        
        # Priority 3: Use config value as fallback
        if actual_vocab_size is None:
            actual_vocab_size = config.get('model', {}).get('vocab_size', 16000)
            logger.warning(f"Using config vocab_size as fallback: {actual_vocab_size}")
        
        # Update config with synchronized vocab size
        if 'model' not in config:
            config['model'] = {}
        
        old_vocab_size = config['model'].get('vocab_size')
        if old_vocab_size != actual_vocab_size:
            if old_vocab_size:
                logger.warning(f"Config vocab_size ({old_vocab_size}) != actual vocab size ({actual_vocab_size})")
            config['model']['vocab_size'] = actual_vocab_size
            logger.info(f"✓ Vocabulary size synchronized: {actual_vocab_size}")
        
        return actual_vocab_size
        
    except Exception as e:
        logger.warning(f"⚠ Could not synchronize vocab size: {e}")
        return config.get('model', {}).get('vocab_size', 16000)
