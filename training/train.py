"""
Main training script for LLM training.
Optimized for CPU execution with comprehensive logging, checkpointing, and error handling.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from loguru import logger
from tqdm import tqdm
import math
import json
import traceback
import sys
from pathlib import Path
import json
sys.path.append(str(Path(__file__).parent.parent))

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.gpt_model import GPTModel
from training.dataset import TextDataset, MultiFileDataset, create_dataloader, split_dataset
from training.utils import (
    ConfigManager, CheckpointManager, DeviceManager, MetricsTracker,
    setup_logging, count_parameters, format_time
)
from training.quantization import QuantizationManager, QuantConfig


class Trainer:
    """Main trainer class for LLM training with improved error handling and features."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        try:
            # Load configuration from JSON and ensure vocabulary consistency
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self._ensure_vocab_consistency()
            self.config_path = config_path
            
            # Setup logging first
            log_config = self.config.get('logging', {})
            setup_logging(
                log_dir=log_config.get('log_dir', "logs"), 
                level=log_config.get('level', "INFO")
            )
            
            # Validate configuration
            self._validate_config()
            
            # Initialize device and optimizations
            self.device = DeviceManager.get_device(
                prefer_cpu=self.config.get('device', {}).get('use_cuda', False) == False
            )
            
            if self.device.type == 'cpu':
                DeviceManager.optimize_for_cpu()
                cpu_threads = self.config.get('device', {}).get('cpu_threads', 0)
                if cpu_threads > 0:
                    torch.set_num_threads(cpu_threads)
            
            # Initialize components
            self.model = None
            self.optimizer = None
            self.scheduler = None
            self.train_loader = None
            self.val_loader = None
            self.checkpoint_manager = None
            self.metrics_tracker = MetricsTracker()
            
            # Paths from config or defaults
            paths = self.config.get('paths', {})
            self.tokenizer_dir = paths.get('tokenizer_dir', "exports/tokenizer")
            self.checkpoint_dir = paths.get('checkpoint_dir', "exports/checkpoints")
            
            # Create directories
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            
            # Training state
            self.current_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.start_time = time.time()
            
            # Performance tracking
            self.step_times = []
            self.memory_usage = []
            
            logger.info("Trainer initialized successfully")
            logger.info(f"Device: {self.device}")
            logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _ensure_vocab_consistency(self):
        """Ensure vocabulary size consistency between tokenizer and model"""
        try:
            # Get actual tokenizer vocab size from data or tokenizer file
            actual_vocab_size = None
            
            # Try to get from tokenizer file
            tokenizer_paths = [
                Path(self.config.get('paths', {}).get('tokenizer_dir', 'exports/tokenizer')) / 'tokenizer.model',
                Path('exports/tokenizer') / 'tokenizer.model',
                Path('tokenizer') / 'tokenizer.model'
            ]
            
            for tokenizer_path in tokenizer_paths:
                if tokenizer_path.exists():
                    import sentencepiece as spm
                    sp = spm.SentencePieceProcessor()
                    sp.load(str(tokenizer_path))
                    actual_vocab_size = sp.vocab_size()
                    break
            
            # If no tokenizer found, use data vocab size if available
            if actual_vocab_size is None and hasattr(self, 'train_dataset'):
                try:
                    actual_vocab_size = self.train_dataset.vocab_size
                except:
                    pass
            
            if actual_vocab_size is not None:
                # Update model vocab size to match actual data
                old_vocab_size = self.config['model'].get('vocab_size')
                if old_vocab_size != actual_vocab_size:
                    logger.warning(f"Config vocab_size ({old_vocab_size}) != data vocab_size ({actual_vocab_size})")
                    logger.info(f"Updating config vocab_size to {actual_vocab_size}")
                    self.config['model']['vocab_size'] = actual_vocab_size
                
                logger.info(f"✓ Vocabulary size synchronized: {actual_vocab_size}")
                return actual_vocab_size
            else:
                logger.warning("⚠ Tokenizer not found, using config vocab_size")
                return self.config['model'].get('vocab_size', 16000)
                
        except Exception as e:
            logger.warning(f"⚠ Could not auto-detect vocab size: {e}")
            return self.config['model'].get('vocab_size', 16000)
    
    def _validate_config(self):
        """Validate configuration structure and required fields."""
        required_sections = ['model', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate model config
        model_required = ['vocab_size', 'embedding_dim', 'num_layers', 'num_heads', 'max_seq_length']
        for field in model_required:
            if field not in self.config['model']:
                raise ValueError(f"Missing required model config field: {field}")
        
        # Validate training config
        training_required = ['batch_size', 'learning_rate', 'num_epochs']
        for field in training_required:
            if field not in self.config['training']:
                raise ValueError(f"Missing required training config field: {field}")
        
        logger.info("Configuration validation passed")
    
    def setup_data(self):
        """Setup datasets and data loaders with improved error handling."""
        logger.info("Setting up datasets...")
        
        try:
            # Check for tokenized data
            data_dir = Path("data/tokens")
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
            data_files = list(data_dir.glob("*.pt"))
            if not data_files:
                raise FileNotFoundError("No tokenized data files found. Please run preprocessing first.")
            
            logger.info(f"Found {len(data_files)} tokenized data file(s)")
            
            # Create dataset
            if len(data_files) == 1:
                # Single file dataset
                data_file = data_files[0]
                logger.info(f"Loading single file dataset: {data_file}")
                dataset = TextDataset(
                    data_path=data_file,
                    block_size=self.config['model']['max_seq_length']
                )
            else:
                # Multi-file dataset
                logger.info(f"Loading multi-file dataset from: {data_dir}")
                dataset = MultiFileDataset(
                    data_dir=data_dir,
                    block_size=self.config['model']['max_seq_length']
                )
            
            # Validate dataset
            if len(dataset) == 0:
                raise ValueError("Dataset is empty")
            
            logger.info(f"Dataset loaded: {len(dataset):,} samples")
            
            # Split into train/validation
            train_ratio = self.config.get('preprocessing', {}).get('train_ratio', 0.9)
            train_dataset, val_dataset = split_dataset(dataset, train_ratio=train_ratio)
            
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                raise ValueError("Train or validation dataset is empty after split")
            
            # Create data loaders
            batch_size = self.config['training']['batch_size']
            num_workers = self.config['training'].get('num_workers', 0)
            pin_memory = self.config['training'].get('pin_memory', False) and self.device.type == 'cuda'
            
            self.train_loader = create_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            self.val_loader = create_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            # Update vocab size in config if needed
            if hasattr(dataset, 'get_vocab_size'):
                actual_vocab_size = dataset.get_vocab_size()
                config_vocab_size = self.config['model']['vocab_size']
                
                if config_vocab_size != actual_vocab_size:
                    logger.warning(f"Config vocab_size ({config_vocab_size}) != data vocab_size ({actual_vocab_size})")
                    logger.info(f"Updating config vocab_size to {actual_vocab_size}")
                    self.config['model']['vocab_size'] = actual_vocab_size
            
            logger.info(f"Data setup complete: {len(train_dataset):,} train, {len(val_dataset):,} val samples")
            logger.info(f"Batch size: {batch_size}, Steps per epoch: {len(self.train_loader)}")
            
        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def setup_model(self):
        """Initialize model, optimizer, and scheduler with improved error handling."""
        logger.info("Setting up model...")
        
        try:
            # Model configuration
            model_config = self.config['model']
            
            # Log model configuration
            logger.info(f"Model configuration: {model_config}")
            
            # Auto-detect vocabulary size from tokenizer if not specified
            vocab_size = model_config.get('vocab_size')
            if vocab_size is None:
                vocab_size = self.tokenizer.vocab_size()
                logger.info(f"Auto-detected vocab_size: {vocab_size}")
            
            # Initialize model with correct vocab size
            self.model = GPTModel(
                vocab_size=vocab_size,
                n_layer=model_config['num_layers'],
                n_head=model_config['num_heads'],
                n_embd=model_config['embedding_dim'],
                block_size=model_config['max_seq_length'],
                dropout=model_config.get('dropout', 0.1)
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Log model info
            n_params = count_parameters(self.model)
            logger.info(f"Model has {n_params:,} trainable parameters")
            
            # Estimate memory usage
            if hasattr(self.model, 'estimate_memory_usage'):
                memory_info = self.model.estimate_memory_usage(
                    batch_size=self.config['training']['batch_size'],
                    sequence_length=model_config['max_seq_length']
                )
                logger.info(f"Estimated memory usage: {memory_info.get('total_mb', 'unknown')} MB")
            
            # Training configuration
            train_config = self.config['training']
            optim_config = self.config.get('optimization', {})
            
            # Optimizer
            optimizer_name = optim_config.get('optimizer', 'adamw').lower()
            if optimizer_name == 'adamw':
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=train_config['learning_rate'],
                    betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.999)),
                    weight_decay=train_config.get('weight_decay', 0.01),
                    eps=optim_config.get('eps', 1e-8)
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
            # Learning rate scheduler
            scheduler_name = optim_config.get('scheduler', 'none').lower()
            if scheduler_name == 'cosine':
                total_steps = len(self.train_loader) * train_config['num_epochs']
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps,
                    eta_min=optim_config.get('min_lr', 1e-6)
                )
            elif scheduler_name == 'linear':
                total_steps = len(self.train_loader) * train_config['num_epochs']
                warmup_steps = train_config.get('warmup_steps', 0)
                
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / max(1, warmup_steps)
                    else:
                        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
                
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            
            # Setup checkpoint manager
            self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
            
            logger.info("Model setup complete")
            if self.scheduler:
                logger.info(f"Using learning rate scheduler: {scheduler_name}")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def save_checkpoint(self, metrics: dict = None, is_best: bool = False):
        """Save model checkpoint with comprehensive state."""
        try:
            if not self.model or not self.optimizer:
                logger.warning("Model or optimizer not initialized, skipping checkpoint save")
                return
            
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'metrics': metrics or {},
                'training_time': time.time() - self.start_time,
                'device': str(self.device)
            }
            
            # Add scheduler state if available
            if self.scheduler:
                checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.current_epoch}.pt")
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save latest checkpoint
            latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
            torch.save(checkpoint_data, latest_path)
            
            # Save best model if this is the best
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(checkpoint_data, best_path)
                logger.info(f"Best model saved: {best_path}")
            
            # Clean up old checkpoints (keep last 3)
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            logger.error(traceback.format_exc())
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Clean up old checkpoint files."""
        try:
            checkpoint_files = list(Path(self.checkpoint_dir).glob("checkpoint_epoch_*.pt"))
            if len(checkpoint_files) > keep_last:
                # Sort by epoch number
                checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                # Remove oldest files
                for file_to_remove in checkpoint_files[:-keep_last]:
                    file_to_remove.unlink()
                    logger.info(f"Removed old checkpoint: {file_to_remove}")
        except Exception as e:
            logger.warning(f"Error cleaning up checkpoints: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        try:
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"Checkpoint loaded: epoch {self.current_epoch}, step {self.global_step}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}m"
        hours = minutes / 60
        return f"{hours:.1f}h"
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with comprehensive error handling and monitoring."""
        self.model.train()
        total_loss = 0.0
        total_steps = len(self.train_loader)
        
        # Initialize progress bar
        pbar = tqdm(
            enumerate(self.train_loader),
            total=total_steps,
            desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}",
            unit="batch",
            leave=True,
            dynamic_ncols=True
        )
        
        start_time = time.time()
        successful_steps = 0
        
        for step, batch in pbar:
            step_start_time = time.time()
            
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                logits, loss = self.model(input_ids, labels)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {step}, skipping")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config['training'].get('gradient_clip_norm', 1.0)
                )
                
                # Check for exploding gradients
                if grad_norm > 10.0:
                    logger.warning(f"Large gradient norm detected: {grad_norm:.2f}")
                
                # Optimizer step
                self.optimizer.step()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                successful_steps += 1
                avg_loss = total_loss / successful_steps
                
                # Track performance
                step_time = time.time() - step_start_time
                self.step_times.append(step_time)
                if len(self.step_times) > 100:  # Keep last 100 times
                    self.step_times.pop(0)
                
                # Calculate progress
                elapsed = time.time() - start_time
                steps_per_sec = successful_steps / elapsed if elapsed > 0 else 0
                remaining_steps = total_steps - step - 1
                eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'speed': f"{steps_per_sec:.1f} steps/s",
                    'eta': self.format_time(eta)
                })
                
                # Log to file at regular intervals
                log_interval = self.config['training'].get('log_every', 50)
                if (step + 1) % log_interval == 0:
                    avg_step_time = sum(self.step_times[-10:]) / len(self.step_times[-10:])
                    logger.info(
                        f"Epoch [{epoch+1}/{self.config['training']['num_epochs']}] "
                        f"Step [{step+1}/{total_steps}] "
                        f"Loss: {avg_loss:.4f} "
                        f"LR: {current_lr:.2e} "
                        f"Speed: {steps_per_sec:.2f} steps/s "
                        f"Avg step time: {avg_step_time:.2f}s "
                        f"Grad norm: {grad_norm:.2f}"
                    )
                
                # Update global step
                self.global_step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"Out of memory at step {step}. Try reducing batch size.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    raise e
                else:
                    logger.error(f"Runtime error in training step {step}: {e}")
                    continue
            except Exception as e:
                logger.error(f"Unexpected error in training step {step}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Close progress bar
        pbar.close()
        
        # Calculate epoch metrics
        if successful_steps > 0:
            avg_epoch_loss = total_loss / successful_steps
        else:
            logger.error("No successful training steps in epoch!")
            avg_epoch_loss = float('inf')
        
        epoch_time = time.time() - start_time
        
        logger.info(
            f"Epoch {epoch + 1} completed in {self.format_time(epoch_time)} - "
            f"Average Loss: {avg_epoch_loss:.4f} "
            f"Successful steps: {successful_steps}/{total_steps}"
        )
        
        return avg_epoch_loss, epoch_time
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model with comprehensive error handling."""
        self.model.eval()
        total_loss = 0.0
        total_steps = len(self.val_loader)
        successful_steps = 0
        
        # Initialize progress bar
        pbar = tqdm(
            self.val_loader,
            total=total_steps,
            desc="Validating",
            unit="batch",
            leave=False,
            dynamic_ncols=True
        )
        
        start_time = time.time()
        
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                try:
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)
                    
                    logits, loss = self.model(input_ids, labels)
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN/Inf validation loss at step {step}, skipping")
                        continue
                    
                    total_loss += loss.item()
                    successful_steps += 1
                    
                    # Update progress
                    avg_loss = total_loss / successful_steps
                    pbar.set_postfix({'val_loss': f"{avg_loss:.4f}"})
                    
                except Exception as e:
                    logger.error(f"Error in validation step {step}: {e}")
                    continue
        
        # Close progress bar
        pbar.close()
        
        # Calculate validation metrics
        if successful_steps > 0:
            avg_loss = total_loss / successful_steps
        else:
            logger.error("No successful validation steps!")
            avg_loss = float('inf')
        
        val_time = time.time() - start_time
        
        logger.info(
            f"Validation completed in {self.format_time(val_time)} - "
            f"Loss: {avg_loss:.4f} "
            f"Successful steps: {successful_steps}/{total_steps}"
        )
        
        return avg_loss, val_time
    
    def train(self) -> None:
        """Main training loop with comprehensive error handling and recovery."""
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config_path}")
        
        try:
            # Setup data first, then model
            self.setup_data()
            self.setup_model()
            
            # Try to load latest checkpoint if exists
            latest_checkpoint = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
            if os.path.exists(latest_checkpoint):
                if self.load_checkpoint(latest_checkpoint):
                    logger.info(f"Resumed training from epoch {self.current_epoch}")
            
            # Training configuration
            epochs = self.config['training']['num_epochs']
            save_every = self.config['training'].get('save_every', 1000)
            eval_every = self.config['training'].get('eval_every', 500)
            
            logger.info(f"Training for {epochs} epochs")
            logger.info(f"Starting from epoch {self.current_epoch}")
            
            # Training loop
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                try:
                    # Train for one epoch
                    train_loss, train_time = self.train_epoch(epoch)
                    
                    # Validate
                    val_loss, val_time = self.validate()
                    
                    # Update metrics
                    if hasattr(self.metrics_tracker, 'update_metrics'):
                        self.metrics_tracker.update_metrics(
                            epoch=epoch,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            learning_rate=self.optimizer.param_groups[0]['lr']
                        )
                    
                    # Check if this is the best model
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        best_str = " (NEW BEST!)"
                    else:
                        best_str = ""
                    
                    # Save checkpoint
                    if (epoch + 1) % save_every == 0 or is_best or epoch == epochs - 1:
                        metrics = {
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'learning_rate': self.optimizer.param_groups[0]['lr']
                        }
                        self.save_checkpoint(metrics=metrics, is_best=is_best)
                    
                    # Log epoch summary
                    total_time = time.time() - self.start_time
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} completed - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}{best_str} "
                        f"Total time: {self.format_time(total_time)}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error in epoch {epoch + 1}: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Save emergency checkpoint
                    try:
                        self.save_checkpoint(is_best=False)
                        logger.info("Emergency checkpoint saved")
                    except:
                        pass
                    
                    # Continue to next epoch if possible
                    continue
            
            # Training completed
            total_training_time = time.time() - self.start_time
            logger.info(f"Training completed successfully in {self.format_time(total_training_time)}!")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            
            # Save final metrics
            try:
                if hasattr(self.metrics_tracker, 'save_history'):
                    metrics_path = "logs/training_history.json"
                    self.metrics_tracker.save_history(metrics_path)
                    logger.info(f"Training history saved to {metrics_path}")
            except Exception as e:
                logger.warning(f"Could not save training history: {e}")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            try:
                # Save current state
                metrics = {
                    'interrupted': True,
                    'total_time': time.time() - self.start_time
                }
                self.save_checkpoint(metrics=metrics)
                logger.info("Checkpoint saved before exit")
            except Exception as e:
                logger.error(f"Error saving checkpoint on interrupt: {e}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            logger.error(traceback.format_exc())
            
            # Save emergency checkpoint
            try:
                self.save_checkpoint(is_best=False)
                logger.info("Emergency checkpoint saved")
            except:
                pass
            
            raise


def main():
    """Main entry point with improved argument handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LLM model with comprehensive error handling")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--tokenizer-dir", type=str, default=None,
                       help="Path to tokenizer directory (overrides config)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Path to checkpoint directory (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only run validation, don't train")
    
    args = parser.parse_args()
    
    try:
        # Validate config file exists
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        
        # Initialize trainer
        trainer = Trainer(config_path=args.config)
        
        # Override paths if provided
        if args.tokenizer_dir:
            trainer.tokenizer_dir = args.tokenizer_dir
        if args.checkpoint_dir:
            trainer.checkpoint_dir = args.checkpoint_dir
            os.makedirs(trainer.checkpoint_dir, exist_ok=True)
        
        # Resume from specific checkpoint if provided
        if args.resume:
            if not trainer.load_checkpoint(args.resume):
                logger.error(f"Failed to load checkpoint: {args.resume}")
                sys.exit(1)
        
        # Start training or validation
        if args.validate_only:
            trainer.setup_model()
            trainer.setup_data()
            val_loss, val_time = trainer.validate()
            logger.info(f"Validation completed: Loss = {val_loss:.4f}")
        else:
            trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()