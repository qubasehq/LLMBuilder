"""
Main training script for LLM training.
Optimized for CPU execution with comprehensive logging and checkpointing.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.gpt_model import GPTModel
from training.dataset import TextDataset, MultiFileDataset, create_dataloader, split_dataset
from training.utils import (
    ConfigManager, CheckpointManager, DeviceManager, MetricsTracker,
    setup_logging, count_parameters, format_time
)


class Trainer:
    """Main trainer class for LLM training."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        # Load configuration
        self.config = ConfigManager.load_config(config_path)
        
        # Setup logging
        setup_logging(log_dir="logs", level="INFO")
        
        # Initialize device and optimizations
        self.device = DeviceManager.get_device(
            prefer_cpu=self.config['train'].get('device', 'cpu') == 'cpu'
        )
        
        if self.device.type == 'cpu':
            DeviceManager.optimize_for_cpu()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.checkpoint_manager = None
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info("Trainer initialized successfully")
    
    def setup_data(self):
        """Setup datasets and data loaders."""
        logger.info("Setting up datasets...")
        
        # Check for tokenized data
        data_dir = Path("data/tokens")
        if not data_dir.exists() or not list(data_dir.glob("*.pt")):
            raise FileNotFoundError(
                "No tokenized data found. Please run preprocessing and tokenizer training first."
            )
        
        try:
            # Create dataset
            if len(list(data_dir.glob("*.pt"))) == 1:
                # Single file dataset
                data_file = list(data_dir.glob("*.pt"))[0]
                dataset = TextDataset(
                    data_path=data_file,
                    block_size=self.config['model']['block_size']
                )
            else:
                # Multi-file dataset
                dataset = MultiFileDataset(
                    data_dir=data_dir,
                    block_size=self.config['model']['block_size']
                )
            
            # Split into train/validation
            train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.9)
            
            # Create data loaders
            self.train_loader = create_dataloader(
                train_dataset,
                batch_size=self.config['train']['batch_size'],
                shuffle=True,
                num_workers=0  # CPU training works better with 0 workers
            )
            
            self.val_loader = create_dataloader(
                val_dataset,
                batch_size=self.config['train']['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            # Update vocab size in config if needed
            vocab_size = dataset.get_vocab_size()
            if self.config['model']['vocab_size'] != vocab_size:
                logger.warning(f"Config vocab_size ({self.config['model']['vocab_size']}) != data vocab_size ({vocab_size})")
                logger.info(f"Updating config vocab_size to {vocab_size}")
                self.config['model']['vocab_size'] = vocab_size
            
            logger.info(f"Data setup complete: {len(train_dataset):,} train, {len(val_dataset):,} val samples")
            
        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            raise
    
    def setup_model(self):
        """Initialize model and optimizer."""
        logger.info("Setting up model...")
        
        try:
            # Create model
            model_config = self.config['model']
            self.model = GPTModel(
                vocab_size=model_config['vocab_size'],
                n_layer=model_config['n_layer'],
                n_head=model_config['n_head'],
                n_embd=model_config['n_embd'],
                block_size=model_config['block_size'],
                dropout=model_config['dropout']
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Log model info
            n_params = count_parameters(self.model)
            logger.info(f"Model has {n_params:,} trainable parameters")
            
            # Estimate memory usage
            memory_info = self.model.estimate_memory_usage(
                batch_size=self.config['train']['batch_size'],
                sequence_length=model_config['block_size']
            )
            logger.info(f"Estimated memory usage: {memory_info['total_mb']:.1f} MB")
            
            # Setup optimizer
            train_config = self.config['train']
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                betas=(0.9, 0.95),
                weight_decay=0.1
            )
            
            # Setup checkpoint manager
            self.checkpoint_manager = CheckpointManager("exports/checkpoints")
            
            logger.info("Model setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits, loss = self.model(input_ids, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Log progress
                if batch_idx % self.config['train']['log_interval'] == 0:
                    avg_loss = total_loss / num_batches
                    elapsed = time.time() - start_time
                    tokens_per_sec = (
                        batch_idx * self.config['train']['batch_size'] * 
                        self.config['model']['block_size'] / elapsed
                    )
                    
                    logger.info(
                        f"Epoch {self.current_epoch}, Step {batch_idx}/{len(self.train_loader)}: "
                        f"Loss {avg_loss:.4f}, Tokens/sec {tokens_per_sec:.0f}"
                    )
                
            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        return {
            'train_loss': avg_loss,
            'epoch_time': epoch_time,
            'tokens_per_sec': len(self.train_loader) * self.config['train']['batch_size'] * 
                            self.config['model']['block_size'] / epoch_time
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits, loss = self.model(input_ids, labels)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation step: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'perplexity': perplexity
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        try:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                loss=metrics.get('val_loss', float('inf')),
                config=self.config,
                metrics=metrics
            )
            
            if is_best:
                # Save best model separately
                best_path = Path("exports/checkpoints/best_model.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'metrics': metrics,
                    'epoch': self.current_epoch
                }, best_path)
                logger.info(f"Best model saved: {best_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        try:
            # Setup
            self.setup_data()
            self.setup_model()
            
            # Training loop
            train_config = self.config['train']
            max_epochs = train_config.get('max_epochs', train_config['max_iters'] // len(self.train_loader))
            
            logger.info(f"Training for {max_epochs} epochs")
            
            for epoch in range(max_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validate
                if epoch % train_config.get('eval_interval', 10) == 0:
                    val_metrics = self.validate()
                    
                    # Combine metrics
                    all_metrics = {**train_metrics, **val_metrics}
                    
                    # Update metrics tracker
                    self.metrics_tracker.update(**all_metrics)
                    self.metrics_tracker.log_epoch(epoch)
                    
                    # Check if best model
                    is_best = val_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['val_loss']
                        logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                    
                    # Save checkpoint
                    self.save_checkpoint(all_metrics, is_best)
                    
                else:
                    # Just update train metrics
                    self.metrics_tracker.update(**train_metrics)
                    self.metrics_tracker.log_epoch(epoch)
            
            # Save final metrics
            self.metrics_tracker.save_history("logs/training_history.json")
            
            logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(self.metrics_tracker.metrics)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main entry point."""
    try:
        # Initialize trainer
        trainer = Trainer(config_path="training/config.yaml")
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

