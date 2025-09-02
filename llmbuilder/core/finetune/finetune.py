#!/usr/bin/env python3
"""
Fine-tuning script for pre-trained LLM models.
Allows loading a pre-trained checkpoint and fine-tuning on new data.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from llmbuilder.utils.lazy_imports import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

# Removed sys.path.append - using proper package imports

from llmbuilder.core.model.gpt_model import GPTModel
from llmbuilder.core.training.dataset import TextDataset
from llmbuilder.core.training.utils import (
    ConfigManager, CheckpointManager, DeviceManager, 
    MetricsTracker, setup_logging
)
from llmbuilder.core.training.train import Trainer


class FineTuner(Trainer):
    """Fine-tuning class that extends the base Trainer."""
    
    def __init__(self, config_path: str, pretrained_model_path: str):
        """
        Initialize fine-tuner.
        
        Args:
            config_path: Path to configuration file
            pretrained_model_path: Path to pre-trained model checkpoint
        """
        super().__init__(config_path)
        self.pretrained_model_path = pretrained_model_path
        
        # Override some training settings for fine-tuning
        self.config.training.learning_rate *= 0.1  # Lower learning rate for fine-tuning
        self.config.training.warmup_steps = min(500, self.config.training.warmup_steps)
        
        logger.info(f"Fine-tuning configuration:")
        logger.info(f"  - Base model: {pretrained_model_path}")
        logger.info(f"  - Learning rate: {self.config.training.learning_rate}")
        logger.info(f"  - Warmup steps: {self.config.training.warmup_steps}")
    
    def load_pretrained_model(self) -> None:
        """Load pre-trained model checkpoint."""
        logger.info(f"Loading pre-trained model from {self.pretrained_model_path}")
        
        if not os.path.exists(self.pretrained_model_path):
            raise FileNotFoundError(f"Pre-trained model not found: {self.pretrained_model_path}")
        
        try:
            checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.success("Successfully loaded pre-trained model weights")
            else:
                # Assume the checkpoint is just the model state dict
                self.model.load_state_dict(checkpoint)
                logger.success("Successfully loaded pre-trained model weights (direct state dict)")
            
            # Optionally load optimizer state for continued training
            if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Loaded optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")
            
            # Load training metadata if available
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch']
                logger.info(f"Resuming from epoch {self.start_epoch}")
            
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
                logger.info(f"Resuming from global step {self.global_step}")
                
        except Exception as e:
            logger.error(f"Failed to load pre-trained model: {e}")
            raise
    
    def setup_model(self) -> None:
        """Setup model and load pre-trained weights."""
        super().setup_model()
        self.load_pretrained_model()
    
    def finetune(self, 
                 train_data_path: str,
                 val_data_path: Optional[str] = None,
                 output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Fine-tune the model on new data.
        
        Args:
            train_data_path: Path to fine-tuning training data
            val_data_path: Path to validation data (optional)
            output_dir: Output directory for fine-tuned model
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting fine-tuning process")
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(self.config.paths.checkpoint_dir, "finetuned")
        os.makedirs(output_dir, exist_ok=True)
        
        # Override checkpoint directory for fine-tuning
        original_checkpoint_dir = self.config.paths.checkpoint_dir
        self.config.paths.checkpoint_dir = output_dir
        self.checkpoint_manager = CheckpointManager(output_dir)
        
        try:
            # Load fine-tuning datasets
            logger.info(f"Loading fine-tuning data from {train_data_path}")
            train_dataset = TextDataset(
                train_data_path,
                self.tokenizer,
                max_length=self.config.model.max_seq_length
            )
            
            val_dataset = None
            if val_data_path and os.path.exists(val_data_path):
                logger.info(f"Loading validation data from {val_data_path}")
                val_dataset = TextDataset(
                    val_data_path,
                    self.tokenizer,
                    max_length=self.config.model.max_seq_length
                )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=self.config.training.num_workers,
                pin_memory=self.config.training.pin_memory and self.device.type == 'cuda'
            )
            
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.training.batch_size,
                    shuffle=False,
                    num_workers=self.config.training.num_workers,
                    pin_memory=self.config.training.pin_memory and self.device.type == 'cuda'
                )
            
            logger.info(f"Fine-tuning dataset size: {len(train_dataset)}")
            if val_dataset:
                logger.info(f"Validation dataset size: {len(val_dataset)}")
            
            # Run training
            results = self._train_loop(train_loader, val_loader)
            
            # Save final fine-tuned model
            final_model_path = os.path.join(output_dir, "finetuned_model.pt")
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch,
                global_step=self.global_step,
                loss=results.get('final_loss', 0.0),
                filepath=final_model_path
            )
            
            logger.success(f"Fine-tuning completed! Model saved to {final_model_path}")
            return results
            
        finally:
            # Restore original checkpoint directory
            self.config.paths.checkpoint_dir = original_checkpoint_dir


def main():
    """Main fine-tuning function."""
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained LLM model")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--pretrained-model", type=str, required=True,
                       help="Path to pre-trained model checkpoint")
    parser.add_argument("--train-data", type=str, required=True,
                       help="Path to fine-tuning training data")
    parser.add_argument("--val-data", type=str, default=None,
                       help="Path to validation data")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for fine-tuned model")
    parser.add_argument("--tokenizer-dir", type=str, default="exports/tokenizer",
                       help="Path to tokenizer directory")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("Starting LLM Fine-tuning")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Pre-trained model: {args.pretrained_model}")
    logger.info(f"Training data: {args.train_data}")
    
    try:
        # Initialize fine-tuner
        finetuner = FineTuner(args.config, args.pretrained_model)
        
        # Load tokenizer
        finetuner.load_tokenizer(args.tokenizer_dir)
        
        # Setup model and training components
        finetuner.setup_model()
        finetuner.setup_training()
        
        # Run fine-tuning
        results = finetuner.finetune(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            output_dir=args.output_dir
        )
        
        logger.success("Fine-tuning completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise


if __name__ == "__main__":
    main()
