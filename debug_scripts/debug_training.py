#!/usr/bin/env python3
"""
Quick debug script to identify the exact training failure point.
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test all required imports."""
    print("=== Testing Imports ===")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        import torch.nn as nn
        print("✓ torch.nn imported")
        
        from loguru import logger
        print("✓ loguru imported")
        
        from tqdm import tqdm
        print("✓ tqdm imported")
        
        # Test project imports
        from model.gpt_model import GPTModel
        print("✓ GPTModel imported")
        
        from training.dataset import TextDataset, create_dataloader, split_dataset
        print("✓ Dataset classes imported")
        
        from training.utils import ConfigManager, MetricsTracker, setup_logging, count_parameters
        print("✓ Training utilities imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n=== Testing Config Loading ===")
    try:
        from training.utils import ConfigManager
        config = ConfigManager.load_config("config.json")
        print(f"✓ Config loaded successfully")
        print(f"  - Keys: {list(config.keys())}")
        
        if 'training' in config:
            print(f"  - Training keys: {list(config['training'].keys())}")
        if 'model' in config:
            print(f"  - Model keys: {list(config['model'].keys())}")
            
        return config
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return None

def test_trainer_init():
    """Test trainer initialization."""
    print("\n=== Testing Trainer Initialization ===")
    try:
        # Import the exact same way as the training script
        sys.path.append(str(Path(__file__).parent))
        
        from training.train import Trainer
        trainer = Trainer("config.json")
        print("✓ Trainer initialized successfully")
        return trainer
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        traceback.print_exc()
        return None

def test_data_setup(trainer):
    """Test data setup."""
    print("\n=== Testing Data Setup ===")
    try:
        trainer.setup_data()
        print("✓ Data setup successful")
        print(f"  - Train batches: {len(trainer.train_loader)}")
        print(f"  - Val batches: {len(trainer.val_loader)}")
        return True
        
    except Exception as e:
        print(f"❌ Data setup failed: {e}")
        traceback.print_exc()
        return False

def test_model_setup(trainer):
    """Test model setup."""
    print("\n=== Testing Model Setup ===")
    try:
        trainer.setup_model()
        print("✓ Model setup successful")
        return True
        
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        traceback.print_exc()
        return False

def test_training_step(trainer):
    """Test a single training step."""
    print("\n=== Testing Single Training Step ===")
    try:
        # Get a batch
        batch = next(iter(trainer.train_loader))
        print(f"✓ Got batch: {[(k, v.shape) for k, v in batch.items()]}")
        
        # Test forward pass
        trainer.model.train()
        input_ids = batch['input_ids'].to(trainer.device)
        labels = batch['labels'].to(trainer.device)
        
        logits, loss = trainer.model(input_ids, labels)
        print(f"✓ Forward pass: loss = {loss.item():.4f}")
        
        # Test backward pass
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        print("✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive debug tests."""
    print("🔍 DEBUGGING TRAINING FAILURE")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ FAILURE: Import issues detected")
        return
    
    # Test 2: Config loading
    config = test_config_loading()
    if not config:
        print("\n❌ FAILURE: Config loading issues")
        return
    
    # Test 3: Trainer initialization
    trainer = test_trainer_init()
    if not trainer:
        print("\n❌ FAILURE: Trainer initialization issues")
        return
    
    # Test 4: Data setup
    if not test_data_setup(trainer):
        print("\n❌ FAILURE: Data setup issues")
        return
    
    # Test 5: Model setup
    if not test_model_setup(trainer):
        print("\n❌ FAILURE: Model setup issues")
        return
    
    # Test 6: Training step
    if not test_training_step(trainer):
        print("\n❌ FAILURE: Training step issues")
        return
    
    print("\n✅ ALL TESTS PASSED!")
    print("Training components are working correctly.")
    print("The issue might be in the main training loop or argument parsing.")

if __name__ == "__main__":
    main()