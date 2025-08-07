# Quick debugging script to identify where the training is hanging
# Add this to your train.py or create a separate debug script

import torch
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.gpt_model import GPTModel
from training.dataset import TextDataset, create_dataloader, split_dataset
from training.utils import ConfigManager

def debug_training_hang(config_path="config.json"):
    """Debug where the training is hanging."""
    
    print("=== DEBUGGING TRAINING HANG ===")
    
    # 1. Load config
    print("1. Loading config...")
    config = ConfigManager.load_config(config_path)
    print(f"   ✓ Config loaded: {list(config.keys())}")
    
    # 2. Test data loading
    print("\n2. Testing data loading...")
    try:
        data_dir = Path("data/tokens")
        data_file = list(data_dir.glob("*.pt"))[0]
        print(f"   Loading from: {data_file}")
        
        dataset = TextDataset(
            data_path=data_file,
            block_size=config['model']['max_seq_length']  # Use correct key
        )
        print(f"   ✓ Dataset loaded: {len(dataset)} samples")
        
        # Test single sample
        sample = dataset[0]
        print(f"   ✓ Sample 0: {[(k, v.shape) for k, v in sample.items()]}")
        
        # Test dataloader creation
        train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.9)
        train_loader = create_dataloader(
            train_dataset,
            batch_size=2,  # Use smaller batch for testing
            shuffle=True,
            num_workers=0
        )
        print(f"   ✓ DataLoader created: {len(train_loader)} batches")
        
        # Test loading first batch
        print("   Testing first batch load...")
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            load_time = time.time() - start_time
            print(f"   ✓ Batch {i} loaded in {load_time:.2f}s: {[(k, v.shape) for k, v in batch.items()]}")
            if i >= 2:  # Test first 3 batches
                break
            start_time = time.time()
            
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Test model creation
    print("\n3. Testing model creation...")
    try:
        device = torch.device('cpu')
        model_config = config['model']
        
        # Create model with correct parameter names
        model = GPTModel(
            vocab_size=model_config['vocab_size'],
            n_layer=model_config['num_layers'],      # Fixed parameter name
            n_head=model_config['num_heads'],        # Fixed parameter name
            n_embd=model_config['embedding_dim'],    # Fixed parameter name
            block_size=model_config['max_seq_length'], # Fixed parameter name
            dropout=model_config['dropout']
        )
        model = model.to(device)
        print(f"   ✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Test single forward pass
    print("\n4. Testing model forward pass...")
    try:
        # Create dummy input
        batch_size = 2
        seq_length = config['model']['max_seq_length']
        vocab_size = config['model']['vocab_size']
        
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
        dummy_labels = dummy_input.clone()
        
        print(f"   Input shape: {dummy_input.shape}")
        
        # Test forward pass with timing
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            logits, loss = model(dummy_input, dummy_labels)
        
        forward_time = time.time() - start_time
        print(f"   ✓ Forward pass completed in {forward_time:.2f}s")
        print(f"   Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Test training step
    print("\n5. Testing single training step...")
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        
        # Get real batch from dataloader
        batch = next(iter(train_loader))
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        print(f"   Real batch shapes: input_ids={input_ids.shape}, labels={labels.shape}")
        
        # Training step with timing
        start_time = time.time()
        
        optimizer.zero_grad()
        logits, loss = model(input_ids, labels)
        loss.backward()
        optimizer.step()
        
        step_time = time.time() - start_time
        print(f"   ✓ Training step completed in {step_time:.2f}s")
        print(f"   Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Performance analysis
    print("\n6. Performance Analysis:")
    print(f"   - Data loading per batch: ~{load_time:.2f}s")
    print(f"   - Forward pass time: {forward_time:.2f}s")
    print(f"   - Full training step: {step_time:.2f}s")
    print(f"   - Estimated time per epoch: {step_time * len(train_loader) / 60:.1f} minutes")
    
    if step_time > 5.0:
        print("   ⚠️  WARNING: Training steps are very slow (>5s each)")
        print("   Consider reducing model size or batch size")
    
    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_training_hang()


# Alternative: Minimal training test
def minimal_training_test():
    """Test just the core training loop components."""
    print("=== MINIMAL TRAINING TEST ===")
    
    # Tiny model for testing
    model = GPTModel(
        vocab_size=1000,
        n_layer=2,      # Very small
        n_head=4,
        n_embd=128,
        block_size=64,  # Short sequences
        dropout=0.1
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Tiny batch
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    print("Testing 5 training steps...")
    model.train()
    
    for step in range(5):
        start = time.time()
        
        optimizer.zero_grad()
        logits, loss = model(input_ids, labels)
        loss.backward()
        optimizer.step()
        
        step_time = time.time() - start
        print(f"Step {step+1}: {step_time:.3f}s, Loss: {loss.item():.4f}")
    
    print("✓ Minimal test completed - basic training loop works")