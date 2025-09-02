#!/usr/bin/env python3
"""
Script to diagnose and fix vocabulary size mismatch between model and tokenizer.

This script helps resolve the common issue where a trained model checkpoint
has a different vocabulary size than the current tokenizer.
"""

import torch
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sentencepiece as spm

def load_tokenizer_info(tokenizer_path: str) -> Dict[str, Any]:
    """Load tokenizer and get vocabulary information."""
    try:
        if tokenizer_path.endswith('.model'):
            # SentencePiece tokenizer
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            vocab_size = sp.get_piece_size()
            tokenizer_type = "SentencePiece"
        else:
            print(f"Unsupported tokenizer format: {tokenizer_path}")
            return {}
        
        return {
            'path': tokenizer_path,
            'type': tokenizer_type,
            'vocab_size': vocab_size,
            'processor': sp if tokenizer_path.endswith('.model') else None
        }
    except Exception as e:
        print(f"Error loading tokenizer {tokenizer_path}: {e}")
        return {}

def analyze_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Analyze model checkpoint to determine vocabulary size."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Look for embedding layers that indicate vocab size
        vocab_size = None
        embedding_shape = None
        lm_head_shape = None
        
        if 'token_embedding.weight' in checkpoint:
            embedding_shape = checkpoint['token_embedding.weight'].shape
            vocab_size = embedding_shape[0]
        elif 'model.token_embedding.weight' in checkpoint:
            embedding_shape = checkpoint['model.token_embedding.weight'].shape
            vocab_size = embedding_shape[0]
        
        if 'lm_head.weight' in checkpoint:
            lm_head_shape = checkpoint['lm_head.weight'].shape
        elif 'model.lm_head.weight' in checkpoint:
            lm_head_shape = checkpoint['model.lm_head.weight'].shape
        
        return {
            'path': checkpoint_path,
            'vocab_size': vocab_size,
            'embedding_shape': embedding_shape,
            'lm_head_shape': lm_head_shape,
            'keys': list(checkpoint.keys())[:10]  # First 10 keys for debugging
        }
    except Exception as e:
        print(f"Error analyzing checkpoint {checkpoint_path}: {e}")
        return {}

def find_matching_tokenizer(target_vocab_size: int) -> Optional[str]:
    """Find a tokenizer that matches the target vocabulary size."""
    tokenizer_paths = [
        'tokenizer/tokenizer.model',
        'tokenizer/sentencepiece.model', 
        'exports/tokenizer/sentencepiece.model',
        'exports/tokenizer/tokenizer.model'
    ]
    
    for path in tokenizer_paths:
        if Path(path).exists():
            info = load_tokenizer_info(path)
            if info.get('vocab_size') == target_vocab_size:
                return path
    
    return None

def resize_embeddings(checkpoint_path: str, new_vocab_size: int, output_path: str):
    """Resize model embeddings to match new vocabulary size."""
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Find embedding layers
        embedding_key = None
        lm_head_key = None
        
        for key in checkpoint.keys():
            if 'token_embedding.weight' in key:
                embedding_key = key
            elif 'lm_head.weight' in key:
                lm_head_key = key
        
        if not embedding_key or not lm_head_key:
            print("Could not find embedding layers in checkpoint")
            return False
        
        old_vocab_size = checkpoint[embedding_key].shape[0]
        embedding_dim = checkpoint[embedding_key].shape[1]
        
        print(f"Resizing from {old_vocab_size} to {new_vocab_size} tokens")
        
        if new_vocab_size > old_vocab_size:
            # Pad with zeros for new tokens
            pad_size = new_vocab_size - old_vocab_size
            
            # Resize token embedding
            old_embedding = checkpoint[embedding_key]
            new_embedding = torch.zeros(new_vocab_size, embedding_dim)
            new_embedding[:old_vocab_size] = old_embedding
            checkpoint[embedding_key] = new_embedding
            
            # Resize LM head
            old_lm_head = checkpoint[lm_head_key]
            new_lm_head = torch.zeros(new_vocab_size, embedding_dim)
            new_lm_head[:old_vocab_size] = old_lm_head
            checkpoint[lm_head_key] = new_lm_head
            
            print(f"Added {pad_size} new token embeddings (initialized to zero)")
            
        elif new_vocab_size < old_vocab_size:
            # Truncate embeddings
            checkpoint[embedding_key] = checkpoint[embedding_key][:new_vocab_size]
            checkpoint[lm_head_key] = checkpoint[lm_head_key][:new_vocab_size]
            
            print(f"Truncated {old_vocab_size - new_vocab_size} token embeddings")
        
        # Save resized checkpoint
        torch.save(checkpoint, output_path)
        print(f"Saved resized checkpoint to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error resizing embeddings: {e}")
        return False

def main():
    """Main function to diagnose and fix vocabulary mismatch."""
    
    print("🔍 Vocabulary Mismatch Diagnostic Tool")
    print("=" * 50)
    
    # Analyze current situation
    print("\n📊 Analyzing current setup...")
    
    # Check tokenizers
    tokenizer_paths = [
        'tokenizer/tokenizer.model',
        'tokenizer/sentencepiece.model',
        'exports/tokenizer/sentencepiece.model'
    ]
    
    tokenizers = {}
    for path in tokenizer_paths:
        if Path(path).exists():
            info = load_tokenizer_info(path)
            if info:
                tokenizers[path] = info
                print(f"✅ Tokenizer: {path} -> {info['vocab_size']} tokens")
    
    if not tokenizers:
        print("❌ No tokenizers found!")
        return
    
    # Check checkpoints
    checkpoint_paths = [
        'exports/checkpoints/latest_checkpoint.pt',
        'exports/checkpoints/best_model.pt'
    ]
    
    checkpoints = {}
    for path in checkpoint_paths:
        if Path(path).exists():
            info = analyze_checkpoint(path)
            if info:
                checkpoints[path] = info
                print(f"✅ Checkpoint: {path} -> {info['vocab_size']} tokens")
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    print(f"\n🔍 Diagnosis:")
    
    # Find mismatches
    mismatches = []
    for cp_path, cp_info in checkpoints.items():
        for tok_path, tok_info in tokenizers.items():
            if cp_info['vocab_size'] != tok_info['vocab_size']:
                mismatches.append({
                    'checkpoint': cp_path,
                    'checkpoint_vocab': cp_info['vocab_size'],
                    'tokenizer': tok_path,
                    'tokenizer_vocab': tok_info['vocab_size']
                })
    
    if not mismatches:
        print("✅ No vocabulary mismatches found!")
        return
    
    print(f"❌ Found {len(mismatches)} vocabulary mismatches:")
    for i, mismatch in enumerate(mismatches, 1):
        print(f"  {i}. Checkpoint {mismatch['checkpoint']} ({mismatch['checkpoint_vocab']} tokens)")
        print(f"     vs Tokenizer {mismatch['tokenizer']} ({mismatch['tokenizer_vocab']} tokens)")
    
    # Provide solutions
    print(f"\n💡 Recommended Solutions:")
    
    # Option 1: Find matching tokenizer
    for cp_path, cp_info in checkpoints.items():
        matching_tokenizer = find_matching_tokenizer(cp_info['vocab_size'])
        if matching_tokenizer:
            print(f"1. Use matching tokenizer for {cp_path}:")
            print(f"   python inference.py --model {cp_path} --tokenizer {matching_tokenizer}")
            continue
    
    # Option 2: Resize embeddings
    print(f"2. Resize model embeddings to match current tokenizer:")
    
    # Use the most recent tokenizer and checkpoint
    current_tokenizer = 'tokenizer/tokenizer.model'
    current_checkpoint = 'exports/checkpoints/latest_checkpoint.pt'
    
    if Path(current_tokenizer).exists() and Path(current_checkpoint).exists():
        tok_info = load_tokenizer_info(current_tokenizer)
        cp_info = analyze_checkpoint(current_checkpoint)
        
        if tok_info and cp_info and tok_info['vocab_size'] != cp_info['vocab_size']:
            print(f"   Resize {current_checkpoint} from {cp_info['vocab_size']} to {tok_info['vocab_size']} tokens")
            
            # Ask user if they want to proceed
            response = input(f"\n🤔 Resize embeddings automatically? (y/n): ").lower().strip()
            
            if response == 'y':
                output_path = current_checkpoint.replace('.pt', '_resized.pt')
                success = resize_embeddings(current_checkpoint, tok_info['vocab_size'], output_path)
                
                if success:
                    print(f"\n✅ Success! Use the resized model:")
                    print(f"   python inference.py --model {output_path} --tokenizer {current_tokenizer}")
                else:
                    print(f"\n❌ Failed to resize embeddings")
            else:
                print(f"\n⏭️  Skipping automatic resize")
    
    print(f"\n📝 Manual Steps:")
    print(f"1. Identify which tokenizer was used during training")
    print(f"2. Use that tokenizer for inference, OR")
    print(f"3. Retrain/fine-tune with the new tokenizer, OR") 
    print(f"4. Resize embeddings (may require additional fine-tuning)")

if __name__ == "__main__":
    main()