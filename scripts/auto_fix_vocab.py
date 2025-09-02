#!/usr/bin/env python3
"""
Automatic vocabulary size synchronization script.

This script automatically detects and fixes vocabulary size mismatches between:
- Tokenizer actual vocab size
- Config file vocab size  
- Model checkpoint vocab size

Usage:
    python scripts/auto_fix_vocab.py
    python scripts/auto_fix_vocab.py --config config.json --tokenizer tokenizer/tokenizer.model
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import sentencepiece as smp
import torch
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def get_tokenizer_vocab_size(tokenizer_path: str) -> Optional[int]:
    """Get vocabulary size from tokenizer."""
    try:
        if not os.path.exists(tokenizer_path):
            return None
            
        sp = smp.SentencePieceProcessor()
        sp.load(tokenizer_path)
        return sp.vocab_size()
    except Exception as e:
        logger.error(f"Failed to load tokenizer {tokenizer_path}: {e}")
        return None

def get_checkpoint_vocab_size(checkpoint_path: str) -> Optional[int]:
    """Get vocabulary size from model checkpoint."""
    try:
        if not os.path.exists(checkpoint_path):
            return None
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        for key in model_state_dict.keys():
            if 'token_embedding.weight' in key:
                return model_state_dict[key].shape[0]
            elif 'embeddings.word_embeddings.weight' in key:
                return model_state_dict[key].shape[0]
        
        return None
    except Exception as e:
        logger.error(f"Failed to analyze checkpoint {checkpoint_path}: {e}")
        return None

def update_config_vocab_size(config_path: str, vocab_size: int) -> bool:
    """Update config file with correct vocab size."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        old_vocab_size = config['model'].get('vocab_size', 0)
        config['model']['vocab_size'] = vocab_size
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Updated {config_path}: {old_vocab_size} -> {vocab_size}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update config {config_path}: {e}")
        return False

def find_files() -> Dict[str, str]:
    """Find relevant files in the project."""
    files = {}
    
    # Find config files
    config_candidates = ['config.json', 'config_gpu.json', 'config_cpu_small.json']
    for config in config_candidates:
        if os.path.exists(config):
            files['config'] = config
            break
    
    # Find tokenizer
    tokenizer_candidates = [
        'tokenizer/tokenizer.model',
        'exports/tokenizer/tokenizer.model',
        'tokenizer/sentencepiece.model'
    ]
    for tokenizer in tokenizer_candidates:
        if os.path.exists(tokenizer):
            files['tokenizer'] = tokenizer
            break
    
    # Find latest checkpoint
    checkpoint_candidates = [
        'exports/checkpoints/latest_checkpoint.pt',
        'exports/checkpoints/best_model.pt'
    ]
    for checkpoint in checkpoint_candidates:
        if os.path.exists(checkpoint):
            files['checkpoint'] = checkpoint
            break
    
    return files

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Auto-fix vocabulary size mismatches")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer file path")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    
    args = parser.parse_args()
    
    logger.info("🔧 Auto Vocabulary Size Fixer")
    logger.info("=" * 40)
    
    # Find files
    if args.config or args.tokenizer or args.checkpoint:
        files = {
            'config': args.config,
            'tokenizer': args.tokenizer, 
            'checkpoint': args.checkpoint
        }
    else:
        logger.info("🔍 Auto-detecting files...")
        files = find_files()
    
    # Check what we found
    logger.info("\n📁 Found files:")
    for file_type, path in files.items():
        if path and os.path.exists(path):
            logger.info(f"  ✅ {file_type}: {path}")
        else:
            logger.info(f"  ❌ {file_type}: Not found")
    
    if not files.get('tokenizer'):
        logger.error("❌ No tokenizer found! Cannot determine correct vocab size.")
        return 1
    
    # Get vocab sizes
    logger.info("\n📊 Analyzing vocabulary sizes...")
    
    tokenizer_vocab = get_tokenizer_vocab_size(files['tokenizer'])
    if not tokenizer_vocab:
        logger.error("❌ Failed to get tokenizer vocab size")
        return 1
    
    logger.info(f"  Tokenizer: {tokenizer_vocab} tokens")
    
    # Check config
    config_vocab = None
    if files.get('config') and os.path.exists(files['config']):
        try:
            with open(files['config'], 'r') as f:
                config = json.load(f)
            config_vocab = config['model'].get('vocab_size', 0)
            logger.info(f"  Config: {config_vocab} tokens")
        except Exception as e:
            logger.error(f"Failed to read config: {e}")
    
    # Check checkpoint
    checkpoint_vocab = None
    if files.get('checkpoint') and os.path.exists(files['checkpoint']):
        checkpoint_vocab = get_checkpoint_vocab_size(files['checkpoint'])
        if checkpoint_vocab:
            logger.info(f"  Checkpoint: {checkpoint_vocab} tokens")
    
    # Determine what needs fixing
    logger.info("\n🔍 Analysis:")
    
    issues = []
    
    if config_vocab and config_vocab != tokenizer_vocab:
        issues.append(f"Config vocab size ({config_vocab}) != Tokenizer ({tokenizer_vocab})")
    
    if checkpoint_vocab and checkpoint_vocab != tokenizer_vocab:
        issues.append(f"Checkpoint vocab size ({checkpoint_vocab}) != Tokenizer ({tokenizer_vocab})")
    
    if not issues:
        logger.info("✅ All vocabulary sizes are consistent!")
        return 0
    
    logger.info("❌ Found issues:")
    for issue in issues:
        logger.info(f"  - {issue}")
    
    # Fix issues
    logger.info(f"\n🔧 Fixes needed:")
    
    if config_vocab and config_vocab != tokenizer_vocab:
        if args.dry_run:
            logger.info(f"  [DRY RUN] Would update config {files['config']}: {config_vocab} -> {tokenizer_vocab}")
        else:
            logger.info(f"  Updating config {files['config']}: {config_vocab} -> {tokenizer_vocab}")
            success = update_config_vocab_size(files['config'], tokenizer_vocab)
            if success:
                logger.info("    ✅ Config updated successfully")
            else:
                logger.error("    ❌ Failed to update config")
    
    if checkpoint_vocab and checkpoint_vocab != tokenizer_vocab:
        logger.info(f"  ⚠️  Checkpoint vocab mismatch detected")
        logger.info(f"     Checkpoint has {checkpoint_vocab} tokens, tokenizer has {tokenizer_vocab}")
        logger.info(f"     Options:")
        logger.info(f"     1. Use inference.py (it will auto-resize embeddings)")
        logger.info(f"     2. Retrain model with current tokenizer")
        logger.info(f"     3. Use fix_vocab_mismatch.py for manual resize")
    
    if not args.dry_run:
        logger.info(f"\n✅ Vocabulary synchronization complete!")
        logger.info(f"   You can now run inference with consistent vocab sizes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())