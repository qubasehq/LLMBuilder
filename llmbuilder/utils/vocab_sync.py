"""
Vocabulary synchronization utilities for LLMBuilder.

This module provides automatic vocabulary size synchronization between:
- Tokenizer actual vocab size
- Config file vocab size  
- Model checkpoint vocab size

Automatically handles mismatches during training and inference.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from loguru import logger
from llmbuilder.utils.lazy_imports import torch, requires_torch

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    smp = None
    logger.warning("SentencePiece not available, some tokenizer features disabled")


class VocabSyncManager:
    """Manages vocabulary size synchronization across the pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize vocab sync manager."""
        self.config_path = config_path
        self.config = None
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.config_path = config_path
            return self.config
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}
    
    def get_tokenizer_vocab_size(self, tokenizer_path: str) -> Optional[int]:
        """Get vocabulary size from tokenizer file."""
        if not HAS_SENTENCEPIECE:
            logger.error("SentencePiece not available")
            return None
            
        try:
            if not os.path.exists(tokenizer_path):
                logger.warning(f"Tokenizer not found: {tokenizer_path}")
                return None
                
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            vocab_size = sp.vocab_size()
            logger.debug(f"Tokenizer {tokenizer_path}: {vocab_size} tokens")
            return vocab_size
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer {tokenizer_path}: {e}")
            return None
    
    def get_checkpoint_vocab_size(self, checkpoint_path: str) -> Optional[int]:
        """Get vocabulary size from model checkpoint."""
        try:
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return None
                
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Look for embedding layers
            for key in model_state_dict.keys():
                if 'token_embedding.weight' in key:
                    vocab_size = model_state_dict[key].shape[0]
                    logger.debug(f"Checkpoint {checkpoint_path}: {vocab_size} tokens")
                    return vocab_size
                elif 'embeddings.word_embeddings.weight' in key:
                    vocab_size = model_state_dict[key].shape[0]
                    logger.debug(f"Checkpoint {checkpoint_path}: {vocab_size} tokens")
                    return vocab_size
            
            logger.warning(f"No embedding layer found in {checkpoint_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze checkpoint {checkpoint_path}: {e}")
            return None
    
    def get_config_vocab_size(self, config_path: Optional[str] = None) -> Optional[int]:
        """Get vocabulary size from config file."""
        config_path = config_path or self.config_path
        if not config_path:
            return None
            
        try:
            if not self.config and config_path:
                self.load_config(config_path)
            
            if self.config and 'model' in self.config:
                vocab_size = self.config['model'].get('vocab_size', 0)
                logger.debug(f"Config {config_path}: {vocab_size} tokens")
                return vocab_size if vocab_size > 0 else None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get config vocab size: {e}")
            return None
    
    def update_config_vocab_size(self, vocab_size: int, config_path: Optional[str] = None) -> bool:
        """Update config file with new vocab size."""
        config_path = config_path or self.config_path
        if not config_path:
            logger.error("No config path specified")
            return False
            
        try:
            if not self.config:
                self.load_config(config_path)
            
            if not self.config:
                logger.error(f"Failed to load config {config_path}")
                return False
            
            old_vocab_size = self.config['model'].get('vocab_size', 0)
            self.config['model']['vocab_size'] = vocab_size
            
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"✓ Updated {config_path}: {old_vocab_size} -> {vocab_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config {config_path}: {e}")
            return False
    
    def find_project_files(self, base_path: str = ".") -> Dict[str, str]:
        """Auto-discover project files."""
        base_path = Path(base_path)
        files = {}
        
        # Find config files
        config_candidates = [
            'config.json',
            'config_gpu.json', 
            'config_cpu_small.json',
            'llmbuilder/templates/configs/default.json'
        ]
        
        for config in config_candidates:
            config_path = base_path / config
            if config_path.exists():
                files['config'] = str(config_path)
                break
        
        # Find tokenizer
        tokenizer_candidates = [
            'tokenizer/tokenizer.model',
            'exports/tokenizer/tokenizer.model',
            'tokenizer/sentencepiece.model',
            'exports/tokenizer/sentencepiece.model'
        ]
        
        for tokenizer in tokenizer_candidates:
            tokenizer_path = base_path / tokenizer
            if tokenizer_path.exists():
                files['tokenizer'] = str(tokenizer_path)
                break
        
        # Find latest checkpoint
        checkpoint_candidates = [
            'exports/checkpoints/latest_checkpoint.pt',
            'exports/checkpoints/best_model.pt',
            'checkpoints/latest_checkpoint.pt',
            'checkpoints/best_model.pt'
        ]
        
        for checkpoint in checkpoint_candidates:
            checkpoint_path = base_path / checkpoint
            if checkpoint_path.exists():
                files['checkpoint'] = str(checkpoint_path)
                break
        
        return files
    
    def analyze_vocab_consistency(self, 
                                tokenizer_path: Optional[str] = None,
                                config_path: Optional[str] = None,
                                checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze vocabulary size consistency across components."""
        
        # Auto-discover files if not provided
        if not any([tokenizer_path, config_path, checkpoint_path]):
            files = self.find_project_files()
            tokenizer_path = tokenizer_path or files.get('tokenizer')
            config_path = config_path or files.get('config') or self.config_path
            checkpoint_path = checkpoint_path or files.get('checkpoint')
        
        analysis = {
            'tokenizer': {'path': tokenizer_path, 'vocab_size': None},
            'config': {'path': config_path, 'vocab_size': None},
            'checkpoint': {'path': checkpoint_path, 'vocab_size': None},
            'consistent': True,
            'issues': [],
            'recommended_vocab_size': None
        }
        
        # Get vocab sizes
        if tokenizer_path:
            analysis['tokenizer']['vocab_size'] = self.get_tokenizer_vocab_size(tokenizer_path)
        
        if config_path:
            analysis['config']['vocab_size'] = self.get_config_vocab_size(config_path)
        
        if checkpoint_path:
            analysis['checkpoint']['vocab_size'] = self.get_checkpoint_vocab_size(checkpoint_path)
        
        # Determine reference vocab size (tokenizer is ground truth)
        if analysis['tokenizer']['vocab_size']:
            analysis['recommended_vocab_size'] = analysis['tokenizer']['vocab_size']
        elif analysis['checkpoint']['vocab_size']:
            analysis['recommended_vocab_size'] = analysis['checkpoint']['vocab_size']
        elif analysis['config']['vocab_size']:
            analysis['recommended_vocab_size'] = analysis['config']['vocab_size']
        
        # Check for inconsistencies
        ref_size = analysis['recommended_vocab_size']
        if ref_size:
            for component, info in analysis.items():
                if component in ['tokenizer', 'config', 'checkpoint']:
                    if info['vocab_size'] and info['vocab_size'] != ref_size:
                        analysis['consistent'] = False
                        analysis['issues'].append(
                            f"{component.title()} vocab size ({info['vocab_size']}) != "
                            f"recommended ({ref_size})"
                        )
        
        return analysis
    
    def auto_sync_vocab_sizes(self, 
                            tokenizer_path: Optional[str] = None,
                            config_path: Optional[str] = None,
                            checkpoint_path: Optional[str] = None,
                            dry_run: bool = False) -> bool:
        """Automatically synchronize vocabulary sizes."""
        
        logger.info("🔧 Auto-syncing vocabulary sizes...")
        
        # Analyze current state
        analysis = self.analyze_vocab_consistency(tokenizer_path, config_path, checkpoint_path)
        
        if analysis['consistent']:
            logger.info("✅ All vocabulary sizes are already consistent!")
            return True
        
        logger.info("❌ Vocabulary size inconsistencies detected:")
        for issue in analysis['issues']:
            logger.info(f"  - {issue}")
        
        recommended_size = analysis['recommended_vocab_size']
        if not recommended_size:
            logger.error("Cannot determine recommended vocabulary size")
            return False
        
        logger.info(f"🎯 Target vocabulary size: {recommended_size}")
        
        # Fix config if needed
        config_info = analysis['config']
        if (config_info['vocab_size'] and 
            config_info['vocab_size'] != recommended_size and 
            config_info['path']):
            
            if dry_run:
                logger.info(f"[DRY RUN] Would update config {config_info['path']}: "
                          f"{config_info['vocab_size']} -> {recommended_size}")
            else:
                logger.info(f"Updating config {config_info['path']}: "
                          f"{config_info['vocab_size']} -> {recommended_size}")
                success = self.update_config_vocab_size(recommended_size, config_info['path'])
                if not success:
                    logger.error("Failed to update config")
                    return False
        
        # Handle checkpoint mismatch
        checkpoint_info = analysis['checkpoint']
        if (checkpoint_info['vocab_size'] and 
            checkpoint_info['vocab_size'] != recommended_size):
            
            logger.warning(f"⚠️  Checkpoint vocab mismatch detected:")
            logger.warning(f"   Checkpoint: {checkpoint_info['vocab_size']} tokens")
            logger.warning(f"   Target: {recommended_size} tokens")
            logger.warning(f"   Solutions:")
            logger.warning(f"   1. Use inference.py (auto-resizes embeddings)")
            logger.warning(f"   2. Retrain model with current tokenizer")
            logger.warning(f"   3. Use resize_checkpoint_embeddings() method")
        
        if not dry_run:
            logger.info("✅ Vocabulary synchronization complete!")
        
        return True
    
    def resize_checkpoint_embeddings(self, 
                                   checkpoint_path: str,
                                   new_vocab_size: int,
                                   output_path: Optional[str] = None) -> bool:
        """Resize checkpoint embeddings to match new vocab size."""
        
        if not output_path:
            output_path = checkpoint_path.replace('.pt', '_resized.pt')
        
        try:
            logger.info(f"Resizing checkpoint embeddings: {checkpoint_path}")
            logger.info(f"Target vocab size: {new_vocab_size}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Find embedding keys
            embedding_key = None
            lm_head_key = None
            
            for key in model_state_dict.keys():
                if 'token_embedding.weight' in key:
                    embedding_key = key
                elif 'lm_head.weight' in key:
                    lm_head_key = key
            
            if not embedding_key or not lm_head_key:
                logger.error("Could not find embedding layers in checkpoint")
                return False
            
            old_token_emb = model_state_dict[embedding_key]
            old_lm_head = model_state_dict[lm_head_key]
            old_vocab_size = old_token_emb.shape[0]
            embedding_dim = old_token_emb.shape[1]
            
            logger.info(f"Resizing from {old_vocab_size} to {new_vocab_size} tokens")
            
            if new_vocab_size > old_vocab_size:
                # Expand embeddings
                pad_size = new_vocab_size - old_vocab_size
                
                new_token_emb = torch.zeros(new_vocab_size, embedding_dim)
                new_token_emb[:old_vocab_size] = old_token_emb
                model_state_dict[embedding_key] = new_token_emb
                
                new_lm_head = torch.zeros(new_vocab_size, embedding_dim)
                new_lm_head[:old_vocab_size] = old_lm_head
                model_state_dict[lm_head_key] = new_lm_head
                
                logger.info(f"✓ Added {pad_size} new token embeddings (zero-initialized)")
                
            elif new_vocab_size < old_vocab_size:
                # Truncate embeddings
                model_state_dict[embedding_key] = old_token_emb[:new_vocab_size]
                model_state_dict[lm_head_key] = old_lm_head[:new_vocab_size]
                
                logger.info(f"✓ Truncated {old_vocab_size - new_vocab_size} token embeddings")
            
            # Save resized checkpoint
            torch.save(checkpoint, output_path)
            logger.info(f"✓ Saved resized checkpoint: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resize checkpoint embeddings: {e}")
            return False


# Convenience functions
def auto_sync_vocab(config_path: Optional[str] = None, 
                   tokenizer_path: Optional[str] = None,
                   checkpoint_path: Optional[str] = None,
                   dry_run: bool = False) -> bool:
    """Convenience function for automatic vocab synchronization."""
    manager = VocabSyncManager(config_path)
    return manager.auto_sync_vocab_sizes(tokenizer_path, config_path, checkpoint_path, dry_run)


def get_vocab_analysis(config_path: Optional[str] = None,
                      tokenizer_path: Optional[str] = None, 
                      checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to analyze vocab consistency."""
    manager = VocabSyncManager(config_path)
    return manager.analyze_vocab_consistency(tokenizer_path, config_path, checkpoint_path)


def sync_config_with_tokenizer(config_path: str, tokenizer_path: str) -> bool:
    """Sync config vocab size with tokenizer."""
    manager = VocabSyncManager(config_path)
    tokenizer_vocab = manager.get_tokenizer_vocab_size(tokenizer_path)
    
    if tokenizer_vocab:
        return manager.update_config_vocab_size(tokenizer_vocab, config_path)
    
    return False