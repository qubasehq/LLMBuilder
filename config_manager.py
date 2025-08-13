#!/usr/bin/env python3
"""
Dynamic configuration manager that synchronizes YAML and JSON configs
Prevents vocabulary size mismatches automatically
"""

import json
import yaml
import os
from pathlib import Path

class DynamicConfigManager:
    """Manages dynamic config synchronization between YAML and JSON"""
    
    def __init__(self, yaml_path="training/config.yaml", json_path="config.json"):
        self.yaml_path = Path(yaml_path)
        self.json_path = Path(json_path)
        self.config = None
        
    def load_config(self):
        """Load configuration from YAML (primary source)"""
        if self.yaml_path.exists():
            with open(self.yaml_path) as f:
                self.config = yaml.safe_load(f)
        return self.config
    
    def sync_vocab_size(self, vocab_size):
        """Synchronize vocabulary size across all configs"""
        
        # Update YAML config
        if self.yaml_path.exists():
            with open(self.yaml_path) as f:
                config = yaml.safe_load(f)
            
            config['model']['vocab_size'] = vocab_size
            config['tokenizer']['vocab_size'] = vocab_size
            
            with open(self.yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"✓ Updated YAML config: vocab_size = {vocab_size}")
        
        # Update JSON config (if not gitignored)
        if self.json_path.exists():
            with open(self.json_path) as f:
                config = json.load(f)
            
            config['model']['vocab_size'] = vocab_size
            
            with open(self.json_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Updated JSON config: vocab_size = {vocab_size}")
        
        return True
    
    def get_vocab_size(self, tokenizer_path="tokenizer/tokenizer.model"):
        """Get actual vocabulary size from tokenizer"""
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(str(tokenizer_path))
            return sp.vocab_size()
        except Exception as e:
            print(f"⚠ Error getting vocab size: {e}")
            return None
    
    def ensure_consistency(self):
        """Ensure all configs have consistent vocabulary sizes"""
        
        vocab_size = self.get_vocab_size()
        if vocab_size is None:
            return False
            
        # Sync all configs
        self.sync_vocab_size(vocab_size)
        
        print(f"✓ All configs synchronized: vocab_size = {vocab_size}")
        return True

if __name__ == "__main__":
    manager = DynamicConfigManager()
    manager.ensure_consistency()
