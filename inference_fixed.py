#!/usr/bin/env python3
"""
Fixed inference script that handles vocabulary size correctly.
"""

import torch
import json
import sys
from pathlib import Path
import sentencepiece as spm

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from model.gpt_model import GPTModel
from training.utils import setup_logging

def load_checkpoint_with_correct_vocab(checkpoint_path, config):
    """Load checkpoint and determine the correct vocabulary size."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get actual vocab size from checkpoint
    model_state = checkpoint['model_state_dict']
    actual_vocab_size = model_state['token_embedding.weight'].shape[0]
    
    print(f"Checkpoint vocabulary size: {actual_vocab_size}")
    
    # Update config with correct vocab size
    config['vocab_size'] = actual_vocab_size
    
    # Initialize model with correct vocab size
    model = GPTModel(config)
    
    # Load the state dict
    model.load_state_dict(model_state)
    
    return model, actual_vocab_size

def create_simple_tokenizer(vocab_size):
    """Create a simple tokenizer for the given vocabulary size."""
    
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.unk_token_id = 1
            self.bos_token_id = 2
            self.eos_token_id = 3
        
        def encode(self, text):
            # Simple character-level encoding for demo
            # In practice, you'd want the original tokenizer
            encoded = []
            for char in text:
                char_id = min(ord(char), self.vocab_size - 1)
                encoded.append(char_id)
            return encoded
        
        def decode(self, token_ids):
            # Simple character-level decoding
            text = ""
            for token_id in token_ids:
                if token_id < 128:  # ASCII range
                    text += chr(token_id)
                else:
                    text += "?"
            return text
    
    return SimpleTokenizer(vocab_size)

def main():
    """Main inference function."""
    
    setup_logging()
    
    print("🚀 Fixed LLM Inference")
    print("=" * 40)
    
    # Load config
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model with correct vocab size
    checkpoint_path = "exports/checkpoints/latest_checkpoint.pt"
    
    try:
        model, actual_vocab_size = load_checkpoint_with_correct_vocab(checkpoint_path, config)
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Vocabulary: {actual_vocab_size:,} tokens")
        
        # Create simple tokenizer (for demo purposes)
        print(f"\n⚠️  Using simple tokenizer (character-level)")
        print(f"   For proper inference, use the original training tokenizer")
        tokenizer = create_simple_tokenizer(actual_vocab_size)
        
        # Set model to evaluation mode
        model.eval()
        
        print(f"\n🎯 Model ready for inference!")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Vocab size: {actual_vocab_size}")
        
        # Interactive mode
        print(f"\n💬 Interactive mode (type 'quit' to exit):")
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                # Encode prompt
                input_ids = tokenizer.encode(prompt)
                input_tensor = torch.tensor([input_ids])
                
                print(f"Input tokens: {len(input_ids)}")
                
                # Generate
                with torch.no_grad():
                    # Simple generation (just one forward pass)
                    outputs = model(input_tensor)
                    logits = outputs.logits[0, -1, :]  # Last token logits
                    
                    # Get top token
                    next_token_id = torch.argmax(logits).item()
                    
                    # Decode
                    generated_text = tokenizer.decode([next_token_id])
                    
                    print(f"Generated: {generated_text}")
                    print(f"Next token ID: {next_token_id}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during generation: {e}")
        
        print(f"\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Solutions:")
        print(f"1. Find the original tokenizer used during training")
        print(f"2. Retrain with a consistent tokenizer")
        print(f"3. Use the LLMBuilder CLI: llmbuilder inference")

if __name__ == "__main__":
    main()