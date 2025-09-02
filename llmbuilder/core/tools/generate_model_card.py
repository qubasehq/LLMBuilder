#!/usr/bin/env python3
"""
Model card generator for LLMBuilder trained models.
Automatically generates model documentation from training metadata.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from llmbuilder.utils.lazy_imports import torch

# Removed sys.path.append - using proper package imports


class ModelCardGenerator:
    """Generate model cards from training metadata and model checkpoints."""
    
    def __init__(self, template_path: Optional[str] = None):
        """Initialize model card generator.
        
        Args:
            template_path: Path to model card template file
        """
        self.project_root = Path(__file__).parent.parent
        self.template_path = template_path or (self.project_root / "templates" / "model_card_template.md")
        self.template_content = self._load_template()
    
    def _load_template(self) -> str:
        """Load model card template."""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
    
    def extract_model_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """Extract metadata from model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            
        Returns:
            Dictionary containing model metadata
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        metadata = {
            'checkpoint_path': str(checkpoint_path),
            'file_size': checkpoint_path.stat().st_size,
            'creation_date': datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat(),
        }
        
        # Extract model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            metadata.update({
                'vocab_size': config.get('vocab_size', 'Unknown'),
                'embedding_dim': config.get('embedding_dim', config.get('hidden_size', 'Unknown')),
                'num_layers': config.get('num_layers', config.get('n_layer', 'Unknown')),
                'num_heads': config.get('num_heads', config.get('n_head', 'Unknown')),
                'hidden_dim': config.get('hidden_dim', config.get('intermediate_size', 'Unknown')),
                'max_seq_length': config.get('max_seq_length', config.get('block_size', 'Unknown')),
                'architecture': config.get('architecture', 'GPT-2'),
            })
        
        # Extract training statistics
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            metadata.update({
                'total_steps': stats.get('step', stats.get('steps', 'Unknown')),
                'final_train_loss': stats.get('train_loss', stats.get('loss', 'Unknown')),
                'final_val_loss': stats.get('val_loss', 'Unknown'),
                'training_duration': stats.get('training_time', 'Unknown'),
                'best_checkpoint_step': stats.get('best_step', 'Unknown'),
            })
        
        # Calculate model size
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
            metadata['model_size'] = total_params
            metadata['model_size_formatted'] = self._format_parameter_count(total_params)
        
        return metadata
    
    def extract_training_metadata(self, training_log_path: Optional[str] = None) -> Dict[str, Any]:
        """Extract metadata from training logs.
        
        Args:
            training_log_path: Path to training log file
            
        Returns:
            Dictionary containing training metadata
        """
        metadata = {}
        
        # Try to find training logs
        if training_log_path:
            log_path = Path(training_log_path)
        else:
            # Look for common log locations
            possible_paths = [
                self.project_root / "logs" / "training_history.json",
                self.project_root / "logs" / "training.log",
                self.project_root / "exports" / "training_stats.json"
            ]
            log_path = None
            for path in possible_paths:
                if path.exists():
                    log_path = path
                    break
        
        if log_path and log_path.exists():
            try:
                if log_path.suffix == '.json':
                    with open(log_path, 'r') as f:
                        log_data = json.load(f)
                    
                    if isinstance(log_data, list) and log_data:
                        # Take the last entry
                        last_entry = log_data[-1]
                        metadata.update({
                            'final_train_loss': last_entry.get('train_loss'),
                            'final_val_loss': last_entry.get('val_loss'),
                            'train_perplexity': last_entry.get('train_perplexity'),
                            'val_perplexity': last_entry.get('val_perplexity'),
                        })
                
            except Exception as e:
                print(f"Warning: Could not parse training log {log_path}: {e}")
        
        return metadata
    
    def extract_dataset_metadata(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract metadata about the training dataset.
        
        Args:
            data_dir: Path to data directory
            
        Returns:
            Dictionary containing dataset metadata
        """
        metadata = {}
        
        if data_dir:
            data_path = Path(data_dir)
        else:
            data_path = self.project_root / "data"
        
        # Check for dataset statistics
        stats_files = [
            data_path / "ingestion_stats.json",
            data_path / "deduplication_stats.json",
            data_path / "dataset_stats.json"
        ]
        
        for stats_file in stats_files:
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    
                    if 'ingestion' in stats_file.name:
                        metadata.update({
                            'num_documents': stats.get('processed_count'),
                            'total_characters': stats.get('total_characters'),
                            'data_sources': stats.get('supported_formats'),
                        })
                    elif 'deduplication' in stats_file.name:
                        metadata.update({
                            'duplicates_removed': stats.get('exact_duplicates_removed', 0) + 
                                                stats.get('near_duplicates_removed', 0),
                            'deduplication_method': 'Hash + Embedding-based',
                        })
                
                except Exception as e:
                    print(f"Warning: Could not parse stats file {stats_file}: {e}")
        
        # Count files in data directories
        for subdir in ['raw', 'cleaned', 'deduped']:
            subdir_path = data_path / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.glob('*.txt')))
                metadata[f'{subdir}_file_count'] = file_count
        
        return metadata
    
    def _format_parameter_count(self, param_count: int) -> str:
        """Format parameter count in human-readable format."""
        if param_count >= 1e9:
            return f"{param_count / 1e9:.1f}B"
        elif param_count >= 1e6:
            return f"{param_count / 1e6:.1f}M"
        elif param_count >= 1e3:
            return f"{param_count / 1e3:.1f}K"
        else:
            return str(param_count)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"
    
    def generate_model_card(self, 
                          model_name: str,
                          checkpoint_path: str,
                          output_path: Optional[str] = None,
                          custom_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate model card from checkpoint and metadata.
        
        Args:
            model_name: Name of the model
            checkpoint_path: Path to model checkpoint
            output_path: Path to save model card (optional)
            custom_metadata: Additional custom metadata
            
        Returns:
            Generated model card content
        """
        # Collect metadata from various sources
        model_metadata = self.extract_model_metadata(checkpoint_path)
        training_metadata = self.extract_training_metadata()
        dataset_metadata = self.extract_dataset_metadata()
        
        # Combine all metadata
        all_metadata = {
            **model_metadata,
            **training_metadata,
            **dataset_metadata,
            **(custom_metadata or {})
        }
        
        # Add default values
        defaults = {
            'MODEL_NAME': model_name,
            'MODEL_VERSION': '1.0.0',
            'CREATION_DATE': datetime.now().strftime('%Y-%m-%d'),
            'AUTHOR': 'LLMBuilder User',
            'LICENSE': 'MIT',
            'BRIEF_DESCRIPTION': f'A {all_metadata.get("architecture", "transformer")} language model trained with LLMBuilder',
            'PYTORCH_VERSION': torch.__version__,
            'LAST_UPDATED': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Merge with collected metadata
        template_vars = {**defaults, **all_metadata}
        
        # Format specific fields
        if 'model_size' in template_vars:
            template_vars['MODEL_SIZE'] = template_vars['model_size_formatted']
        
        if 'file_size' in template_vars:
            template_vars['PYTORCH_SIZE'] = self._format_file_size(template_vars['file_size'])
        
        # Fill in template
        model_card_content = self.template_content
        
        for key, value in template_vars.items():
            placeholder = f"{{{key.upper()}}}"
            if placeholder in model_card_content:
                model_card_content = model_card_content.replace(placeholder, str(value))
        
        # Clean up any remaining placeholders
        import re
        remaining_placeholders = re.findall(r'\{[A-Z_]+\}', model_card_content)
        for placeholder in remaining_placeholders:
            model_card_content = model_card_content.replace(placeholder, 'N/A')
        
        # Save to file if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(model_card_content)
            
            print(f"Model card saved to: {output_file}")
        
        return model_card_content
    
    def generate_from_config(self, config_path: str) -> str:
        """Generate model card from configuration file.
        
        Args:
            config_path: Path to model card configuration JSON
            
        Returns:
            Generated model card content
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return self.generate_model_card(
            model_name=config['model_name'],
            checkpoint_path=config['checkpoint_path'],
            output_path=config.get('output_path'),
            custom_metadata=config.get('metadata', {})
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate model cards for LLMBuilder trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate model card from checkpoint
  python tools/generate_model_card.py \\
    --name "MyModel-v1.0" \\
    --checkpoint exports/checkpoints/best_model.pt \\
    --output models/MyModel/MODEL_CARD.md
  
  # Generate from configuration file
  python tools/generate_model_card.py --config model_card_config.json
  
  # Generate with custom metadata
  python tools/generate_model_card.py \\
    --name "MyModel" \\
    --checkpoint exports/checkpoints/best_model.pt \\
    --metadata '{"author": "John Doe", "license": "Apache-2.0"}'
        """
    )
    
    parser.add_argument('--name', type=str, help='Model name')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output path for model card')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--template', type=str, help='Custom template file path')
    parser.add_argument('--metadata', type=str, help='Custom metadata as JSON string')
    parser.add_argument('--print', action='store_true', help='Print model card to stdout')
    
    args = parser.parse_args()
    
    try:
        generator = ModelCardGenerator(template_path=args.template)
        
        if args.config:
            # Generate from configuration file
            model_card = generator.generate_from_config(args.config)
        elif args.name and args.checkpoint:
            # Generate from command line arguments
            custom_metadata = {}
            if args.metadata:
                custom_metadata = json.loads(args.metadata)
            
            model_card = generator.generate_model_card(
                model_name=args.name,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                custom_metadata=custom_metadata
            )
        else:
            parser.error("Either --config or both --name and --checkpoint are required")
        
        if args.print:
            print("\n" + "="*80)
            print("GENERATED MODEL CARD")
            print("="*80)
            print(model_card)
        
        print("✅ Model card generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating model card: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()