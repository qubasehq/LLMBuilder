"""
CLI commands for vocabulary management and synchronization.
"""

import click
from pathlib import Path
from loguru import logger


@click.group()
def vocab():
    """Vocabulary management commands."""
    pass


@vocab.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Config file path')
@click.option('--tokenizer', '-t', type=click.Path(exists=True),
              help='Tokenizer file path')
@click.option('--checkpoint', '-m', type=click.Path(exists=True),
              help='Model checkpoint path')
@click.option('--dry-run', is_flag=True,
              help='Show what would be changed without making changes')
def sync(config, tokenizer, checkpoint, dry_run):
    """Automatically synchronize vocabulary sizes across components."""
    
    logger.info("🔧 LLMBuilder Vocabulary Synchronization")
    logger.info("=" * 50)
    
    try:
        from llmbuilder.utils.vocab_sync import VocabSyncManager
        manager = VocabSyncManager(config)
        success = manager.auto_sync_vocab_sizes(
            tokenizer_path=tokenizer,
            config_path=config,
            checkpoint_path=checkpoint,
            dry_run=dry_run
        )
        
        if success:
            if dry_run:
                logger.info("✅ Dry run completed successfully")
            else:
                logger.info("✅ Vocabulary synchronization completed successfully")
        else:
            logger.error("❌ Vocabulary synchronization failed")
            raise click.ClickException("Synchronization failed")
            
    except Exception as e:
        logger.error(f"Error during vocab sync: {e}")
        raise click.ClickException(str(e))


@vocab.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Config file path')
@click.option('--tokenizer', '-t', type=click.Path(exists=True),
              help='Tokenizer file path')
@click.option('--checkpoint', '-m', type=click.Path(exists=True),
              help='Model checkpoint path')
def analyze(config, tokenizer, checkpoint):
    """Analyze vocabulary size consistency across components."""
    
    logger.info("🔍 Vocabulary Consistency Analysis")
    logger.info("=" * 40)
    
    try:
        from llmbuilder.utils.vocab_sync import VocabSyncManager
        manager = VocabSyncManager(config)
        analysis = manager.analyze_vocab_consistency(
            tokenizer_path=tokenizer,
            config_path=config,
            checkpoint_path=checkpoint
        )
        
        # Display results
        logger.info("\n📊 Component Analysis:")
        
        for component in ['tokenizer', 'config', 'checkpoint']:
            info = analysis[component]
            if info['path']:
                status = "✅" if info['vocab_size'] else "❌"
                vocab_str = f"{info['vocab_size']} tokens" if info['vocab_size'] else "Not found"
                logger.info(f"  {status} {component.title()}: {info['path']} -> {vocab_str}")
            else:
                logger.info(f"  ⚪ {component.title()}: Not found")
        
        logger.info(f"\n🎯 Recommended vocab size: {analysis['recommended_vocab_size']}")
        
        if analysis['consistent']:
            logger.info("✅ All vocabulary sizes are consistent!")
        else:
            logger.info("❌ Inconsistencies detected:")
            for issue in analysis['issues']:
                logger.info(f"  - {issue}")
            
            logger.info(f"\n💡 Run 'llmbuilder vocab sync' to fix these issues")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise click.ClickException(str(e))


@vocab.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('new_vocab_size', type=int)
@click.option('--output', '-o', type=click.Path(),
              help='Output path for resized checkpoint')
def resize(checkpoint_path, new_vocab_size, output):
    """Resize checkpoint embeddings to match new vocabulary size."""
    
    logger.info(f"🔧 Resizing Checkpoint Embeddings")
    logger.info(f"Input: {checkpoint_path}")
    logger.info(f"Target vocab size: {new_vocab_size}")
    
    if not output:
        output = str(Path(checkpoint_path).with_suffix('')) + '_resized.pt'
    
    logger.info(f"Output: {output}")
    
    try:
        from llmbuilder.utils.vocab_sync import VocabSyncManager
        manager = VocabSyncManager()
        success = manager.resize_checkpoint_embeddings(
            checkpoint_path=checkpoint_path,
            new_vocab_size=new_vocab_size,
            output_path=output
        )
        
        if success:
            logger.info("✅ Checkpoint resizing completed successfully")
            logger.info(f"Resized checkpoint saved to: {output}")
        else:
            logger.error("❌ Checkpoint resizing failed")
            raise click.ClickException("Resizing failed")
            
    except Exception as e:
        logger.error(f"Error during resize: {e}")
        raise click.ClickException(str(e))


@vocab.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Config file path')
@click.option('--tokenizer', '-t', type=click.Path(exists=True),
              help='Tokenizer file path')
def info(config, tokenizer):
    """Display vocabulary information for components."""
    
    logger.info("ℹ️  Vocabulary Information")
    logger.info("=" * 30)
    
    try:
        from llmbuilder.utils.vocab_sync import VocabSyncManager
        manager = VocabSyncManager(config)
        
        # Auto-discover files if not provided
        if not config and not tokenizer:
            files = manager.find_project_files()
            config = files.get('config')
            tokenizer = files.get('tokenizer')
        
        if tokenizer:
            vocab_size = manager.get_tokenizer_vocab_size(tokenizer)
            if vocab_size:
                logger.info(f"📝 Tokenizer: {tokenizer}")
                logger.info(f"   Vocabulary size: {vocab_size:,} tokens")
            else:
                logger.warning(f"⚠️  Could not read tokenizer: {tokenizer}")
        
        if config:
            vocab_size = manager.get_config_vocab_size(config)
            if vocab_size:
                logger.info(f"⚙️  Config: {config}")
                logger.info(f"   Vocabulary size: {vocab_size:,} tokens")
            else:
                logger.warning(f"⚠️  Could not read config: {config}")
        
        # Check for checkpoints
        files = manager.find_project_files()
        checkpoint = files.get('checkpoint')
        if checkpoint:
            vocab_size = manager.get_checkpoint_vocab_size(checkpoint)
            if vocab_size:
                logger.info(f"💾 Checkpoint: {checkpoint}")
                logger.info(f"   Vocabulary size: {vocab_size:,} tokens")
        
    except Exception as e:
        logger.error(f"Error getting vocab info: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    vocab()