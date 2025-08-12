#!/usr/bin/env python3
"""
Simple test for DeduplicationPipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dedup import DeduplicationPipeline
from loguru import logger

def main():
    logger.info("Testing DeduplicationPipeline import...")
    
    # Test basic initialization
    pipeline = DeduplicationPipeline(
        use_hash_dedup=True,
        use_embedding_dedup=False
    )
    
    logger.info("✅ DeduplicationPipeline initialized successfully!")
    return True

if __name__ == "__main__":
    main()