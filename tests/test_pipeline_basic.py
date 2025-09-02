#!/usr/bin/env python3
"""
Basic test for DeduplicationPipeline functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llmbuilder.core.data.dedup import DeduplicationPipeline
from loguru import logger

def test_basic_pipeline():
    """Test basic pipeline functionality."""
    logger.info("Testing basic pipeline functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "output"
        
        # Create test files
        content1 = "This is the first test document with some content."
        content2 = "This is the first test document with some content."  # Exact duplicate
        content3 = "This is a different document with unique content."
        
        files = []
        for i, content in enumerate([content1, content2, content3], 1):
            file_path = test_dir / f"test_{i}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            files.append(str(file_path))
        
        # Test hash-only pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False
        )
        
        # Process files
        stats = pipeline.process_files(files, str(output_dir))
        
        logger.info(f"Results:")
        logger.info(f"  Input files: {stats.total_documents}")
        logger.info(f"  Output files: {stats.final_document_count}")
        logger.info(f"  Exact duplicates removed: {stats.exact_duplicates_removed}")
        logger.info(f"  Processing time: {stats.processing_time:.3f}s")
        
        # Should find 1 duplicate (2 identical files -> 1 kept)
        if stats.exact_duplicates_removed == 1:
            logger.info("✅ Pipeline working correctly!")
            return True
        else:
            logger.error(f"❌ Expected 1 duplicate removed, got {stats.exact_duplicates_removed}")
            return False

def main():
    logger.info("🔍 Testing DeduplicationPipeline...")
    
    try:
        if test_basic_pipeline():
            logger.info("🎉 Basic pipeline test passed!")
        else:
            logger.error("❌ Basic pipeline test failed!")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()