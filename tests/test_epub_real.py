#!/usr/bin/env python3
"""
Test EPUB extraction with the real EPUB file.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llmbuilder.core.data.ingest import DocumentIngester, EPUBExtractor
from loguru import logger


def test_real_epub():
    """Test EPUB extraction with the real file."""
    epub_file = Path("tests/Dana Priest, William M. Arkin - Top Secret America.epub")
    
    if not epub_file.exists():
        logger.error(f"EPUB file not found: {epub_file}")
        return False
    
    logger.info(f"Testing EPUB extraction with: {epub_file}")
    logger.info(f"File size: {epub_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Test with EPUBExtractor directly
    extractor = EPUBExtractor()
    text = extractor.extract(epub_file)
    
    if text:
        logger.info(f"✅ EPUB extraction successful!")
        logger.info(f"Extracted text length: {len(text):,} characters")
        logger.info(f"Word count: {len(text.split()):,} words")
        
        # Show first 500 characters
        logger.info(f"First 500 characters:")
        logger.info(f"'{text[:500]}...'")
        
        # Show some statistics
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        logger.info(f"Total lines: {len(lines):,}")
        logger.info(f"Non-empty lines: {len(non_empty_lines):,}")
        
        return True
    else:
        logger.error("❌ EPUB extraction failed!")
        return False


def test_epub_with_ingester():
    """Test EPUB with the full DocumentIngester."""
    epub_file = Path("tests/Dana Priest, William M. Arkin - Top Secret America.epub")
    
    if not epub_file.exists():
        logger.error(f"EPUB file not found: {epub_file}")
        return False
    
    logger.info("Testing with DocumentIngester...")
    
    # Create output directory
    output_dir = Path("tests/epub_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize ingester
    ingester = DocumentIngester(
        output_dir=str(output_dir),
        max_file_size_mb=50  # Increase limit for larger EPUB
    )
    
    # Process the EPUB file
    metadata = ingester.ingest_file(epub_file)
    
    if metadata:
        logger.info(f"✅ DocumentIngester processing successful!")
        logger.info(f"Processing method: {metadata.processing_method}")
        logger.info(f"Character count: {metadata.character_count:,}")
        logger.info(f"Word count: {metadata.word_count:,}")
        logger.info(f"Processing time: {metadata.processing_time:.2f} seconds")
        logger.info(f"File type: {metadata.file_type}")
        
        # Check output file
        output_file = output_dir / f"{epub_file.stem}_cleaned.txt"
        if output_file.exists():
            logger.info(f"Output file created: {output_file}")
            logger.info(f"Output file size: {output_file.stat().st_size / 1024:.1f} KB")
            
            # Show first few lines of output
            with open(output_file, 'r', encoding='utf-8') as f:
                first_lines = f.read(1000)
                logger.info(f"First 1000 characters of output:")
                logger.info(f"'{first_lines}...'")
        
        return True
    else:
        logger.error("❌ DocumentIngester processing failed!")
        return False


def main():
    """Run EPUB tests."""
    logger.info("🔍 Testing EPUB extraction with real file...")
    
    success = True
    
    try:
        # Test 1: Direct extractor
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Direct EPUBExtractor")
        logger.info("="*50)
        if not test_real_epub():
            success = False
        
        # Test 2: Full ingester
        logger.info("\n" + "="*50)
        logger.info("TEST 2: DocumentIngester")
        logger.info("="*50)
        if not test_epub_with_ingester():
            success = False
        
        if success:
            logger.info("\n🎉 All EPUB tests passed!")
        else:
            logger.error("\n❌ Some EPUB tests failed!")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()