#!/usr/bin/env python3
"""
Test HashDeduplicator for exact duplicate detection.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llmbuilder.core.data.dedup import HashDeduplicator, DeduplicationPipeline
from loguru import logger


def create_test_files(test_dir: Path):
    """Create test files with known duplicates."""
    
    # Original content
    content1 = """
    This is the first test document.
    It contains some sample text for testing deduplication.
    The content should be unique and identifiable.
    """
    
    # Exact duplicate
    content2 = content1
    
    # Near duplicate (different whitespace)
    content3 = """
    This is the first test document.
    It contains some sample text for testing deduplication.
    The content should be unique and identifiable.
    """
    
    # Different content
    content4 = """
    This is a completely different document.
    It has different text and should not be considered a duplicate.
    The content is unique and serves as a control.
    """
    
    # Case variation (should be duplicate if case normalization is enabled)
    content5 = """
    THIS IS THE FIRST TEST DOCUMENT.
    IT CONTAINS SOME SAMPLE TEXT FOR TESTING DEDUPLICATION.
    THE CONTENT SHOULD BE UNIQUE AND IDENTIFIABLE.
    """
    
    # Create files
    files = {
        'doc1.txt': content1,
        'doc2.txt': content2,  # Exact duplicate of doc1
        'doc3.txt': content3,  # Near duplicate (whitespace)
        'doc4.txt': content4,  # Unique content
        'doc5.txt': content5,  # Case variation
    }
    
    created_files = []
    for filename, content in files.items():
        file_path = test_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(str(file_path))
    
    return created_files


def test_hash_deduplicator_basic():
    """Test basic hash deduplication functionality."""
    logger.info("Testing basic HashDeduplicator functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        files = create_test_files(test_dir)
        
        # Initialize deduplicator
        dedup = HashDeduplicator(
            normalize_whitespace=True,
            normalize_case=False,  # Keep case sensitivity for this test
            min_length=10
        )
        
        # Add documents
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                dedup.add_document(file_path, content)
        
        # Find duplicates
        duplicate_groups = dedup.find_duplicates()
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        for i, group in enumerate(duplicate_groups):
            logger.info(f"Group {i + 1}: {[Path(f).name for f in group]}")
        
        # Verify results
        if len(duplicate_groups) >= 1:
            # Should find doc1.txt and doc2.txt as duplicates
            found_exact_duplicates = False
            for group in duplicate_groups:
                group_names = [Path(f).name for f in group]
                if 'doc1.txt' in group_names and 'doc2.txt' in group_names:
                    found_exact_duplicates = True
                    break
            
            if found_exact_duplicates:
                logger.info("✅ Found expected exact duplicates (doc1.txt, doc2.txt)")
                return True
            else:
                logger.error("❌ Did not find expected exact duplicates")
                return False
        else:
            logger.error("❌ No duplicate groups found")
            return False


def test_hash_deduplicator_normalization():
    """Test different normalization options."""
    logger.info("Testing HashDeduplicator normalization options...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        files = create_test_files(test_dir)
        
        # Test with case normalization enabled
        dedup_case_norm = HashDeduplicator(
            normalize_whitespace=True,
            normalize_case=True,  # Enable case normalization
            min_length=10
        )
        
        # Add documents
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                dedup_case_norm.add_document(file_path, content)
        
        # Find duplicates
        duplicate_groups = dedup_case_norm.find_duplicates()
        
        logger.info(f"With case normalization: {len(duplicate_groups)} duplicate groups")
        
        # Should find more duplicates with case normalization
        found_case_duplicates = False
        for group in duplicate_groups:
            group_names = [Path(f).name for f in group]
            if 'doc1.txt' in group_names and 'doc5.txt' in group_names:
                found_case_duplicates = True
                break
        
        if found_case_duplicates:
            logger.info("✅ Case normalization working correctly")
            return True
        else:
            logger.warning("⚠️ Case normalization may not be working as expected")
            return False


def test_hash_deduplicator_statistics():
    """Test statistics generation."""
    logger.info("Testing HashDeduplicator statistics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        files = create_test_files(test_dir)
        
        dedup = HashDeduplicator()
        
        # Add documents
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                dedup.add_document(file_path, content)
        
        # Find duplicates
        dedup.find_duplicates()
        
        # Get statistics
        stats = dedup.get_statistics()
        
        logger.info(f"Statistics: {stats}")
        
        # Verify statistics
        expected_fields = [
            'total_files', 'unique_content_hashes', 'duplicate_groups',
            'duplicate_files_removed', 'files_kept', 'deduplication_ratio'
        ]
        
        for field in expected_fields:
            if field not in stats:
                logger.error(f"❌ Missing statistics field: {field}")
                return False
        
        if stats['total_files'] == len(files):
            logger.info("✅ Statistics generation working correctly")
            return True
        else:
            logger.error(f"❌ Incorrect total_files: expected {len(files)}, got {stats['total_files']}")
            return False


def test_hash_deduplicator_file_selection():
    """Test file selection strategies."""
    logger.info("Testing file selection strategies...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        files = create_test_files(test_dir)
        
        dedup = HashDeduplicator()
        
        # Add documents
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                dedup.add_document(file_path, content)
        
        # Find duplicates
        dedup.find_duplicates()
        
        # Test different selection strategies
        strategies = ['first', 'shortest_name']
        
        for strategy in strategies:
            files_to_keep = dedup.get_files_to_keep(keep_strategy=strategy)
            logger.info(f"Strategy '{strategy}': keeping {len(files_to_keep)} files")
            
            # Should keep fewer files than total (due to duplicates)
            if len(files_to_keep) < len(files):
                logger.info(f"✅ Strategy '{strategy}' working correctly")
            else:
                logger.warning(f"⚠️ Strategy '{strategy}' may not be removing duplicates")
        
        return True


def test_deduplication_pipeline():
    """Test the complete deduplication pipeline."""
    logger.info("Testing DeduplicationPipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "output"
        
        files = create_test_files(test_dir)
        
        # Test hash-only pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False  # Disable embedding for this test
        )
        
        try:
            stats = pipeline.process_files(files, str(output_dir))
            
            logger.info(f"Pipeline results:")
            logger.info(f"  Input files: {stats.total_documents}")
            logger.info(f"  Output files: {stats.final_document_count}")
            logger.info(f"  Exact duplicates removed: {stats.exact_duplicates_removed}")
            logger.info(f"  Processing time: {stats.processing_time:.3f}s")
            
            # Verify output files exist
            output_files = list(output_dir.glob("*.txt"))
            if len(output_files) == stats.final_document_count:
                logger.info("✅ Pipeline processing successful")
                
                # Check statistics file
                stats_file = output_dir / "deduplication_stats.json"
                if stats_file.exists():
                    logger.info("✅ Statistics file created")
                    return True
                else:
                    logger.error("❌ Statistics file not created")
                    return False
            else:
                logger.error(f"❌ Expected {stats.final_document_count} output files, found {len(output_files)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False


def main():
    """Run all hash deduplicator tests."""
    logger.info("🔍 Testing HashDeduplicator...")
    
    tests = [
        ("Basic Functionality", test_hash_deduplicator_basic),
        ("Normalization Options", test_hash_deduplicator_normalization),
        ("Statistics Generation", test_hash_deduplicator_statistics),
        ("File Selection Strategies", test_hash_deduplicator_file_selection),
        ("Deduplication Pipeline", test_deduplication_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"HASH DEDUPLICATOR TEST RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All hash deduplicator tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    main()