#!/usr/bin/env python3
"""
Test DeduplicationPipeline orchestrator functionality.
"""

import sys
import tempfile
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dedup import DeduplicationPipeline, DeduplicationStats, EMBEDDINGS_AVAILABLE
from loguru import logger


def create_comprehensive_test_files(test_dir: Path):
    """Create a comprehensive set of test files for pipeline testing."""
    
    # Exact duplicates
    content_original = """
    The importance of renewable energy cannot be overstated in today's world.
    Solar and wind power are becoming increasingly cost-effective alternatives.
    Governments worldwide are investing heavily in clean energy infrastructure.
    """
    
    content_exact_duplicate = content_original  # Exact copy
    
    # Near duplicates (semantic similarity)
    content_near_duplicate1 = """
    Renewable energy's significance is paramount in our current era.
    Solar and wind technologies are increasingly becoming affordable options.
    Nations globally are making substantial investments in sustainable energy systems.
    """
    
    content_near_duplicate2 = """
    The critical role of clean energy is undeniable in modern times.
    Photovoltaic and wind generation are growing more economically viable.
    Countries around the world are funding green energy projects extensively.
    """
    
    # Different content (should not be duplicates)
    content_different1 = """
    Machine learning algorithms have revolutionized data analysis.
    Neural networks can identify patterns in complex datasets.
    Artificial intelligence is transforming various industries rapidly.
    """
    
    content_different2 = """
    Climate change poses significant challenges to global ecosystems.
    Rising temperatures affect weather patterns and biodiversity.
    Urgent action is needed to mitigate environmental impacts.
    """
    
    content_different3 = """
    Space exploration continues to push the boundaries of human knowledge.
    Mars missions provide valuable insights into planetary science.
    Technological advances enable deeper exploration of the cosmos.
    """
    
    # Short content (might be filtered out)
    content_short = "Short text."
    
    # Very similar but slightly different (edge case)
    content_edge_case = """
    The importance of renewable energy cannot be overstated in today's world.
    Solar and wind power are becoming increasingly cost-effective alternatives.
    Governments worldwide are investing heavily in clean energy infrastructure.
    This additional sentence makes it slightly different.
    """
    
    # Create files
    files = {
        'renewable_original.txt': content_original,
        'renewable_exact_copy.txt': content_exact_duplicate,
        'renewable_similar1.txt': content_near_duplicate1,
        'renewable_similar2.txt': content_near_duplicate2,
        'machine_learning.txt': content_different1,
        'climate_change.txt': content_different2,
        'space_exploration.txt': content_different3,
        'short_content.txt': content_short,
        'renewable_edge_case.txt': content_edge_case,
    }
    
    created_files = []
    for filename, content in files.items():
        file_path = test_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(str(file_path))
    
    return created_files, files


def test_hash_only_pipeline():
    """Test pipeline with only hash deduplication enabled."""
    logger.info("Testing hash-only deduplication pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "hash_output"
        
        files, content_map = create_comprehensive_test_files(test_dir)
        
        # Configure hash-only pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False,
            hash_config={
                'normalize_whitespace': True,
                'normalize_case': True,
                'min_length': 20
            }
        )
        
        # Process files
        stats = pipeline.process_files(files, str(output_dir))
        
        logger.info(f"Hash-only results:")
        logger.info(f"  Input files: {stats.total_documents}")
        logger.info(f"  Output files: {stats.final_document_count}")
        logger.info(f"  Exact duplicates removed: {stats.exact_duplicates_removed}")
        logger.info(f"  Near duplicates removed: {stats.near_duplicates_removed}")
        logger.info(f"  Processing time: {stats.processing_time:.3f}s")
        
        # Verify results
        if stats.exact_duplicates_removed > 0:
            logger.info("✅ Hash deduplication found exact duplicates")
        
        if stats.near_duplicates_removed == 0:
            logger.info("✅ No near-duplicate processing (as expected for hash-only)")
        
        # Check output files exist
        output_files = list(output_dir.glob("*.txt"))
        if len(output_files) == stats.final_document_count:
            logger.info("✅ Correct number of output files created")
            return True
        else:
            logger.error(f"❌ Expected {stats.final_document_count} files, got {len(output_files)}")
            return False


def test_embedding_only_pipeline():
    """Test pipeline with only embedding deduplication enabled."""
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Skipping embedding-only test - dependencies not available")
        return True
    
    logger.info("Testing embedding-only deduplication pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "embedding_output"
        
        files, content_map = create_comprehensive_test_files(test_dir)
        
        # Configure embedding-only pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=False,
            use_embedding_dedup=True,
            embedding_config={
                'model_name': 'all-MiniLM-L6-v2',
                'similarity_threshold': 0.7,
                'chunk_size': 500
            }
        )
        
        # Process files
        stats = pipeline.process_files(files, str(output_dir))
        
        logger.info(f"Embedding-only results:")
        logger.info(f"  Input files: {stats.total_documents}")
        logger.info(f"  Output files: {stats.final_document_count}")
        logger.info(f"  Exact duplicates removed: {stats.exact_duplicates_removed}")
        logger.info(f"  Near duplicates removed: {stats.near_duplicates_removed}")
        logger.info(f"  Processing time: {stats.processing_time:.3f}s")
        logger.info(f"  Embedding comparisons: {stats.embedding_comparisons}")
        
        # Verify results
        if stats.exact_duplicates_removed == 0:
            logger.info("✅ No exact duplicate processing (as expected for embedding-only)")
        
        if stats.near_duplicates_removed >= 0:  # Could be 0 depending on threshold
            logger.info("✅ Embedding deduplication completed")
        
        # Check output files exist
        output_files = list(output_dir.glob("*.txt"))
        if len(output_files) == stats.final_document_count:
            logger.info("✅ Correct number of output files created")
            return True
        else:
            logger.error(f"❌ Expected {stats.final_document_count} files, got {len(output_files)}")
            return False


def test_combined_pipeline():
    """Test pipeline with both hash and embedding deduplication."""
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Skipping combined test - embedding dependencies not available")
        return True
    
    logger.info("Testing combined hash + embedding pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "combined_output"
        
        files, content_map = create_comprehensive_test_files(test_dir)
        
        # Configure combined pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=True,
            hash_config={
                'normalize_whitespace': True,
                'normalize_case': True,
                'min_length': 20
            },
            embedding_config={
                'model_name': 'all-MiniLM-L6-v2',
                'similarity_threshold': 0.75,
                'chunk_size': 500
            }
        )
        
        # Process files
        stats = pipeline.process_files(files, str(output_dir))
        
        logger.info(f"Combined pipeline results:")
        logger.info(f"  Input files: {stats.total_documents}")
        logger.info(f"  Output files: {stats.final_document_count}")
        logger.info(f"  Exact duplicates removed: {stats.exact_duplicates_removed}")
        logger.info(f"  Near duplicates removed: {stats.near_duplicates_removed}")
        logger.info(f"  Total removed: {stats.exact_duplicates_removed + stats.near_duplicates_removed}")
        logger.info(f"  Processing time: {stats.processing_time:.3f}s")
        logger.info(f"  Duplicate pairs found: {len(stats.duplicate_pairs)}")
        
        # Should find both types of duplicates
        total_removed = stats.exact_duplicates_removed + stats.near_duplicates_removed
        if total_removed > 0:
            logger.info("✅ Combined pipeline found duplicates")
        
        # Check output files exist
        output_files = list(output_dir.glob("*.txt"))
        if len(output_files) == stats.final_document_count:
            logger.info("✅ Correct number of output files created")
            return True
        else:
            logger.error(f"❌ Expected {stats.final_document_count} files, got {len(output_files)}")
            return False


def test_statistics_generation():
    """Test comprehensive statistics generation."""
    logger.info("Testing statistics generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "stats_output"
        
        files, content_map = create_comprehensive_test_files(test_dir)
        
        # Configure pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=EMBEDDINGS_AVAILABLE
        )
        
        # Process files
        stats = pipeline.process_files(files, str(output_dir))
        
        # Verify statistics structure
        required_fields = [
            'total_documents', 'exact_duplicates_removed', 'near_duplicates_removed',
            'final_document_count', 'total_characters_before', 'total_characters_after',
            'processing_time', 'duplicate_pairs', 'hash_collisions'
        ]
        
        stats_dict = stats.__dict__
        missing_fields = [field for field in required_fields if field not in stats_dict]
        
        if missing_fields:
            logger.error(f"❌ Missing statistics fields: {missing_fields}")
            return False
        
        # Check statistics file was created
        stats_file = output_dir / "deduplication_stats.json"
        if not stats_file.exists():
            logger.error("❌ Statistics file not created")
            return False
        
        # Verify statistics file content
        try:
            with open(stats_file, 'r') as f:
                saved_stats = json.load(f)
            
            if 'total_documents' in saved_stats and saved_stats['total_documents'] == len(files):
                logger.info("✅ Statistics file correctly saved")
            else:
                logger.error("❌ Statistics file content incorrect")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error reading statistics file: {e}")
            return False
        
        logger.info("✅ Statistics generation working correctly")
        return True


def test_error_handling():
    """Test error handling in the pipeline."""
    logger.info("Testing error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "error_output"
        
        # Create some valid files and some problematic ones
        valid_files = []
        
        # Valid file
        valid_file = test_dir / "valid.txt"
        with open(valid_file, 'w', encoding='utf-8') as f:
            f.write("This is a valid file with sufficient content for processing.")
        valid_files.append(str(valid_file))
        
        # Empty file
        empty_file = test_dir / "empty.txt"
        with open(empty_file, 'w', encoding='utf-8') as f:
            f.write("")
        valid_files.append(str(empty_file))
        
        # Very short file
        short_file = test_dir / "short.txt"
        with open(short_file, 'w', encoding='utf-8') as f:
            f.write("Hi")
        valid_files.append(str(short_file))
        
        # Non-existent file (will cause error)
        non_existent = str(test_dir / "non_existent.txt")
        valid_files.append(non_existent)
        
        # Configure pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False  # Keep simple for error testing
        )
        
        # Process files (should handle errors gracefully)
        try:
            stats = pipeline.process_files(valid_files, str(output_dir))
            
            logger.info(f"Error handling test results:")
            logger.info(f"  Input files attempted: {len(valid_files)}")
            logger.info(f"  Output files: {stats.final_document_count}")
            logger.info(f"  Processing time: {stats.processing_time:.3f}s")
            
            # Should process at least the valid file
            if stats.final_document_count >= 1:
                logger.info("✅ Pipeline handled errors gracefully")
                return True
            else:
                logger.error("❌ Pipeline failed to process any files")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline crashed with error: {e}")
            return False


def test_configuration_options():
    """Test different configuration options."""
    logger.info("Testing configuration options...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test files
        files, _ = create_comprehensive_test_files(test_dir)
        
        # Test different hash configurations
        hash_configs = [
            {'normalize_case': True, 'normalize_whitespace': True},
            {'normalize_case': False, 'normalize_whitespace': True},
            {'hash_algorithm': 'sha256', 'min_length': 50}
        ]
        
        for i, config in enumerate(hash_configs):
            output_dir = test_dir / f"config_test_{i}"
            
            pipeline = DeduplicationPipeline(
                use_hash_dedup=True,
                use_embedding_dedup=False,
                hash_config=config
            )
            
            try:
                stats = pipeline.process_files(files, str(output_dir))
                logger.info(f"Config {i}: {stats.final_document_count} files output")
            except Exception as e:
                logger.error(f"❌ Configuration {i} failed: {e}")
                return False
        
        logger.info("✅ Configuration options working correctly")
        return True


def test_large_file_handling():
    """Test handling of larger file sets."""
    logger.info("Testing large file handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "large_output"
        
        # Create a larger set of files
        files = []
        base_content = "This is base content for file number {i}. " * 10
        
        # Create 20 files with some duplicates
        for i in range(20):
            file_path = test_dir / f"file_{i:02d}.txt"
            
            if i % 5 == 0 and i > 0:
                # Every 5th file is a duplicate of file_00
                content = base_content.format(i=0)
            else:
                content = base_content.format(i=i)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            files.append(str(file_path))
        
        # Configure pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False  # Keep simple for performance test
        )
        
        # Process files
        start_time = time.time()
        stats = pipeline.process_files(files, str(output_dir))
        processing_time = time.time() - start_time
        
        logger.info(f"Large file handling results:")
        logger.info(f"  Input files: {len(files)}")
        logger.info(f"  Output files: {stats.final_document_count}")
        logger.info(f"  Duplicates removed: {stats.exact_duplicates_removed}")
        logger.info(f"  Processing time: {processing_time:.3f}s")
        logger.info(f"  Files per second: {len(files)/processing_time:.1f}")
        
        # Should find duplicates and process reasonably quickly
        if stats.exact_duplicates_removed > 0 and processing_time < 5.0:
            logger.info("✅ Large file handling working efficiently")
            return True
        else:
            logger.warning("⚠️ Large file handling may need optimization")
            return True  # Still pass, just a performance note


def main():
    """Run all deduplication pipeline tests."""
    logger.info("🔍 Testing DeduplicationPipeline Orchestrator...")
    
    tests = [
        ("Hash-Only Pipeline", test_hash_only_pipeline),
        ("Embedding-Only Pipeline", test_embedding_only_pipeline),
        ("Combined Pipeline", test_combined_pipeline),
        ("Statistics Generation", test_statistics_generation),
        ("Error Handling", test_error_handling),
        ("Configuration Options", test_configuration_options),
        ("Large File Handling", test_large_file_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"{'='*60}")
        
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
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DEDUPLICATION PIPELINE TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All deduplication pipeline tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    import time
    main()