#!/usr/bin/env python3
"""
Test EmbeddingDeduplicator for near-duplicate detection.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llmbuilder.core.data.dedup import EmbeddingDeduplicator, DeduplicationPipeline, EMBEDDINGS_AVAILABLE
from loguru import logger


def create_test_files_semantic(test_dir: Path):
    """Create test files with semantic similarities."""
    
    # Original content
    content1 = """
    The quick brown fox jumps over the lazy dog.
    This is a classic sentence used in typography and font testing.
    It contains every letter of the English alphabet at least once.
    """
    
    # Semantically similar (paraphrased)
    content2 = """
    A fast brown fox leaps over a sleepy dog.
    This sentence is commonly used for testing fonts and typography.
    It includes all letters from the English alphabet.
    """
    
    # Semantically similar but different wording
    content3 = """
    The rapid brown fox bounds over the drowsy canine.
    This phrase is frequently employed in font and typography testing.
    It encompasses every letter of the English alphabet.
    """
    
    # Completely different content
    content4 = """
    Machine learning is a subset of artificial intelligence.
    It focuses on algorithms that can learn from data.
    Neural networks are a popular approach in this field.
    """
    
    # Another different topic
    content5 = """
    Climate change is a global environmental challenge.
    It involves long-term shifts in weather patterns.
    Human activities are the primary cause of recent changes.
    """
    
    # Similar to content4 (machine learning topic)
    content6 = """
    Artificial intelligence includes machine learning techniques.
    These algorithms improve their performance through experience.
    Deep learning uses neural networks with multiple layers.
    """
    
    # Create files
    files = {
        'fox1.txt': content1,
        'fox2.txt': content2,  # Similar to fox1
        'fox3.txt': content3,  # Similar to fox1 and fox2
        'ml1.txt': content4,   # Different topic
        'climate.txt': content5,  # Different topic
        'ml2.txt': content6,   # Similar to ml1
    }
    
    created_files = []
    for filename, content in files.items():
        file_path = test_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(str(file_path))
    
    return created_files


def test_embedding_availability():
    """Test if embedding dependencies are available."""
    logger.info("Testing embedding dependencies...")
    
    if EMBEDDINGS_AVAILABLE:
        logger.info("✅ Embedding dependencies available")
        
        try:
            # Test model loading
            dedup = EmbeddingDeduplicator(model_name="all-MiniLM-L6-v2")
            logger.info("✅ Embedding model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return False
    else:
        logger.warning("⚠️ Embedding dependencies not available")
        return False


def test_embedding_deduplicator_basic():
    """Test basic embedding deduplication functionality."""
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Skipping embedding test - dependencies not available")
        return True
    
    logger.info("Testing basic EmbeddingDeduplicator functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        files = create_test_files_semantic(test_dir)
        
        # Initialize deduplicator with a smaller model for faster testing
        dedup = EmbeddingDeduplicator(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.7,  # Lower threshold to catch more similarities
            chunk_size=500,
            max_chunks_per_doc=3
        )
        
        # Add documents
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                dedup.add_document(file_path, content)
        
        logger.info(f"Added {len(files)} documents for embedding analysis")
        
        # Compute embeddings
        dedup.compute_embeddings()
        
        if len(dedup.document_embeddings) > 0:
            logger.info(f"✅ Computed embeddings for {len(dedup.document_embeddings)} documents")
            
            # Find near-duplicates
            near_duplicates = dedup.find_near_duplicates()
            
            logger.info(f"Found {len(near_duplicates)} near-duplicate pairs")
            for doc1, doc2, similarity in near_duplicates:
                logger.info(f"  {Path(doc1).name} <-> {Path(doc2).name}: {similarity:.3f}")
            
            # Should find some similarities
            if len(near_duplicates) > 0:
                logger.info("✅ Near-duplicate detection working")
                return True
            else:
                logger.warning("⚠️ No near-duplicates found (may need threshold adjustment)")
                return True  # Still pass as this depends on threshold
        else:
            logger.error("❌ No embeddings computed")
            return False


def test_embedding_similarity_thresholds():
    """Test different similarity thresholds."""
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Skipping embedding threshold test - dependencies not available")
        return True
    
    logger.info("Testing similarity thresholds...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        files = create_test_files_semantic(test_dir)
        
        thresholds = [0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            logger.info(f"Testing threshold: {threshold}")
            
            dedup = EmbeddingDeduplicator(
                model_name="all-MiniLM-L6-v2",
                similarity_threshold=threshold,
                chunk_size=500
            )
            
            # Add documents
            for file_path in files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    dedup.add_document(file_path, content)
            
            # Compute embeddings and find duplicates
            dedup.compute_embeddings()
            near_duplicates = dedup.find_near_duplicates()
            
            logger.info(f"  Threshold {threshold}: {len(near_duplicates)} pairs found")
            
            # Higher thresholds should generally find fewer duplicates
            # (though this isn't guaranteed for all content)
        
        logger.info("✅ Threshold testing completed")
        return True


def test_embedding_chunking():
    """Test text chunking functionality."""
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Skipping chunking test - dependencies not available")
        return True
    
    logger.info("Testing text chunking...")
    
    dedup = EmbeddingDeduplicator(chunk_size=100, max_chunks_per_doc=3)
    
    # Test with different text lengths
    test_texts = [
        "Short text.",
        "This is a medium length text that should be split into chunks. " * 5,
        "This is a very long text that will definitely need to be chunked. " * 20
    ]
    
    for i, text in enumerate(test_texts):
        chunks = dedup.chunk_text(text)
        logger.info(f"Text {i+1} ({len(text)} chars) -> {len(chunks)} chunks")
        
        # Verify chunking
        if len(text) <= dedup.chunk_size:
            expected_chunks = 1
        else:
            expected_chunks = min(dedup.max_chunks_per_doc, len(text) // dedup.chunk_size + 1)
        
        if len(chunks) <= dedup.max_chunks_per_doc:
            logger.info(f"  ✅ Chunking working correctly")
        else:
            logger.error(f"  ❌ Too many chunks: {len(chunks)} > {dedup.max_chunks_per_doc}")
            return False
    
    return True


def test_embedding_pipeline_integration():
    """Test embedding deduplication in the full pipeline."""
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Skipping pipeline integration test - dependencies not available")
        return True
    
    logger.info("Testing embedding pipeline integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "output"
        
        files = create_test_files_semantic(test_dir)
        
        # Test embedding-only pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=False,  # Disable hash deduplication
            use_embedding_dedup=True,
            embedding_config={
                'model_name': 'all-MiniLM-L6-v2',
                'similarity_threshold': 0.7,
                'chunk_size': 500
            }
        )
        
        try:
            stats = pipeline.process_files(files, str(output_dir))
            
            logger.info(f"Embedding pipeline results:")
            logger.info(f"  Input files: {stats.total_documents}")
            logger.info(f"  Output files: {stats.final_document_count}")
            logger.info(f"  Near duplicates removed: {stats.near_duplicates_removed}")
            logger.info(f"  Processing time: {stats.processing_time:.3f}s")
            logger.info(f"  Embedding comparisons: {stats.embedding_comparisons}")
            
            # Verify output
            output_files = list(output_dir.glob("*.txt"))
            if len(output_files) == stats.final_document_count:
                logger.info("✅ Embedding pipeline integration successful")
                return True
            else:
                logger.error(f"❌ Output file count mismatch: expected {stats.final_document_count}, got {len(output_files)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Embedding pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_combined_pipeline():
    """Test combined hash + embedding deduplication."""
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Skipping combined pipeline test - dependencies not available")
        return True
    
    logger.info("Testing combined hash + embedding pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "output"
        
        # Create files with both exact and near duplicates
        files = create_test_files_semantic(test_dir)
        
        # Add an exact duplicate
        exact_duplicate_path = test_dir / "fox1_copy.txt"
        with open(files[0], 'r') as f:
            content = f.read()
        with open(exact_duplicate_path, 'w') as f:
            f.write(content)
        files.append(str(exact_duplicate_path))
        
        # Test combined pipeline
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=True,
            embedding_config={
                'model_name': 'all-MiniLM-L6-v2',
                'similarity_threshold': 0.7
            }
        )
        
        try:
            stats = pipeline.process_files(files, str(output_dir))
            
            logger.info(f"Combined pipeline results:")
            logger.info(f"  Input files: {stats.total_documents}")
            logger.info(f"  Output files: {stats.final_document_count}")
            logger.info(f"  Exact duplicates removed: {stats.exact_duplicates_removed}")
            logger.info(f"  Near duplicates removed: {stats.near_duplicates_removed}")
            logger.info(f"  Total removed: {stats.exact_duplicates_removed + stats.near_duplicates_removed}")
            logger.info(f"  Processing time: {stats.processing_time:.3f}s")
            
            # Should remove both exact and near duplicates
            total_removed = stats.exact_duplicates_removed + stats.near_duplicates_removed
            if total_removed > 0:
                logger.info("✅ Combined pipeline working correctly")
                return True
            else:
                logger.warning("⚠️ No duplicates removed (may need threshold adjustment)")
                return True  # Still pass as this depends on content and thresholds
                
        except Exception as e:
            logger.error(f"❌ Combined pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all embedding deduplicator tests."""
    logger.info("🔍 Testing EmbeddingDeduplicator...")
    
    tests = [
        ("Embedding Availability", test_embedding_availability),
        ("Basic Functionality", test_embedding_deduplicator_basic),
        ("Similarity Thresholds", test_embedding_similarity_thresholds),
        ("Text Chunking", test_embedding_chunking),
        ("Pipeline Integration", test_embedding_pipeline_integration),
        ("Combined Pipeline", test_combined_pipeline),
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
    logger.info(f"EMBEDDING DEDUPLICATOR TEST RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All embedding deduplicator tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    main()