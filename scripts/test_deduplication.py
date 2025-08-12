#!/usr/bin/env python3
"""
CLI script to test deduplication functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dedup import DeduplicationPipeline
import time

def main():
    print("🔍 Testing DeduplicationPipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "output"
        
        print(f"Test directory: {test_dir}")
        
        # Create test files
        content1 = "The quick brown fox jumps over the lazy dog. This is a test document."
        content2 = "The quick brown fox jumps over the lazy dog. This is a test document."  # Exact duplicate
        content3 = "A fast brown fox leaps over a sleepy dog. This is a test document."  # Similar content
        content4 = "Machine learning is transforming the world of technology."  # Different content
        
        files = []
        contents = [content1, content2, content3, content4]
        names = ["original.txt", "duplicate.txt", "similar.txt", "different.txt"]
        
        for name, content in zip(names, contents):
            file_path = test_dir / name
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            files.append(str(file_path))
        
        print(f"Created {len(files)} test files")
        
        # Test hash-only deduplication
        print("\n--- Hash-Only Deduplication ---")
        pipeline_hash = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False
        )
        
        start_time = time.time()
        stats_hash = pipeline_hash.process_files(files, str(output_dir / "hash"))
        hash_time = time.time() - start_time
        
        print(f"Hash deduplication results:")
        print(f"  Input files: {stats_hash.total_documents}")
        print(f"  Output files: {stats_hash.final_document_count}")
        print(f"  Exact duplicates removed: {stats_hash.exact_duplicates_removed}")
        print(f"  Processing time: {hash_time:.3f}s")
        
        # Test combined deduplication (if embeddings available)
        try:
            print("\n--- Combined Hash + Embedding Deduplication ---")
            pipeline_combined = DeduplicationPipeline(
                use_hash_dedup=True,
                use_embedding_dedup=True,
                embedding_config={'similarity_threshold': 0.7}
            )
            
            start_time = time.time()
            stats_combined = pipeline_combined.process_files(files, str(output_dir / "combined"))
            combined_time = time.time() - start_time
            
            print(f"Combined deduplication results:")
            print(f"  Input files: {stats_combined.total_documents}")
            print(f"  Output files: {stats_combined.final_document_count}")
            print(f"  Exact duplicates removed: {stats_combined.exact_duplicates_removed}")
            print(f"  Near duplicates removed: {stats_combined.near_duplicates_removed}")
            print(f"  Total removed: {stats_combined.exact_duplicates_removed + stats_combined.near_duplicates_removed}")
            print(f"  Processing time: {combined_time:.3f}s")
            
        except Exception as e:
            print(f"Combined deduplication not available: {e}")
        
        print("\n✅ DeduplicationPipeline test completed successfully!")
        print(f"Test files created in: {test_dir}")

if __name__ == "__main__":
    main()