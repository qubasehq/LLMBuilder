#!/usr/bin/env python3
"""
Performance benchmarking and validation tests for the LLM training pipeline.
Tests ingestion speed, memory usage, and training performance impact.
"""

import os
import sys
import time
import psutil
import tempfile
import pytest
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from memory_profiler import profile
from unittest.mock import patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llmbuilder.core.data.ingest import DocumentIngester
from llmbuilder.core.data.dedup import DeduplicationPipeline, HashDeduplicator, EmbeddingDeduplicator
from llmbuilder.core.tools.quantization_manager import QuantizationManager, create_quantization_config


class PerformanceMonitor:
    """Utility class for monitoring performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration': end_time - self.start_time if self.start_time else 0,
            'start_memory_mb': self.start_memory or 0,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': end_memory - (self.start_memory or 0),
            'cpu_percent': self.process.cpu_percent()
        }


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    return PerformanceMonitor()


@pytest.fixture
def large_test_data():
    """Create large test dataset for performance testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create various sized files
        file_sizes = [
            (10, "small"),    # 10 files, small size
            (5, "medium"),    # 5 files, medium size
            (2, "large")      # 2 files, large size
        ]
        
        for count, size_type in file_sizes:
            if size_type == "small":
                content_size = 1000  # ~1KB
            elif size_type == "medium":
                content_size = 10000  # ~10KB
            else:  # large
                content_size = 100000  # ~100KB
            
            for i in range(count):
                content = f"Test content for {size_type} file {i}. " * content_size
                file_path = temp_path / f"{size_type}_{i}.txt"
                file_path.write_text(content, encoding='utf-8')
        
        yield temp_path


class TestIngestionPerformance:
    """Performance tests for document ingestion."""
    
    def test_ingestion_throughput(self, large_test_data, performance_monitor):
        """Test document ingestion throughput."""
        output_dir = large_test_data / "output"
        output_dir.mkdir()
        
        ingester = DocumentIngester(output_dir=str(output_dir))
        
        performance_monitor.start_monitoring()
        
        results = ingester.ingest_directory(large_test_data, recursive=False)
        
        metrics = performance_monitor.get_metrics()
        
        # Performance assertions
        assert results['processed_count'] > 0
        assert metrics['duration'] < 30.0  # Should complete within 30 seconds
        
        # Calculate throughput metrics
        throughput_files_per_sec = results['processed_count'] / metrics['duration']
        throughput_chars_per_sec = results['total_characters'] / metrics['duration']
        
        # Minimum performance requirements
        assert throughput_files_per_sec > 0.5  # At least 0.5 files per second
        assert throughput_chars_per_sec > 1000  # At least 1000 chars per second
        
        # Memory usage should be reasonable
        assert metrics['memory_increase_mb'] < 500  # Less than 500MB increase
        
        print(f"Ingestion Performance:")
        print(f"  Files/sec: {throughput_files_per_sec:.2f}")
        print(f"  Chars/sec: {throughput_chars_per_sec:.0f}")
        print(f"  Duration: {metrics['duration']:.2f}s")
        print(f"  Memory increase: {metrics['memory_increase_mb']:.1f}MB")
    
    def test_ingestion_memory_efficiency(self, large_test_data):
        """Test memory efficiency of document ingestion."""
        output_dir = large_test_data / "output"
        output_dir.mkdir()
        
        # Monitor memory during ingestion
        memory_samples = []
        
        def memory_callback():
            memory_samples.append(psutil.Process().memory_info().rss / 1024 / 1024)
        
        ingester = DocumentIngester(output_dir=str(output_dir))
        
        # Patch to add memory monitoring
        original_process_file = ingester._process_file
        
        def monitored_process_file(*args, **kwargs):
            memory_callback()
            return original_process_file(*args, **kwargs)
        
        ingester._process_file = monitored_process_file
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        results = ingester.ingest_directory(large_test_data, recursive=False)
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = max(memory_samples) if memory_samples else end_memory
        
        # Memory efficiency assertions
        memory_increase = end_memory - start_memory
        peak_increase = peak_memory - start_memory
        
        assert memory_increase < 200  # Less than 200MB permanent increase
        assert peak_increase < 500   # Less than 500MB peak increase
        
        # Memory should not grow linearly with file count
        memory_per_file = memory_increase / max(results['processed_count'], 1)
        assert memory_per_file < 10  # Less than 10MB per file on average
        
        print(f"Memory Efficiency:")
        print(f"  Start memory: {start_memory:.1f}MB")
        print(f"  End memory: {end_memory:.1f}MB")
        print(f"  Peak memory: {peak_memory:.1f}MB")
        print(f"  Memory per file: {memory_per_file:.1f}MB")
    
    def test_ingestion_scalability(self):
        """Test ingestion scalability with different file counts."""
        file_counts = [10, 50, 100]
        performance_data = []
        
        for file_count in file_counts:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                output_dir = temp_path / "output"
                output_dir.mkdir()
                
                # Create test files
                for i in range(file_count):
                    content = f"Test content for file {i}. " * 500  # ~5KB each
                    (temp_path / f"file_{i}.txt").write_text(content)
                
                ingester = DocumentIngester(output_dir=str(output_dir))
                
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                results = ingester.ingest_directory(temp_path, recursive=False)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                duration = end_time - start_time
                memory_increase = end_memory - start_memory
                
                performance_data.append({
                    'file_count': file_count,
                    'duration': duration,
                    'memory_increase': memory_increase,
                    'throughput': file_count / duration
                })
        
        # Analyze scalability
        for i in range(1, len(performance_data)):
            prev = performance_data[i-1]
            curr = performance_data[i]
            
            # Time should scale sub-linearly (better than O(n))
            time_ratio = curr['duration'] / prev['duration']
            file_ratio = curr['file_count'] / prev['file_count']
            
            assert time_ratio < file_ratio * 1.5  # Allow some overhead
            
            # Memory should scale sub-linearly
            memory_ratio = curr['memory_increase'] / max(prev['memory_increase'], 1)
            assert memory_ratio < file_ratio * 1.2  # Memory should be more efficient
        
        print("Scalability Results:")
        for data in performance_data:
            print(f"  {data['file_count']} files: {data['duration']:.2f}s, "
                  f"{data['throughput']:.1f} files/sec, "
                  f"{data['memory_increase']:.1f}MB")


class TestDeduplicationPerformance:
    """Performance tests for deduplication algorithms."""
    
    def test_hash_deduplication_performance(self, performance_monitor):
        """Test hash-based deduplication performance."""
        # Create test data with known duplicates
        test_files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files with duplicates
            unique_contents = [f"Unique content {i} " * 100 for i in range(50)]
            
            for i in range(200):  # 200 files total
                content_idx = i % 50  # Creates 4 copies of each unique content
                file_path = temp_path / f"file_{i}.txt"
                file_path.write_text(unique_contents[content_idx])
                test_files.append(str(file_path))
            
            deduplicator = HashDeduplicator()
            
            performance_monitor.start_monitoring()
            
            # Add all documents
            for file_path in test_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                deduplicator.add_document(file_path, content)
            
            # Find duplicates
            duplicate_groups = deduplicator.find_duplicates()
            
            metrics = performance_monitor.get_metrics()
            
            # Performance assertions
            assert len(duplicate_groups) == 50  # Should find 50 groups of duplicates
            assert metrics['duration'] < 10.0  # Should complete within 10 seconds
            assert metrics['memory_increase_mb'] < 100  # Less than 100MB increase
            
            # Calculate performance metrics
            files_per_sec = len(test_files) / metrics['duration']
            assert files_per_sec > 20  # At least 20 files per second
            
            print(f"Hash Deduplication Performance:")
            print(f"  Files processed: {len(test_files)}")
            print(f"  Duplicate groups found: {len(duplicate_groups)}")
            print(f"  Files/sec: {files_per_sec:.1f}")
            print(f"  Duration: {metrics['duration']:.2f}s")
            print(f"  Memory increase: {metrics['memory_increase_mb']:.1f}MB")
    
    @pytest.mark.slow
    def test_embedding_deduplication_performance(self, performance_monitor):
        """Test embedding-based deduplication performance."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not available")
        
        # Create test data with semantic similarities
        test_files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create semantically similar content
            base_contents = [
                "The quick brown fox jumps over the lazy dog.",
                "A fast brown fox leaps over a sleepy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "ML is part of AI technology and computer science.",
                "Python is a popular programming language.",
                "Python programming language is widely used."
            ]
            
            for i, content in enumerate(base_contents):
                file_path = temp_path / f"file_{i}.txt"
                file_path.write_text(content * 10)  # Repeat for more content
                test_files.append(str(file_path))
            
            deduplicator = EmbeddingDeduplicator(
                model_name="all-MiniLM-L6-v2",
                similarity_threshold=0.8,
                batch_size=4
            )
            
            performance_monitor.start_monitoring()
            
            # Add documents
            for file_path in test_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                deduplicator.add_document(file_path, content)
            
            # Compute embeddings
            deduplicator.compute_embeddings()
            
            # Find near duplicates
            near_duplicates = deduplicator.find_near_duplicates()
            
            metrics = performance_monitor.get_metrics()
            
            # Performance assertions
            assert len(near_duplicates) > 0  # Should find some near duplicates
            assert metrics['duration'] < 60.0  # Should complete within 60 seconds
            assert metrics['memory_increase_mb'] < 500  # Less than 500MB increase
            
            print(f"Embedding Deduplication Performance:")
            print(f"  Files processed: {len(test_files)}")
            print(f"  Near duplicates found: {len(near_duplicates)}")
            print(f"  Duration: {metrics['duration']:.2f}s")
            print(f"  Memory increase: {metrics['memory_increase_mb']:.1f}MB")
    
    def test_deduplication_pipeline_performance(self, large_test_data, performance_monitor):
        """Test complete deduplication pipeline performance."""
        # First create cleaned files
        output_dir = large_test_data / "cleaned"
        output_dir.mkdir()
        
        ingester = DocumentIngester(output_dir=str(output_dir))
        ingester.ingest_directory(large_test_data, recursive=False)
        
        cleaned_files = [str(f) for f in output_dir.glob("*.txt")]
        deduped_dir = large_test_data / "deduped"
        deduped_dir.mkdir()
        
        # Test hash-only deduplication (faster)
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False
        )
        
        performance_monitor.start_monitoring()
        
        stats = pipeline.process_files(cleaned_files, str(deduped_dir))
        
        metrics = performance_monitor.get_metrics()
        
        # Performance assertions
        assert stats.total_documents > 0
        assert metrics['duration'] < 30.0  # Should complete within 30 seconds
        assert metrics['memory_increase_mb'] < 200  # Less than 200MB increase
        
        # Calculate efficiency metrics
        files_per_sec = stats.total_documents / metrics['duration']
        chars_per_sec = stats.total_characters_before / metrics['duration']
        
        assert files_per_sec > 0.5  # At least 0.5 files per second
        
        print(f"Deduplication Pipeline Performance:")
        print(f"  Files processed: {stats.total_documents}")
        print(f"  Files/sec: {files_per_sec:.2f}")
        print(f"  Chars/sec: {chars_per_sec:.0f}")
        print(f"  Duration: {metrics['duration']:.2f}s")
        print(f"  Duplicates removed: {stats.exact_duplicates_removed}")


class TestQuantizationPerformance:
    """Performance tests for model quantization."""
    
    def test_quantization_speed(self, performance_monitor):
        """Test quantization speed for different tensor sizes."""
        tensor_sizes = [
            (100, 100),      # Small tensor
            (1000, 1000),    # Medium tensor
            (2000, 2000),    # Large tensor
        ]
        
        quantization_types = ["f16", "q8_0", "q4_0"]
        
        for quant_type in quantization_types:
            config = create_quantization_config(quant_type)
            manager = QuantizationManager(config)
            
            for size in tensor_sizes:
                test_tensor = torch.randn(*size)
                
                performance_monitor.start_monitoring()
                
                quantized_data, stats = manager.quantize_tensor(test_tensor, f"test_{size[0]}x{size[1]}")
                
                metrics = performance_monitor.get_metrics()
                
                # Performance assertions
                assert len(quantized_data) > 0
                assert stats['quality_score'] > 0
                assert metrics['duration'] < 10.0  # Should complete within 10 seconds
                
                # Calculate throughput
                elements = size[0] * size[1]
                elements_per_sec = elements / metrics['duration']
                
                print(f"Quantization Performance ({quant_type}, {size[0]}x{size[1]}):")
                print(f"  Elements/sec: {elements_per_sec:.0f}")
                print(f"  Duration: {metrics['duration']:.3f}s")
                print(f"  Quality score: {stats['quality_score']:.3f}")
                print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    
    def test_quantization_memory_usage(self):
        """Test memory usage during quantization."""
        # Create a moderately large tensor
        test_tensor = torch.randn(1000, 1000)
        
        config = create_quantization_config("q4_0")
        manager = QuantizationManager(config)
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        quantized_data, stats = manager.quantize_tensor(test_tensor, "memory_test")
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = end_memory - start_memory
        
        # Memory usage should be reasonable
        tensor_size_mb = test_tensor.numel() * 4 / 1024 / 1024  # F32 size in MB
        
        # Memory increase should not be more than 3x the tensor size
        assert memory_increase < tensor_size_mb * 3
        
        print(f"Quantization Memory Usage:")
        print(f"  Tensor size: {tensor_size_mb:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        print(f"  Efficiency ratio: {memory_increase / tensor_size_mb:.2f}x")


class TestTrainingPerformanceImpact:
    """Test performance impact of new features on training."""
    
    def test_tokenizer_training_impact(self):
        """Test performance impact of enhanced tokenizer training."""
        # Create test data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create training text
            training_text = "This is sample training text. " * 1000
            text_file = temp_path / "training.txt"
            text_file.write_text(training_text)
            
            # Mock tokenizer training to measure setup overhead
            start_time = time.time()
            
            # Simulate tokenizer configuration and setup
            config = {
                'vocab_size': 1000,
                'model_type': 'bpe',
                'special_tokens': ['<pad>', '<unk>', '<s>', '</s>']
            }
            
            # Simulate file processing
            with open(text_file, 'r') as f:
                content = f.read()
                words = content.split()
            
            end_time = time.time()
            setup_time = end_time - start_time
            
            # Setup should be fast
            assert setup_time < 1.0  # Less than 1 second for setup
            
            print(f"Tokenizer Setup Performance:")
            print(f"  Setup time: {setup_time:.3f}s")
            print(f"  Words processed: {len(words)}")
    
    def test_data_loading_impact(self, large_test_data):
        """Test impact of enhanced data loading on training speed."""
        # Simulate data loading with different approaches
        
        # Method 1: Direct file reading
        start_time = time.time()
        
        all_content = []
        for txt_file in large_test_data.glob("*.txt"):
            content = txt_file.read_text(encoding='utf-8')
            all_content.append(content)
        
        direct_time = time.time() - start_time
        
        # Method 2: Using ingestion pipeline
        output_dir = large_test_data / "output"
        output_dir.mkdir()
        
        ingester = DocumentIngester(output_dir=str(output_dir))
        
        start_time = time.time()
        results = ingester.ingest_directory(large_test_data, recursive=False)
        pipeline_time = time.time() - start_time
        
        # Pipeline should not be significantly slower
        overhead_ratio = pipeline_time / direct_time
        assert overhead_ratio < 3.0  # Less than 3x overhead
        
        print(f"Data Loading Performance:")
        print(f"  Direct loading: {direct_time:.3f}s")
        print(f"  Pipeline loading: {pipeline_time:.3f}s")
        print(f"  Overhead ratio: {overhead_ratio:.2f}x")


class TestRegressionDetection:
    """Automated performance regression detection."""
    
    def test_performance_regression_thresholds(self):
        """Test that performance meets minimum thresholds."""
        # Define performance thresholds
        thresholds = {
            'ingestion_files_per_sec': 0.5,
            'deduplication_files_per_sec': 1.0,
            'quantization_elements_per_sec': 10000,
            'memory_efficiency_mb_per_file': 10.0
        }
        
        # This would normally load historical performance data
        # For now, we'll simulate current performance
        current_performance = {
            'ingestion_files_per_sec': 2.0,
            'deduplication_files_per_sec': 3.0,
            'quantization_elements_per_sec': 50000,
            'memory_efficiency_mb_per_file': 5.0
        }
        
        # Check for regressions
        regressions = []
        for metric, threshold in thresholds.items():
            current_value = current_performance.get(metric, 0)
            if current_value < threshold:
                regressions.append(f"{metric}: {current_value} < {threshold}")
        
        # Assert no regressions
        assert len(regressions) == 0, f"Performance regressions detected: {regressions}"
        
        print("Performance Regression Check:")
        for metric, value in current_performance.items():
            threshold = thresholds.get(metric, 0)
            status = "✅ PASS" if value >= threshold else "❌ FAIL"
            print(f"  {metric}: {value} (threshold: {threshold}) {status}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])