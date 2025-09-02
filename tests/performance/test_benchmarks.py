#!/usr/bin/env python3
"""
Performance benchmarks for LLMBuilder components.
"""

import pytest
import time
import psutil
import os
from pathlib import Path
import json
import sys
from click.testing import CliRunner

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llmbuilder.cli.main import cli


class TestDataProcessingPerformance:
    """Performance tests for data processing operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.process = psutil.Process(os.getpid())
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_file_processing_performance(self):
        """Test performance with large files."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'perf-test'])
            assert result.exit_code == 0
            
            project_path = Path('perf-test')
            data_dir = project_path / 'data' / 'raw'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create large test file
            large_file = data_dir / 'large_dataset.jsonl'
            
            start_time = time.time()
            with open(large_file, 'w') as f:
                for i in range(10000):  # 10K records
                    text = f"This is sample text number {i}. " * 20  # ~500 chars each
                    f.write(json.dumps({"text": text, "id": i}) + '\n')
            
            file_creation_time = time.time() - start_time
            file_size_mb = large_file.stat().st_size / (1024 * 1024)
            
            print(f"Created {file_size_mb:.2f}MB file in {file_creation_time:.2f}s")
            
            # Measure processing performance
            initial_memory = self.process.memory_info().rss / (1024 * 1024)
            
            start_time = time.time()
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            processing_time = time.time() - start_time
            
            final_memory = self.process.memory_info().rss / (1024 * 1024)
            memory_used = final_memory - initial_memory
            
            print(f"Processing took {processing_time:.2f}s")
            print(f"Memory used: {memory_used:.2f}MB")
            
            # Performance assertions
            assert processing_time < 60  # Should complete within 1 minute
            assert memory_used < 1000    # Should use less than 1GB
            
            # Throughput calculation
            if processing_time > 0:
                throughput_mb_per_sec = file_size_mb / processing_time
                print(f"Throughput: {throughput_mb_per_sec:.2f} MB/s")
                assert throughput_mb_per_sec > 0.1  # At least 0.1 MB/s
    
    @pytest.mark.performance
    def test_many_small_files_performance(self):
        """Test performance with many small files."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'many-files-test'])
            assert result.exit_code == 0
            
            project_path = Path('many-files-test')
            data_dir = project_path / 'data' / 'raw'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create many small files
            num_files = 1000
            start_time = time.time()
            
            for i in range(num_files):
                file_path = data_dir / f'file_{i:04d}.txt'
                file_path.write_text(f"Content of file {i}")
            
            file_creation_time = time.time() - start_time
            print(f"Created {num_files} files in {file_creation_time:.2f}s")
            
            # Measure processing performance
            start_time = time.time()
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            processing_time = time.time() - start_time
            
            print(f"Processing {num_files} files took {processing_time:.2f}s")
            
            # Performance assertions
            assert processing_time < 30  # Should complete within 30 seconds
            
            if processing_time > 0:
                files_per_sec = num_files / processing_time
                print(f"Throughput: {files_per_sec:.2f} files/s")
                assert files_per_sec > 10  # At least 10 files per second
    
    @pytest.mark.performance
    def test_deduplication_performance(self):
        """Test deduplication performance."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'dedup-test'])
            assert result.exit_code == 0
            
            project_path = Path('dedup-test')
            data_dir = project_path / 'data' / 'raw'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dataset with duplicates
            dataset_file = data_dir / 'dataset_with_dupes.jsonl'
            
            start_time = time.time()
            with open(dataset_file, 'w') as f:
                # Create base content
                base_texts = [
                    "This is unique text number {}.",
                    "Another unique piece of content {}.",
                    "Different text sample {}."
                ]
                
                # Write original content
                for i in range(1000):
                    text = base_texts[i % len(base_texts)].format(i)
                    f.write(json.dumps({"text": text}) + '\n')
                
                # Add duplicates
                for i in range(500):  # 50% duplicates
                    text = base_texts[i % len(base_texts)].format(i)
                    f.write(json.dumps({"text": text}) + '\n')
            
            file_creation_time = time.time() - start_time
            print(f"Created dataset with duplicates in {file_creation_time:.2f}s")
            
            # Measure deduplication performance
            start_time = time.time()
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(data_dir),
                '--output', str(project_path / 'data' / 'cleaned'),
                '--deduplicate'
            ])
            dedup_time = time.time() - start_time
            
            print(f"Deduplication took {dedup_time:.2f}s")
            
            # Performance assertions
            assert dedup_time < 30  # Should complete within 30 seconds
            assert isinstance(result.exit_code, int)


class TestCLIPerformance:
    """Performance tests for CLI operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.performance
    def test_cli_startup_time(self):
        """Test CLI startup performance."""
        start_time = time.time()
        result = self.runner.invoke(cli, ['--help'])
        startup_time = time.time() - start_time
        
        print(f"CLI startup time: {startup_time:.3f}s")
        
        # Should start quickly
        assert startup_time < 2.0  # Less than 2 seconds
        assert result.exit_code == 0
    
    @pytest.mark.performance
    def test_config_operations_performance(self):
        """Test configuration operations performance."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'config-perf'])
            assert result.exit_code == 0
            
            # Test config set performance
            start_time = time.time()
            for i in range(100):
                result = self.runner.invoke(cli, [
                    'config', 'set',
                    f'test.key_{i}', f'value_{i}'
                ])
            set_time = time.time() - start_time
            
            print(f"100 config set operations took {set_time:.3f}s")
            
            # Test config get performance
            start_time = time.time()
            for i in range(100):
                result = self.runner.invoke(cli, [
                    'config', 'get',
                    f'test.key_{i}'
                ])
            get_time = time.time() - start_time
            
            print(f"100 config get operations took {get_time:.3f}s")
            
            # Performance assertions
            assert set_time < 10.0  # Less than 10 seconds for 100 operations
            assert get_time < 5.0   # Less than 5 seconds for 100 operations
    
    @pytest.mark.performance
    def test_project_initialization_performance(self):
        """Test project initialization performance."""
        with self.runner.isolated_filesystem():
            # Test multiple project initializations
            times = []
            
            for i in range(5):
                start_time = time.time()
                result = self.runner.invoke(cli, ['init', f'perf-project-{i}'])
                init_time = time.time() - start_time
                times.append(init_time)
                
                assert result.exit_code == 0
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            
            print(f"Average init time: {avg_time:.3f}s")
            print(f"Max init time: {max_time:.3f}s")
            
            # Performance assertions
            assert avg_time < 2.0  # Average less than 2 seconds
            assert max_time < 5.0  # Max less than 5 seconds


class TestMemoryUsage:
    """Memory usage tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.process = psutil.Process(os.getpid())
    
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'memory-test'])
            assert result.exit_code == 0
            
            # Get baseline memory
            baseline_memory = self.process.memory_info().rss / (1024 * 1024)
            
            # Perform repeated operations
            for i in range(10):
                # Create and process data
                project_path = Path(f'memory-test-{i}')
                result = self.runner.invoke(cli, ['init', str(project_path)])
                
                if result.exit_code == 0:
                    # Create small dataset
                    data_dir = project_path / 'data' / 'raw'
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(data_dir / 'test.jsonl', 'w') as f:
                        for j in range(100):
                            f.write(json.dumps({"text": f"Sample {j}"}) + '\n')
                    
                    # Process data
                    result = self.runner.invoke(cli, [
                        'data', 'prepare',
                        '--input', str(data_dir),
                        '--output', str(project_path / 'data' / 'cleaned')
                    ])
                
                # Check memory after each iteration
                current_memory = self.process.memory_info().rss / (1024 * 1024)
                memory_increase = current_memory - baseline_memory
                
                print(f"Iteration {i}: Memory increase: {memory_increase:.2f}MB")
                
                # Memory should not grow excessively
                assert memory_increase < 200  # Less than 200MB increase
    
    @pytest.mark.performance
    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        with self.runner.isolated_filesystem():
            # Setup project
            result = self.runner.invoke(cli, ['init', 'large-memory-test'])
            assert result.exit_code == 0
            
            project_path = Path('large-memory-test')
            data_dir = project_path / 'data' / 'raw'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create large dataset
            large_file = data_dir / 'large_dataset.jsonl'
            file_size_mb = 0
            
            with open(large_file, 'w') as f:
                for i in range(5000):  # 5K records
                    text = f"Large text content number {i}. " * 50  # ~1.25KB each
                    f.write(json.dumps({"text": text}) + '\n')
            
            file_size_mb = large_file.stat().st_size / (1024 * 1024)
            print(f"Created {file_size_mb:.2f}MB dataset")
            
            # Monitor memory during processing
            initial_memory = self.process.memory_info().rss / (1024 * 1024)
            
            result = self.runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(data_dir),
                '--output', str(project_path / 'data' / 'cleaned')
            ])
            
            peak_memory = self.process.memory_info().rss / (1024 * 1024)
            memory_used = peak_memory - initial_memory
            
            print(f"Peak memory usage: {memory_used:.2f}MB")
            print(f"Memory efficiency: {file_size_mb / memory_used:.2f} (file_size/memory_used)")
            
            # Memory usage should be reasonable relative to file size
            assert memory_used < file_size_mb * 5  # Less than 5x file size
            assert isinstance(result.exit_code, int)


class TestConcurrencyPerformance:
    """Concurrency performance tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations."""
        import threading
        import tempfile
        
        results = []
        times = []
        
        def concurrent_operation(thread_id):
            """Perform operation in separate thread."""
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                start_time = time.time()
                
                # Initialize project
                result = self.runner.invoke(cli, ['init', str(temp_path / f'concurrent-{thread_id}')])
                
                if result.exit_code == 0:
                    # Create and process data
                    project_path = temp_path / f'concurrent-{thread_id}'
                    data_dir = project_path / 'data' / 'raw'
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(data_dir / 'test.jsonl', 'w') as f:
                        for i in range(100):
                            f.write(json.dumps({"text": f"Thread {thread_id} sample {i}"}) + '\n')
                    
                    result = self.runner.invoke(cli, [
                        'data', 'prepare',
                        '--input', str(data_dir),
                        '--output', str(project_path / 'data' / 'cleaned')
                    ])
                
                end_time = time.time()
                
                results.append(result.exit_code)
                times.append(end_time - start_time)
        
        # Start multiple concurrent threads
        threads = []
        num_threads = 3
        
        overall_start = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=concurrent_operation, args=[i])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        overall_time = time.time() - overall_start
        
        print(f"Concurrent operations completed in {overall_time:.2f}s")
        print(f"Individual times: {[f'{t:.2f}s' for t in times]}")
        print(f"Average individual time: {sum(times)/len(times):.2f}s")
        
        # All operations should complete
        assert len(results) == num_threads
        assert all(isinstance(code, int) for code in results)
        
        # Concurrent execution should be reasonably efficient
        sequential_estimate = sum(times)
        efficiency = sequential_estimate / overall_time if overall_time > 0 else 0
        print(f"Concurrency efficiency: {efficiency:.2f}x")


if __name__ == '__main__':
    pytest.main([__file__])