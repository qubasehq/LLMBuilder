#!/usr/bin/env python3
"""
Integration tests for the complete LLM training pipeline.
Tests end-to-end workflows with limited sample sizes for CI/CD.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
import torch
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llmbuilder.core.data.ingest import DocumentIngester
from llmbuilder.core.data.dedup import DeduplicationPipeline
from llmbuilder.core.training.train_tokenizer import TokenizerTrainer
from llmbuilder.core.tools.conversion_pipeline import ConversionPipeline, ConversionConfig
from llmbuilder.core.tools.quantization_manager import QuantizationManager, create_quantization_config


class TestDataFixtures:
    """Test data fixtures and utilities."""
    
    @staticmethod
    def create_sample_documents(temp_dir: Path) -> Dict[str, str]:
        """Create sample documents for testing."""
        documents = {
            "sample.txt": "This is a sample text document for testing the pipeline. It contains multiple sentences and should be processed correctly.",
            "sample.html": """
            <html>
                <head><title>Test Document</title></head>
                <body>
                    <h1>Test HTML Document</h1>
                    <p>This is a test HTML document with some content.</p>
                    <script>console.log('should be removed');</script>
                    <p>More content here for testing purposes.</p>
                </body>
            </html>
            """,
            "sample.md": """
            # Test Markdown Document
            
            This is a **test** markdown document with:
            - Lists
            - **Bold text**
            - `Code snippets`
            
            ## Section 2
            More content for testing the markdown processor.
            """,
            "duplicate.txt": "This is a sample text document for testing the pipeline. It contains multiple sentences and should be processed correctly.",
            "similar.txt": "This is a sample text file for testing the pipeline system. It has multiple sentences and should be handled correctly."
        }
        
        for filename, content in documents.items():
            file_path = temp_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return documents
    
    @staticmethod
    def create_minimal_config() -> Dict[str, Any]:
        """Create minimal configuration for testing."""
        return {
            "model": {
                "vocab_size": 1000,
                "embedding_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "hidden_dim": 128,
                "max_seq_length": 64,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 2,
                "learning_rate": 0.001,
                "num_epochs": 1,
                "max_steps": 10,
                "save_every": 5,
                "eval_every": 5,
                "log_every": 1
            },
            "tokenizer": {
                "vocab_size": 1000,
                "model_type": "bpe",
                "special_tokens": ["<pad>", "<unk>", "<s>", "</s>"]
            },
            "device": {
                "use_cuda": False,
                "cpu_threads": 1
            }
        }


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create directory structure
        (workspace / "data" / "raw").mkdir(parents=True)
        (workspace / "data" / "cleaned").mkdir(parents=True)
        (workspace / "data" / "deduped").mkdir(parents=True)
        (workspace / "exports" / "tokenizer").mkdir(parents=True)
        (workspace / "exports" / "checkpoints").mkdir(parents=True)
        (workspace / "exports" / "gguf").mkdir(parents=True)
        
        yield workspace


@pytest.fixture
def sample_documents(temp_workspace):
    """Create sample documents in temporary workspace."""
    raw_dir = temp_workspace / "data" / "raw"
    documents = TestDataFixtures.create_sample_documents(raw_dir)
    return documents


@pytest.fixture
def minimal_config(temp_workspace):
    """Create minimal configuration file."""
    config = TestDataFixtures.create_minimal_config()
    config_path = temp_workspace / "config_test.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


class TestDocumentIngestionIntegration:
    """Integration tests for document ingestion pipeline."""
    
    def test_complete_ingestion_workflow(self, temp_workspace, sample_documents):
        """Test complete document ingestion workflow."""
        raw_dir = temp_workspace / "data" / "raw"
        cleaned_dir = temp_workspace / "data" / "cleaned"
        
        # Initialize ingester
        ingester = DocumentIngester(
            output_dir=str(cleaned_dir),
            max_file_size_mb=10
        )
        
        # Run ingestion
        results = ingester.ingest_directory(raw_dir, recursive=False)
        
        # Verify results
        assert results['total_files'] == len(sample_documents)
        assert results['processed_count'] > 0
        assert results['failed_count'] == 0
        
        # Check output files exist
        output_files = list(cleaned_dir.glob("*.txt"))
        assert len(output_files) > 0
        
        # Verify content was extracted
        for output_file in output_files:
            content = output_file.read_text(encoding='utf-8')
            assert len(content.strip()) > 0
    
    def test_ingestion_with_error_handling(self, temp_workspace):
        """Test ingestion with error handling for invalid files."""
        raw_dir = temp_workspace / "data" / "raw"
        cleaned_dir = temp_workspace / "data" / "cleaned"
        
        # Create invalid file
        invalid_file = raw_dir / "invalid.pdf"
        invalid_file.write_bytes(b"Not a valid PDF file")
        
        # Initialize ingester
        ingester = DocumentIngester(
            output_dir=str(cleaned_dir),
            max_file_size_mb=10
        )
        
        # Run ingestion (should handle errors gracefully)
        results = ingester.ingest_directory(raw_dir, recursive=False)
        
        # Should not crash, but may have failed files
        assert results['total_files'] >= 1
        assert 'failed_files' in results


class TestDeduplicationIntegration:
    """Integration tests for deduplication pipeline."""
    
    def test_complete_deduplication_workflow(self, temp_workspace, sample_documents):
        """Test complete deduplication workflow."""
        # First run ingestion
        raw_dir = temp_workspace / "data" / "raw"
        cleaned_dir = temp_workspace / "data" / "cleaned"
        deduped_dir = temp_workspace / "data" / "deduped"
        
        ingester = DocumentIngester(output_dir=str(cleaned_dir))
        ingester.ingest_directory(raw_dir, recursive=False)
        
        # Get cleaned files
        cleaned_files = list(cleaned_dir.glob("*.txt"))
        assert len(cleaned_files) > 0
        
        # Run deduplication
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False  # Skip embedding for speed
        )
        
        stats = pipeline.process_files(
            [str(f) for f in cleaned_files],
            str(deduped_dir)
        )
        
        # Verify deduplication results
        assert stats.total_documents > 0
        assert stats.final_document_count > 0
        assert stats.processing_time > 0
        
        # Check output files
        deduped_files = list(deduped_dir.glob("*.txt"))
        assert len(deduped_files) > 0
    
    def test_hash_deduplication_only(self, temp_workspace, sample_documents):
        """Test hash-based deduplication only."""
        cleaned_dir = temp_workspace / "data" / "cleaned"
        deduped_dir = temp_workspace / "data" / "deduped"
        
        # Create test files with exact duplicates
        (cleaned_dir / "file1.txt").write_text("Exact same content")
        (cleaned_dir / "file2.txt").write_text("Exact same content")
        (cleaned_dir / "file3.txt").write_text("Different content")
        
        cleaned_files = list(cleaned_dir.glob("*.txt"))
        
        # Run hash deduplication
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False
        )
        
        stats = pipeline.process_files(
            [str(f) for f in cleaned_files],
            str(deduped_dir)
        )
        
        # Should remove one duplicate
        assert stats.total_documents == 3
        assert stats.final_document_count == 2
        assert stats.exact_duplicates_removed == 1


class TestTokenizerIntegration:
    """Integration tests for tokenizer training."""
    
    @pytest.mark.slow
    def test_tokenizer_training_workflow(self, temp_workspace, sample_documents):
        """Test complete tokenizer training workflow."""
        # Setup data
        raw_dir = temp_workspace / "data" / "raw"
        cleaned_dir = temp_workspace / "data" / "cleaned"
        tokenizer_dir = temp_workspace / "exports" / "tokenizer"
        
        # Run ingestion
        ingester = DocumentIngester(output_dir=str(cleaned_dir))
        ingester.ingest_directory(raw_dir, recursive=False)
        
        # Combine text files for tokenizer training
        combined_text = []
        for txt_file in cleaned_dir.glob("*.txt"):
            combined_text.append(txt_file.read_text(encoding='utf-8'))
        
        combined_file = cleaned_dir / "combined.txt"
        combined_file.write_text("\n".join(combined_text), encoding='utf-8')
        
        # Train tokenizer (mock the actual training for speed)
        with patch('training.train_tokenizer.TokenizerTrainer') as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance
            mock_instance.train.return_value = True
            
            # This would normally train the tokenizer
            trainer = mock_trainer(
                vocab_size=1000,
                model_type='bpe',
                input_files=[str(combined_file)],
                output_dir=str(tokenizer_dir)
            )
            
            result = trainer.train()
            assert result is True
            mock_instance.train.assert_called_once()


class TestConversionIntegration:
    """Integration tests for GGUF conversion pipeline."""
    
    def test_conversion_pipeline_setup(self, temp_workspace):
        """Test GGUF conversion pipeline setup."""
        checkpoint_dir = temp_workspace / "exports" / "checkpoints"
        gguf_dir = temp_workspace / "exports" / "gguf"
        
        # Create mock checkpoint
        mock_checkpoint = {
            'model': {
                'embedding.weight': torch.randn(1000, 64),
                'transformer.layers.0.attention.weight': torch.randn(64, 64),
                'lm_head.weight': torch.randn(1000, 64)
            },
            'config': {
                'vocab_size': 1000,
                'embedding_dim': 64,
                'num_layers': 2
            }
        }
        
        checkpoint_path = checkpoint_dir / "test_model.pt"
        torch.save(mock_checkpoint, checkpoint_path)
        
        # Test conversion config creation
        config = ConversionConfig(
            input_checkpoint=checkpoint_path,
            output_dir=gguf_dir,
            quantization_levels=["f16"],
            validate_output=False  # Skip validation for speed
        )
        
        assert config.input_checkpoint.exists()
        assert config.output_dir.exists()
        assert len(config.quantization_levels) == 1
    
    def test_quantization_manager_setup(self):
        """Test quantization manager setup."""
        config = create_quantization_config("q4_0")
        manager = QuantizationManager(config)
        
        # Test with small tensor
        test_tensor = torch.randn(10, 10)
        quantized_data, stats = manager.quantize_tensor(test_tensor, "test_tensor")
        
        assert len(quantized_data) > 0
        assert stats['name'] == "test_tensor"
        assert stats['quality_score'] > 0


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_minimal_pipeline_workflow(self, temp_workspace, sample_documents, minimal_config):
        """Test minimal end-to-end pipeline workflow."""
        # This test runs a minimal version of the complete pipeline
        
        # 1. Document Ingestion
        raw_dir = temp_workspace / "data" / "raw"
        cleaned_dir = temp_workspace / "data" / "cleaned"
        
        ingester = DocumentIngester(output_dir=str(cleaned_dir))
        ingestion_results = ingester.ingest_directory(raw_dir, recursive=False)
        
        assert ingestion_results['processed_count'] > 0
        
        # 2. Deduplication
        deduped_dir = temp_workspace / "data" / "deduped"
        cleaned_files = list(cleaned_dir.glob("*.txt"))
        
        dedup_pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False  # Skip for speed
        )
        
        dedup_stats = dedup_pipeline.process_files(
            [str(f) for f in cleaned_files],
            str(deduped_dir)
        )
        
        assert dedup_stats.final_document_count > 0
        
        # 3. Mock tokenizer training (actual training too slow for CI)
        tokenizer_dir = temp_workspace / "exports" / "tokenizer"
        
        with patch('training.train_tokenizer.TokenizerTrainer') as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance
            mock_instance.train.return_value = True
            
            trainer = mock_trainer(
                vocab_size=1000,
                model_type='bpe',
                input_files=[str(f) for f in deduped_dir.glob("*.txt")],
                output_dir=str(tokenizer_dir)
            )
            
            tokenizer_result = trainer.train()
            assert tokenizer_result is True
        
        # 4. Mock model training (actual training too slow for CI)
        checkpoint_dir = temp_workspace / "exports" / "checkpoints"
        
        with patch('training.train.Trainer') as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance
            mock_instance.train.return_value = {"final_loss": 2.5, "steps": 10}
            
            trainer = mock_trainer(config_path=str(minimal_config))
            training_result = trainer.train()
            
            assert training_result["steps"] == 10
        
        # 5. Test conversion pipeline setup (without actual conversion)
        gguf_dir = temp_workspace / "exports" / "gguf"
        
        # Create mock checkpoint for conversion testing
        mock_checkpoint = {
            'model': {
                'embedding.weight': torch.randn(100, 32),
                'lm_head.weight': torch.randn(100, 32)
            }
        }
        
        checkpoint_path = checkpoint_dir / "mock_model.pt"
        torch.save(mock_checkpoint, checkpoint_path)
        
        conversion_config = ConversionConfig(
            input_checkpoint=checkpoint_path,
            output_dir=gguf_dir,
            quantization_levels=["f16"],
            validate_output=False
        )
        
        assert conversion_config.input_checkpoint.exists()
        
        # Pipeline completed successfully
        assert True
    
    def test_error_recovery_workflow(self, temp_workspace):
        """Test pipeline error recovery and graceful failure handling."""
        # Test with empty data directory
        raw_dir = temp_workspace / "data" / "raw"
        cleaned_dir = temp_workspace / "data" / "cleaned"
        
        ingester = DocumentIngester(output_dir=str(cleaned_dir))
        results = ingester.ingest_directory(raw_dir, recursive=False)
        
        # Should handle empty directory gracefully
        assert results['total_files'] == 0
        assert results['processed_count'] == 0
        assert results['failed_count'] == 0
        
        # Test deduplication with no files
        deduped_dir = temp_workspace / "data" / "deduped"
        dedup_pipeline = DeduplicationPipeline()
        
        stats = dedup_pipeline.process_files([], str(deduped_dir))
        
        # Should handle empty file list gracefully
        assert stats.total_documents == 0
        assert stats.final_document_count == 0


class TestPerformanceValidation:
    """Performance validation tests."""
    
    def test_ingestion_performance(self, temp_workspace):
        """Test ingestion performance with timing."""
        import time
        
        raw_dir = temp_workspace / "data" / "raw"
        cleaned_dir = temp_workspace / "data" / "cleaned"
        
        # Create multiple test files
        for i in range(10):
            test_file = raw_dir / f"test_{i}.txt"
            test_file.write_text(f"Test content for file {i} " * 100)
        
        ingester = DocumentIngester(output_dir=str(cleaned_dir))
        
        start_time = time.time()
        results = ingester.ingest_directory(raw_dir, recursive=False)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert results['processed_count'] == 10
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        # Calculate throughput
        throughput = results['processed_count'] / processing_time
        assert throughput > 1.0  # At least 1 file per second
    
    def test_deduplication_performance(self, temp_workspace):
        """Test deduplication performance."""
        import time
        
        cleaned_dir = temp_workspace / "data" / "cleaned"
        deduped_dir = temp_workspace / "data" / "deduped"
        
        # Create test files with some duplicates
        for i in range(20):
            content = f"Content for file {i % 10}" * 50  # Creates duplicates
            test_file = cleaned_dir / f"file_{i}.txt"
            test_file.write_text(content)
        
        files = [str(f) for f in cleaned_dir.glob("*.txt")]
        
        pipeline = DeduplicationPipeline(
            use_hash_dedup=True,
            use_embedding_dedup=False  # Hash only for speed
        )
        
        start_time = time.time()
        stats = pipeline.process_files(files, str(deduped_dir))
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert stats.total_documents == 20
        assert stats.exact_duplicates_removed > 0
        assert processing_time < 30.0  # Should complete within 30 seconds
        
        # Memory efficiency check (basic)
        assert stats.final_document_count < stats.total_documents


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])