"""
Tests for the LanguageFilter module.

This module tests the language detection and filtering functionality
including various detection backends and fallback mechanisms.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from training.advanced_cleaning.modules.language_filter import LanguageFilter, LanguageDetectionResult
from training.advanced_cleaning.data_models import CleaningOperationType


class TestLanguageFilter:
    """Test LanguageFilter functionality."""
    
    def test_module_initialization_default(self):
        """Test basic module initialization with default settings."""
        filter_module = LanguageFilter()
        
        assert filter_module.module_name == "LanguageFilter"
        assert filter_module.get_operation_type() == CleaningOperationType.LANGUAGE_FILTERING
        assert filter_module.is_enabled() is True
        assert filter_module.target_languages == ['en']
        assert filter_module.confidence_threshold == 0.8
        assert filter_module.detection_backend is not None
    
    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        config = {
            'target_languages': ['en', 'es', 'fr'],
            'confidence_threshold': 0.9,
            'detection_method': 'langdetect',
            'fallback_to_heuristics': False
        }
        
        filter_module = LanguageFilter(config)
        
        assert filter_module.target_languages == ['en', 'es', 'fr']
        assert filter_module.confidence_threshold == 0.9
        assert filter_module.detection_method == 'langdetect'
        assert filter_module.fallback_to_heuristics is False
    
    def test_heuristic_detection_english(self):
        """Test heuristic language detection for English text."""
        # Force heuristic mode by mocking langdetect to fail
        config = {'detection_method': 'heuristic'}
        
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'langdetect':
                    raise ImportError("Langdetect not available")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            filter_module = LanguageFilter(config)
            
            english_texts = [
                "This is a comprehensive guide to network security and cybersecurity best practices.",
                "Firewalls are essential components that help protect networks from unauthorized access.",
                "The implementation of encryption algorithms ensures data confidentiality and integrity.",
                "Security audits should be performed regularly to identify potential vulnerabilities."
            ]
            
            for text in english_texts:
                result = filter_module.detect_language(text)
                
                assert result.detected_language == "en"
                assert result.confidence > 0.0
                assert result.is_target_language is True
                assert result.method_used == "heuristic"
    
    def test_heuristic_detection_non_english(self):
        """Test heuristic detection with non-English-like text."""
        # Force heuristic mode by mocking langdetect to fail
        config = {'detection_method': 'heuristic'}
        
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'langdetect':
                    raise ImportError("Langdetect not available")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            filter_module = LanguageFilter(config)
            
            non_english_texts = [
                "12345 67890 !@#$% ^&*()",  # Numbers and symbols
                "αβγδε ζηθικ λμνξο",  # Greek characters
                "测试文本内容",  # Chinese characters
            ]
            
            for text in non_english_texts:
                result = filter_module.detect_language(text)
                
                # Should either detect as unknown or have low confidence
                assert result.detected_language in ["en", "unknown"]
                if result.detected_language == "en":
                    assert result.confidence < 0.6  # Low confidence for non-English
            
            # Test random letter combinations separately (might have higher confidence due to alphabetic content)
            random_text = "xyz qwerty asdfgh"
            result = filter_module.detect_language(random_text)
            # This might be detected as English with moderate confidence due to alphabetic content
            assert result.detected_language in ["en", "unknown"]
    
    def test_text_filtering_accept(self):
        """Test text filtering that accepts content."""
        config = {
            'target_languages': ['en'],
            'confidence_threshold': 0.5,
            'detection_method': 'heuristic'
        }
        filter_module = LanguageFilter(config)
        
        text = "Network security is crucial for protecting digital assets and preventing cyber attacks."
        
        filtered_text, operation = filter_module.clean_text(text)
        
        assert filtered_text == text  # Text should be preserved
        assert operation.success is True
        assert operation.items_removed == 0
        assert operation.original_length == len(text)
        assert operation.final_length == len(text)
        assert "Language accepted" in operation.description
        assert operation.metadata["detected_language"] == "en"
        assert operation.metadata["is_target_language"] is True
    
    def test_text_filtering_reject(self):
        """Test text filtering that rejects content."""
        config = {
            'target_languages': ['en'],
            'confidence_threshold': 0.8,
            'detection_method': 'heuristic'
        }
        filter_module = LanguageFilter(config)
        
        # Text that should be rejected (low English confidence)
        text = "12345 xyz qwerty !@#$%"
        
        filtered_text, operation = filter_module.clean_text(text)
        
        assert filtered_text == ""  # Text should be removed
        assert operation.success is True
        assert operation.items_removed == 1
        assert operation.original_length == len(text)
        assert operation.final_length == 0
        assert "Language filtered" in operation.description
    
    def test_filter_by_language_method(self):
        """Test the filter_by_language convenience method."""
        config = {
            'target_languages': ['en'],
            'confidence_threshold': 0.6,
            'detection_method': 'heuristic'
        }
        filter_module = LanguageFilter(config)
        
        # English text should be kept
        english_text = "Cybersecurity involves protecting systems from digital attacks."
        should_keep, result = filter_module.filter_by_language(english_text)
        
        assert should_keep is True
        assert result.detected_language == "en"
        assert result.is_target_language is True
        assert result.confidence >= 0.6
        
        # Non-English text should be rejected (use text with very low English indicators)
        non_english_text = "12345 !@#$% ζηθικ λμνξο"
        should_keep, result = filter_module.filter_by_language(non_english_text)
        
        assert should_keep is False
    
    def test_multiple_target_languages(self):
        """Test filtering with multiple target languages."""
        config = {
            'target_languages': ['en', 'es', 'fr'],
            'confidence_threshold': 0.5,
            'detection_method': 'heuristic'
        }
        filter_module = LanguageFilter(config)
        
        # Mock detection to return Spanish
        with patch.object(filter_module, 'detect_language') as mock_detect:
            mock_detect.return_value = LanguageDetectionResult(
                detected_language="es",
                confidence=0.9,
                is_target_language=True,
                method_used="mock"
            )
            
            text = "Texto en español sobre ciberseguridad"
            filtered_text, operation = filter_module.clean_text(text)
            
            assert filtered_text == text
            assert operation.items_removed == 0
            assert operation.metadata["detected_language"] == "es"
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        filter_module = LanguageFilter()
        
        test_cases = ["", "   ", "\n\n\n", "\t\t"]
        
        for text in test_cases:
            filtered_text, operation = filter_module.clean_text(text)
            
            assert filtered_text == text
            assert operation.success is True
            assert operation.items_removed == 0
            assert "No text to process" in operation.description
    
    def test_short_text_handling(self):
        """Test handling of very short text."""
        filter_module = LanguageFilter()
        
        short_text = "Hi"
        result = filter_module.detect_language(short_text)
        
        assert result.detected_language == "unknown"
        assert result.confidence == 0.0
        assert result.is_target_language is False
        assert "too_short" in result.method_used
    
    def test_text_preparation_for_detection(self):
        """Test text cleaning before language detection."""
        filter_module = LanguageFilter()
        
        messy_text = """
        This   is    a   test   with   excessive   whitespace.
        
        Visit https://example.com for more info.
        Contact us at test@example.com for questions.
        The year 2023 was significant. Call 555-1234.
        """
        
        clean_text = filter_module._prepare_text_for_detection(messy_text)
        
        # Should remove URLs, emails, and excessive whitespace
        assert "https://example.com" not in clean_text
        assert "test@example.com" not in clean_text
        assert "   " not in clean_text  # No excessive whitespace
        assert "This is a test" in clean_text
    
    def test_fasttext_detection_success(self):
        """Test successful FastText language detection."""
        config = {'detection_method': 'fasttext'}
        
        # Mock the fasttext import and model
        with patch('builtins.__import__') as mock_import:
            mock_fasttext = Mock()
            mock_model = Mock()
            mock_model.predict.return_value = (['__label__en'], [0.95])
            mock_fasttext.load_model.return_value = mock_model
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'fasttext':
                    return mock_fasttext
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            filter_module = LanguageFilter(config)
            filter_module.detector = mock_model
            filter_module.detection_backend = 'fasttext'
            
            text = "This is English text for testing."
            result = filter_module.detect_language(text)
            
            assert result.detected_language == "en"
            assert result.confidence == 0.95
            assert result.is_target_language is True
            assert result.method_used == "fasttext"
    
    def test_fasttext_initialization_failure(self):
        """Test FastText initialization failure and fallback."""
        config = {'detection_method': 'fasttext'}
        
        # Mock fasttext import to raise an exception
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'fasttext':
                    raise ImportError("FastText not available")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            filter_module = LanguageFilter(config)
            
            # Should fall back to heuristic detection
            assert filter_module.detection_backend == "heuristic"
    
    def test_langdetect_real_functionality(self):
        """Test real langdetect functionality when available."""
        config = {'detection_method': 'langdetect'}
        filter_module = LanguageFilter(config)
        
        # Test English text
        english_text = "This is a comprehensive cybersecurity guide for network protection."
        result = filter_module.detect_language(english_text)
        
        assert result.detected_language == "en"
        assert result.confidence > 0.8
        assert result.is_target_language is True
        assert result.method_used == "langdetect"
        
        # Test non-English text (if langdetect can detect it)
        spanish_text = "Esta es una guía completa de ciberseguridad para la protección de redes."
        result = filter_module.detect_language(spanish_text)
        
        # Should detect as Spanish (not English)
        assert result.detected_language != "en"
        assert result.is_target_language is False  # Since target is ['en']
        assert result.method_used == "langdetect"
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            'target_languages': ['en', 'es'],
            'confidence_threshold': 0.75,
            'detection_method': 'auto',
            'fallback_to_heuristics': True
        }
        
        filter_module = LanguageFilter(valid_config)
        errors = filter_module.validate_config()
        assert len(errors) == 0
        
        # Invalid configurations
        invalid_configs = [
            {'target_languages': 'not a list'},
            {'target_languages': []},
            {'target_languages': ['en', 123]},
            {'confidence_threshold': 'not a number'},
            {'confidence_threshold': -0.5},
            {'confidence_threshold': 1.5},
            {'detection_method': 'invalid_method'},
            {'fallback_to_heuristics': 'not a boolean'}
        ]
        
        for invalid_config in invalid_configs:
            filter_module = LanguageFilter(invalid_config)
            errors = filter_module.validate_config()
            assert len(errors) > 0
    
    def test_statistics_generation(self):
        """Test statistics generation."""
        config = {
            'target_languages': ['en', 'fr'],
            'confidence_threshold': 0.85,
            'detection_method': 'auto',
            'fallback_to_heuristics': True
        }
        
        filter_module = LanguageFilter(config)
        stats = filter_module.get_statistics()
        
        assert stats['target_languages'] == ['en', 'fr']
        assert stats['confidence_threshold'] == 0.85
        assert stats['detection_method'] == 'auto'
        assert stats['fallback_to_heuristics'] is True
        assert 'detection_backend' in stats
        assert 'backend_available' in stats
        assert isinstance(stats['heuristic_patterns'], int)
    
    def test_operation_metadata(self):
        """Test that operations include proper metadata."""
        filter_module = LanguageFilter()
        
        text = "Network security requires comprehensive protection strategies."
        filtered_text, operation = filter_module.clean_text(text)
        
        assert operation.operation_type == CleaningOperationType.LANGUAGE_FILTERING
        assert operation.module_name == "LanguageFilter"
        assert operation.success is True
        
        # Check metadata
        metadata = operation.metadata
        assert 'detected_language' in metadata
        assert 'confidence' in metadata
        assert 'is_target_language' in metadata
        assert 'detection_method' in metadata
        assert 'target_languages' in metadata
        assert 'confidence_threshold' in metadata
    
    def test_confidence_threshold_filtering(self):
        """Test filtering based on confidence thresholds."""
        # High threshold - should reject low confidence detections
        high_threshold_config = {
            'confidence_threshold': 0.9,
            'detection_method': 'heuristic'
        }
        filter_module = LanguageFilter(high_threshold_config)
        
        # Mock low confidence detection
        with patch.object(filter_module, 'detect_language') as mock_detect:
            mock_detect.return_value = LanguageDetectionResult(
                detected_language="en",
                confidence=0.7,  # Below threshold
                is_target_language=True,
                method_used="mock"
            )
            
            text = "Some text"
            filtered_text, operation = filter_module.clean_text(text)
            
            assert filtered_text == ""  # Should be filtered out
            assert operation.items_removed == 1
            assert "Language filtered" in operation.description
    
    def test_language_detection_result_class(self):
        """Test LanguageDetectionResult class functionality."""
        result = LanguageDetectionResult(
            detected_language="en",
            confidence=0.95,
            is_target_language=True,
            method_used="fasttext"
        )
        
        assert result.detected_language == "en"
        assert result.confidence == 0.95
        assert result.is_target_language is True
        assert result.method_used == "fasttext"
    
    def test_backend_initialization_auto_mode(self):
        """Test automatic backend selection."""
        config = {'detection_method': 'auto'}
        
        # Mock both fasttext and langdetect as unavailable
        with patch('training.advanced_cleaning.modules.language_filter.LanguageFilter._try_initialize_fasttext', return_value=False), \
             patch('training.advanced_cleaning.modules.language_filter.LanguageFilter._try_initialize_langdetect', return_value=False):
            
            filter_module = LanguageFilter(config)
            
            # Should fall back to heuristic
            assert filter_module.detection_backend == "heuristic"


if __name__ == "__main__":
    pytest.main([__file__])