"""
Tests for the BoilerplateRemover module.

This module tests the boilerplate removal functionality including
copyright notices, legal disclaimers, quiz patterns, and multiple choice sequences.
"""

import pytest
from training.advanced_cleaning.modules.boilerplate_remover import BoilerplateRemover
from training.advanced_cleaning.data_models import CleaningOperationType


class TestBoilerplateRemover:
    """Test BoilerplateRemover functionality."""
    
    def test_module_initialization(self):
        """Test basic module initialization."""
        remover = BoilerplateRemover()
        
        assert remover.module_name == "BoilerplateRemover"
        assert remover.get_operation_type() == CleaningOperationType.BOILERPLATE_REMOVAL
        assert remover.is_enabled() is True
        assert len(remover.compiled_patterns) > 0
    
    def test_custom_patterns_configuration(self):
        """Test initialization with custom patterns."""
        config = {
            'custom_patterns': [
                r'(?i)custom boilerplate pattern',
                r'(?i)another custom pattern'
            ]
        }
        
        remover = BoilerplateRemover(config)
        
        # Should have default patterns plus custom patterns
        expected_total = len(remover.default_patterns) + 2
        assert len(remover.compiled_patterns) == expected_total
    
    def test_copyright_removal(self):
        """Test removal of copyright notices."""
        remover = BoilerplateRemover()
        
        test_cases = [
            "This is important content.\nCopyright 2023 Example Corp.\nMore important content.",
            "Content here.\n© 2023 All rights reserved.\nMore content.",
            "Text content.\nAll rights reserved by the author.\nContinued text.",
            "Some text.\nProprietary and confidential information.\nMore text."
        ]
        
        for text in test_cases:
            cleaned_text, operation = remover.clean_text(text)
            
            # Should remove copyright lines
            assert "copyright" not in cleaned_text.lower()
            assert "©" not in cleaned_text
            assert "all rights reserved" not in cleaned_text.lower()
            assert "proprietary and confidential" not in cleaned_text.lower()
            
            # Should preserve other content
            assert "important content" in cleaned_text or "content" in cleaned_text
            assert operation.success is True
            assert operation.items_removed > 0
    
    def test_quiz_pattern_removal(self):
        """Test removal of quiz and question patterns."""
        remover = BoilerplateRemover()
        
        text = """
        This is educational content about cybersecurity.
        
        Quiz: Test your knowledge
        Question 1: What is a firewall?
        Answer: A network security device
        Correct answer is option B
        
        Choose the correct answer from the following:
        Which of the following is true about encryption?
        
        More educational content continues here.
        """
        
        cleaned_text, operation = remover.clean_text(text)
        
        # Should remove quiz-related lines
        assert "quiz" not in cleaned_text.lower()
        assert "question 1" not in cleaned_text.lower()
        assert "correct answer" not in cleaned_text.lower()
        assert "choose the correct" not in cleaned_text.lower()
        assert "which of the following" not in cleaned_text.lower()
        
        # Should preserve educational content
        assert "educational content" in cleaned_text
        assert "cybersecurity" in cleaned_text
        assert operation.success is True
    
    def test_multiple_choice_removal(self):
        """Test removal of multiple choice sequences."""
        remover = BoilerplateRemover()
        
        text = """
        Cybersecurity is important for protecting systems.
        
        a. This is option A about firewalls.
        b. This is option B about encryption.
        c. This is option C about malware.
        d. This is option D about authentication.
        
        The correct implementation involves multiple layers.
        """
        
        cleaned_text, operation = remover.clean_text(text)
        
        # Should remove multiple choice options
        assert "a. This is option A" not in cleaned_text
        assert "b. This is option B" not in cleaned_text
        assert "c. This is option C" not in cleaned_text
        assert "d. This is option D" not in cleaned_text
        
        # Should preserve other content
        assert "Cybersecurity is important" in cleaned_text
        assert "correct implementation" in cleaned_text
        assert operation.success is True
        assert operation.items_removed >= 4  # At least 4 choice lines removed
    
    def test_different_choice_formats(self):
        """Test removal of different multiple choice formats."""
        remover = BoilerplateRemover()
        
        test_cases = [
            # Format: a) b) c) d)
            """
            Content before choices.
            a) First option here
            b) Second option here
            c) Third option here
            Content after choices.
            """,
            # Format: (a) (b) (c) (d)
            """
            Content before choices.
            (a) First option here
            (b) Second option here
            (c) Third option here
            Content after choices.
            """,
            # Format: 1. 2. 3. 4.
            """
            Content before choices.
            1. First numbered option
            2. Second numbered option
            3. Third numbered option
            Content after choices.
            """
        ]
        
        for text in test_cases:
            cleaned_text, operation = remover.clean_text(text)
            
            # Should remove choice sequences
            assert "First option" not in cleaned_text
            assert "Second option" not in cleaned_text
            assert "Third option" not in cleaned_text
            
            # Should preserve surrounding content
            assert "Content before" in cleaned_text
            assert "Content after" in cleaned_text
            assert operation.success is True
    
    def test_header_footer_removal(self):
        """Test removal of headers and footers."""
        remover = BoilerplateRemover({'remove_headers': True, 'remove_footers': True})
        
        text = """
        Page 1
        Chapter 5: Network Security
        ===========================
        
        This is the main content about network security.
        Firewalls are essential components of network defense.
        
        ===========================
        Continued on next page
        End of chapter
        """
        
        cleaned_text, operation = remover.clean_text(text)
        
        # Should remove headers and footers
        assert "Page 1" not in cleaned_text
        assert "Chapter 5" not in cleaned_text
        assert "Continued on next page" not in cleaned_text
        assert "End of chapter" not in cleaned_text
        assert "===" not in cleaned_text
        
        # Should preserve main content
        assert "main content about network security" in cleaned_text
        assert "Firewalls are essential" in cleaned_text
        assert operation.success is True
    
    def test_min_line_length_filtering(self):
        """Test filtering of lines that are too short."""
        remover = BoilerplateRemover({'min_line_length': 15})
        
        text = """
        This is a longer line that should be preserved.
        Short.
        OK
        This is another longer line with cybersecurity content.
        No.
        """
        
        cleaned_text, operation = remover.clean_text(text)
        
        # Should remove short lines
        assert "Short." not in cleaned_text
        assert "OK" not in cleaned_text
        assert "No." not in cleaned_text
        
        # Should preserve longer lines
        assert "longer line that should be preserved" in cleaned_text
        assert "cybersecurity content" in cleaned_text
        assert operation.success is True
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        remover = BoilerplateRemover()
        
        test_cases = ["", "   ", "\n\n\n", "\t\t"]
        
        for text in test_cases:
            cleaned_text, operation = remover.clean_text(text)
            
            assert cleaned_text == text  # Should return unchanged
            assert operation.success is True
            assert operation.items_removed == 0
    
    def test_no_boilerplate_text(self):
        """Test processing of text with no boilerplate content."""
        remover = BoilerplateRemover()
        
        text = """
        Network security is a critical aspect of cybersecurity.
        Firewalls help protect against unauthorized access.
        Encryption ensures data confidentiality and integrity.
        Regular security audits help identify vulnerabilities.
        """
        
        cleaned_text, operation = remover.clean_text(text)
        
        # Should preserve all content
        assert cleaned_text.strip() == text.strip()
        assert operation.success is True
        assert operation.items_removed == 0
        assert "no boilerplate detected" in operation.description
    
    def test_mixed_content_processing(self):
        """Test processing of text with mixed boilerplate and valid content."""
        remover = BoilerplateRemover()
        
        text = """
        Cybersecurity Fundamentals
        
        Network security involves multiple layers of protection.
        
        © 2023 Educational Institution
        All rights reserved.
        
        Firewalls are the first line of defense against attacks.
        
        Quiz Question: What is encryption?
        a. A method of hiding data
        b. A type of firewall
        c. A network protocol
        
        Encryption algorithms use mathematical functions to secure data.
        
        Disclaimer: This content is for educational purposes only.
        """
        
        cleaned_text, operation = remover.clean_text(text)
        
        # Should preserve educational content
        assert "Network security involves" in cleaned_text
        assert "Firewalls are the first line" in cleaned_text
        assert "Encryption algorithms use" in cleaned_text
        
        # Should remove boilerplate
        assert "© 2023" not in cleaned_text
        assert "All rights reserved" not in cleaned_text
        assert "Quiz Question" not in cleaned_text
        assert "a. A method" not in cleaned_text
        assert "Disclaimer" not in cleaned_text
        
        assert operation.success is True
        assert operation.items_removed > 0
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            'custom_patterns': ['(?i)test pattern'],
            'remove_headers': True,
            'remove_footers': False,
            'min_line_length': 5,
            'max_consecutive_choices': 3
        }
        
        remover = BoilerplateRemover(valid_config)
        errors = remover.validate_config()
        assert len(errors) == 0
        
        # Invalid configurations
        invalid_configs = [
            {'custom_patterns': 'not a list'},
            {'custom_patterns': ['[invalid regex']},
            {'min_line_length': 'not a number'},
            {'min_line_length': -1},
            {'remove_headers': 'not a boolean'},
            {'max_consecutive_choices': 0}
        ]
        
        for invalid_config in invalid_configs:
            remover = BoilerplateRemover(invalid_config)
            errors = remover.validate_config()
            assert len(errors) > 0
    
    def test_statistics_generation(self):
        """Test statistics generation."""
        config = {
            'custom_patterns': ['pattern1', 'pattern2'],
            'remove_headers': True,
            'min_line_length': 10
        }
        
        remover = BoilerplateRemover(config)
        stats = remover.get_statistics()
        
        assert 'total_patterns' in stats
        assert 'default_patterns' in stats
        assert 'custom_patterns' in stats
        assert stats['custom_patterns'] == 2
        assert stats['remove_headers'] is True
        assert stats['min_line_length'] == 10
    
    def test_operation_metadata(self):
        """Test that operations include proper metadata."""
        remover = BoilerplateRemover()
        
        text = """
        Valid content here.
        Copyright 2023 Test Corp.
        More valid content.
        """
        
        cleaned_text, operation = remover.clean_text(text)
        
        assert operation.operation_type == CleaningOperationType.BOILERPLATE_REMOVAL
        assert operation.module_name == "BoilerplateRemover"
        assert operation.success is True
        assert operation.original_length > operation.final_length
        assert operation.items_removed > 0
        
        # Check metadata
        assert 'original_lines' in operation.metadata
        assert 'final_lines' in operation.metadata
        assert 'removed_patterns' in operation.metadata
        assert 'patterns_used' in operation.metadata
    
    def test_consecutive_choice_threshold(self):
        """Test configurable consecutive choice threshold."""
        # Test with threshold of 3 (should not remove 2 choices)
        remover = BoilerplateRemover({'max_consecutive_choices': 3})
        
        text = """
        Content before.
        a. First choice
        b. Second choice
        Content after.
        """
        
        cleaned_text, operation = remover.clean_text(text)
        
        # Should preserve choices (only 2, threshold is 3)
        assert "a. First choice" in cleaned_text
        assert "b. Second choice" in cleaned_text
        
        # Test with threshold of 2 (should remove 2 choices)
        remover = BoilerplateRemover({'max_consecutive_choices': 2})
        
        cleaned_text, operation = remover.clean_text(text)
        
        # Should remove choices (2 choices, threshold is 2)
        assert "a. First choice" not in cleaned_text
        assert "b. Second choice" not in cleaned_text
        assert "Content before" in cleaned_text
        assert "Content after" in cleaned_text


if __name__ == "__main__":
    pytest.main([__file__])