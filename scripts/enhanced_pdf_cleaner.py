#!/usr/bin/env python3
"""
Enhanced PDF text cleaner to fix common extraction issues.
Addresses problems like scattered page numbers, reference numbers, and fragmented text.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class EnhancedPDFTextCleaner:
    """Advanced PDF text cleaner that handles common extraction artifacts."""
    
    def __init__(self):
        self.name = "EnhancedPDFTextCleaner"
        
        # Common patterns that indicate non-content text
        self.noise_patterns = [
            # Page numbers (standalone numbers)
            r'^\s*\d{1,4}\s*$',
            
            # Reference numbers in brackets
            r'^\s*\[\s*\d+\s*\]\s*$',
            
            # Table of contents entries (number followed by dots/spaces and number)
            r'^\s*\d+\.?\s*[\.\s]+\d+\s*$',
            
            # Headers/footers with just numbers and basic words
            r'^\s*(?:page|chapter|section)?\s*\d+\s*$',
            
            # Single letters or very short fragments
            r'^\s*[a-zA-Z]\s*$',
            
            # Lines with mostly punctuation
            r'^\s*[^\w\s]*\s*$',
            
            # Repeated characters (OCR artifacts)
            r'^(.)\1{4,}$',
            
            # Common PDF metadata lines
            r'^\s*(?:created|modified|author|title|subject):\s*.*$',
            
            # Navigation elements
            r'^\s*(?:next|previous|back|home|contents?|index)\s*$',
        ]
        
        # Patterns for reference numbers that appear mid-text
        self.inline_reference_patterns = [
            # Reference numbers like [123] or (123)
            r'\[\s*\d{1,4}\s*\]',
            r'\(\s*\d{1,4}\s*\)',
            
            # Superscript-style references
            r'\s+\d{1,3}\s+(?=[A-Z])',  # Number followed by capital letter
            
            # Page references scattered in text
            r'\s+\d{1,4}\s+(?=\w)',  # Isolated numbers in text flow
        ]
        
        # Common cybersecurity/technical terms to preserve context
        self.technical_terms = {
            'nmap', 'port', 'scan', 'vulnerability', 'exploit', 'payload',
            'tcp', 'udp', 'http', 'https', 'ssl', 'tls', 'dns', 'ip',
            'firewall', 'proxy', 'vpn', 'authentication', 'authorization',
            'encryption', 'decryption', 'hash', 'algorithm', 'protocol',
            'security', 'penetration', 'testing', 'reconnaissance',
            'enumeration', 'brute', 'force', 'injection', 'xss', 'csrf',
            'sql', 'database', 'server', 'client', 'network', 'system'
        }
    
    def clean_pdf_text(self, text: str) -> str:
        """
        Clean PDF extracted text by removing common artifacts.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Cleaned text with artifacts removed
        """
        if not text:
            return ""
        
        logger.info("Starting enhanced PDF text cleaning...")
        
        # Step 1: Basic normalization
        text = self._normalize_whitespace(text)
        
        # Step 2: Remove obvious noise lines
        text = self._remove_noise_lines(text)
        
        # Step 3: Clean inline references and scattered numbers
        text = self._clean_inline_references(text)
        
        # Step 4: Fix fragmented sentences
        text = self._fix_fragmented_sentences(text)
        
        # Step 5: Remove orphaned single words/characters
        text = self._remove_orphaned_fragments(text)
        
        # Step 6: Reconstruct paragraphs
        text = self._reconstruct_paragraphs(text)
        
        # Step 7: Final cleanup
        text = self._final_cleanup(text)
        
        logger.info("PDF text cleaning completed")
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Remove trailing spaces from lines
        lines = [line.rstrip() for line in text.split('\n')]
        
        return '\n'.join(lines)
    
    def _remove_noise_lines(self, text: str) -> str:
        """Remove lines that are clearly noise/artifacts."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines (we'll handle paragraph breaks later)
            if not line:
                cleaned_lines.append('')
                continue
            
            # Check against noise patterns
            is_noise = False
            for pattern in self.noise_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_noise = True
                    logger.debug(f"Removing noise line: '{line}'")
                    break
            
            if not is_noise:
                # Additional heuristic checks
                if self._is_likely_noise(line):
                    logger.debug(f"Removing likely noise: '{line}'")
                    continue
                
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_likely_noise(self, line: str) -> bool:
        """Additional heuristics to identify noise lines."""
        line = line.strip()
        
        # Very short lines with no meaningful content
        if len(line) < 3:
            return True
        
        # Lines with mostly numbers and minimal text
        alpha_chars = len(re.findall(r'[a-zA-Z]', line))
        digit_chars = len(re.findall(r'\d', line))
        
        if len(line) > 0:
            alpha_ratio = alpha_chars / len(line)
            digit_ratio = digit_chars / len(line)
            
            # If line is mostly digits with very little text
            if digit_ratio > 0.7 and alpha_ratio < 0.2:
                return True
            
            # If line has very few alphabetic characters
            if alpha_ratio < 0.3 and len(line) < 20:
                return True
        
        # Lines that look like headers/footers
        header_footer_indicators = [
            'page', 'chapter', 'section', 'figure', 'table',
            'appendix', 'index', 'contents', 'bibliography'
        ]
        
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in header_footer_indicators):
            # If it's just the indicator with numbers/minimal text
            words = re.findall(r'\b\w+\b', line_lower)
            if len(words) <= 3:
                return True
        
        return False
    
    def _clean_inline_references(self, text: str) -> str:
        """Remove inline reference numbers and scattered digits."""
        # Remove reference numbers in brackets/parentheses
        for pattern in self.inline_reference_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Handle scattered numbers that break text flow
        # Look for patterns like "word 123 word" where 123 is likely a page/ref number
        text = re.sub(r'(\w+)\s+\d{1,4}\s+([a-z])', r'\1 \2', text)
        
        # Remove standalone numbers between words (but preserve technical numbers)
        # This is tricky - we want to keep legitimate numbers but remove page numbers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Split line into words
            words = line.split()
            cleaned_words = []
            
            for i, word in enumerate(words):
                # Check if word is just a number
                if re.match(r'^\d{1,4}$', word):
                    # Look at context to decide if it's a legitimate number
                    prev_word = words[i-1] if i > 0 else ""
                    next_word = words[i+1] if i < len(words)-1 else ""
                    
                    # Keep numbers that seem legitimate
                    if self._is_legitimate_number(word, prev_word, next_word):
                        cleaned_words.append(word)
                    else:
                        logger.debug(f"Removing scattered number: '{word}' (context: '{prev_word}' ... '{next_word}')")
                        # Don't add the word (effectively removing it)
                        pass
                else:
                    cleaned_words.append(word)
            
            cleaned_lines.append(' '.join(cleaned_words))
        
        return '\n'.join(cleaned_lines)
    
    def _is_legitimate_number(self, number: str, prev_word: str, next_word: str) -> bool:
        """Determine if a number should be kept based on context."""
        # Technical contexts where numbers are important
        technical_contexts = [
            'port', 'tcp', 'udp', 'http', 'https', 'ssl', 'tls',
            'version', 'ip', 'address', 'protocol', 'rfc',
            'cve', 'exploit', 'vulnerability', 'scan', 'nmap'
        ]
        
        prev_lower = prev_word.lower()
        next_lower = next_word.lower()
        
        # Keep numbers in technical contexts
        if any(term in prev_lower or term in next_lower for term in technical_contexts):
            return True
        
        # Keep numbers that are part of technical terms
        if any(term in prev_lower for term in self.technical_terms):
            return True
        
        # Keep numbers followed by units or technical terms
        if next_lower in ['mb', 'gb', 'kb', 'bytes', 'bits', 'ms', 'seconds', 'minutes']:
            return True
        
        # Keep numbers in version-like contexts
        if '.' in prev_word or '.' in next_word:
            return True
        
        # Keep numbers that are clearly part of technical specifications
        if re.match(r'[a-zA-Z]+\d+', prev_word) or re.match(r'\d+[a-zA-Z]+', next_word):
            return True
        
        # Remove standalone numbers that are likely page/reference numbers
        # These typically have no meaningful context
        if not prev_word or not next_word:
            return False
        
        # If surrounded by common words, likely a page number
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        if prev_lower in common_words or next_lower in common_words:
            return False
        
        # Default: keep the number if we're not sure
        return True
    
    def _fix_fragmented_sentences(self, text: str) -> str:
        """Fix sentences that were broken across lines."""
        lines = text.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                fixed_lines.append('')
                i += 1
                continue
            
            # Check if current line looks like a sentence fragment
            if self._is_sentence_fragment(current_line) and i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                
                # If next line continues the sentence, merge them
                if next_line and self._should_merge_lines(current_line, next_line):
                    merged_line = current_line + ' ' + next_line
                    fixed_lines.append(merged_line)
                    i += 2  # Skip next line since we merged it
                    continue
            
            fixed_lines.append(current_line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _is_sentence_fragment(self, line: str) -> bool:
        """Check if a line looks like an incomplete sentence."""
        line = line.strip()
        
        # Empty lines are not fragments
        if not line:
            return False
        
        # Lines ending with sentence terminators are complete
        if line.endswith(('.', '!', '?', ':', ';')):
            return False
        
        # Very short lines might be fragments
        if len(line) < 50:
            return True
        
        # Lines that don't start with capital letters might be continuations
        if line and not line[0].isupper():
            return True
        
        return False
    
    def _should_merge_lines(self, line1: str, line2: str) -> bool:
        """Determine if two lines should be merged."""
        # Don't merge if either line is empty
        if not line1.strip() or not line2.strip():
            return False
        
        # Don't merge if second line starts with capital (likely new sentence)
        if line2[0].isupper():
            # Exception: if first line clearly ends mid-word or mid-phrase
            if line1.endswith((',', 'and', 'or', 'but', 'the', 'a', 'an')):
                return True
            return False
        
        # Merge if first line ends with comma or conjunction
        if line1.endswith((',', 'and', 'or', 'but')):
            return True
        
        # Merge if second line starts with lowercase (continuation)
        if line2[0].islower():
            return True
        
        return False
    
    def _remove_orphaned_fragments(self, text: str) -> str:
        """Remove single words or very short fragments that are likely artifacts."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                cleaned_lines.append('')
                continue
            
            # Remove very short lines that are likely artifacts
            words = line.split()
            
            # Single character lines (except 'I' or 'a')
            if len(words) == 1 and len(words[0]) == 1:
                if words[0].lower() not in ['i', 'a']:
                    logger.debug(f"Removing orphaned character: '{line}'")
                    continue
            
            # Single word lines that are likely artifacts
            if len(words) == 1:
                word = words[0].lower()
                # Keep important single words
                if word in self.technical_terms or len(word) > 8:
                    cleaned_lines.append(line)
                else:
                    # Check if it's a common artifact
                    if re.match(r'^\d+$', word) or len(word) < 3:
                        logger.debug(f"Removing orphaned word: '{line}'")
                        continue
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _reconstruct_paragraphs(self, text: str) -> str:
        """Reconstruct proper paragraph structure."""
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            # Empty line indicates paragraph break
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            # Add line to current paragraph
            current_paragraph.append(line)
        
        # Don't forget the last paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Filter out very short paragraphs that are likely artifacts
        meaningful_paragraphs = []
        for para in paragraphs:
            if len(para.split()) >= 5:  # At least 5 words
                meaningful_paragraphs.append(para)
            else:
                logger.debug(f"Removing short paragraph: '{para[:50]}...'")
        
        return '\n\n'.join(meaningful_paragraphs)
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and normalization."""
        # Remove excessive whitespace
        text = re.sub(r' +', ' ', text)
        
        # Normalize paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


def clean_pdf_file(input_file: Path, output_file: Path = None) -> bool:
    """
    Clean a single PDF text file.
    
    Args:
        input_file: Path to input text file (extracted from PDF)
        output_file: Path to output cleaned file (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        
        # Clean the text
        cleaner = EnhancedPDFTextCleaner()
        cleaned_text = cleaner.clean_pdf_text(raw_text)
        
        # Determine output file
        if output_file is None:
            output_file = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
        
        # Write cleaned text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        logger.info(f"Cleaned text saved to: {output_file}")
        logger.info(f"Original length: {len(raw_text):,} characters")
        logger.info(f"Cleaned length: {len(cleaned_text):,} characters")
        logger.info(f"Reduction: {((len(raw_text) - len(cleaned_text)) / len(raw_text) * 100):.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning PDF text file {input_file}: {e}")
        return False


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean PDF extracted text files")
    parser.add_argument("input_file", help="Input text file (extracted from PDF)")
    parser.add_argument("-o", "--output", help="Output file (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else None
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    success = clean_pdf_file(input_path, output_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())