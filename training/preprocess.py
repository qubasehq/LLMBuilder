"""
Data preprocessing script for LLM training.
Extracts and cleans text from PDF, DOCX, and TXT files.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Document processing imports
try:
    import fitz  # PyMuPDF
except ImportError:
    logger.warning("PyMuPDF not available. PDF processing will be disabled.")
    fitz = None

try:
    import docx2txt
except ImportError:
    logger.warning("docx2txt not available. DOCX processing will be disabled.")
    docx2txt = None

try:
    import markdown
    from bs4 import BeautifulSoup
except ImportError:
    logger.warning("markdown/beautifulsoup4 not available. Enhanced markdown processing will be disabled.")
    markdown = None
    BeautifulSoup = None

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from training.utils import setup_logging


class TextProcessor:
    """Handles text extraction and cleaning from various file formats."""
    
    def __init__(self, 
                 min_length: int = 100,
                 max_length: int = 1000000,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 normalize_whitespace: bool = True):
        """
        Initialize text processor.
        
        Args:
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize_whitespace: Whether to normalize whitespace
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Additional cleaning patterns
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.special_chars_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]')
        self.repeated_chars_pattern = re.compile(r'(.)\1{4,}')  # Remove 5+ repeated chars
        self.markdown_link_pattern = re.compile(r'\[([^\]]+)\]\([^\)]+\)')  # [text](url)
        self.markdown_image_pattern = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')  # ![alt](url)
        
        logger.info("Text processor initialized")
    
    def extract_from_txt(self, file_path: Path) -> Optional[str]:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return text
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            return None
    
    def extract_from_pdf(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF file with enhanced error handling."""
        if fitz is None:
            logger.warning(f"Skipping PDF {file_path}: PyMuPDF not available")
            return None
        
        try:
            # Try to open PDF with different methods
            doc = None
            text = ""
            
            try:
                doc = fitz.open(file_path)
            except Exception as e:
                logger.warning(f"Failed to open PDF {file_path} normally: {e}")
                # Try to open as stream for corrupted PDFs
                try:
                    with open(file_path, 'rb') as f:
                        doc = fitz.open(stream=f.read(), filetype="pdf")
                    logger.info(f"Successfully opened PDF {file_path} as stream")
                except Exception as e2:
                    logger.error(f"Failed to open PDF {file_path} as stream: {e2}")
                    return None
            
            if doc is None:
                return None
                
            # Extract text from all pages
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    # If regular text extraction fails, try OCR-like extraction
                    if not page_text.strip():
                        try:
                            # Try different text extraction methods
                            page_text = page.get_text("text", flags=11)  # More aggressive extraction
                        except:
                            page_text = ""
                    
                    text += page_text + "\n"
                    
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num} of {file_path}: {e}")
                    continue
            
            doc.close()
            
            if not text.strip():
                logger.warning(f"No text extracted from PDF {file_path}")
                return None
                
            return text
            
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            return None
    
    def extract_from_docx(self, file_path: Path) -> Optional[str]:
        """Extract text from DOCX file."""
        if docx2txt is None:
            logger.warning(f"Skipping DOCX {file_path}: docx2txt not available")
            return None
        
        try:
            text = docx2txt.process(str(file_path))
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX file {file_path}: {e}")
            return None
    
    def extract_from_md(self, file_path: Path) -> Optional[str]:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # If markdown library is available, convert to HTML then extract text
            if markdown is not None and BeautifulSoup is not None:
                try:
                    # Convert markdown to HTML
                    html = markdown.markdown(content, extensions=['extra', 'codehilite'])
                    # Extract text from HTML
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    return text
                except Exception as e:
                    logger.warning(f"Failed to process markdown with library for {file_path}: {e}")
                    # Fall back to basic processing
            
            # Basic markdown processing - remove common markdown syntax
            text = content
            
            # Remove markdown headers
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
            
            # Remove markdown links but keep text: [text](url) -> text
            text = self.markdown_link_pattern.sub(r'\1', text)
            
            # Remove markdown images
            text = self.markdown_image_pattern.sub('', text)
            
            # Remove markdown code blocks
            text = re.sub(r'```[\s\S]*?```', '', text)
            text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
            
            # Remove markdown emphasis
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
            text = re.sub(r'__([^_]+)__', r'\1', text)  # Bold
            text = re.sub(r'_([^_]+)_', r'\1', text)  # Italic
            
            # Remove markdown lists
            text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
            
            # Remove horizontal rules
            text = re.sub(r'^[\s]*[-*_]{3,}[\s]*$', '', text, flags=re.MULTILINE)
            
            return text
            
        except Exception as e:
            logger.error(f"Error reading Markdown file {file_path}: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text with enhanced cleaning."""
        if not text:
            return ""
        
        # Remove special control characters
        text = self.special_chars_pattern.sub('', text)
        
        # Remove HTML tags (in case any remain)
        text = self.html_tag_pattern.sub('', text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Remove repeated characters (like ===== or -----)
        text = self.repeated_chars_pattern.sub(r'\1\1\1', text)  # Keep max 3 repetitions
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove lines that are mostly special characters or numbers
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Skip very short lines
                continue
            # Skip lines that are mostly numbers, punctuation, or special chars
            if len(re.sub(r'[\d\s\W]', '', line)) < len(line) * 0.3:
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def process_file(self, file_path: Path) -> Optional[str]:
        """Process a single file and extract clean text."""
        logger.info(f"Processing {file_path}")
        
        # Extract text based on file extension
        text = None
        ext = file_path.suffix.lower()
        
        if ext == '.txt':
            text = self.extract_from_txt(file_path)
        elif ext == '.pdf':
            text = self.extract_from_pdf(file_path)
        elif ext == '.docx':
            text = self.extract_from_docx(file_path)
        elif ext == '.md':
            text = self.extract_from_md(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None
        
        if text is None:
            logger.warning(f"Failed to extract text from {file_path}")
            return None
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Check length constraints
        if len(cleaned_text) < self.min_length:
            logger.warning(f"Text too short ({len(cleaned_text)} chars): {file_path}")
            return None
        
        if len(cleaned_text) > self.max_length:
            logger.warning(f"Text too long ({len(cleaned_text)} chars), truncating: {file_path}")
            cleaned_text = cleaned_text[:self.max_length]
        
        logger.info(f"Extracted {len(cleaned_text)} characters from {file_path}")
        return cleaned_text


class DataPreprocessor:
    """Main data preprocessing pipeline."""
    
    def __init__(self, 
                 raw_data_dir: str = "data/raw",
                 cleaned_data_dir: str = "data/cleaned",
                 **processor_kwargs):
        """
        Initialize data preprocessor.
        
        Args:
            raw_data_dir: Directory containing raw data files
            cleaned_data_dir: Directory to save cleaned data
            **processor_kwargs: Arguments for TextProcessor
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.cleaned_data_dir = Path(cleaned_data_dir)
        self.processor = TextProcessor(**processor_kwargs)
        
        # Create output directory
        self.cleaned_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data preprocessor initialized")
        logger.info(f"Raw data dir: {self.raw_data_dir}")
        logger.info(f"Cleaned data dir: {self.cleaned_data_dir}")
    
    def find_files(self) -> List[Path]:
        """Find all supported files in raw data directory."""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")
        
        supported_extensions = ['.txt', '.pdf', '.docx', '.md']
        files = []
        
        for ext in supported_extensions:
            files.extend(self.raw_data_dir.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(files)} files to process")
        return sorted(files)
    
    def process_all(self) -> Dict[str, Any]:
        """Process all files in the raw data directory."""
        logger.info("Starting data preprocessing...")
        
        files = self.find_files()
        
        if not files:
            logger.warning("No files found to process")
            return {
                'processed': 0,
                'failed': 0,
                'total_chars': 0,
                'output_files': []
            }
        
        processed_count = 0
        failed_count = 0
        total_chars = 0
        output_files = []
        
        for file_path in files:
            try:
                # Process file
                cleaned_text = self.processor.process_file(file_path)
                
                if cleaned_text is None:
                    failed_count += 1
                    continue
                
                # Save cleaned text
                output_filename = f"{file_path.stem}_cleaned.txt"
                output_path = self.cleaned_data_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                processed_count += 1
                total_chars += len(cleaned_text)
                output_files.append(output_path)
                
                logger.info(f"Saved cleaned text to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                failed_count += 1
                continue
        
        # Create combined file
        if output_files:
            combined_path = self.cleaned_data_dir / "combined_text.txt"
            with open(combined_path, 'w', encoding='utf-8') as combined_file:
                for output_file in output_files:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        combined_file.write(f.read())
                        combined_file.write("\n\n")  # Separator between documents
            
            logger.info(f"Created combined file: {combined_path}")
            output_files.append(combined_path)
        
        results = {
            'processed': processed_count,
            'failed': failed_count,
            'total_chars': total_chars,
            'output_files': output_files
        }
        
        logger.info(f"Preprocessing complete: {processed_count} processed, {failed_count} failed")
        logger.info(f"Total characters: {total_chars:,}")
        
        return results


def main():
    """Main entry point."""
    # Setup logging
    setup_logging(log_dir="logs", level="INFO")
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            min_length=50,  # Minimum 50 characters
            max_length=500000,  # Maximum 500k characters per file
            remove_urls=True,
            remove_emails=True,
            normalize_whitespace=True
        )
        
        # Process all files
        results = preprocessor.process_all()
        
        # Print summary
        logger.info("=== Preprocessing Summary ===")
        logger.info(f"Files processed: {results['processed']}")
        logger.info(f"Files failed: {results['failed']}")
        logger.info(f"Total characters: {results['total_chars']:,}")
        logger.info(f"Output files: {len(results['output_files'])}")
        
        if results['processed'] == 0:
            logger.warning("No files were successfully processed!")
            logger.info("Please check that you have files in the data/raw directory")
            logger.info("Supported formats: .txt, .pdf, .docx, .md")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

