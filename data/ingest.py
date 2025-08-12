"""
Enhanced data ingestion system for LLM training.
Supports multiple document formats with OCR fallback and comprehensive error handling.
"""

import os
import re
import sys
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger
import time
import hashlib

# Document processing imports with graceful fallbacks
try:
    import fitz  # PyMuPDF
except ImportError:
    logger.warning("PyMuPDF not available. PDF processing will be disabled.")
    fitz = None

try:
    from bs4 import BeautifulSoup
    import requests
except ImportError:
    logger.warning("BeautifulSoup4 not available. HTML processing will be disabled.")
    BeautifulSoup = None

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    logger.warning("ebooklib not available. EPUB processing will be disabled.")
    ebooklib = None
    epub = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    logger.warning("pytesseract/PIL not available. OCR processing will be disabled.")
    pytesseract = None
    Image = None

try:
    import markdown
except ImportError:
    logger.warning("markdown not available. Enhanced markdown processing will be disabled.")
    markdown = None

try:
    import docx2txt
except ImportError:
    logger.warning("docx2txt not available. DOCX processing will be disabled.")
    docx2txt = None

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    file_path: Path
    file_type: str
    file_size: int
    processing_method: str  # "direct", "ocr", "hybrid"
    extraction_confidence: float
    character_count: int
    word_count: int
    language_detected: Optional[str]
    processing_time: float
    hash_signature: str
    embedding_signature: Optional[str] = None


class HTMLExtractor:
    """Extract clean text from HTML documents using BeautifulSoup with comprehensive cleaning."""
    
    def __init__(self, preserve_structure: bool = True):
        """
        Initialize HTML extractor.
        
        Args:
            preserve_structure: Whether to preserve paragraph structure in output
        """
        self.name = "HTMLExtractor"
        self.preserve_structure = preserve_structure
        
        # Define elements to remove completely
        self.remove_elements = [
            'script', 'style', 'noscript', 'iframe', 'object', 'embed',
            'nav', 'header', 'footer', 'aside', 'menu', 'menuitem',
            'form', 'input', 'button', 'select', 'textarea', 'label',
            'meta', 'link', 'base', 'title', 'head'
        ]
        
        # Define elements that typically contain navigation/UI content
        self.ui_elements = [
            'nav', 'sidebar', 'breadcrumb', 'pagination', 'toolbar',
            'menubar', 'tablist', 'tab', 'dialog', 'alertdialog'
        ]
        
    def extract(self, file_path: Path) -> Optional[str]:
        """Extract text from HTML file with comprehensive cleaning."""
        if BeautifulSoup is None:
            logger.warning(f"Skipping HTML {file_path}: BeautifulSoup not available")
            return None
            
        try:
            # Read file with multiple encoding attempts
            content = self._read_html_file(file_path)
            if not content:
                return None
            
            # Parse HTML with appropriate parser
            soup = self._parse_html(content)
            if not soup:
                return None
            
            # Clean the HTML structure
            self._clean_html_structure(soup)
            
            # Extract text with structure preservation
            text = self._extract_text_with_structure(soup)
            
            # Final text cleaning
            cleaned_text = self._clean_extracted_text(text)
            
            return cleaned_text if cleaned_text else None
            
        except Exception as e:
            logger.error(f"Error extracting HTML from {file_path}: {e}")
            return None
    
    def _read_html_file(self, file_path: Path) -> Optional[str]:
        """Read HTML file with multiple encoding attempts."""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        logger.debug(f"Successfully read HTML with {encoding} encoding")
                        return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading with {encoding}: {e}")
                continue
        
        logger.error(f"Failed to read HTML file with any encoding: {file_path}")
        return None
    
    def _parse_html(self, content: str) -> Optional[BeautifulSoup]:
        """Parse HTML content with appropriate parser."""
        # Try different parsers in order of preference
        parsers = ['html.parser', 'lxml', 'html5lib']
        
        for parser in parsers:
            try:
                soup = BeautifulSoup(content, parser)
                if soup:
                    logger.debug(f"Successfully parsed HTML with {parser}")
                    return soup
            except Exception as e:
                logger.debug(f"Parser {parser} failed: {e}")
                continue
        
        # Fallback to basic parser
        try:
            soup = BeautifulSoup(content, 'html.parser')
            logger.warning("Using fallback HTML parser")
            return soup
        except Exception as e:
            logger.error(f"All HTML parsers failed: {e}")
            return None
    
    def _clean_html_structure(self, soup: BeautifulSoup) -> None:
        """Remove unwanted HTML elements and clean structure."""
        # Remove unwanted elements completely
        for element_name in self.remove_elements:
            for element in soup.find_all(element_name):
                element.decompose()
        
        # Remove elements with specific classes/IDs that indicate UI/navigation
        ui_indicators = [
            'nav', 'navigation', 'menu', 'sidebar', 'header', 'footer',
            'breadcrumb', 'pagination', 'toolbar', 'ads', 'advertisement',
            'social', 'share', 'comment', 'comments', 'related', 'popup',
            'modal', 'overlay', 'banner', 'promo', 'promotion'
        ]
        
        for indicator in ui_indicators:
            # Remove by class
            for element in soup.find_all(class_=re.compile(indicator, re.I)):
                element.decompose()
            # Remove by ID
            for element in soup.find_all(id=re.compile(indicator, re.I)):
                element.decompose()
        
        # Remove elements with ARIA roles that indicate UI components
        ui_roles = ['navigation', 'banner', 'complementary', 'contentinfo', 'search', 'form']
        for role in ui_roles:
            for element in soup.find_all(attrs={'role': role}):
                element.decompose()
        
        # Remove hidden elements
        for element in soup.find_all(style=re.compile(r'display\s*:\s*none', re.I)):
            element.decompose()
        
        # Remove elements with minimal text content that are likely UI
        for element in soup.find_all(['div', 'span', 'section']):
            text = element.get_text(strip=True)
            if len(text) < 10 and any(keyword in text.lower() for keyword in 
                                    ['menu', 'nav', 'skip', 'login', 'search', 'home', 'back']):
                element.decompose()
    
    def _extract_text_with_structure(self, soup: BeautifulSoup) -> str:
        """Extract text while preserving document structure."""
        if not self.preserve_structure:
            return soup.get_text(separator=' ', strip=True)
        
        text_parts = []
        
        # Process main content areas first
        main_content = soup.find('main') or soup.find('article') or soup.find(class_=re.compile(r'content|main|body', re.I))
        
        if main_content:
            text_parts.append(self._extract_structured_text(main_content))
        else:
            # Process body or entire document
            body = soup.find('body') or soup
            text_parts.append(self._extract_structured_text(body))
        
        return '\n\n'.join(filter(None, text_parts))
    
    def _extract_structured_text(self, element) -> str:
        """Extract text from an element while preserving structure."""
        text_parts = []
        
        # Handle different element types
        for child in element.children:
            if hasattr(child, 'name') and child.name:
                # This is an HTML element
                if child.name in ['p', 'div', 'section', 'article']:
                    # Paragraph-like elements
                    text = child.get_text(separator=' ', strip=True)
                    if text and len(text) > 10:  # Filter out very short paragraphs
                        text_parts.append(text)
                
                elif child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Headers
                    text = child.get_text(strip=True)
                    if text:
                        text_parts.append(f"\n{text}\n")
                
                elif child.name in ['ul', 'ol']:
                    # Lists
                    list_items = []
                    for li in child.find_all('li'):
                        item_text = li.get_text(separator=' ', strip=True)
                        if item_text:
                            list_items.append(f"• {item_text}")
                    if list_items:
                        text_parts.append('\n'.join(list_items))
                
                elif child.name in ['table']:
                    # Tables - extract as structured text
                    table_text = self._extract_table_text(child)
                    if table_text:
                        text_parts.append(table_text)
                
                elif child.name in ['blockquote']:
                    # Blockquotes
                    text = child.get_text(separator=' ', strip=True)
                    if text:
                        text_parts.append(f'"{text}"')
                
                else:
                    # Other elements - recurse
                    text = self._extract_structured_text(child)
                    if text:
                        text_parts.append(text)
            else:
                # This is a text node (NavigableString)
                text = str(child).strip()
                if text and len(text) > 5:
                    text_parts.append(text)
        
        return '\n'.join(filter(None, text_parts))
    
    def _extract_table_text(self, table) -> str:
        """Extract text from HTML table in a readable format."""
        rows = []
        
        for tr in table.find_all('tr'):
            cells = []
            for cell in tr.find_all(['td', 'th']):
                cell_text = cell.get_text(separator=' ', strip=True)
                if cell_text:
                    cells.append(cell_text)
            
            if cells:
                rows.append(' | '.join(cells))
        
        return '\n'.join(rows) if rows else ''
    
    def _clean_extracted_text(self, text: str) -> str:
        """Final cleaning of extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove lines that are mostly punctuation or very short
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            # Skip lines that are mostly punctuation or numbers
            alpha_chars = len(re.findall(r'[a-zA-Z]', line))
            if alpha_chars < len(line) * 0.3:
                continue
            
            # Skip lines that look like navigation or UI elements
            if any(keyword in line.lower() for keyword in 
                  ['click here', 'read more', 'continue reading', 'skip to', 'go to', 'home page']):
                continue
            
            cleaned_lines.append(line)
        
        # Rejoin and final cleanup
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()


class EPUBExtractor:
    """Extract text from EPUB files using ebooklib with comprehensive processing."""
    
    def __init__(self, extract_metadata: bool = True, preserve_structure: bool = True):
        """
        Initialize EPUB extractor.
        
        Args:
            extract_metadata: Whether to include book metadata in output
            preserve_structure: Whether to preserve chapter structure
        """
        self.name = "EPUBExtractor"
        self.extract_metadata = extract_metadata
        self.preserve_structure = preserve_structure
        
        # Elements to remove from EPUB content
        self.remove_elements = [
            'script', 'style', 'noscript', 'iframe', 'object', 'embed',
            'nav', 'header', 'footer', 'aside', 'menu',
            'form', 'input', 'button', 'select', 'textarea',
            'meta', 'link', 'base', 'title'
        ]
        
    def extract(self, file_path: Path) -> Optional[str]:
        """Extract text from EPUB file with comprehensive processing."""
        if ebooklib is None or epub is None:
            logger.warning(f"Skipping EPUB {file_path}: ebooklib not available")
            return None
            
        try:
            # Read EPUB file
            book = self._read_epub_file(file_path)
            if not book:
                return None
            
            # Extract metadata if requested
            metadata_text = ""
            if self.extract_metadata:
                metadata_text = self._extract_metadata(book)
            
            # Extract main content
            content_text = self._extract_content(book)
            
            # Combine metadata and content
            parts = [part for part in [metadata_text, content_text] if part]
            full_text = '\n\n'.join(parts)
            
            # Final cleaning
            cleaned_text = self._clean_extracted_text(full_text)
            
            return cleaned_text if cleaned_text else None
            
        except Exception as e:
            logger.error(f"Error extracting EPUB from {file_path}: {e}")
            return None
    
    def _read_epub_file(self, file_path: Path) -> Optional[epub.EpubBook]:
        """Read EPUB file with error handling."""
        try:
            book = epub.read_epub(str(file_path))
            logger.debug(f"Successfully opened EPUB: {file_path}")
            return book
        except Exception as e:
            logger.error(f"Failed to read EPUB file {file_path}: {e}")
            return None
    
    def _extract_metadata(self, book: epub.EpubBook) -> str:
        """Extract metadata from EPUB book."""
        metadata_parts = []
        
        try:
            # Title
            title = book.get_metadata('DC', 'title')
            if title:
                title_text = title[0][0] if title[0] else ""
                if title_text:
                    metadata_parts.append(f"Title: {title_text}")
            
            # Author(s)
            authors = book.get_metadata('DC', 'creator')
            if authors:
                author_names = []
                for author in authors:
                    if author[0]:
                        author_names.append(author[0])
                if author_names:
                    metadata_parts.append(f"Author(s): {', '.join(author_names)}")
            
            # Description
            descriptions = book.get_metadata('DC', 'description')
            if descriptions and descriptions[0][0]:
                desc_text = descriptions[0][0]
                # Clean HTML from description
                if BeautifulSoup:
                    soup = BeautifulSoup(desc_text, 'html.parser')
                    desc_text = soup.get_text(separator=' ', strip=True)
                metadata_parts.append(f"Description: {desc_text}")
            
            # Subject/Keywords
            subjects = book.get_metadata('DC', 'subject')
            if subjects:
                subject_list = [subj[0] for subj in subjects if subj[0]]
                if subject_list:
                    metadata_parts.append(f"Subjects: {', '.join(subject_list)}")
            
        except Exception as e:
            logger.warning(f"Error extracting EPUB metadata: {e}")
        
        return '\n'.join(metadata_parts) if metadata_parts else ""
    
    def _extract_content(self, book: epub.EpubBook) -> str:
        """Extract main content from EPUB book."""
        content_parts = []
        
        try:
            # Get reading order if available
            spine_items = []
            for item_id, linear in book.spine:
                item = book.get_item_with_id(item_id)
                if item:
                    spine_items.append(item)
            
            # If no spine, get all document items
            if not spine_items:
                spine_items = [item for item in book.get_items() 
                             if item.get_type() == ebooklib.ITEM_DOCUMENT]
            
            # Process each content item
            for item in spine_items:
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    item_text = self._extract_item_text(item)
                    if item_text:
                        if self.preserve_structure:
                            # Add chapter separator
                            content_parts.append(f"\n--- Chapter ---\n{item_text}")
                        else:
                            content_parts.append(item_text)
            
        except Exception as e:
            logger.error(f"Error extracting EPUB content: {e}")
        
        return '\n\n'.join(content_parts) if content_parts else ""
    
    def _extract_item_text(self, item) -> str:
        """Extract text from a single EPUB item."""
        try:
            # Get item content
            content = item.get_content()
            if not content:
                return ""
            
            # Parse HTML content
            if not BeautifulSoup:
                # Fallback: basic text extraction
                text = content.decode('utf-8', errors='ignore')
                # Remove basic HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                return re.sub(r'\s+', ' ', text).strip()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove unwanted elements
            for element_name in self.remove_elements:
                for element in soup.find_all(element_name):
                    element.decompose()
            
            # Remove elements with specific classes that indicate navigation
            nav_classes = ['nav', 'navigation', 'toc', 'contents', 'index']
            for nav_class in nav_classes:
                for element in soup.find_all(class_=re.compile(nav_class, re.I)):
                    element.decompose()
            
            # Extract structured text
            if self.preserve_structure:
                text = self._extract_structured_epub_text(soup)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error extracting text from EPUB item: {e}")
            return ""
    
    def _extract_structured_epub_text(self, soup) -> str:
        """Extract text from EPUB content while preserving structure."""
        text_parts = []
        
        # Process different elements
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'blockquote']):
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Headers
                text = element.get_text(strip=True)
                if text and len(text) > 1:
                    text_parts.append(f"\n{text}\n")
            
            elif element.name == 'p':
                # Paragraphs
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 10:  # Filter very short paragraphs
                    text_parts.append(text)
            
            elif element.name == 'blockquote':
                # Blockquotes
                text = element.get_text(separator=' ', strip=True)
                if text:
                    text_parts.append(f'"{text}"')
            
            elif element.name == 'div':
                # Divs - only if they contain substantial text
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 20:
                    # Check if this div doesn't contain other block elements
                    if not element.find(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        text_parts.append(text)
        
        # Handle lists
        for ul in soup.find_all(['ul', 'ol']):
            list_items = []
            for li in ul.find_all('li'):
                item_text = li.get_text(separator=' ', strip=True)
                if item_text:
                    list_items.append(f"• {item_text}")
            if list_items:
                text_parts.append('\n'.join(list_items))
        
        return '\n'.join(filter(None, text_parts))
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted EPUB text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove chapter separators if they're too frequent
        chapter_count = text.count('--- Chapter ---')
        if chapter_count > 50:  # Too many chapters, probably not real chapters
            text = text.replace('--- Chapter ---', '')
        
        # Remove lines that are mostly punctuation or very short
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            # Skip lines that are mostly punctuation or numbers
            alpha_chars = len(re.findall(r'[a-zA-Z]', line))
            if alpha_chars < len(line) * 0.3:
                continue
            
            # Skip common EPUB navigation elements
            if any(keyword in line.lower() for keyword in 
                  ['table of contents', 'next chapter', 'previous chapter', 
                   'go to', 'click here', 'page', 'chapter']):
                if len(line) < 50:  # Only skip short navigation lines
                    continue
            
            cleaned_lines.append(line)
        
        # Rejoin and final cleanup
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()


class EnhancedPDFExtractor:
    """Enhanced PDF extractor with OCR fallback."""
    
    def __init__(self, ocr_languages: List[str] = None, ocr_page_limit: int = 10, ocr_dpi: int = 300):
        self.name = "EnhancedPDFExtractor"
        self.ocr_languages = ocr_languages or ['eng']
        self.ocr_page_limit = ocr_page_limit
        self.ocr_dpi = ocr_dpi
        self.confidence_threshold = 0.5
        
    def extract(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF with OCR fallback."""
        if fitz is None:
            logger.warning(f"Skipping PDF {file_path}: PyMuPDF not available")
            return None
            
        try:
            # Try direct text extraction first
            text, confidence = self._extract_direct(file_path)
            
            # If direct extraction failed or has low confidence, try OCR
            if not text or confidence < 0.5:
                logger.info(f"Attempting OCR for {file_path} (confidence: {confidence:.2f})")
                ocr_text = self._extract_with_ocr(file_path)
                if ocr_text:
                    return ocr_text
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF from {file_path}: {e}")
            return None
    
    def _extract_direct(self, file_path: Path) -> tuple[Optional[str], float]:
        """Extract text directly from PDF."""
        try:
            doc = fitz.open(file_path)
            text_parts = []
            total_chars = 0
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        text_parts.append(page_text)
                        total_chars += len(page_text)
                        
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                    continue
            
            doc.close()
            
            if not text_parts:
                return None, 0.0
            
            full_text = '\n'.join(text_parts)
            
            # Calculate confidence based on text characteristics
            confidence = self._calculate_text_confidence(full_text)
            
            return full_text, confidence
            
        except Exception as e:
            logger.error(f"Error in direct PDF extraction: {e}")
            return None, 0.0
    
    def _extract_with_ocr(self, file_path: Path) -> Optional[str]:
        """Extract text using OCR with enhanced configuration."""
        if pytesseract is None or Image is None:
            logger.warning("OCR not available: pytesseract or PIL not installed")
            return None
        
        # Check if Tesseract is actually available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract OCR engine not found: {e}")
            logger.info("Please install Tesseract OCR. See INSTALL_TESSERACT.md for instructions.")
            return None
            
        try:
            doc = fitz.open(file_path)
            text_parts = []
            successful_pages = 0
            
            # Limit pages for OCR to avoid excessive processing time
            max_pages = min(doc.page_count, self.ocr_page_limit)
            logger.info(f"Starting OCR processing for {max_pages} pages (limit: {self.ocr_page_limit})")
            
            for page_num in range(max_pages):
                try:
                    page = doc[page_num]
                    
                    # Convert page to image with configurable DPI
                    zoom_factor = self.ocr_dpi / 72.0  # 72 DPI is default
                    matrix = fitz.Matrix(zoom_factor, zoom_factor)
                    pix = page.get_pixmap(matrix=matrix)
                    img_data = pix.tobytes("png")
                    
                    # OCR the image
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Enhanced OCR configuration
                    # OEM 3: Default, based on what is available
                    # PSM 6: Uniform block of text
                    config = f'--oem 3 --psm 6 -l {"+".join(self.ocr_languages)}'
                    
                    # Add additional OCR options for better accuracy
                    config += ' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}"\'-/\\@#$%^&*+=<>|~`'
                    
                    ocr_text = pytesseract.image_to_string(image, config=config)
                    
                    if ocr_text.strip():
                        # Clean OCR artifacts
                        cleaned_text = self._clean_ocr_text(ocr_text)
                        if cleaned_text:
                            text_parts.append(cleaned_text)
                            successful_pages += 1
                            logger.debug(f"OCR page {page_num + 1}: {len(cleaned_text)} characters")
                        
                except Exception as e:
                    logger.warning(f"Error OCR processing page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
            if text_parts:
                combined_text = '\n\n'.join(text_parts)
                logger.info(f"OCR completed: {successful_pages}/{max_pages} pages processed, {len(combined_text)} characters extracted")
                return combined_text
            
            logger.warning("OCR processing completed but no text was extracted")
            return None
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return None
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean common OCR artifacts and errors."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove lines with mostly special characters (OCR noise)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Skip very short lines
                continue
            
            # Skip lines that are mostly non-alphabetic (likely OCR errors)
            alpha_chars = len(re.findall(r'[a-zA-Z]', line))
            if alpha_chars < len(line) * 0.3:  # Less than 30% alphabetic
                continue
            
            # Skip lines with excessive repeated characters (OCR artifacts)
            if re.search(r'(.)\1{5,}', line):  # 6+ repeated characters
                continue
            
            cleaned_lines.append(line)
        
        # Rejoin lines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Fix common OCR character substitutions
        ocr_fixes = {
            r'\b0\b': 'O',  # Zero to O
            r'\b1\b': 'I',  # One to I (in context)
            r'rn': 'm',     # Common OCR error
            r'cl': 'd',     # Common OCR error
            r'vv': 'w',     # Common OCR error
        }
        
        for pattern, replacement in ocr_fixes.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        return cleaned_text.strip()
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text using multiple heuristics."""
        if not text:
            return 0.0
        
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Metric 1: Readable character ratio
        readable_chars = len(re.findall(r'[a-zA-Z0-9\s.,!?;:()"\'-]', text))
        readable_ratio = readable_chars / total_chars
        
        # Metric 2: Word density (words per 100 characters)
        words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        word_density = words / (total_chars / 100) if total_chars > 0 else 0
        word_density_score = min(word_density / 15, 1.0)  # Normalize to 0-1
        
        # Metric 3: Average word length (good text has reasonable word lengths)
        if words > 0:
            avg_word_length = sum(len(word) for word in re.findall(r'\b[a-zA-Z]+\b', text)) / words
            # Optimal average word length is around 4-6 characters
            word_length_score = 1.0 - abs(avg_word_length - 5) / 10
            word_length_score = max(0, min(1, word_length_score))
        else:
            word_length_score = 0.0
        
        # Metric 4: Sentence structure (presence of punctuation)
        sentences = len(re.findall(r'[.!?]+', text))
        sentence_density = sentences / (total_chars / 500) if total_chars > 0 else 0
        sentence_score = min(sentence_density, 1.0)
        
        # Metric 5: Check for excessive repeated characters (indicates OCR errors)
        repeated_chars = len(re.findall(r'(.)\1{3,}', text))  # 4+ repeated chars
        repetition_penalty = min(repeated_chars / (total_chars / 100), 0.5)
        
        # Combine metrics with weights
        confidence = (
            readable_ratio * 0.3 +
            word_density_score * 0.25 +
            word_length_score * 0.2 +
            sentence_score * 0.15 +
            (1.0 - repetition_penalty) * 0.1
        )
        
        # Adjust for text length
        if total_chars > 1000:
            confidence *= 1.05  # Small bonus for longer texts
        elif total_chars < 50:
            confidence *= 0.5   # Significant penalty for very short texts
        elif total_chars < 100:
            confidence *= 0.7   # Moderate penalty for short texts
        
        return min(confidence, 1.0)


class EnhancedMarkdownExtractor:
    """Enhanced markdown extractor with better formatting handling."""
    
    def __init__(self):
        self.name = "EnhancedMarkdownExtractor"
        
    def extract(self, file_path: Path) -> Optional[str]:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Use markdown library if available for better processing
            if markdown is not None and BeautifulSoup is not None:
                try:
                    # Convert markdown to HTML
                    html = markdown.markdown(
                        content, 
                        extensions=['extra', 'codehilite', 'toc', 'tables']
                    )
                    
                    # Extract clean text from HTML
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    
                    # Additional cleanup for markdown artifacts that survived HTML conversion
                    text = self._clean_markdown_artifacts(text)
                    
                    return self._clean_text(text)
                    
                except Exception as e:
                    logger.warning(f"Failed to process markdown with library: {e}")
                    # Fall back to basic processing
            
            # Basic markdown processing
            return self._basic_markdown_processing(content)
            
        except Exception as e:
            logger.error(f"Error extracting Markdown from {file_path}: {e}")
            return None
    
    def _basic_markdown_processing(self, content: str) -> str:
        """Basic markdown processing without external libraries."""
        text = content
        
        # Remove markdown headers but keep the text
        text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
        
        # Remove markdown links but keep text: [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove markdown images
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        
        # Remove code blocks (multiline first, then inline)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
        
        # Remove emphasis markers (be more careful with nested patterns)
        text = re.sub(r'\*\*([^*]+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\1', text)  # Italic (not part of bold)
        text = re.sub(r'__([^_]+?)__', r'\1', text)  # Bold
        text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'\1', text)  # Italic (not part of bold)
        
        # Remove list markers
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules
        text = re.sub(r'^[\s]*[-*_]{3,}[\s]*$', '', text, flags=re.MULTILINE)
        
        return self._clean_text(text)
    
    def _clean_markdown_artifacts(self, text: str) -> str:
        """Clean markdown artifacts that survived HTML conversion."""
        # Remove spaced-out markdown syntax that HTML conversion creates
        text = re.sub(r'\*\s*\*\s*([^*]+?)\s*\*\s*\*', r'\1', text)  # **bold** with spaces
        text = re.sub(r'(?<!\*)\*\s+([^*]+?)\s+\*(?!\*)', r'\1', text)  # *italic* with spaces
        text = re.sub(r'__\s*([^_]+?)\s*__', r'\1', text)  # __bold__ with spaces
        text = re.sub(r'(?<!_)_\s+([^_]+?)\s+_(?!_)', r'\1', text)  # _italic_ with spaces
        
        # Remove spaced-out links: [ text ] ( url )
        text = re.sub(r'\[\s*([^\]]+?)\s*\]\s*\(\s*[^\)]+?\s*\)', r'\1', text)
        
        # Remove spaced-out images: ! [ alt ] ( url )
        text = re.sub(r'!\s*\[\s*([^\]]*?)\s*\]\s*\(\s*[^\)]+?\s*\)', '', text)
        
        # Remove spaced-out code blocks: ` ` ` language ... ` ` `
        text = re.sub(r'`\s*`\s*`[\s\S]*?`\s*`\s*`', '', text)
        
        # Remove spaced-out inline code: ` code `
        text = re.sub(r'`\s+([^`]+?)\s+`', r'\1', text)
        
        # Remove header markers with spaces: # # # Header
        text = re.sub(r'#{1,6}\s+', '', text)
        
        # Remove horizontal rules with spaces: - - -
        text = re.sub(r'[\-*_]\s+[\-*_]\s+[\-*_][\s\-*_]*', '', text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()


class DocumentIngester:
    """Main document ingestion orchestrator."""
    
    def __init__(self, 
                 output_dir: str = "data/cleaned",
                 ocr_languages: List[str] = None,
                 max_file_size_mb: int = 100):
        """
        Initialize document ingester.
        
        Args:
            output_dir: Directory to save cleaned text files
            ocr_languages: Languages for OCR processing
            max_file_size_mb: Maximum file size to process (MB)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        
        # Initialize extractors
        self.extractors = {
            '.html': HTMLExtractor(),
            '.htm': HTMLExtractor(),
            '.epub': EPUBExtractor(),
            '.pdf': EnhancedPDFExtractor(ocr_languages or ['eng']),
            '.md': EnhancedMarkdownExtractor(),
            '.markdown': EnhancedMarkdownExtractor(),
            '.txt': self._extract_txt,
            '.docx': self._extract_docx,
        }
        
        logger.info(f"DocumentIngester initialized with output dir: {self.output_dir}")
        logger.info(f"Supported formats: {list(self.extractors.keys())}")
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        return list(self.extractors.keys())
    
    def ingest_file(self, file_path: Path) -> Optional[DocumentMetadata]:
        """
        Ingest single file and return metadata.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            DocumentMetadata if successful, None otherwise
        """
        start_time = time.time()
        
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                logger.warning(f"File too large ({file_size / 1024 / 1024:.1f}MB): {file_path}")
                return None
            
            # Check if file type is supported
            ext = file_path.suffix.lower()
            if ext not in self.extractors:
                logger.warning(f"Unsupported file type {ext}: {file_path}")
                return None
            
            logger.info(f"Processing {file_path} ({file_size / 1024:.1f}KB)")
            
            # Extract text
            extractor = self.extractors[ext]
            if callable(extractor):
                text = extractor(file_path)
                processing_method = "direct"
            else:
                text = extractor.extract(file_path)
                processing_method = getattr(extractor, 'name', 'unknown')
            
            if not text:
                logger.warning(f"No text extracted from {file_path}")
                return None
            
            # Clean and validate text
            cleaned_text = self._clean_text(text)
            if len(cleaned_text) < 50:  # Minimum text length
                logger.warning(f"Text too short ({len(cleaned_text)} chars): {file_path}")
                return None
            
            # Save cleaned text
            output_file = self.output_dir / f"{file_path.stem}_cleaned.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Calculate metadata
            processing_time = time.time() - start_time
            word_count = len(cleaned_text.split())
            hash_signature = hashlib.md5(cleaned_text.encode()).hexdigest()
            
            metadata = DocumentMetadata(
                file_path=file_path,
                file_type=ext,
                file_size=file_size,
                processing_method=processing_method,
                extraction_confidence=1.0,  # TODO: Implement confidence calculation
                character_count=len(cleaned_text),
                word_count=word_count,
                language_detected=None,  # TODO: Implement language detection
                processing_time=processing_time,
                hash_signature=hash_signature
            )
            
            logger.info(f"Successfully processed {file_path}: {len(cleaned_text)} chars, {word_count} words")
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def ingest_directory(self, input_dir: Path, recursive: bool = True) -> Dict[str, Any]:
        """
        Batch ingest all supported files in directory.
        
        Args:
            input_dir: Directory containing files to process
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Starting batch ingestion from {input_dir}")
        
        # Find all supported files
        files = []
        supported_exts = set(self.get_supported_formats())
        
        if recursive:
            for ext in supported_exts:
                files.extend(input_dir.rglob(f"*{ext}"))
        else:
            for ext in supported_exts:
                files.extend(input_dir.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files
        processed_files = []
        failed_files = []
        total_chars = 0
        total_words = 0
        
        for file_path in files:
            metadata = self.ingest_file(file_path)
            if metadata:
                processed_files.append(metadata)
                total_chars += metadata.character_count
                total_words += metadata.word_count
            else:
                failed_files.append(str(file_path))
        
        # Create summary
        results = {
            'processed_count': len(processed_files),
            'failed_count': len(failed_files),
            'total_files': len(files),
            'total_characters': total_chars,
            'total_words': total_words,
            'processed_files': [str(m.file_path) for m in processed_files],
            'failed_files': failed_files,
            'output_directory': str(self.output_dir),
            'supported_formats': self.get_supported_formats()
        }
        
        logger.info(f"Batch ingestion complete: {len(processed_files)}/{len(files)} files processed")
        logger.info(f"Total content: {total_chars:,} characters, {total_words:,} words")
        
        return results
    
    def _extract_txt(self, file_path: Path) -> Optional[str]:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            return None
    
    def _extract_docx(self, file_path: Path) -> Optional[str]:
        """Extract text from DOCX file."""
        if docx2txt is None:
            logger.warning(f"Skipping DOCX {file_path}: docx2txt not available")
            return None
        
        try:
            return docx2txt.process(str(file_path))
        except Exception as e:
            logger.error(f"Error reading DOCX file {file_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced document ingestion")
    parser.add_argument("--input-dir", required=True, help="Input directory")
    parser.add_argument("--output-dir", default="data/cleaned", help="Output directory")
    parser.add_argument("--max-size", type=int, default=100, help="Max file size (MB)")
    parser.add_argument("--ocr-lang", nargs='+', default=['eng'], help="OCR languages")
    
    args = parser.parse_args()
    
    # Initialize ingester
    ingester = DocumentIngester(
        output_dir=args.output_dir,
        ocr_languages=args.ocr_lang,
        max_file_size_mb=args.max_size
    )
    
    # Process directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    results = ingester.ingest_directory(input_path)
    
    # Print results
    print(f"\n=== Ingestion Results ===")
    print(f"Processed: {results['processed_count']}/{results['total_files']} files")
    print(f"Failed: {results['failed_count']} files")
    print(f"Total content: {results['total_characters']:,} characters")
    print(f"Output directory: {results['output_directory']}")


if __name__ == "__main__":
    main()