"""
Comprehensive tests for EPUBExtractor.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llmbuilder.core.data.ingest import EPUBExtractor
from loguru import logger

# Check if ebooklib is available for testing
try:
    import ebooklib
    from ebooklib import epub
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False
    logger.warning("ebooklib not available - EPUB tests will be skipped")


def create_test_epub(output_path: Path, title: str = "Test Book", author: str = "Test Author"):
    """Create a simple test EPUB file."""
    if not EBOOKLIB_AVAILABLE:
        return False
    
    try:
        # Create a new EPUB book
        book = epub.EpubBook()
        
        # Set metadata
        book.set_identifier('test-book-id')
        book.set_title(title)
        book.set_language('en')
        book.add_author(author)
        book.add_metadata('DC', 'description', 'A test book for EPUB extraction testing')
        book.add_metadata('DC', 'subject', 'Testing')
        book.add_metadata('DC', 'subject', 'EPUB')
        
        # Create chapters
        chapter1_content = """
        <html>
        <head><title>Chapter 1</title></head>
        <body>
            <h1>Chapter 1: Introduction</h1>
            <p>This is the first chapter of our test book. It contains some introductory text that should be extracted properly.</p>
            <p>Here's another paragraph with <strong>bold text</strong> and <em>italic text</em> to test formatting handling.</p>
            <blockquote>This is a quoted passage that should be preserved in the extraction.</blockquote>
        </body>
        </html>
        """
        
        chapter2_content = """
        <html>
        <head><title>Chapter 2</title></head>
        <body>
            <h1>Chapter 2: Main Content</h1>
            <p>This is the second chapter with more substantial content for testing.</p>
            <h2>Section 2.1</h2>
            <p>This section contains a list:</p>
            <ul>
                <li>First list item</li>
                <li>Second list item</li>
                <li>Third list item</li>
            </ul>
            <h2>Section 2.2</h2>
            <p>This section has more text content that should be extracted and cleaned properly.</p>
            <div class="content">
                <p>This is content inside a div that should also be extracted.</p>
            </div>
        </body>
        </html>
        """
        
        chapter3_content = """
        <html>
        <head><title>Chapter 3</title></head>
        <body>
            <h1>Chapter 3: Conclusion</h1>
            <p>This is the final chapter of our test book.</p>
            <p>It contains concluding remarks and should demonstrate proper text extraction from multiple chapters.</p>
            
            <!-- Navigation elements that should be removed -->
            <nav class="navigation">
                <a href="#prev">Previous Chapter</a>
                <a href="#next">Next Chapter</a>
            </nav>
            
            <script>
                // This script should be removed
                console.log('Chapter 3 loaded');
            </script>
        </body>
        </html>
        """
        
        # Create EPUB items
        c1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
        c1.content = chapter1_content
        
        c2 = epub.EpubHtml(title='Chapter 2', file_name='chap_02.xhtml', lang='en')
        c2.content = chapter2_content
        
        c3 = epub.EpubHtml(title='Chapter 3', file_name='chap_03.xhtml', lang='en')
        c3.content = chapter3_content
        
        # Add chapters to book
        book.add_item(c1)
        book.add_item(c2)
        book.add_item(c3)
        
        # Create table of contents
        book.toc = (epub.Link("chap_01.xhtml", "Chapter 1", "chap_01"),
                   epub.Link("chap_02.xhtml", "Chapter 2", "chap_02"),
                   epub.Link("chap_03.xhtml", "Chapter 3", "chap_03"))
        
        # Add navigation files
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Define CSS style
        style = '''
        body { font-family: Arial, sans-serif; }
        h1 { color: #333; }
        p { margin: 1em 0; }
        '''
        nav_css = epub.EpubItem(uid="nav_css", file_name="style/nav.css", 
                               media_type="text/css", content=style)
        book.add_item(nav_css)
        
        # Create spine
        book.spine = ['nav', c1, c2, c3]
        
        # Write EPUB file
        epub.write_epub(str(output_path), book, {})
        
        logger.info(f"Created test EPUB: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create test EPUB: {e}")
        return False


def test_basic_epub_extraction():
    """Test basic EPUB text extraction."""
    if not EBOOKLIB_AVAILABLE:
        logger.info("Skipping EPUB test - ebooklib not available")
        return True
    
    logger.info("Testing basic EPUB extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        epub_file = test_dir / "test_book.epub"
        
        # Create test EPUB
        if not create_test_epub(epub_file):
            logger.error("Failed to create test EPUB")
            return False
        
        # Test extraction
        extractor = EPUBExtractor()
        result = extractor.extract(epub_file)
        
        if result:
            logger.info(f"EPUB extraction successful: {len(result)} characters")
            logger.info(f"Sample: {result[:300]}...")
            
            # Check that metadata was extracted
            assert "Title: Test Book" in result
            assert "Author(s): Test Author" in result
            assert "Description: A test book" in result
            
            # Check that main content was extracted
            assert "Chapter 1: Introduction" in result
            assert "first chapter of our test book" in result
            assert "Chapter 2: Main Content" in result
            assert "Second list item" in result
            assert "Chapter 3: Conclusion" in result
            
            # Check that unwanted elements were removed
            assert "console.log" not in result
            assert "Previous Chapter" not in result
            assert "Next Chapter" not in result
            
            logger.info("✓ Basic EPUB extraction test passed")
            return True
        else:
            logger.error("✗ Basic EPUB extraction failed")
            return False


def test_epub_without_metadata():
    """Test EPUB extraction without metadata."""
    if not EBOOKLIB_AVAILABLE:
        logger.info("Skipping EPUB test - ebooklib not available")
        return True
    
    logger.info("Testing EPUB extraction without metadata...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        epub_file = test_dir / "test_book_no_meta.epub"
        
        # Create test EPUB
        if not create_test_epub(epub_file):
            logger.error("Failed to create test EPUB")
            return False
        
        # Test extraction without metadata
        extractor = EPUBExtractor(extract_metadata=False)
        result = extractor.extract(epub_file)
        
        if result:
            logger.info(f"EPUB extraction (no metadata) successful: {len(result)} characters")
            logger.info(f"Sample: {result[:200]}...")
            
            # Check that metadata was NOT extracted
            assert "Title: Test Book" not in result
            assert "Author(s): Test Author" not in result
            
            # Check that main content was still extracted
            assert "Chapter 1: Introduction" in result
            assert "first chapter of our test book" in result
            
            logger.info("✓ EPUB extraction without metadata test passed")
            return True
        else:
            logger.error("✗ EPUB extraction without metadata failed")
            return False


def test_epub_structure_preservation():
    """Test EPUB structure preservation options."""
    if not EBOOKLIB_AVAILABLE:
        logger.info("Skipping EPUB test - ebooklib not available")
        return True
    
    logger.info("Testing EPUB structure preservation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        epub_file = test_dir / "test_book_structure.epub"
        
        # Create test EPUB
        if not create_test_epub(epub_file):
            logger.error("Failed to create test EPUB")
            return False
        
        # Test with structure preservation
        extractor_structured = EPUBExtractor(preserve_structure=True, extract_metadata=False)
        result_structured = extractor_structured.extract(epub_file)
        
        # Test without structure preservation
        extractor_flat = EPUBExtractor(preserve_structure=False, extract_metadata=False)
        result_flat = extractor_flat.extract(epub_file)
        
        if result_structured and result_flat:
            logger.info(f"Structured result: {len(result_structured)} chars")
            logger.info(f"Flat result: {len(result_flat)} chars")
            
            # Structured result should have chapter separators
            structured_chapters = result_structured.count('--- Chapter ---')
            flat_chapters = result_flat.count('--- Chapter ---')
            
            logger.info(f"Structured chapters: {structured_chapters}, Flat chapters: {flat_chapters}")
            
            # Both should contain the same core content
            assert "Chapter 1: Introduction" in result_structured
            assert "Chapter 1: Introduction" in result_flat
            assert "first chapter" in result_structured
            assert "first chapter" in result_flat
            
            logger.info("✓ EPUB structure preservation test passed")
            return True
        else:
            logger.error("✗ EPUB structure preservation test failed")
            return False


def test_epub_with_complex_content():
    """Test EPUB extraction with complex HTML content."""
    if not EBOOKLIB_AVAILABLE:
        logger.info("Skipping EPUB test - ebooklib not available")
        return True
    
    logger.info("Testing EPUB with complex content...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        epub_file = test_dir / "complex_book.epub"
        
        # Create test EPUB with complex content
        if not create_test_epub(epub_file, "Complex Book", "Complex Author"):
            logger.error("Failed to create complex test EPUB")
            return False
        
        # Test extraction
        extractor = EPUBExtractor(extract_metadata=True, preserve_structure=True)
        result = extractor.extract(epub_file)
        
        if result:
            logger.info(f"Complex EPUB extraction successful: {len(result)} characters")
            
            # Check content quality
            word_count = len(result.split())
            char_count = len(result)
            
            logger.info(f"Extracted {word_count} words, {char_count} characters")
            
            # Should have substantial content
            assert word_count > 50
            assert char_count > 500
            
            # Should contain structured content
            assert "Chapter" in result
            assert ":" in result  # Headers should have colons
            
            logger.info("✓ Complex EPUB extraction test passed")
            return True
        else:
            logger.error("✗ Complex EPUB extraction failed")
            return False


def test_epub_error_handling():
    """Test EPUB extractor error handling."""
    logger.info("Testing EPUB error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Test with non-existent file
        extractor = EPUBExtractor()
        result = extractor.extract(test_dir / "nonexistent.epub")
        
        # Should return None for non-existent file
        assert result is None
        
        # Test with invalid file (create a text file with .epub extension)
        invalid_epub = test_dir / "invalid.epub"
        with open(invalid_epub, 'w') as f:
            f.write("This is not an EPUB file")
        
        result = extractor.extract(invalid_epub)
        
        # Should return None for invalid EPUB
        assert result is None
        
        logger.info("✓ EPUB error handling test passed")
        return True


def main():
    """Run all EPUB extractor tests."""
    logger.info("Starting EPUBExtractor comprehensive tests...")
    
    if not EBOOKLIB_AVAILABLE:
        logger.warning("ebooklib not available - most EPUB tests will be skipped")
        logger.info("To install: pip install ebooklib")
        return True  # Don't fail if library is not available
    
    tests = [
        test_basic_epub_extraction,
        test_epub_without_metadata,
        test_epub_structure_preservation,
        test_epub_with_complex_content,
        test_epub_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    logger.info(f"\n=== EPUBExtractor Test Results ===")
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        logger.info("🎉 All EPUBExtractor tests passed!")
    else:
        logger.error(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)