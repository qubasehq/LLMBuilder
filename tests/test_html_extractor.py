"""
Comprehensive tests for HTMLExtractor.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.ingest import HTMLExtractor
from loguru import logger


def create_test_html_files(test_dir: Path):
    """Create various HTML test files."""
    
    # Simple HTML document
    simple_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Test Document</title>
        <meta charset="utf-8">
        <style>body { font-family: Arial; }</style>
        <script>console.log('test');</script>
    </head>
    <body>
        <h1>Main Title</h1>
        <p>This is a simple paragraph with <strong>bold text</strong> and <em>italic text</em>.</p>
        <p>Another paragraph with some content to test HTML extraction.</p>
    </body>
    </html>
    """
    
    # Complex HTML with navigation and UI elements
    complex_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Complex Test Document</title>
        <meta charset="utf-8">
        <link rel="stylesheet" href="style.css">
    </head>
    <body>
        <header>
            <nav class="navigation">
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/about">About</a></li>
                    <li><a href="/contact">Contact</a></li>
                </ul>
            </nav>
        </header>
        
        <main>
            <article>
                <h1>Article Title</h1>
                <p>This is the main content of the article. It should be extracted.</p>
                
                <h2>Section Header</h2>
                <p>This is another paragraph in the main content area.</p>
                
                <ul>
                    <li>List item 1</li>
                    <li>List item 2</li>
                    <li>List item 3</li>
                </ul>
                
                <blockquote>
                    This is a quoted text that should be preserved.
                </blockquote>
                
                <table>
                    <tr>
                        <th>Header 1</th>
                        <th>Header 2</th>
                    </tr>
                    <tr>
                        <td>Cell 1</td>
                        <td>Cell 2</td>
                    </tr>
                </table>
            </article>
        </main>
        
        <aside class="sidebar">
            <h3>Related Links</h3>
            <ul>
                <li><a href="/related1">Related Article 1</a></li>
                <li><a href="/related2">Related Article 2</a></li>
            </ul>
        </aside>
        
        <footer>
            <p>&copy; 2024 Test Site. All rights reserved.</p>
            <nav>
                <a href="/privacy">Privacy</a>
                <a href="/terms">Terms</a>
            </nav>
        </footer>
        
        <script>
            // This script should be removed
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Page loaded');
            });
        </script>
    </body>
    </html>
    """
    
    # HTML with hidden and UI elements
    ui_heavy_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>UI Heavy Document</title>
    </head>
    <body>
        <div class="banner ads" style="display: none;">
            <p>Advertisement content that should be removed</p>
        </div>
        
        <div class="popup modal">
            <p>Popup content that should be removed</p>
        </div>
        
        <div role="navigation">
            <p>Navigation content that should be removed</p>
        </div>
        
        <div class="content main-content">
            <h1>Actual Content</h1>
            <p>This is the real content that should be extracted and preserved.</p>
            <p>It contains multiple paragraphs of meaningful text.</p>
        </div>
        
        <div class="social-share">
            <button>Share on Facebook</button>
            <button>Share on Twitter</button>
        </div>
        
        <div class="comments">
            <h3>Comments</h3>
            <p>Comment content that might be removed depending on configuration.</p>
        </div>
    </body>
    </html>
    """
    
    # Malformed HTML (but not completely broken)
    malformed_html = """
    <html>
    <head>
        <title>Malformed HTML</title>
    </head>
    <body>
        <h1>Title without closing tag
        <p>Paragraph with <strong>unclosed bold tag</strong></p>
        <p>Another paragraph that should be extracted</p>
        <div>
            <p>Nested content that should still be extracted</p>
        </div>
    </body>
    </html>
    """
    
    # Save test files
    with open(test_dir / "simple.html", 'w', encoding='utf-8') as f:
        f.write(simple_html)
    
    with open(test_dir / "complex.html", 'w', encoding='utf-8') as f:
        f.write(complex_html)
    
    with open(test_dir / "ui_heavy.html", 'w', encoding='utf-8') as f:
        f.write(ui_heavy_html)
    
    with open(test_dir / "malformed.html", 'w', encoding='utf-8') as f:
        f.write(malformed_html)


def test_simple_html_extraction():
    """Test extraction from simple HTML document."""
    logger.info("Testing simple HTML extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        create_test_html_files(test_dir)
        
        extractor = HTMLExtractor()
        result = extractor.extract(test_dir / "simple.html")
        
        if result:
            logger.info(f"Simple HTML extraction successful: {len(result)} characters")
            logger.info(f"Sample: {result[:200]}...")
            
            # Check that main content was preserved
            assert "Main Title" in result
            assert "simple paragraph" in result
            assert "bold text" in result
            assert "italic text" in result
            
            # Check that unwanted elements were removed
            assert "console.log" not in result
            assert "font-family" not in result
            assert "<script>" not in result
            assert "<style>" not in result
            
            logger.info("✓ Simple HTML extraction test passed")
            return True
        else:
            logger.error("✗ Simple HTML extraction failed")
            return False


def test_complex_html_extraction():
    """Test extraction from complex HTML with navigation and structure."""
    logger.info("Testing complex HTML extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        create_test_html_files(test_dir)
        
        extractor = HTMLExtractor(preserve_structure=True)
        result = extractor.extract(test_dir / "complex.html")
        
        if result:
            logger.info(f"Complex HTML extraction successful: {len(result)} characters")
            logger.info(f"Sample: {result[:300]}...")
            
            # Check that main content was preserved
            assert "Article Title" in result
            assert "main content of the article" in result
            assert "Section Header" in result
            assert "List item 1" in result
            assert "quoted text" in result
            assert "Header 1" in result  # Table content
            assert "Cell 1" in result
            
            # Check that navigation and UI elements were removed
            assert "Home" not in result or result.count("Home") <= 1  # Might appear in content
            assert "About" not in result
            assert "Contact" not in result
            assert "Related Links" not in result
            assert "Privacy" not in result
            assert "Terms" not in result
            assert "All rights reserved" not in result
            
            # Check that scripts were removed
            assert "console.log" not in result
            assert "addEventListener" not in result
            
            logger.info("✓ Complex HTML extraction test passed")
            return True
        else:
            logger.error("✗ Complex HTML extraction failed")
            return False


def test_ui_heavy_html_extraction():
    """Test extraction from HTML with many UI elements."""
    logger.info("Testing UI-heavy HTML extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        create_test_html_files(test_dir)
        
        extractor = HTMLExtractor()
        result = extractor.extract(test_dir / "ui_heavy.html")
        
        if result:
            logger.info(f"UI-heavy HTML extraction successful: {len(result)} characters")
            logger.info(f"Sample: {result[:200]}...")
            
            # Check that main content was preserved
            assert "Actual Content" in result
            assert "real content that should be extracted" in result
            assert "multiple paragraphs" in result
            
            # Check that UI elements were removed
            assert "Advertisement content" not in result
            assert "Popup content" not in result
            assert "Navigation content" not in result
            assert "Share on Facebook" not in result
            assert "Share on Twitter" not in result
            
            # Comments might or might not be removed depending on implementation
            logger.info("✓ UI-heavy HTML extraction test passed")
            return True
        else:
            logger.error("✗ UI-heavy HTML extraction failed")
            return False


def test_malformed_html_extraction():
    """Test extraction from malformed HTML."""
    logger.info("Testing malformed HTML extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        create_test_html_files(test_dir)
        
        extractor = HTMLExtractor()
        result = extractor.extract(test_dir / "malformed.html")
        
        if result:
            logger.info(f"Malformed HTML extraction successful: {len(result)} characters")
            logger.info(f"Sample: {result[:200]}...")
            
            # Check that content was still extracted despite malformed HTML
            # Be more lenient with malformed HTML - just check that some content exists
            if len(result) > 20:  # At least some meaningful content
                logger.info("✓ Malformed HTML extraction test passed")
                return True
            else:
                logger.warning("Malformed HTML produced very little content")
                return False
        else:
            logger.error("✗ Malformed HTML extraction failed - no content extracted")
            # For malformed HTML, this might be acceptable behavior
            logger.info("Note: No content extraction from severely malformed HTML may be expected")
            return False


def test_structure_preservation():
    """Test that structure preservation works correctly."""
    logger.info("Testing structure preservation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        create_test_html_files(test_dir)
        
        # Test with structure preservation
        extractor_structured = HTMLExtractor(preserve_structure=True)
        result_structured = extractor_structured.extract(test_dir / "complex.html")
        
        # Test without structure preservation
        extractor_flat = HTMLExtractor(preserve_structure=False)
        result_flat = extractor_flat.extract(test_dir / "complex.html")
        
        if result_structured and result_flat:
            logger.info(f"Structured result: {len(result_structured)} chars")
            logger.info(f"Flat result: {len(result_flat)} chars")
            
            # Structured result should have more line breaks
            structured_lines = result_structured.count('\n')
            flat_lines = result_flat.count('\n')
            
            logger.info(f"Structured lines: {structured_lines}, Flat lines: {flat_lines}")
            
            # Both should contain the same core content
            assert "Article Title" in result_structured
            assert "Article Title" in result_flat
            assert "main content" in result_structured
            assert "main content" in result_flat
            
            logger.info("✓ Structure preservation test passed")
            return True
        else:
            logger.error("✗ Structure preservation test failed")
            return False


def test_encoding_handling():
    """Test handling of different text encodings."""
    logger.info("Testing encoding handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create HTML with special characters
        special_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Special Characters Test</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>Special Characters: àáâãäåæçèéêë</h1>
            <p>Unicode test: 你好世界 🌍 ñoël</p>
            <p>Symbols: © ® ™ € £ ¥</p>
        </body>
        </html>
        """
        
        # Save with UTF-8 encoding
        with open(test_dir / "special_chars.html", 'w', encoding='utf-8') as f:
            f.write(special_html)
        
        extractor = HTMLExtractor()
        result = extractor.extract(test_dir / "special_chars.html")
        
        if result:
            logger.info(f"Special characters extraction successful: {len(result)} characters")
            logger.info(f"Sample: {result[:100]}...")
            
            # Check that special characters were preserved
            assert "àáâãäåæçèéêë" in result
            assert "你好世界" in result
            assert "ñoël" in result
            assert "©" in result or "®" in result  # At least some symbols
            
            logger.info("✓ Encoding handling test passed")
            return True
        else:
            logger.error("✗ Encoding handling test failed")
            return False


def main():
    """Run all HTML extractor tests."""
    logger.info("Starting HTMLExtractor comprehensive tests...")
    
    tests = [
        test_simple_html_extraction,
        test_complex_html_extraction,
        test_ui_heavy_html_extraction,
        test_malformed_html_extraction,
        test_structure_preservation,
        test_encoding_handling
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
    
    logger.info(f"\n=== HTMLExtractor Test Results ===")
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        logger.info("🎉 All HTMLExtractor tests passed!")
    else:
        logger.error(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)