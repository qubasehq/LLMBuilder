# Enhanced Data Ingestion System

This directory contains the enhanced data ingestion system that supports multiple document formats with OCR capabilities.

## Features

### Supported Formats
- **HTML/HTM**: Clean text extraction with BeautifulSoup
- **EPUB**: Full text extraction from e-books
- **PDF**: Direct text extraction with OCR fallback for scanned documents
- **Markdown**: Enhanced processing with proper formatting removal
- **TXT**: Plain text files
- **DOCX**: Microsoft Word documents

### Key Capabilities
- **OCR Support**: Automatic OCR processing for scanned PDFs using Tesseract
- **Multi-language OCR**: Support for multiple languages (English, French, German, etc.)
- **Error Handling**: Robust error handling with detailed logging
- **Progress Tracking**: Real-time progress reporting during batch processing
- **Metadata Collection**: Comprehensive metadata for each processed document
- **File Size Limits**: Configurable maximum file size limits

## Quick Start

### Basic Usage

```bash
# Process all files in data/raw directory
python scripts/run_ingestion.py --input data/raw --output data/cleaned

# Process with OCR support for multiple languages
python scripts/run_ingestion.py --input data/raw --output data/cleaned --ocr-lang eng fra deu

# Process with custom file size limit (50MB)
python scripts/run_ingestion.py --input data/raw --output data/cleaned --max-size 50
```

### Python API

```python
from data.ingest import DocumentIngester

# Initialize ingester
ingester = DocumentIngester(
    output_dir="data/cleaned",
    ocr_languages=['eng', 'fra'],
    max_file_size_mb=100
)

# Process single file
metadata = ingester.ingest_file(Path("document.pdf"))

# Process entire directory
results = ingester.ingest_directory(Path("data/raw"))
```

## Installation

### Required Dependencies

```bash
pip install ebooklib pytesseract Pillow requests
```

### System Dependencies

For OCR functionality, you need to install Tesseract:

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-deu
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

## Configuration

### OCR Languages

Supported language codes for OCR:
- `eng`: English
- `fra`: French  
- `deu`: German
- `spa`: Spanish
- `ita`: Italian
- `por`: Portuguese
- `rus`: Russian
- `chi_sim`: Chinese Simplified
- `jpn`: Japanese

### File Size Limits

Default maximum file size is 100MB. Adjust based on your system resources:

```python
ingester = DocumentIngester(max_file_size_mb=50)  # 50MB limit
```

## Output Format

All processed documents are saved as clean text files with the suffix `_cleaned.txt`. The system preserves:
- Main content text
- Paragraph structure
- Basic formatting (converted to plain text)

The system removes:
- HTML/XML tags
- Navigation elements
- Scripts and styles
- Excessive whitespace
- Special characters

## Error Handling

The system includes comprehensive error handling:
- **File-level errors**: Skip problematic files, continue processing others
- **Format-specific fallbacks**: OCR fallback for corrupted PDFs
- **Memory management**: Process large files efficiently
- **Progress tracking**: Detailed progress with error counts

## Metadata

Each processed document generates metadata including:
- File path and type
- Processing method used
- Extraction confidence score
- Character and word counts
- Processing time
- Content hash signature

## Integration

### With Existing Preprocessing

```python
# Use enhanced preprocessing
python scripts/enhanced_preprocess.py --use-enhanced --input-dir data/raw
```

### With Training Pipeline

The cleaned text files can be used directly with the existing training pipeline:

```bash
# Train tokenizer on cleaned data
python training/train_tokenizer.py --input data/cleaned

# Start model training
python training/train.py --config config.json
```

## Troubleshooting

### Common Issues

1. **OCR not working**: Ensure Tesseract is installed and in PATH
2. **Memory errors**: Reduce max file size or process files individually
3. **Import errors**: Install all required dependencies
4. **Permission errors**: Check file permissions in input/output directories

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python scripts/run_ingestion.py --input data/raw --verbose
```

## Performance Tips

1. **Batch Processing**: Process multiple files at once for efficiency
2. **OCR Optimization**: Limit OCR to first 10 pages of large PDFs
3. **Memory Management**: Use appropriate file size limits
4. **Language Selection**: Only specify OCR languages you need

## Testing

Run the test suite to verify installation:

```bash
python tests/test_ingestion.py
```

This will test all extractors and the complete ingestion workflow.