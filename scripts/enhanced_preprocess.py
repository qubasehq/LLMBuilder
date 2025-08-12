#!/usr/bin/env python3
"""
Enhanced preprocessing script that uses the new ingestion system.
This script demonstrates how to integrate the enhanced ingestion with existing preprocessing.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.ingest import DocumentIngester
from training.preprocess import DataPreprocessor
from loguru import logger


def main():
    """Enhanced preprocessing with new ingestion capabilities."""
    parser = argparse.ArgumentParser(
        description="Enhanced preprocessing with multi-format ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Input directory with raw documents (default: data/raw)"
    )
    
    parser.add_argument(
        "--cleaned-dir",
        default="data/cleaned",
        help="Output directory for cleaned text (default: data/cleaned)"
    )
    
    parser.add_argument(
        "--use-enhanced",
        action="store_true",
        help="Use enhanced ingestion system (supports HTML, EPUB, OCR)"
    )
    
    parser.add_argument(
        "--ocr-lang",
        nargs='+',
        default=['eng'],
        help="OCR languages for scanned PDFs (default: eng)"
    )
    
    parser.add_argument(
        "--max-size",
        type=int,
        default=100,
        help="Maximum file size in MB (default: 100)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    cleaned_path = Path(args.cleaned_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)
    
    logger.info("Starting enhanced preprocessing...")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {cleaned_path}")
    logger.info(f"Enhanced ingestion: {args.use_enhanced}")
    
    if args.use_enhanced:
        # Use new enhanced ingestion system
        logger.info("Using enhanced ingestion system with multi-format support")
        
        ingester = DocumentIngester(
            output_dir=str(cleaned_path),
            ocr_languages=args.ocr_lang,
            max_file_size_mb=args.max_size
        )
        
        logger.info(f"Supported formats: {', '.join(ingester.get_supported_formats())}")
        
        # Process all files
        results = ingester.ingest_directory(input_path, recursive=True)
        
        # Display results
        logger.info("Enhanced ingestion completed!")
        logger.info(f"Processed: {results['processed_count']}/{results['total_files']} files")
        logger.info(f"Total content: {results['total_characters']:,} characters")
        
        if results['failed_count'] > 0:
            logger.warning(f"Failed to process {results['failed_count']} files")
            for failed_file in results['failed_files'][:5]:  # Show first 5
                logger.warning(f"  - {failed_file}")
        
    else:
        # Use existing preprocessing system
        logger.info("Using existing preprocessing system")
        
        preprocessor = DataPreprocessor(
            raw_data_dir=str(input_path),
            cleaned_data_dir=str(cleaned_path),
            min_length=50,
            max_length=500000,
            remove_urls=True,
            remove_emails=True,
            normalize_whitespace=True
        )
        
        # Process all files
        results = preprocessor.process_all()
        
        # Display results
        logger.info("Standard preprocessing completed!")
        logger.info(f"Processed: {results['processed']} files")
        logger.info(f"Failed: {results['failed']} files")
        logger.info(f"Total content: {results['total_chars']:,} characters")
    
    # Show next steps
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print("Next steps:")
    print("1. Review cleaned text files")
    print("2. Run deduplication (if implemented)")
    print("3. Train tokenizer:")
    print(f"   python training/train_tokenizer.py --input {cleaned_path}")
    print("4. Start model training:")
    print("   python training/train.py --config config.json")
    print("="*50)


if __name__ == "__main__":
    main()