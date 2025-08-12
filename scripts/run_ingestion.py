#!/usr/bin/env python3
"""
CLI script to run enhanced document ingestion.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.ingest import DocumentIngester
from loguru import logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced document ingestion for LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in data/raw directory
  python scripts/run_ingestion.py --input data/raw --output data/cleaned
  
  # Process with OCR support for multiple languages
  python scripts/run_ingestion.py --input data/raw --output data/cleaned --ocr-lang eng fra deu
  
  # Process with custom file size limit
  python scripts/run_ingestion.py --input data/raw --output data/cleaned --max-size 50
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing documents to process"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="data/cleaned",
        help="Output directory for cleaned text files (default: data/cleaned)"
    )
    
    parser.add_argument(
        "--max-size",
        type=int,
        default=100,
        help="Maximum file size to process in MB (default: 100)"
    )
    
    parser.add_argument(
        "--ocr-lang",
        nargs='+',
        default=['eng'],
        help="OCR languages for scanned PDFs (default: eng)"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Validate input directory
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_path}")
        sys.exit(1)
    
    # Initialize ingester
    logger.info(f"Initializing DocumentIngester...")
    logger.info(f"  Input directory: {input_path}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Max file size: {args.max_size}MB")
    logger.info(f"  OCR languages: {args.ocr_lang}")
    logger.info(f"  Recursive: {args.recursive}")
    
    ingester = DocumentIngester(
        output_dir=args.output,
        ocr_languages=args.ocr_lang,
        max_file_size_mb=args.max_size
    )
    
    # Show supported formats
    logger.info(f"Supported formats: {', '.join(ingester.get_supported_formats())}")
    
    # Process directory
    logger.info("Starting document ingestion...")
    results = ingester.ingest_directory(input_path, recursive=args.recursive)
    
    # Display results
    print("\n" + "="*50)
    print("INGESTION RESULTS")
    print("="*50)
    print(f"Total files found: {results['total_files']}")
    print(f"Successfully processed: {results['processed_count']}")
    print(f"Failed to process: {results['failed_count']}")
    print(f"Success rate: {results['processed_count']/results['total_files']*100:.1f}%")
    print()
    print(f"Total content extracted:")
    print(f"  Characters: {results['total_characters']:,}")
    print(f"  Words: {results['total_words']:,}")
    print()
    print(f"Output directory: {results['output_directory']}")
    print(f"Supported formats: {', '.join(results['supported_formats'])}")
    
    if results['failed_files']:
        print(f"\nFailed files ({len(results['failed_files'])}):")
        for failed_file in results['failed_files'][:10]:  # Show first 10
            print(f"  - {failed_file}")
        if len(results['failed_files']) > 10:
            print(f"  ... and {len(results['failed_files']) - 10} more")
    
    print("\n" + "="*50)
    
    if results['processed_count'] > 0:
        logger.info("✅ Ingestion completed successfully!")
        
        # Show next steps
        print("\nNext steps:")
        print("1. Review the cleaned text files in the output directory")
        print("2. Run deduplication if needed")
        print("3. Train tokenizer on the cleaned data")
        print("4. Start model training")
        
    else:
        logger.error("❌ No files were successfully processed!")
        logger.info("Check the logs above for specific error messages")
        sys.exit(1)


if __name__ == "__main__":
    main()