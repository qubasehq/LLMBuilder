# Data Processing

Data is the foundation of any successful language model. LLMBuilder provides tools for loading, cleaning, and preparing text data from various sources.

## Overview

LLMBuilder's data processing pipeline handles:
- Loading documents from various formats
- Cleaning and normalizing text
- Removing duplicate content
- Preparing datasets for training

## Supported File Formats

LLMBuilder can process various document formats:
- Plain Text (.txt)
- HTML (.html, .htm)
- Markdown (.md)
- EPUB (.epub)
- PDF (.pdf)

## Quick Start

### Basic Data Loading

```bash
# Process text files
llmbuilder data load \
  --input ./documents \
  --output processed_text.txt \
  --format txt \
  --clean
```

### Remove Duplicates

```bash
# Remove exact duplicates
llmbuilder data deduplicate \
  --input processed_text.txt \
  --output clean_text.txt \
  --exact
```

### Python API

```python
from llmbuilder.data.ingest import IngestionPipeline
from llmbuilder.data.dedup import DeduplicationPipeline

# Process documents
pipeline = IngestionPipeline()
results = pipeline.process_directory("./raw_documents", "./processed")

# Remove duplicates
dedup_pipeline = DeduplicationPipeline()
dedup_results = dedup_pipeline.process_file(
    "./processed/combined.txt",
    "./clean/deduplicated.txt"
)
```

## Data Processing Features

### Multi-Format Document Ingestion

LLMBuilder can process multiple document formats simultaneously:

```python
from llmbuilder.data.ingest import IngestionPipeline

# Configure ingestion
config = {
    "supported_formats": ["html", "markdown", "epub", "pdf", "txt"],
    "batch_size": 100,
    "num_workers": 4
}

pipeline = IngestionPipeline(**config)
results = pipeline.process_directory("./documents", "./processed")
```

### Deduplication

Remove duplicate content to improve training quality:

```python
from llmbuilder.data.dedup import DeduplicationPipeline

# Configure deduplication
dedup_config = {
    "enable_exact_deduplication": True,
    "enable_semantic_deduplication": True,
    "similarity_threshold": 0.85
}

dedup = DeduplicationPipeline(**dedup_config)
results = dedup.process_file(
    input_file="./processed/raw_text.txt",
    output_file="./clean/deduplicated.txt"
)
```

## Next Steps

- **[Tokenization Guide](tokenization.md)** - Train tokenizers
- **[Training Guide](training.md)** - Train models
- **[Configuration Guide](configuration.md)** - Configure processing

<div align="center">
  <p>Start with simple text files and gradually work with more complex formats.</p>
</div>