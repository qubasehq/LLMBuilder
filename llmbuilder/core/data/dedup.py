"""
Deduplication system for LLM training data.
Removes exact and near-duplicate content to improve training quality.
"""

import os
import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from loguru import logger
import time

# Optional dependencies for embedding-based deduplication
try:
# Lazy import: from sentence_transformers import \1
from llmbuilder.utils.lazy_imports import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers or sklearn not available. Embedding-based deduplication will be disabled.")
    EMBEDDINGS_AVAILABLE = False


@dataclass
class DeduplicationStats:
    """Statistics for deduplication process."""
    total_documents: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    final_document_count: int
    total_characters_before: int
    total_characters_after: int
    processing_time: float
    duplicate_pairs: List[Tuple[str, str, float]]
    hash_collisions: int
    embedding_comparisons: int = 0


class HashDeduplicator:
    """Exact duplicate detection using normalized hashing."""
    
    def __init__(self, 
                 hash_algorithm: str = 'md5',
                 normalize_whitespace: bool = True,
                 normalize_case: bool = True,
                 remove_punctuation: bool = False,
                 min_length: int = 50):
        """
        Initialize hash deduplicator.
        
        Args:
            hash_algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
            normalize_whitespace: Normalize whitespace before hashing
            normalize_case: Convert to lowercase before hashing
            remove_punctuation: Remove punctuation before hashing
            min_length: Minimum text length to consider for deduplication
        """
        self.hash_algorithm = hash_algorithm
        self.normalize_whitespace = normalize_whitespace
        self.normalize_case = normalize_case
        self.remove_punctuation = remove_punctuation
        self.min_length = min_length
        
        # Hash function mapping
        self.hash_functions = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256
        }
        
        if hash_algorithm not in self.hash_functions:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
        
        # Storage for hashes and file mappings
        self.content_hashes: Dict[str, List[str]] = defaultdict(list)  # hash -> [file_paths]
        self.file_hashes: Dict[str, str] = {}  # file_path -> hash
        self.duplicate_groups: List[List[str]] = []
        
        logger.info(f"HashDeduplicator initialized with {hash_algorithm} hashing")
        logger.info(f"Normalization: whitespace={normalize_whitespace}, case={normalize_case}, punctuation={remove_punctuation}")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text before hashing."""
        if not text:
            return ""
        
        normalized = text
        
        # Normalize whitespace
        if self.normalize_whitespace:
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = normalized.strip()
        
        # Normalize case
        if self.normalize_case:
            normalized = normalized.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def calculate_hash(self, text: str) -> str:
        """Calculate hash for normalized text."""
        normalized_text = self.normalize_text(text)
        
        if len(normalized_text) < self.min_length:
            return None
        
        hash_func = self.hash_functions[self.hash_algorithm]
        return hash_func(normalized_text.encode('utf-8')).hexdigest()
    
    def add_document(self, file_path: str, content: str) -> bool:
        """
        Add document to deduplication index.
        
        Args:
            file_path: Path to the document file
            content: Text content of the document
            
        Returns:
            True if document was added, False if too short or invalid
        """
        if len(content) < self.min_length:
            logger.debug(f"Skipping short document: {file_path} ({len(content)} chars)")
            return False
        
        content_hash = self.calculate_hash(content)
        if not content_hash:
            return False
        
        # Store mappings
        self.content_hashes[content_hash].append(file_path)
        self.file_hashes[file_path] = content_hash
        
        logger.debug(f"Added document: {file_path} -> {content_hash[:8]}...")
        return True
    
    def find_duplicates(self) -> List[List[str]]:
        """
        Find groups of duplicate documents.
        
        Returns:
            List of duplicate groups, each group is a list of file paths
        """
        duplicate_groups = []
        
        for content_hash, file_paths in self.content_hashes.items():
            if len(file_paths) > 1:
                duplicate_groups.append(file_paths)
                logger.info(f"Found duplicate group: {len(file_paths)} files with hash {content_hash[:8]}...")
                for file_path in file_paths:
                    logger.debug(f"  - {file_path}")
        
        self.duplicate_groups = duplicate_groups
        return duplicate_groups
    
    def get_files_to_keep(self, keep_strategy: str = 'first') -> Set[str]:
        """
        Determine which files to keep from duplicate groups.
        
        Args:
            keep_strategy: Strategy for choosing which duplicate to keep
                          'first': Keep first file in each group
                          'shortest_name': Keep file with shortest name
                          'longest_content': Keep file with longest content (requires content access)
        
        Returns:
            Set of file paths to keep
        """
        files_to_keep = set()
        
        # Add all non-duplicate files
        all_files = set(self.file_hashes.keys())
        duplicate_files = set()
        
        for group in self.duplicate_groups:
            duplicate_files.update(group)
        
        files_to_keep.update(all_files - duplicate_files)
        
        # Choose one file from each duplicate group
        for group in self.duplicate_groups:
            if keep_strategy == 'first':
                chosen = group[0]
            elif keep_strategy == 'shortest_name':
                chosen = min(group, key=lambda x: len(Path(x).name))
            else:  # Default to first
                chosen = group[0]
            
            files_to_keep.add(chosen)
            logger.debug(f"Keeping {chosen} from duplicate group of {len(group)} files")
        
        return files_to_keep
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        total_files = len(self.file_hashes)
        duplicate_files = sum(len(group) - 1 for group in self.duplicate_groups)  # -1 because we keep one from each group
        unique_files = total_files - duplicate_files
        
        return {
            'total_files': total_files,
            'unique_content_hashes': len(self.content_hashes),
            'duplicate_groups': len(self.duplicate_groups),
            'duplicate_files_removed': duplicate_files,
            'files_kept': unique_files,
            'deduplication_ratio': duplicate_files / total_files if total_files > 0 else 0.0,
            'hash_algorithm': self.hash_algorithm,
            'normalization_settings': {
                'whitespace': self.normalize_whitespace,
                'case': self.normalize_case,
                'punctuation': self.remove_punctuation
            }
        }


class EmbeddingDeduplicator:
    """Near-duplicate detection using sentence embeddings."""
    
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.85,
                 chunk_size: int = 1000,
                 max_chunks_per_doc: int = 5,
                 batch_size: int = 32):
        """
        Initialize embedding deduplicator.
        
        Args:
            model_name: SentenceTransformer model name
            similarity_threshold: Cosine similarity threshold for near-duplicates
            chunk_size: Size of text chunks for embedding
            max_chunks_per_doc: Maximum chunks per document to process
            batch_size: Batch size for embedding computation
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers and sklearn required for embedding deduplication")
        
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.batch_size = batch_size
        
        # Initialize model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Storage
        self.document_embeddings: Dict[str, np.ndarray] = {}
        self.document_chunks: Dict[str, List[str]] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
        self.near_duplicate_pairs: List[Tuple[str, str, float]] = []
        
        logger.info(f"EmbeddingDeduplicator initialized with {model_name}")
        logger.info(f"Similarity threshold: {similarity_threshold}, chunk size: {chunk_size}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for embedding."""
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split by sentences first, then group into chunks
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Limit number of chunks per document
        if len(chunks) > self.max_chunks_per_doc:
            # Take chunks from beginning, middle, and end
            step = len(chunks) // self.max_chunks_per_doc
            chunks = [chunks[i * step] for i in range(self.max_chunks_per_doc)]
        
        return chunks
    
    def add_document(self, file_path: str, content: str) -> bool:
        """Add document for embedding-based deduplication."""
        if len(content) < 100:  # Skip very short documents
            return False
        
        chunks = self.chunk_text(content)
        if not chunks:
            return False
        
        self.document_chunks[file_path] = chunks
        logger.debug(f"Added document for embedding: {file_path} ({len(chunks)} chunks)")
        return True
    
    def compute_embeddings(self) -> None:
        """Compute embeddings for all document chunks."""
        if not self.document_chunks:
            logger.warning("No documents added for embedding computation")
            return
        
        logger.info(f"Computing embeddings for {len(self.document_chunks)} documents...")
        
        # Collect all chunks with document references
        all_chunks = []
        chunk_to_doc = []
        
        for file_path, chunks in self.document_chunks.items():
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_to_doc.append(file_path)
        
        logger.info(f"Processing {len(all_chunks)} chunks in batches of {self.batch_size}")
        
        # Compute embeddings in batches
        chunk_embeddings = []
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            chunk_embeddings.extend(batch_embeddings)
            
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(all_chunks)} chunks")
        
        # Average embeddings per document
        for file_path in self.document_chunks.keys():
            doc_embeddings = []
            for i, doc in enumerate(chunk_to_doc):
                if doc == file_path:
                    doc_embeddings.append(chunk_embeddings[i])
            
            if doc_embeddings:
                # Average the chunk embeddings for the document
                self.document_embeddings[file_path] = np.mean(doc_embeddings, axis=0)
        
        logger.info(f"Computed embeddings for {len(self.document_embeddings)} documents")
    
    def find_near_duplicates(self) -> List[Tuple[str, str, float]]:
        """Find near-duplicate pairs based on embedding similarity."""
        if not self.document_embeddings:
            logger.warning("No embeddings computed. Call compute_embeddings() first.")
            return []
        
        logger.info(f"Finding near-duplicates with threshold {self.similarity_threshold}")
        
        # Get document paths and embeddings
        doc_paths = list(self.document_embeddings.keys())
        embeddings = np.array([self.document_embeddings[path] for path in doc_paths])
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(embeddings)
        
        # Find pairs above threshold
        near_duplicates = []
        n_docs = len(doc_paths)
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                similarity = self.similarity_matrix[i, j]
                if similarity >= self.similarity_threshold:
                    near_duplicates.append((doc_paths[i], doc_paths[j], similarity))
        
        self.near_duplicate_pairs = near_duplicates
        logger.info(f"Found {len(near_duplicates)} near-duplicate pairs")
        
        return near_duplicates
    
    def get_files_to_remove(self, keep_strategy: str = 'first') -> Set[str]:
        """Get files to remove based on near-duplicate detection."""
        files_to_remove = set()
        
        # Group near-duplicates
        duplicate_groups = defaultdict(set)
        for doc1, doc2, similarity in self.near_duplicate_pairs:
            # Create groups of similar documents
            found_group = None
            for group_id, group in duplicate_groups.items():
                if doc1 in group or doc2 in group:
                    found_group = group_id
                    break
            
            if found_group is not None:
                duplicate_groups[found_group].update([doc1, doc2])
            else:
                new_group_id = len(duplicate_groups)
                duplicate_groups[new_group_id] = {doc1, doc2}
        
        # Choose files to remove from each group
        for group in duplicate_groups.values():
            group_list = list(group)
            if len(group_list) > 1:
                if keep_strategy == 'first':
                    to_keep = group_list[0]
                else:
                    to_keep = group_list[0]  # Default
                
                files_to_remove.update(group_list[1:])
                logger.debug(f"Near-duplicate group: keeping {to_keep}, removing {len(group_list) - 1} files")
        
        return files_to_remove


class DeduplicationPipeline:
    """Main deduplication pipeline orchestrator."""
    
    def __init__(self,
                 use_hash_dedup: bool = True,
                 use_embedding_dedup: bool = True,
                 hash_config: Optional[Dict] = None,
                 embedding_config: Optional[Dict] = None):
        """
        Initialize deduplication pipeline.
        
        Args:
            use_hash_dedup: Enable exact duplicate detection
            use_embedding_dedup: Enable near-duplicate detection
            hash_config: Configuration for HashDeduplicator
            embedding_config: Configuration for EmbeddingDeduplicator
        """
        self.use_hash_dedup = use_hash_dedup
        self.use_embedding_dedup = use_embedding_dedup and EMBEDDINGS_AVAILABLE
        
        # Initialize deduplicators
        self.hash_dedup = None
        self.embedding_dedup = None
        
        if self.use_hash_dedup:
            hash_config = hash_config or {}
            self.hash_dedup = HashDeduplicator(**hash_config)
        
        if self.use_embedding_dedup:
            if not EMBEDDINGS_AVAILABLE:
                logger.warning("Embedding deduplication disabled: required packages not available")
                self.use_embedding_dedup = False
            else:
                embedding_config = embedding_config or {}
                self.embedding_dedup = EmbeddingDeduplicator(**embedding_config)
        
        logger.info(f"DeduplicationPipeline initialized: hash={self.use_hash_dedup}, embedding={self.use_embedding_dedup}")
    
    def process_files(self, input_files: List[str], output_dir: str) -> DeduplicationStats:
        """
        Process files for deduplication.
        
        Args:
            input_files: List of input file paths
            output_dir: Directory to save deduplicated files
            
        Returns:
            DeduplicationStats with processing results
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting deduplication of {len(input_files)} files")
        
        # Load and process files
        file_contents = {}
        total_chars_before = 0
        
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    file_contents[file_path] = content
                    total_chars_before += len(content)
                    
                    # Add to deduplicators
                    if self.hash_dedup:
                        self.hash_dedup.add_document(file_path, content)
                    
                    if self.embedding_dedup:
                        self.embedding_dedup.add_document(file_path, content)
                        
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue
        
        # Find duplicates
        files_to_remove = set()
        duplicate_pairs = []
        
        # Hash-based deduplication
        if self.hash_dedup:
            logger.info("Finding exact duplicates...")
            hash_duplicates = self.hash_dedup.find_duplicates()
            hash_files_to_keep = self.hash_dedup.get_files_to_keep()
            hash_files_to_remove = set(file_contents.keys()) - hash_files_to_keep
            files_to_remove.update(hash_files_to_remove)
            
            # Add to duplicate pairs for reporting
            for group in hash_duplicates:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        duplicate_pairs.append((group[i], group[j], 1.0))  # Exact match
        
        # Embedding-based deduplication
        embedding_comparisons = 0
        if self.embedding_dedup:
            logger.info("Computing embeddings...")
            self.embedding_dedup.compute_embeddings()
            
            logger.info("Finding near-duplicates...")
            near_duplicates = self.embedding_dedup.find_near_duplicates()
            embedding_files_to_remove = self.embedding_dedup.get_files_to_remove()
            
            # Only remove if not already removed by hash deduplication
            for file_path in embedding_files_to_remove:
                if file_path not in files_to_remove:
                    files_to_remove.add(file_path)
            
            duplicate_pairs.extend(near_duplicates)
            embedding_comparisons = len(self.embedding_dedup.document_embeddings) ** 2
        
        # Copy non-duplicate files to output directory
        files_kept = []
        total_chars_after = 0
        
        for file_path in file_contents.keys():
            if file_path not in files_to_remove:
                # Copy file to output directory
                input_path = Path(file_path)
                output_file = output_path / input_path.name
                
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        content = file_contents[file_path]
                        f.write(content)
                        total_chars_after += len(content)
                        files_kept.append(str(output_file))
                except Exception as e:
                    logger.error(f"Error writing {output_file}: {e}")
        
        # Calculate statistics
        processing_time = time.time() - start_time
        exact_duplicates = len(files_to_remove) if self.hash_dedup else 0
        near_duplicates = len(files_to_remove) - exact_duplicates if self.embedding_dedup else 0
        
        stats = DeduplicationStats(
            total_documents=len(input_files),
            exact_duplicates_removed=exact_duplicates,
            near_duplicates_removed=near_duplicates,
            final_document_count=len(files_kept),
            total_characters_before=total_chars_before,
            total_characters_after=total_chars_after,
            processing_time=processing_time,
            duplicate_pairs=duplicate_pairs,
            hash_collisions=0,  # TODO: Implement if needed
            embedding_comparisons=embedding_comparisons
        )
        
        # Save statistics
        stats_file = output_path / "deduplication_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, indent=2, default=str)
        
        logger.info(f"Deduplication complete in {processing_time:.2f}s")
        logger.info(f"Files: {len(input_files)} -> {len(files_kept)} ({len(files_to_remove)} removed)")
        logger.info(f"Characters: {total_chars_before:,} -> {total_chars_after:,}")
        
        return stats


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deduplicate text files")
    parser.add_argument("--input-dir", required=True, help="Input directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--hash-only", action="store_true", help="Use only hash deduplication")
    parser.add_argument("--embedding-only", action="store_true", help="Use only embedding deduplication")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, help="Similarity threshold for near-duplicates")
    
    args = parser.parse_args()
    
    # Find input files
    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob("*.txt"))
    
    if not input_files:
        logger.error(f"No .txt files found in {input_dir}")
        return
    
    # Configure pipeline
    use_hash = not args.embedding_only
    use_embedding = not args.hash_only
    
    embedding_config = {
        'similarity_threshold': args.similarity_threshold
    }
    
    # Run deduplication
    pipeline = DeduplicationPipeline(
        use_hash_dedup=use_hash,
        use_embedding_dedup=use_embedding,
        embedding_config=embedding_config
    )
    
    stats = pipeline.process_files([str(f) for f in input_files], args.output_dir)
    
    # Print results
    print(f"\n=== Deduplication Results ===")
    print(f"Input files: {stats.total_documents}")
    print(f"Output files: {stats.final_document_count}")
    print(f"Exact duplicates removed: {stats.exact_duplicates_removed}")
    print(f"Near duplicates removed: {stats.near_duplicates_removed}")
    print(f"Processing time: {stats.processing_time:.2f}s")
    print(f"Character reduction: {stats.total_characters_before:,} -> {stats.total_characters_after:,}")


if __name__ == "__main__":
    main()