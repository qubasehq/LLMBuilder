#!/usr/bin/env python3
"""
Smart data downloader for LLMBuilder.
Downloads PDF, DOCX, and TXT files based on topics/keywords.
"""

import os
import requests
import urllib.request
from pathlib import Path
from typing import List, Dict
import argparse
from loguru import logger


class SmartDataDownloader:
    """Download training data based on topics and keywords."""
    
    def __init__(self, download_dir: str = "data/raw"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def download_gutenberg_books(self, book_ids: List[int]) -> Dict[str, bool]:
        """Download books from Project Gutenberg."""
        results = {}
        base_url = "https://www.gutenberg.org/files/{}/{}-0.txt"
        
        for book_id in book_ids:
            try:
                filename = f"gutenberg_{book_id}.txt"
                filepath = self.download_dir / filename
                
                if filepath.exists():
                    logger.info(f"Already downloaded: {filename}")
                    results[filename] = True
                    continue
                
                url = base_url.format(book_id, book_id)
                urllib.request.urlretrieve(url, filepath)
                results[filename] = True
                logger.success(f"Downloaded: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download book {book_id}: {e}")
                results[f"gutenberg_{book_id}.txt"] = False
                
        return results
    
    def download_sample_texts(self, topic: str, count: int = 3) -> Dict[str, bool]:
        """Download sample texts based on topic."""
        topic_sources = {
            "technology": [
                "https://raw.githubusercontent.com/microsoft/Docs/master/azure-docs/articles/virtual-machines/linux/overview.md",
                "https://raw.githubusercontent.com/pytorch/pytorch/master/README.md",
                "https://raw.githubusercontent.com/tensorflow/tensorflow/master/README.md",
                "https://raw.githubusercontent.com/nodejs/node/main/README.md",
                "https://raw.githubusercontent.com/golang/go/master/README.md",
            ],
            "science": [
                "https://raw.githubusercontent.com/scipy/scipy/master/README.rst",
                "https://raw.githubusercontent.com/numpy/numpy/main/README.md",
                "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/README.rst",
                "https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md",
                "https://raw.githubusercontent.com/matplotlib/matplotlib/main/README.rst",
            ],
            "literature": [
                (1342, "pride_prejudice"),  # Pride and Prejudice
                (84, "frankenstein"),       # Frankenstein
                (11, "alice_wonderland"),   # Alice in Wonderland
                (1661, "grimm_fairy_tales"), # Grimm's Fairy Tales
                (2701, "moby_dick"),        # Moby Dick
                (64317, "dracula"),         # Dracula
                (174, "picture_dorian_gray"), # Picture of Dorian Gray
                (64317, "dracula"),         # Dracula
                (28054, "world_history"),   # World History
            ],
            "philosophy": [
                (3207, "plato_republic"),    # Plato's Republic
                (13476, "aristotle_ethics"), # Aristotle's Ethics
                (3200, "meditations"),      # Marcus Aurelius
                (3201, "nichomachean_ethics"), # Aristotle's Nicomachean Ethics
                (408, "critique_pure_reason"), # Kant's Critique of Pure Reason
                (1998, "utilitarianism"),   # Mill's Utilitarianism
            ],
            "history": [
                (19712, "history_rome"),    # History of Rome
                (1998, "decline_fall"),     # Decline and Fall of Rome
                (28054, "world_history"),   # World History
                (28054, "world_history"),   # World History
                (28054, "world_history"),   # World History
            ],
            "business": [
                (27042, "wealth_nations"),  # Adam Smith's Wealth of Nations
                (3300, "art_of_war"),       # Sun Tzu's Art of War
                (7370, "i_ching"),         # I Ching
                (3207, "plato_republic"),   # Plato's Republic
            ],
            "health": [
                (40702, "medical_essays"),  # Medical Essays
                (27042, "wealth_nations"),  # Economics related to health
                (3207, "plato_republic"),   # Philosophy of health
                (1998, "utilitarianism"),   # Ethics in medicine
            ],
            "education": [
                (40702, "medical_essays"),  # Educational content
                (3207, "plato_republic"),   # Educational philosophy
                (1998, "utilitarianism"),   # Educational ethics
                (28054, "world_history"),   # Educational history
            ],
            "environment": [
                (28054, "world_history"),   # Environmental history
                (1998, "utilitarianism"),   # Environmental ethics
                (3207, "plato_republic"),   # Environmental philosophy
                (27042, "wealth_nations"),  # Environmental economics
            ],
            "food": [
                (64317, "dracula"),         # Cultural food references
                (174, "picture_dorian_gray"), # Food in literature
                (1342, "pride_prejudice"),  # Food in social contexts
                (84, "frankenstein"),       # Food in gothic literature
            ],
            "travel": [
                (28054, "world_history"),   # Travel narratives
                (2701, "moby_dick"),        # Maritime travel
                (1342, "pride_prejudice"),  # Social travel
                (11, "alice_wonderland"),   # Fantasy travel
            ],
            "religion": [
                (3207, "plato_republic"),   # Religious philosophy
                (1998, "utilitarianism"),   # Religious ethics
                (7370, "i_ching"),         # Religious texts
                (3300, "art_of_war"),      # Spiritual warfare
            ],
            "language": [
                (3207, "plato_republic"),   # Philosophy of language
                (1998, "utilitarianism"),   # Language ethics
                (28054, "world_history"),   # Language history
                (40702, "medical_essays"),  # Medical language
            ],
            "media": [
                (64317, "dracula"),         # Media in literature
                (174, "picture_dorian_gray"), # Media and art
                (1342, "pride_prejudice"),  # Media in society
                (84, "frankenstein"),       # Media and science
            ]
        }
        
        if topic.lower() in topic_sources:
            if topic.lower() == "literature" or topic.lower() == "philosophy" or topic.lower() == "history":
                # Use Gutenberg books
                book_info = topic_sources[topic.lower()]
                book_ids = [book[0] for book in book_info[:count]]
                return self.download_gutenberg_books(book_ids)
            else:
                # Download from URLs
                urls = topic_sources[topic.lower()][:count]
                return self.download_from_urls(urls, topic)
        else:
            logger.warning(f"Topic '{topic}' not found. Available: {list(topic_sources.keys())}")
            return {}
    
    def download_from_urls(self, urls: List[str], topic: str) -> Dict[str, bool]:
        """Download from direct URLs."""
        results = {}
        
        for i, url in enumerate(urls):
            try:
                filename = f"{topic}_{i+1}.txt"
                filepath = self.download_dir / filename
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                results[filename] = True
                logger.success(f"Downloaded: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download from {url}: {e}")
                results[f"{topic}_{i+1}.txt"] = False
                
        return results
    
    def create_sample_corpus(self, topics: List[str] = None) -> Dict[str, int]:
        """Create a sample corpus with diverse content."""
        if topics is None:
            topics = ["literature", "science", "technology"]
            
        stats = {"downloaded": 0, "failed": 0}
        
        for topic in topics:
            logger.info(f"Downloading {topic} content...")
            results = self.download_sample_texts(topic)
            
            stats["downloaded"] += sum(results.values())
            stats["failed"] += len(results) - sum(results.values())
            
        return stats


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Download training data")
    parser.add_argument("--topic", help="Topic to download (literature, science, technology, philosophy, history)")
    parser.add_argument("--count", type=int, default=3, help="Number of files to download")
    parser.add_argument("--corpus", action="store_true", help="Download sample corpus")
    parser.add_argument("--book", type=int, action="append", help="Gutenberg book ID to download")
    
    args = parser.parse_args()
    
    downloader = SmartDataDownloader()
    
    if args.corpus:
        stats = downloader.create_sample_corpus()
        logger.info(f"Corpus created: {stats['downloaded']} files downloaded, {stats['failed']} failed")
        
    elif args.book:
        results = downloader.download_gutenberg_books(args.book)
        logger.info(f"Books downloaded: {sum(results.values())}/{len(results)}")
        
    elif args.topic:
        results = downloader.download_sample_texts(args.topic, args.count)
        logger.info(f"Topic '{args.topic}': {sum(results.values())}/{len(results)} files downloaded")
        
    else:
        # Interactive mode
        print("Available topics: literature, science, technology, philosophy, history")
        topic = input("Enter topic: ")
        count = int(input("Number of files (default 3): ") or "3")
        
        results = downloader.download_sample_texts(topic, count)
        logger.info(f"Results: {sum(results.values())}/{len(results)} files downloaded")


if __name__ == "__main__":
    main()
