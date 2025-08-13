"""
Language Filter Module for Advanced Cybersecurity Dataset Cleaning.

This module implements language detection and filtering capabilities to ensure
training data contains only content in target languages with configurable
confidence thresholds.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger

from ..base_module import CleaningModule
from ..data_models import CleaningOperation, CleaningOperationType


class LanguageDetectionResult:
    """Result of language detection operation."""
    
    def __init__(self, detected_language: str, confidence: float, 
                 is_target_language: bool, method_used: str):
        self.detected_language = detected_language
        self.confidence = confidence
        self.is_target_language = is_target_language
        self.method_used = method_used


class LanguageFilter(CleaningModule):
    """
    Filters content by language using configurable detection methods and
    confidence thresholds.
    
    This module supports multiple language detection backends including
    fasttext and langdetect, with graceful fallback when dependencies
    are not available.
    """
    
    def _initialize_module(self) -> None:
        """Initialize the language detection system and configuration."""
        # Configuration options with validation
        self.target_languages = self.config.get('target_languages', ['en'])
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.detection_method = self.config.get('detection_method', 'auto')
        self.fallback_to_heuristics = self.config.get('fallback_to_heuristics', True)
        
        # Validate and normalize target languages
        if isinstance(self.target_languages, str):
            # Handle case where string is passed instead of list
            self.target_languages = [self.target_languages]
        
        if isinstance(self.target_languages, list):
            # Normalize target languages to lowercase, filtering out invalid entries
            normalized_languages = []
            for lang in self.target_languages:
                if isinstance(lang, str) and len(lang) >= 2:
                    normalized_languages.append(lang.lower())
                else:
                    logger.warning(f"Invalid language code ignored: {lang}")
            self.target_languages = normalized_languages or ['en']  # Fallback to English
        else:
            logger.warning(f"Invalid target_languages type: {type(self.target_languages)}, using default ['en']")
            self.target_languages = ['en']
        
        # Initialize detection backend
        self.detector = None
        self.detection_backend = None
        
        self._initialize_detection_backend()
        
        # Heuristic patterns for basic language detection
        self._initialize_heuristic_patterns()
        
        logger.info(f"LanguageFilter initialized with backend: {self.detection_backend}, "
                   f"targets: {self.target_languages}, threshold: {self.confidence_threshold}")
    
    def _initialize_detection_backend(self) -> None:
        """Initialize the language detection backend."""
        if self.detection_method == 'fasttext':
            self._try_initialize_fasttext()
        elif self.detection_method == 'langdetect':
            self._try_initialize_langdetect()
        else:  # auto
            # Try fasttext first, then langdetect, then heuristics
            if not self._try_initialize_fasttext():
                if not self._try_initialize_langdetect():
                    self._initialize_heuristic_only()
    
    def _try_initialize_fasttext(self) -> bool:
        """Try to initialize fasttext language detection."""
        try:
            import fasttext
            
            # Try to load the language identification model
            # This will download the model if not present
            try:
                self.detector = fasttext.load_model('lid.176.bin')
                self.detection_backend = 'fasttext'
                logger.info("FastText language detection initialized successfully")
                return True
            except Exception as e:
                logger.warning(f"Failed to load FastText model: {e}")
                return False
                
        except ImportError:
            logger.debug("FastText not available, trying alternative backends")
            return False
    
    def _try_initialize_langdetect(self) -> bool:
        """Try to initialize langdetect language detection."""
        try:
            from langdetect import detect, detect_langs, LangDetectException
            from langdetect.lang_detect_exception import LangDetectException
            
            # Test the library with a simple detection
            try:
                detect("This is a test sentence in English.")
                self.detector = detect_langs  # Use detect_langs for confidence scores
                self.detection_backend = 'langdetect'
                logger.info("Langdetect language detection initialized successfully")
                return True
            except Exception as e:
                logger.warning(f"Failed to initialize langdetect: {e}")
                return False
                
        except ImportError:
            logger.debug("Langdetect not available, trying heuristic fallback")
            return False
    
    def _initialize_heuristic_only(self) -> None:
        """Initialize heuristic-only language detection."""
        self.detection_backend = 'heuristic'
        logger.warning("No language detection libraries available, using heuristic fallback")
    
    def _initialize_heuristic_patterns(self) -> None:
        """Initialize heuristic patterns for basic language detection."""
        # Common English patterns and indicators
        self.english_patterns = [
            re.compile(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', re.IGNORECASE),
            re.compile(r'\b(is|are|was|were|have|has|had|will|would|can|could)\b', re.IGNORECASE),
            re.compile(r'\b(this|that|these|those|what|where|when|why|how)\b', re.IGNORECASE),
        ]
        
        # Character frequency patterns for different languages
        self.char_patterns = {
            'en': re.compile(r'[a-zA-Z]'),
            'es': re.compile(r'[a-zA-ZñáéíóúüÑÁÉÍÓÚÜ]'),
            'fr': re.compile(r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]'),
            'de': re.compile(r'[a-zA-ZäöüßÄÖÜ]'),
        }
    
    def get_operation_type(self) -> CleaningOperationType:
        """Return the operation type for this module."""
        return CleaningOperationType.LANGUAGE_FILTERING
    
    def clean_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, CleaningOperation]:
        """
        Filter text based on language detection.
        
        Args:
            text: The text to analyze and potentially filter
            metadata: Optional metadata about the text
            
        Returns:
            Tuple of (filtered_text, cleaning_operation)
        """
        if not text or not text.strip():
            operation = self._create_operation(
                description="No text to process",
                original_length=len(text),
                final_length=len(text),
                success=True
            )
            return text, operation
        
        original_length = len(text)
        
        # Detect language
        detection_result = self.detect_language(text)
        
        # Determine if text should be kept
        should_keep = detection_result.is_target_language
        
        # If confidence is too low, apply stricter filtering
        if detection_result.confidence < self.confidence_threshold:
            should_keep = False
        
        # Prepare result
        if should_keep:
            filtered_text = text
            description = f"Language accepted: {detection_result.detected_language} " \
                         f"(confidence: {detection_result.confidence:.3f})"
            items_removed = 0
        else:
            filtered_text = ""
            description = f"Language filtered: {detection_result.detected_language} " \
                         f"(confidence: {detection_result.confidence:.3f}, threshold: {self.confidence_threshold})"
            items_removed = 1
        
        # Create operation record
        operation = self._create_operation(
            description=f"Language filtering: {description}",
            original_length=original_length,
            final_length=len(filtered_text),
            items_removed=items_removed,
            success=True,
            metadata={
                "detected_language": detection_result.detected_language,
                "confidence": detection_result.confidence,
                "is_target_language": detection_result.is_target_language,
                "detection_method": detection_result.method_used,
                "target_languages": self.target_languages,
                "confidence_threshold": self.confidence_threshold
            }
        )
        
        return filtered_text, operation
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of the provided text.
        
        Args:
            text: The text to analyze
            
        Returns:
            LanguageDetectionResult with detection details
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                detected_language="unknown",
                confidence=0.0,
                is_target_language=False,
                method_used=self.detection_backend or "none"
            )
        
        # Clean text for detection (remove excessive whitespace, special chars)
        clean_text = self._prepare_text_for_detection(text)
        
        if len(clean_text) < 10:  # Too short for reliable detection
            return LanguageDetectionResult(
                detected_language="unknown",
                confidence=0.0,
                is_target_language=False,
                method_used=f"{self.detection_backend}_too_short"
            )
        
        # Try detection with available backend
        if self.detection_backend == 'fasttext':
            return self._detect_with_fasttext(clean_text)
        elif self.detection_backend == 'langdetect':
            return self._detect_with_langdetect(clean_text)
        else:
            return self._detect_with_heuristics(clean_text)
    
    def _prepare_text_for_detection(self, text: str) -> str:
        """Prepare text for language detection by cleaning it."""
        # Remove excessive whitespace
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs, emails, and other non-linguistic content
        clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
        clean_text = re.sub(r'\S+@\S+\.\S+', '', clean_text)
        clean_text = re.sub(r'\b\d+\b', '', clean_text)  # Remove standalone numbers
        
        # Remove excessive punctuation
        clean_text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def _detect_with_fasttext(self, text: str) -> LanguageDetectionResult:
        """Detect language using FastText."""
        try:
            predictions = self.detector.predict(text, k=1)
            
            # FastText returns labels like '__label__en' and probabilities
            detected_lang = predictions[0][0].replace('__label__', '')
            confidence = float(predictions[1][0])
            
            is_target = detected_lang.lower() in self.target_languages
            
            return LanguageDetectionResult(
                detected_language=detected_lang,
                confidence=confidence,
                is_target_language=is_target,
                method_used="fasttext"
            )
            
        except Exception as e:
            logger.warning(f"FastText detection failed: {e}")
            return self._detect_with_heuristics(text)
    
    def _detect_with_langdetect(self, text: str) -> LanguageDetectionResult:
        """Detect language using langdetect."""
        try:
            from langdetect import detect_langs, LangDetectException
            
            detections = detect_langs(text)
            
            if detections:
                # Get the most confident detection
                best_detection = detections[0]
                detected_lang = best_detection.lang
                confidence = best_detection.prob
                
                is_target = detected_lang.lower() in self.target_languages
                
                return LanguageDetectionResult(
                    detected_language=detected_lang,
                    confidence=confidence,
                    is_target_language=is_target,
                    method_used="langdetect"
                )
            else:
                return self._detect_with_heuristics(text)
                
        except Exception as e:
            logger.warning(f"Langdetect detection failed: {e}")
            return self._detect_with_heuristics(text)
    
    def _detect_with_heuristics(self, text: str) -> LanguageDetectionResult:
        """Detect language using simple heuristics."""
        # Count English indicators
        english_score = 0
        total_words = len(text.split())
        
        if total_words == 0:
            return LanguageDetectionResult(
                detected_language="unknown",
                confidence=0.0,
                is_target_language=False,
                method_used="heuristic_no_words"
            )
        
        # Check English patterns
        for pattern in self.english_patterns:
            matches = len(pattern.findall(text))
            english_score += matches
        
        # Calculate confidence based on English indicators
        # Use a more generous scoring system
        base_confidence = min(english_score / total_words, 1.0)
        
        # Boost confidence for longer texts with English characteristics
        if total_words >= 5:
            # Check for common English word patterns (expanded list)
            english_words = len(re.findall(r'\b(the|and|or|but|in|on|at|to|for|of|with|by|is|are|was|were|have|has|had|will|would|can|could|this|that|these|those|what|where|when|why|how|a|an|from|into|through|during|before|after|above|below|up|down|out|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|s|t|can|will|just|don|should|now)\b', text, re.IGNORECASE))
            if english_words > 0:
                word_ratio = english_words / total_words
                base_confidence = max(base_confidence, word_ratio * 1.2)  # Boost confidence more for English words
        
        # Check for alphabetic content ratio and English-like characteristics
        alphabetic_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0:
            alpha_ratio = alphabetic_chars / total_chars
            # Boost confidence for mostly alphabetic content
            if alpha_ratio > 0.8:  # Mostly alphabetic content
                # Additional check for English-like word patterns
                words = text.split()
                english_like_words = 0
                for word in words:
                    # Check for English-like characteristics (vowel patterns, common endings)
                    if re.search(r'[aeiou]', word.lower()) and len(word) > 2:
                        english_like_words += 1
                
                if english_like_words > 0:
                    english_like_ratio = english_like_words / len(words)
                    base_confidence = max(base_confidence, english_like_ratio * 0.7)
        
        confidence = min(base_confidence, 1.0)
        
        # Simple heuristic: if confidence > 0.2, assume English
        if confidence > 0.2:
            detected_lang = "en"
        else:
            detected_lang = "unknown"
        
        is_target = detected_lang.lower() in self.target_languages
        
        return LanguageDetectionResult(
            detected_language=detected_lang,
            confidence=confidence,
            is_target_language=is_target,
            method_used="heuristic"
        )
    
    def filter_by_language(self, text: str) -> Tuple[bool, LanguageDetectionResult]:
        """
        Check if text should be kept based on language filtering.
        
        Args:
            text: The text to check
            
        Returns:
            Tuple of (should_keep, detection_result)
        """
        detection_result = self.detect_language(text)
        
        should_keep = (detection_result.is_target_language and 
                      detection_result.confidence >= self.confidence_threshold)
        
        return should_keep, detection_result
    
    def validate_config(self) -> List[str]:
        """
        Validate the module's configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = super().validate_config()
        
        # Validate target_languages
        target_languages = self.config.get('target_languages', ['en'])
        if not isinstance(target_languages, list):
            errors.append("target_languages must be a list")
        elif not target_languages:
            errors.append("target_languages cannot be empty")
        else:
            for i, lang in enumerate(target_languages):
                if not isinstance(lang, str) or len(lang) < 2:
                    errors.append(f"target_languages[{i}] must be a valid language code")
        
        # Validate confidence_threshold
        confidence_threshold = self.config.get('confidence_threshold', 0.8)
        if not isinstance(confidence_threshold, (int, float)) or not (0.0 <= confidence_threshold <= 1.0):
            errors.append("confidence_threshold must be a number between 0.0 and 1.0")
        
        # Validate detection_method
        detection_method = self.config.get('detection_method', 'auto')
        valid_methods = ['auto', 'fasttext', 'langdetect', 'heuristic']
        if detection_method not in valid_methods:
            errors.append(f"detection_method must be one of: {valid_methods}")
        
        # Validate boolean parameters
        boolean_params = ['fallback_to_heuristics']
        for param in boolean_params:
            if param in self.config and not isinstance(self.config[param], bool):
                errors.append(f"{param} must be a boolean")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the language filter configuration.
        
        Returns:
            Dictionary containing module statistics
        """
        return {
            "target_languages": self.target_languages,
            "confidence_threshold": self.confidence_threshold,
            "detection_method": self.detection_method,
            "detection_backend": self.detection_backend,
            "fallback_to_heuristics": self.fallback_to_heuristics,
            "backend_available": self.detection_backend is not None,
            "heuristic_patterns": len(self.english_patterns) if hasattr(self, 'english_patterns') else 0
        }