"""
Thai text pretokenization utilities.

This module provides functions for detecting and segmenting Thai text using the pythainlp
library for word segmentation.
"""

from functools import cache

import regex


def has_thai(text: str) -> bool:
    """
    Check if the given text contains Thai characters.

    Args:
        text: The input text to check for Thai characters

    Returns:
        True if Thai characters are found, False otherwise
    """
    # Match any character in the Thai Unicode script
    return bool(regex.search(r'[\p{Thai}]', text))


@cache
def get_thai_segmenter():
    """
    Get a cached instance of the PyThaiNLP Thai word tokenizer.

    PyThaiNLP is a popular Thai text processing library that uses a combination of
    dictionary-based matching and statistical models to segment Thai text into words.
    The tokenizer is cached to avoid repeated initialization overhead.

    Returns:
        pythainlp word tokenization function for text segmentation

    Raises:
        ImportError: If the pythainlp library is not installed
    """
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:
        print("Error: pythainlp library not found. Please install it with: pip install pythainlp")
        raise

    return word_tokenize


def segment_thai(text: str) -> list[str]:
    """
    Segment Thai text into words.

    Thai is written without spaces between words, so this uses PyThaiNLP's word
    tokenizer to break Thai text into individual words. This preprocessing step
    helps the tokenizer better understand Thai text structure.

    Args:
        text: The Thai text to segment

    Returns:
        List of Thai words

    Example:
        >>> segment_thai("สวัสดีโลก")
        ["สวัสดี", "โลก"]
    """
    word_tokenize = get_thai_segmenter()
    segments = word_tokenize(text)
    # Filter out empty segments the tokenizer may produce
    return [segment for segment in segments if segment]
