import pytest
import torch

from words_segmentation.pretokenizer import (
    WordStoppingCriteria,
    is_word_complete,
    text_to_words,
    utf8_chunks_grapheme_safe,
)


class MockTokenizer:
    """Mock tokenizer for testing WordStoppingCriteria."""

    def decode(self, token_ids):
        """Simple mock decode that converts token IDs to characters."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Simple mapping: use chr() for decoding, allow control characters
        return ''.join(chr(tid % 128) for tid in token_ids)


def test_utf8_chunks_english():
    """Test utf8_chunks_grapheme_safe with English text."""
    text = "hello world"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=5))
    assert chunks == ["hello", " worl", "d"]
    assert len(chunks) == 3
    assert all(len(chunk.encode('utf-8')) <= 5 for chunk in chunks)


def test_utf8_chunks_hebrew():
    """Test utf8_chunks_grapheme_safe with Hebrew text."""
    text = "עמית מוריוסף"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=8))
    assert chunks == ['עמית', ' מור', 'יוסף']
    assert "".join(chunks) == text
    assert len(chunks) == 3
    assert all(len(chunk.encode('utf-8')) <= 8 for chunk in chunks)


def test_utf8_chunks_emoji():
    """Test utf8_chunks_grapheme_safe with basic emoji."""
    text = "hello 😀 world"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=8))
    assert chunks == ['hello ', '😀 wor', 'ld']
    assert "".join(chunks) == text
    assert len(chunks) == 3
    assert all(len(chunk.encode('utf-8')) <= 8 for chunk in chunks)


def test_utf8_chunks_long_emoji_cluster():
    """Test utf8_chunks_grapheme_safe with complex emoji cluster."""
    text = "👩‍👩‍👧‍👦"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=5))
    assert chunks == [text]
    assert len(chunks) == 1
    assert len(text.encode('utf-8')) > 5


def test_utf8_chunks_single_grapheme():
    """Test special case of single grapheme cluster."""
    text = "a"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=16))
    assert chunks == ["a"]


def test_utf8_chunks_mixed_content():
    """Test with mixed English, Hebrew, and emoji."""
    text = "hello עמית 👋"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=10))
    assert "".join(chunks) == text


def test_text_to_words_json():
    """Test text_to_words with JSON string."""
    json_text = '{"name": "test", "value": 123}'
    words = text_to_words(json_text, max_bytes=10)
    assert words == ['{"name": ', '"test", ', '"value": ', "123}"]
    assert "".join(words) == json_text
    assert len(words) == 4


def test_text_to_words_long_string():
    """Test text_to_words with long string."""
    long_text = ("This is a very long string that should be split into multiple chunks "
                 "when processed by the text_to_words function with appropriate byte limits.")
    words = text_to_words(long_text, max_bytes=8)
    assert "".join(words) == long_text


def test_text_to_words_short_string():
    """Test text_to_words with short string."""
    short_text = "hi"
    words = text_to_words(short_text, max_bytes=16)
    assert words == ["hi"]


def test_text_to_words_multiline_code():
    """Test similar to the processor test example."""
    text = """
    def foo():
        return "bar"
    """.strip()
    words = text_to_words(text, max_bytes=10)
    assert words == ['def ', 'foo():\n', '        ', 'return ', '"bar"']
    assert "".join(words) == text
    assert len(words) == 5
    assert 'def ' in words
    assert '        ' in words


def test_text_to_words_whitespace():
    """Test proper whitespace handling."""
    text = "hello    world"
    words = text_to_words(text, max_bytes=8)
    assert words == ["hello ", "   ", "world"]
    assert "".join(words) == text
    assert len(words) == 3
    assert "   " in words


def test_text_to_words_mixed_unicode():
    """Test with mixed content including unicode."""
    text = "hello עמית! 🌟 {'key': 'value'}"
    words = text_to_words(text, max_bytes=10)
    assert "".join(words) == text


def test_text_to_words_empty():
    """Test with empty string."""
    words = text_to_words("", max_bytes=16)
    assert words == []


def test_text_to_words_only_whitespace():
    """Test with only whitespace."""
    text = "   \n\t  "
    words = text_to_words(text, max_bytes=16)
    assert text == words[0]
    assert "".join(words) == text
    assert len(words) == 1
    assert all(c.isspace() for c in words[0])


def test_text_to_words_json_unicode():
    """Test JSON containing unicode characters."""
    json_text = '{"message":"שלום world 🌍","count": 42}'
    words = text_to_words(json_text, max_bytes=6)
    assert "".join(words) == json_text
    assert len(words) > 5
    assert any("🌍" in word for word in words)


def test_is_word_complete_control_tokens():
    """Test is_word_complete with control tokens."""
    assert is_word_complete("\x01")
    assert is_word_complete("\x02")
    assert is_word_complete("\x03")
    assert is_word_complete("\x08")
    assert is_word_complete("\x7F")


def test_is_word_complete_words_with_space():
    """Test is_word_complete with words that have trailing space."""
    assert is_word_complete("hello ")
    assert is_word_complete("world ")
    assert is_word_complete("test ")
    assert is_word_complete("עמית ")
    assert is_word_complete("🌟 ")


def test_is_word_complete_incomplete_words():
    """Test is_word_complete with incomplete words (no trailing space)."""
    assert not is_word_complete("hello")
    assert not is_word_complete("world")
    assert not is_word_complete("test")
    assert not is_word_complete("עמית")
    assert not is_word_complete("🌟")


def test_is_word_complete_whitespace_only():
    """Test is_word_complete with whitespace-only strings."""
    assert not is_word_complete(" ")
    assert not is_word_complete("  ")
    assert not is_word_complete("\n")
    assert not is_word_complete("\t")
    assert not is_word_complete("   \n\t  ")


def test_is_word_complete_empty_string():
    """Test is_word_complete with empty string."""
    assert not is_word_complete("")

def test_is_word_complete_unicode_with_space():
    """Test is_word_complete with unicode characters and trailing space."""
    assert is_word_complete("שלום ")
    assert is_word_complete("مرحبا ")
    assert is_word_complete("こんにちは ")
    assert not is_word_complete("שלום")
    assert not is_word_complete("مرحبا")
    assert not is_word_complete("こんにちは")


def test_word_stopping_criteria_basic():
    """Test WordStoppingCriteria with basic functionality on CPU."""
    tokenizer = MockTokenizer()
    criteria = WordStoppingCriteria(tokenizer)

    # Test with complete word (has trailing space) - ASCII 'h','e','l','l','o',' ' = 104,101,108,108,111,32
    input_ids = torch.tensor([[104, 101, 108, 108, 111, 32]])  # "hello "
    scores = torch.zeros((1, 100))
    result = criteria(input_ids, scores)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.bool
    assert result.device.type == "cpu"
    assert result.shape == (1,)
    assert result[0].item() is True  # "hello " is complete


def test_word_stopping_criteria_incomplete():
    """Test WordStoppingCriteria with incomplete word."""
    tokenizer = MockTokenizer()
    criteria = WordStoppingCriteria(tokenizer)

    # Test with incomplete word (no trailing space) - ASCII 'h','e','l','l','o' = 104,101,108,108,111
    input_ids = torch.tensor([[104, 101, 108, 108, 111]])  # "hello"
    scores = torch.zeros((1, 100))
    result = criteria(input_ids, scores)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.bool
    assert result.device.type == "cpu"
    assert result.shape == (1,)
    assert result[0].item() is False  # "hello" is incomplete


def test_word_stopping_criteria_batch():
    """Test WordStoppingCriteria with batch of inputs."""
    tokenizer = MockTokenizer()
    criteria = WordStoppingCriteria(tokenizer)

    # Batch with mixed complete and incomplete words (same length with padding)
    input_ids = torch.tensor([
        [104, 101, 108, 108, 111, 32],  # "hello " - complete
        [104, 101, 108, 108, 111, 0],    # "hello" (with padding) - incomplete
        [119, 111, 114, 108, 100, 32],   # "world " - complete
    ])
    scores = torch.zeros((3, 100))
    result = criteria(input_ids, scores)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.bool
    assert result.device.type == "cpu"
    assert result.shape == (3,)
    assert result[0].item() is True   # "hello " is complete
    assert result[1].item() is False  # "hello" is incomplete
    assert result[2].item() is True   # "world " is complete


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_word_stopping_criteria_cuda_device():
    """Test WordStoppingCriteria respects CUDA device."""
    tokenizer = MockTokenizer()
    criteria = WordStoppingCriteria(tokenizer)

    # Test with input on CUDA
    input_ids = torch.tensor([[104, 101, 108, 108, 111, 32]], device="cuda")  # "hello "
    scores = torch.zeros((1, 100), device="cuda")
    result = criteria(input_ids, scores)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.bool
    assert result.device.type == "cuda"  # Should match input device
    assert result.shape == (1,)
    assert result[0].item() is True


def test_word_stopping_criteria_control_token():
    """Test WordStoppingCriteria with control tokens."""
    tokenizer = MockTokenizer()
    criteria = WordStoppingCriteria(tokenizer)

    # Control token (e.g., \x01)
    input_ids = torch.tensor([[1]])  # Control token
    scores = torch.zeros((1, 100))
    result = criteria(input_ids, scores)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.bool
    assert result.shape == (1,)
    assert result[0].item() is True  # Control tokens are always complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
