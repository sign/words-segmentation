import pytest
import torch

from words_segmentation.pretokenizer import WordStoppingCriteria


class MockTokenizer:
    """Mock tokenizer for testing WordStoppingCriteria."""

    def decode(self, token_ids):
        """Simple mock decode that converts token IDs to characters."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Simple mapping: use chr() for decoding, allow control characters
        return ''.join(chr(tid % 128) for tid in token_ids)


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
