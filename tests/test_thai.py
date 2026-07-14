import pytest

from words_segmentation.languages import segment_text
from words_segmentation.thai import has_thai, segment_thai


def test_has_thai_simple():
    """Test has_thai with simple Thai text."""
    assert has_thai("สวัสดี")
    assert has_thai("ภาษาไทย")
    assert has_thai("ผมรักเมืองไทย")


def test_has_thai_mixed_content():
    """Test has_thai with mixed Thai and other characters."""
    assert has_thai("hello สวัสดี")
    assert has_thai("mixed ไทย")
    assert has_thai("123 ภาษาไทย abc")
    assert has_thai("English ไทย ผสม")


def test_has_thai_no_thai():
    """Test has_thai with non-Thai text."""
    assert not has_thai("hello")
    assert not has_thai("English text")
    assert not has_thai("123456")
    assert not has_thai("!@#$%^&*()")
    assert not has_thai("こんにちは")  # Japanese
    assert not has_thai("你好")  # Chinese
    assert not has_thai("ລາວ")  # Lao


def test_has_thai_empty_string():
    """Test has_thai with empty string."""
    assert not has_thai("")


def test_has_thai_whitespace_only():
    """Test has_thai with whitespace only."""
    assert not has_thai(" ")
    assert not has_thai("\n\t")
    assert not has_thai("   ")


def test_segment_thai_simple():
    """Test segment_thai with a single Thai word."""
    result = segment_thai("สวัสดี")
    assert result == ["สวัสดี"]


def test_segment_thai_empty():
    """Test segment_thai with empty string."""
    result = segment_thai("")
    assert result == []


def test_segment_thai_sentence():
    """Test segment_thai with a Thai sentence without spaces."""
    result = segment_thai("ผมรักเมืองไทย")
    assert result == ["ผม", "รัก", "เมือง", "ไทย"]


def test_segment_thai_compound_words():
    """Test segment_thai with compound words."""
    result = segment_thai("การประมวลผลภาษาธรรมชาติ")
    assert "".join(result) == "การประมวลผลภาษาธรรมชาติ"
    assert len(result) > 1


def test_segment_text_dispatches_thai():
    """Test that Thai text is routed to segment_thai via LANGUAGE_SPECS."""
    result = list(segment_text("ผมรักเมืองไทย"))
    assert result == [["ผม", "รัก", "เมือง", "ไทย"]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
