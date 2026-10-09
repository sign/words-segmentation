import pytest

from words_segmentation.languages import segment_text
from words_segmentation.pretokenizer import text_to_words
from words_segmentation.signwriting import segment_signwriting


def test_segment_single_sign():
    sign = "𝠀񆄱񈠣񍉡𝠃𝤛𝤵񍉡𝣴𝣵񆄱𝤌𝤆񈠣𝤉𝤚"
    result = segment_signwriting(sign)
    assert result == [sign]

def test_segment_single_sign_no_prefix():
    sign = "𝠃𝤛𝤵񍉡𝣴𝣵񆄱𝤌𝤆񈠣𝤉𝤚"
    result = segment_signwriting(sign)
    assert result == [sign]

def test_segment_with_space():
    signs = [
        "𝠀񀀒񀀚񋚥񋛩𝠃𝤟𝤩񋛩𝣵𝤐񀀒𝤇𝣤񋚥𝤐𝤆񀀚𝣮𝣭",
        "𝠀񂇢񂇈񆙡񋎥񋎵𝠃𝤛𝤬񂇈𝤀𝣺񂇢𝤄𝣻񋎥𝤄𝤗񋎵𝤃𝣟񆙡𝣱𝣸",
        "𝠃𝤙𝤞񀀙𝣷𝤀񅨑𝣼𝤀񆉁𝣳𝣮"
    ]
    result = segment_signwriting(" ".join(signs))
    assert result == [signs[0] + " ", signs[1] + " ", signs[2]]

def test_segment_no_space():
    signs = [
        "𝠀񀀒񀀚񋚥񋛩𝠃𝤟𝤩񋛩𝣵𝤐񀀒𝤇𝣤񋚥𝤐𝤆񀀚𝣮𝣭",
        "𝠀񂇢񂇈񆙡񋎥񋎵𝠃𝤛𝤬񂇈𝤀𝣺񂇢𝤄𝣻񋎥𝤄𝤗񋎵𝤃𝣟񆙡𝣱𝣸",
        "𝠃𝤙𝤞񀀙𝣷𝤀񅨑𝣼𝤀񆉁𝣳𝣮"
    ]
    result = segment_signwriting("".join(signs))
    assert result == signs

def test_segment_keeps_incomplete_signs():
    """Characters that do not form a complete sign are kept, each its own word."""
    text = "𝠀񆄱񈠣"  # A sign prefix, without its box
    assert segment_signwriting(text) == list(text)


def test_signs_are_words_within_text():
    """SWU symbols (plane 4) belong to SignWriting, though Unicode does not assign them a script."""
    signs = ["𝠀񆄱񈠣񍉡𝠃𝤛𝤵񍉡𝣴𝣵񆄱𝤌𝤆񈠣𝤉𝤚", "𝠃𝤙𝤞񀀙𝣷𝤀񅨑𝣼𝤀񆉁𝣳𝣮"]
    text = f"<ase>\x0e{signs[0]} {signs[1]}\x0f<en> hello"
    words = text_to_words(text)
    assert signs[0] in words
    assert signs[1] in words
    assert "".join(words) == text


def test_segment_text_is_lossless():
    text = "𝠀񆄱񈠣񍉡𝠃𝤛𝤵񍉡𝣴𝣵񆄱𝤌𝤆񈠣𝤉𝤚 and 𝠃𝤙𝤞 text"
    assert "".join(word for words in segment_text(text) for word in words) == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
