import re

from signwriting.formats.swu import re_swu

# Each sign is a word (with its trailing space, like other words); characters outside a complete sign are kept,
# each its own word
_SIGN_OR_CHARACTER = re.compile(f"(?:{re_swu['sign']}) ?|.", re.DOTALL)


def segment_signwriting(text: str) -> list[str]:
    return _SIGN_OR_CHARACTER.findall(text)
