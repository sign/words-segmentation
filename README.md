# Words Segmentation

This repository contains a pretokenizer that segments text into "words" for further processing.

We define three classes of tokens:

1. `C0` Control tokens (always atomic)
2. "Words" = runs of non-space, non-control + optional single trailing whitespace
3. Whitespace runs

For any script where the default is not suitable, you can implement a custom pretokenizer.
Modify `LANGUAGE_SPECS` in [languages.py](./words_segmentation/languages.py) to add a custom function for specific scripts.

For example:

```python
LANGUAGE_SPECS: Dict[str, LanguageSpec] = {
    "Chinese": {
        "scripts": ("Han",),
        "callback": segment_chinese,
    },
    "Japanese": {
        "scripts": ("Han", "Hiragana", "Katakana"),
        "callback": segment_japanese,
    },
}
```

Then, with a `max_bytes` parameter, we split long words into smaller chunks while preserving
Unicode grapheme boundaries.

## Usage

Install:
```bash
pip install words-segmentation
```

Pretokenize text using a Huggingface Tokenizer implementation:
```python
from words_segmentation.tokenizer import WordsSegmentationTokenizer

pretokenizer = WordsSegmentationTokenizer(max_bytes=16)
tokens = pretokenizer.tokenize("hello world! 我爱北京天安门 👩‍👩‍👧‍👦")
# ['hello ', 'world! ', '我', '爱', '北京', '天安门', ' ', '👩‍👩‍👧‍👦‍']
```


## [Writing systems without word boundaries](https://en.wikipedia.org/wiki/Category:Writing_systems_without_word_boundaries)

Perhaps there will come a day when we could have a universal pretokenizer that works for all languages.
Until then, we need to handle some writing systems with custom logic.
We implement custom fallback pretoknizers for the following writing systems:

- [x] [Chinese characters](https://en.wikipedia.org/wiki/Chinese_characters) - using [jieba](https://github.com/fxsjy/jieba)
- [x] [Japanese writing system](https://en.wikipedia.org/wiki/Japanese_writing_system) - using [fugashi](https://github.com/polm/fugashi)
- [ ] [Balinese script](https://en.wikipedia.org/wiki/Balinese_script)
- [ ] [Burmese alphabet](https://en.wikipedia.org/wiki/Burmese_alphabet)
- [ ] [Chữ Hán](https://en.wikipedia.org/wiki/Ch%E1%BB%AF_H%C3%A1n)
- [ ] [Chữ Nôm](https://en.wikipedia.org/wiki/Ch%E1%BB%AF_N%C3%B4m)
- [ ] [Hanja](https://en.wikipedia.org/wiki/Hanja)
- [ ] [Javanese script](https://en.wikipedia.org/wiki/Javanese_script)
- [ ] [Khmer script](https://en.wikipedia.org/wiki/Khmer_script)
- [ ] [Lao script](https://en.wikipedia.org/wiki/Lao_script)
- [ ] [ʼPhags-pa script](https://en.wikipedia.org/wiki/%CA%BCPhags-pa_script)
- [ ] [Rasm](https://en.wikipedia.org/wiki/Rasm)
- [ ] [Sawndip](https://en.wikipedia.org/wiki/Sawndip)
- [ ] [Scriptio continua](https://en.wikipedia.org/wiki/Scriptio_continua)
- [ ] [S'gaw Karen alphabet](https://en.wikipedia.org/wiki/S%27gaw_Karen_alphabet)
- [ ] [Tai Tham script](https://en.wikipedia.org/wiki/Tai_Tham_script)
- [ ] [Thai script](https://en.wikipedia.org/wiki/Thai_script)
- [ ] [Tibetan script](https://en.wikipedia.org/wiki/Tibetan_script)
- [ ] [Vietnamese alphabet](https://en.wikipedia.org/wiki/Vietnamese_alphabet)
- [ ] [Western Pwo alphabet](https://en.wikipedia.org/wiki/Western_Pwo_alphabet)


## Cite

If you use this code in your research, please consider citing the work:

```bibtex
@misc{moryossef2025words,
  title={Words Segmentation: A Word Level Pre-tokenizer for Languages of the World},
  author={Moryossef, Amit},
  howpublished={\url{https://github.com/sign/words-segmentation}},
  year={2025}
}
```