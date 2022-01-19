import emoji
import torch
from scml import nlp as snlp
from typing import AnyStr


__all__ = [
    "pre1",
    "pre2",
    "digit_frac",
    "letter_frac",
    "space_frac",
    "punc_frac",
    "upper_frac",
    "Dataset",
]


def pre1(s: AnyStr) -> str:
    """
    Preprocess Stage 1: Preserve case and punctuation.
    """
    res: str = snlp.to_str(s)
    res = emoji.demojize(res)
    res = snlp.emoji_shortcode_to_text(res)
    res = snlp.to_ascii(res)
    res = " ".join(res.split())
    return res


_slang = snlp.Slang()


def pre2(s: str) -> str:
    """
    Preprocess Stage 2: Prepare output for transformer models, embeddings
    """
    res = s
    res = _slang.expand(res)
    res = snlp.expand_contractions(res)
    res = " ".join(res.split())
    return res


def digit_frac(s: str) -> float:
    return snlp.count_digit(s) / len(s)  # type: ignore


def letter_frac(s: str) -> float:
    return snlp.count_alpha(s) / len(s)  # type: ignore


def space_frac(s: str) -> float:
    return snlp.count_space(s) / len(s)  # type: ignore


def punc_frac(s: str) -> float:
    return snlp.count_punctuation(s) / len(s)  # type: ignore


def upper_frac(s: str) -> float:
    return snlp.count_upper(s) / len(s)  # type: ignore


# noinspection PyUnresolvedReferences
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


from .detoxify import *

__all__ += detoxify.__all__  # type: ignore
