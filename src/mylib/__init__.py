import torch
from scml import nlp as snlp

__all__ = [
    "preprocess",
    "digit_frac",
    "letter_frac",
    "space_frac",
    "punc_frac",
    "upper_frac",
    "Dataset",
]


def preprocess(s: str) -> str:
    """
    Preserve case and punctuation.
    """
    res: str = snlp.to_ascii(s)
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
