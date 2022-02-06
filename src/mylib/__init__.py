import emoji
import numpy as np
import pandas as pd
import spacy
import torch
from scml import nlp as snlp
from typing import AnyStr, Dict, Union


__all__ = [
    "pre1",
    "pre2",
    "pre3",
    "digit_frac",
    "letter_frac",
    "space_frac",
    "punc_frac",
    "upper_frac",
    "repeat_char_frac",
    "repeat_substring_frac",
    "Dataset",
    "comp_metric",
]


def comp_metric(
    preds: Dict[str, Union[int, float]], validation_data: pd.DataFrame
) -> float:
    scores = []
    for t in validation_data.itertuples():
        less = getattr(t, "less_toxic")
        more = getattr(t, "more_toxic")
        s = 0
        if preds[less] < preds[more]:
            s = 1
        scores.append(s)
    return float(np.mean(scores))


_emoticon = snlp.EmoticonToText(prefix=" [", suffix="] ")
_slang = snlp.SlangExpansion(keep_original_term=True)
_contraction = snlp.ContractionExpansion()
_r_char = snlp.RepeatingCharacter(max_times=3, letters=True, punctuation=True)
_r_substring = snlp.RepeatingSubstring(
    min_length=3, max_times=1, letters=True, punctuation=True
)
nlp = spacy.load("en_core_web_lg", exclude=["textcat"])


def pre1(s: AnyStr) -> str:
    """
    Preprocess Stage 1: Preserve case and punctuation.
    """
    res: str = snlp.to_str(s)
    # remove extra whitespace before preprocess
    res = snlp.collapse_whitespace(res)
    res = snlp.strip_xml(res, replacement=" ")
    res = snlp.strip_url(res, replacement=" ")
    res = snlp.strip_ip_address(res, replacement=" ")
    res = emoji.demojize(res)
    res = snlp.emoji_shortcode_to_text(res)
    res = _emoticon.apply(res)
    res = snlp.to_ascii(res)
    # make sure no extra whitespace after preprocess
    res = snlp.collapse_whitespace(res)
    return res


def pre2(s: str) -> str:
    """
    Preprocess Stage 2: Prepare output for transformer models, embeddings
    """
    res = s
    res = _r_substring.collapse(res)
    res = _r_char.collapse(res)
    res = _slang.apply(res)
    res = _contraction.apply(res)
    res = snlp.collapse_whitespace(res)
    return res


def pre3(s: str) -> str:
    """
    Preprocess Stage 3: Prepare output for TF-IDF features
    """
    res = s
    res = snlp.strip_punctuation(res, replacement=" ")
    doc = nlp(res)
    tokens = []
    for token in doc:
        # some lemma has uppercase char
        tokens.append(token.lemma_)
    res = " ".join(tokens)
    res = res.lower()
    res = snlp.collapse_whitespace(res)
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


def repeat_char_frac(s: str) -> float:
    return _r_char.count(s) / len(s)  # type: ignore


def repeat_substring_frac(s: str) -> float:
    return _r_substring.count_char(s) / len(s)  # type: ignore


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
