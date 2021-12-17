from detoxify import Detoxify
from typing import List, Dict
from scml import nlp as snlp

__all__ = [
    "preprocess",
    "digit_frac",
    "letter_frac",
    "space_frac",
    "punc_frac",
    "upper_frac",
    "detoxify_labels",
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


def detoxify_labels(
    sentences: List[str], checkpoint: str, device, batch_size: int
) -> Dict[str, List[float]]:
    model = Detoxify(
        checkpoint=checkpoint,
        device=device,
    )
    i = 0
    res: Dict[str, List[float]] = {}
    while i < len(sentences):
        batch = sentences[i : i + batch_size]
        tmp = model.predict(batch)
        if len(res) == 0:
            res = tmp
        else:
            for k in res.keys():
                res[k] += tmp[k]
        i += batch_size
    return res
