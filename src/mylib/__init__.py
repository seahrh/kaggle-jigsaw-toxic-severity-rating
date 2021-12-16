from scml import nlp as snlp

__all__ = ["preprocess"]


def preprocess(s: str) -> str:
    """
    Preserve case and punctuation.
    """
    res = snlp.to_ascii(s)
    res = snlp.expand_contractions(res)
    res = " ".join(res.split())
    return res
