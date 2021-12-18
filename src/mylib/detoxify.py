from typing import List, Dict

import torch
import transformers

__all__ = ["Detoxify", "detoxify_labels"]


def get_model_and_tokenizer(
    model_name: str,
    tokenizer_name: str,
    num_classes: int,
    config_dir: str,
    model_max_length: int,
    state_dict,
):
    model = getattr(transformers, model_name).from_pretrained(
        pretrained_model_name_or_path=None,
        config=config_dir,
        num_labels=num_classes,
        state_dict=state_dict,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        config_dir, model_max_length=model_max_length
    )
    return model, tokenizer


def load_checkpoint(checkpoint, config_dir: str, model_max_length: int):
    if checkpoint is None:
        raise ValueError("checkpoint path must be provided")
    else:
        loaded = torch.load(checkpoint)
        if "config" not in loaded or "state_dict" not in loaded:
            raise ValueError(
                "Checkpoint needs to contain the config it was trained \
                    with as well as the state dict"
            )
    class_names = loaded["config"]["dataset"]["args"]["classes"]
    # standardise class names between models
    change_names = {
        "toxic": "toxicity",
        "identity_hate": "identity_attack",
        "severe_toxic": "severe_toxicity",
    }
    class_names = [change_names.get(cl, cl) for cl in class_names]
    model, tokenizer = get_model_and_tokenizer(
        model_name=loaded["config"]["arch"]["args"]["model_name"],
        tokenizer_name=loaded["config"]["arch"]["args"]["tokenizer_name"],
        num_classes=loaded["config"]["arch"]["args"]["num_classes"],
        config_dir=config_dir,
        model_max_length=model_max_length,
        state_dict=loaded["state_dict"],
    )
    return model, tokenizer, class_names


class Detoxify:
    """Detoxify
    Easily predict if a comment or list of comments is toxic.
    Can initialize 5 different model types from model type or checkpoint path:
        - original:
            model trained on data from the Jigsaw Toxic Comment
            Classification Challenge
        - unbiased:
            model trained on data from the Jigsaw Unintended Bias in
            Toxicity Classification Challenge
        - multilingual:
            model trained on data from the Jigsaw Multilingual
            Toxic Comment Classification Challenge
        - original-small:
            lightweight version of the original model
        - unbiased-small:
            lightweight version of the unbiased model
    Args:
        checkpoint(str): checkpoint path, defaults to None
        device(str or torch.device): accepts any torch.device input or
                                     torch.device object, defaults to cpu
    Returns:
        results(dict): dictionary of output scores for each class
    """

    def __init__(
        self, checkpoint, config_dir: str, device="cpu", model_max_length: int = 512
    ):
        super(Detoxify, self).__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            checkpoint=checkpoint,
            config_dir=config_dir,
            model_max_length=model_max_length,
        )
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.model.device)
        out = self.model(**inputs)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i]
                if isinstance(text, str)
                else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
        return results


def detoxify_labels(
    sentences: List[str],
    checkpoint: str,
    config_dir: str,
    model_max_length: int,
    device,
    batch_size: int,
) -> Dict[str, List[float]]:
    model = Detoxify(
        checkpoint=checkpoint,
        config_dir=config_dir,
        model_max_length=model_max_length,
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
