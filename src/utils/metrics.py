"""Evaluation metrics for IWSLT: BLEU, chrF++, COMET."""

import re
import string
from typing import Optional

import sacrebleu
from comet import download_model, load_from_checkpoint


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation per IWSLT eval protocol."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_bleu(hypotheses: list[str], references: list[str], normalize: bool = True) -> dict:
    if normalize:
        hypotheses = [normalize_text(h) for h in hypotheses]
        references = [normalize_text(r) for r in references]
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return {"bleu": result.score, "bleu_signature": str(result)}


def compute_chrf(hypotheses: list[str], references: list[str], normalize: bool = True) -> dict:
    if normalize:
        hypotheses = [normalize_text(h) for h in hypotheses]
        references = [normalize_text(r) for r in references]
    result = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return {"chrf++": result.score}


def compute_comet(
    hypotheses: list[str],
    references: list[str],
    sources: Optional[list[str]] = None,
    model_path: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 32,
    gpus: int = 1,
) -> dict:
    model_path_local = download_model(model_path)
    model = load_from_checkpoint(model_path_local)

    if sources is None:
        sources = [""] * len(hypotheses)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {"comet": output.system_score, "comet_scores": output.scores}


def evaluate_all(
    hypotheses: list[str],
    references: list[str],
    sources: Optional[list[str]] = None,
    compute_comet_score: bool = True,
) -> dict:
    results = {}
    results.update(compute_bleu(hypotheses, references))
    results.update(compute_chrf(hypotheses, references))
    if compute_comet_score:
        comet_result = compute_comet(hypotheses, references, sources)
        results["comet"] = comet_result["comet"]
    return results
