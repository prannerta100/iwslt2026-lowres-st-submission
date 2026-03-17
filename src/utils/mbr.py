"""Minimum Bayes Risk (MBR) decoding for combining hypotheses from multiple systems.

Combines N-best lists from cascaded (Whisper+NLLB) and E2E (SeamlessM4T) systems.
Scores each hypothesis against all others using a utility metric (chrF/BLEU/COMET).
Selects the hypothesis with the highest expected utility.
"""

import sacrebleu
import numpy as np
from typing import Optional


def chrf_score(hypothesis: str, reference: str) -> float:
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score


def bleu_score(hypothesis: str, reference: str) -> float:
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score


SCORING_FNS = {
    "chrf": chrf_score,
    "bleu": bleu_score,
}


def mbr_decode(
    hypotheses: list[str],
    scoring_metric: str = "chrf",
    weights: Optional[list[float]] = None,
) -> tuple[str, int, float]:
    """Select the hypothesis with highest expected utility under MBR.

    Args:
        hypotheses: Combined N-best list from all systems
        scoring_metric: Utility function ("chrf" or "bleu")
        weights: Optional per-hypothesis prior weights (e.g., from model scores)

    Returns:
        (best_hypothesis, best_index, best_score)
    """
    if not hypotheses:
        return "", 0, 0.0

    # Deduplicate while preserving order
    seen = set()
    unique = []
    orig_indices = []
    for i, h in enumerate(hypotheses):
        h_stripped = h.strip()
        if h_stripped not in seen:
            seen.add(h_stripped)
            unique.append(h_stripped)
            orig_indices.append(i)

    if len(unique) == 1:
        return unique[0], orig_indices[0], 100.0

    score_fn = SCORING_FNS[scoring_metric]
    n = len(unique)

    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights[:n])
        weights = weights / weights.sum()

    # Compute pairwise utility matrix
    utility_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                utility_matrix[i][j] = score_fn(unique[i], unique[j])

    # Expected utility for each hypothesis
    expected_utility = utility_matrix @ weights

    best_idx = int(np.argmax(expected_utility))
    return unique[best_idx], orig_indices[best_idx], float(expected_utility[best_idx])


def mbr_decode_batch(
    all_hypotheses: list[list[str]],
    scoring_metric: str = "chrf",
) -> list[str]:
    """MBR decode a batch of N-best lists (one per source utterance)."""
    results = []
    for hyps in all_hypotheses:
        best, _, _ = mbr_decode(hyps, scoring_metric)
        results.append(best)
    return results
