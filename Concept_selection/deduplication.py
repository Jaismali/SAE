"""
Module 3.1 - Concept Selection
Step 5: Deduplication.

Two candidates whose decoder vectors point in nearly the same direction
represent (functionally) the same underlying concept, even if they were
extracted as separate SAE features. This module detects those pairs via
cosine similarity and keeps exactly one representative per duplicate
cluster, discarding the rest -- with the discard decision fully traceable.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

from contract import ConceptCandidate

# Two decoder vectors at or above this cosine similarity are treated as
# duplicates. Chosen to sit below the synthetic near-duplicate pairs'
# observed similarity (>0.999, see test_synthetic_backend.py) and well
# above the similarity observed between independently-generated,
# genuinely distinct synthetic candidates (which stayed well under 0.95
# across 780 sampled pairs). Like the monosemanticity threshold, this is
# a placeholder tuned to synthetic data and must be re-validated against
# real decoder-vector geometry once real SAE features are available.
DUPLICATE_COSINE_SIMILARITY_THRESHOLD = 0.95


@dataclass
class DeduplicationResult:
    kept: List[ConceptCandidate]
    discarded: List[ConceptCandidate]
    discard_reasons: dict  # feature_id -> reason string (which kept feature_id it duplicated)


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _is_duplicate_of_any_kept(
    candidate: ConceptCandidate,
    kept_so_far: List[ConceptCandidate],
    threshold: float,
) -> Tuple[bool, str]:
    """Checks candidate against every already-kept representative.
    Returns (is_duplicate, duplicated_feature_id_or_empty)."""
    for kept_candidate in kept_so_far:
        similarity = cosine_similarity(candidate.decoder_vector, kept_candidate.decoder_vector)
        if similarity >= threshold:
            return True, kept_candidate.feature_id
    return False, ""


def deduplicate_by_decoder_vector(
    candidates: List[ConceptCandidate],
    threshold: float = DUPLICATE_COSINE_SIMILARITY_THRESHOLD,
) -> DeduplicationResult:
    """Greedy, order-preserving deduplication: walks candidates in input
    order, keeping the first occurrence of each duplicate cluster and
    discarding later ones that are too similar to something already kept.
    """
    kept: List[ConceptCandidate] = []
    discarded: List[ConceptCandidate] = []
    discard_reasons = {}

    for candidate in candidates:
        is_duplicate, duplicated_of = _is_duplicate_of_any_kept(candidate, kept, threshold)
        if is_duplicate:
            discarded.append(candidate)
            discard_reasons[candidate.feature_id] = (
                f"cosine similarity to kept feature '{duplicated_of}' "
                f"exceeds threshold {threshold:.3f}"
            )
        else:
            kept.append(candidate)

    return DeduplicationResult(kept=kept, discarded=discarded, discard_reasons=discard_reasons)
