"""
Module 3.1 - Concept Selection
Held-out context sampling for real auto-interp scoring.

A genuine Bills et al.-style auto-interp score needs HELD-OUT contexts
with REAL, MEASURED activation values for the simulator's predictions
to be checked against -- not just a feature's top-10 activating
tokens, which is all real_backend.py's ConceptCandidate currently
carries. This module closes that gap by reusing the SAME full
per-token activation matrix real_backend.py's
_collect_activations_for_all_features() already computes, rather than
requiring a second, separate activation-extraction pass.

Pure, testable logic only -- no model/GPU dependency. Takes a raw
numpy activation column and token list (already computed elsewhere)
and returns held-out (context_text, true_activation) pairs.
"""

import random
from typing import List, Tuple

import numpy as np


def sample_held_out_contexts(
    feature_activations: np.ndarray,
    tokens: List[str],
    excluded_indices: List[int],
    num_samples: int = 10,
    window_size: int = 5,
    seed: int = None,
    near_top_fraction: float = 0.5,
) -> List[Tuple[str, float]]:
    """Samples held-out (context_text, true_activation) pairs for one
    feature, EXCLUDING token positions already used as top-activating
    examples (so the simulator is genuinely tested on unseen data, not
    re-shown the same positions the explainer already saw).

    "TOP-AND-RANDOM" SAMPLING, per Bills et al.'s standard method --
    NOT pure random. Confirmed necessary by a real pilot run: pure
    random sampling (an earlier version of this function) produced
    ALL-ZERO held-out true activations for genuinely sparse features
    (feat_0/1/2 in a real n=48 pilot batch, out of 8,686 total tokens),
    since a sparse feature's non-top-10 positions are overwhelmingly
    non-activating. All-zero true activations have zero variance,
    which forces pearson_correlation()'s own safeguard to return 0.0
    regardless of simulator quality -- corrupting the score for a
    reason that has nothing to do with the judge's actual competence.

    Fix: guarantee a fraction of the held-out set comes from the
    NEXT-highest-activating positions (immediately after the excluded
    top-K), so real activation variance is present regardless of
    the feature's overall sparsity. The remaining fraction is
    genuinely random, to also test specificity (does the simulator
    correctly predict LOW on truly unrelated text).

    Args:
        feature_activations: 1D array of this feature's activation at
            every token position.
        tokens: token strings at each position, same length.
        excluded_indices: positions already shown to the explainer.
        num_samples: how many held-out contexts to sample total.
        window_size: how many tokens of surrounding context to join
            into a readable text snippet around each sampled position.
        seed: optional random seed for reproducible sampling.
        near_top_fraction: fraction of num_samples drawn from the
            next-highest-activating available positions (guarantees
            real variance); the remainder is genuinely random.

    Returns:
        List of (context_text, true_activation_value) tuples, order
        shuffled so presentation order doesn't trivially reveal which
        were "near-top" vs "random" -- length up to num_samples,
        fewer if not enough positions are available after exclusion.

    Raises:
        ValueError: if feature_activations and tokens have mismatched
            lengths.
    """
    if feature_activations.shape[0] != len(tokens):
        raise ValueError(
            f"feature_activations length ({feature_activations.shape[0]}) "
            f"must match tokens length ({len(tokens)})"
        )

    total_positions = len(tokens)
    excluded_set = set(excluded_indices)
    available_indices = [i for i in range(total_positions) if i not in excluded_set]

    if not available_indices:
        return []

    rng = random.Random(seed)
    sample_size = min(num_samples, len(available_indices))
    near_top_count = round(sample_size * near_top_fraction)
    random_count = sample_size - near_top_count

    # Near-top: the highest-activating positions among what's available
    # (i.e. immediately below the excluded top-K) -- deterministic
    # selection, since the whole point is guaranteeing real activation
    # signal is present, not leaving it to chance.
    available_sorted_by_activation = sorted(
        available_indices, key=lambda idx: feature_activations[idx], reverse=True
    )
    near_top_indices = available_sorted_by_activation[:near_top_count]

    # Random: genuinely random draw from whatever's left (excluding
    # both the originally-excluded top-K AND the near-top set just used).
    remaining_pool = [i for i in available_indices if i not in set(near_top_indices)]
    random_indices = rng.sample(remaining_pool, min(random_count, len(remaining_pool)))

    selected_indices = near_top_indices + random_indices
    rng.shuffle(selected_indices)  # avoid trivially-sorted presentation order

    results = []
    for idx in selected_indices:
        window_start = max(0, idx - window_size)
        window_end = min(total_positions, idx + 1)
        context_text = "".join(tokens[window_start:window_end])
        true_activation = float(feature_activations[idx])
        results.append((context_text, true_activation))

    return results


def has_sufficient_variance(true_activations: List[float], min_variance: float = 1e-6) -> bool:
    """Checks whether a held-out set's TRUE activation values have
    enough variance to make a correlation score meaningful at all.

    Added after a real pilot run showed 10/48 candidates still
    producing all-zero true activations even after the top-and-random
    sampling fix -- these are features where the reference corpus
    simply doesn't contain enough of the feature's activating examples
    (a corpus-size limitation, the same root issue as Diagnostic 8's
    non-convergence finding, resurfacing here). Calling the LLM judge
    on these wastes compute AND produces a misleading 0.000 score that
    looks like "no correlation" when it actually means "no data to
    correlate against." This check lets a caller skip the LLM call
    entirely and categorize these separately, rather than silently
    scoring them as if they were real judgments.
    """
    if len(true_activations) < 2:
        return False
    mean_val = sum(true_activations) / len(true_activations)
    variance = sum((v - mean_val) ** 2 for v in true_activations) / len(true_activations)
    return variance > min_variance
