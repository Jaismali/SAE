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
) -> List[Tuple[str, float]]:
    """Samples held-out (context_text, true_activation) pairs for one
    feature, EXCLUDING token positions already used as top-activating
    examples (so the simulator is genuinely tested on unseen data, not
    re-shown the same positions the explainer already saw).

    Args:
        feature_activations: 1D array of this feature's activation at
            every token position (same convention as
            activation_statistics.py's functions).
        tokens: token strings at each position, same length.
        excluded_indices: positions already shown to the explainer --
            excluded from held-out sampling to avoid leakage.
        num_samples: how many held-out contexts to sample.
        window_size: how many tokens of surrounding context to join
            into a readable text snippet around each sampled position.
        seed: optional random seed for reproducible sampling -- default
            None means non-reproducible, matching this project's
            established pattern of requiring explicit opt-in for
            determinism rather than assuming it silently.

    Returns:
        List of (context_text, true_activation_value) tuples, length
        min(num_samples, available positions after exclusion).

    Raises:
        ValueError: if feature_activations and tokens have mismatched
            lengths -- must fail loudly, matching
            activation_statistics.py's existing convention for this
            exact kind of misalignment risk.
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
    sampled_indices = rng.sample(available_indices, sample_size)

    results = []
    for idx in sampled_indices:
        window_start = max(0, idx - window_size)
        window_end = min(total_positions, idx + 1)
        context_text = "".join(tokens[window_start:window_end])
        true_activation = float(feature_activations[idx])
        results.append((context_text, true_activation))

    return results
