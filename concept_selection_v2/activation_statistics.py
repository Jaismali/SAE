"""
Module 3.1 - Concept Selection
Real backend, Part 1: activation-statistics functions.

These functions are deliberately separated from any model/SAE loading
code. They take plain numeric arrays and token lists as input, so they
can be fully unit-tested with synthetic data -- independent of whether
a GPU, SAELens, or a real corpus is available. real_backend.py wires
these into the actual SAELens/model pipeline.
"""

from typing import List, Sequence

import numpy as np


def compute_activation_frequency(
    feature_activations: np.ndarray, threshold: float = 0.0
) -> float:
    """Fraction of tokens where this feature fires above `threshold`.

    Args:
        feature_activations: 1D array of this feature's activation value
            at every token position in the reference corpus pass.
        threshold: activations at or below this count as "not firing".
            Default 0.0 matches JumpReLU's own sparsity convention
            (exactly zero = did not fire).

    Returns:
        A value in [0, 1]. Returns 0.0 for an empty array rather than
        raising or returning NaN, since "no tokens observed" and
        "never fires" should not be conflated -- callers should check
        array length separately if that distinction matters to them.
    """
    if feature_activations.size == 0:
        return 0.0
    firing_mask = feature_activations > threshold
    return float(np.count_nonzero(firing_mask)) / float(feature_activations.size)


def compute_mean_nonzero_magnitude(
    feature_activations: np.ndarray, threshold: float = 0.0
) -> float:
    """Mean activation magnitude, restricted to tokens where it fired.

    Returns 0.0 (not NaN) when the feature never fires, so this value
    always satisfies the contract's `mean_activation_magnitude >= 0`
    check without special-casing "never fires" downstream.
    """
    if feature_activations.size == 0:
        return 0.0
    firing_values = feature_activations[feature_activations > threshold]
    if firing_values.size == 0:
        return 0.0
    return float(np.mean(firing_values))


def get_top_activating_tokens(
    feature_activations: np.ndarray,
    tokens: Sequence[str],
    k: int = 10,
) -> List[str]:
    """Return the k tokens with the highest activation for this feature.

    Args:
        feature_activations: 1D array, same length as `tokens`.
        tokens: the token string at each corresponding position.
        k: max number of top tokens to return.

    Raises:
        ValueError: if `feature_activations` and `tokens` have mismatched
            lengths. This must fail loudly -- a silent truncation here
            would misattribute activations to the wrong tokens, which
            would corrupt every downstream interpretability signal.
    """
    if feature_activations.shape[0] != len(tokens):
        raise ValueError(
            f"feature_activations length ({feature_activations.shape[0]}) "
            f"must match tokens length ({len(tokens)})"
        )
    if feature_activations.size == 0:
        return []

    k = min(k, feature_activations.size)
    top_indices = np.argsort(feature_activations)[::-1][:k]
    return [tokens[i] for i in top_indices]


def extract_decoder_vector(decoder_weight_matrix: np.ndarray, feature_idx: int) -> List[float]:
    """Extract one feature's decoder direction as a plain float list.

    Args:
        decoder_weight_matrix: SAE decoder weights, shape
            (num_features, d_model) -- SAELens's W_dec convention.
        feature_idx: which feature's row to extract.

    Raises:
        IndexError: if feature_idx is out of range for the given matrix
            (propagates numpy's own IndexError rather than swallowing it,
            since an out-of-range feature index is a caller bug that
            should surface immediately, not produce a zero vector that
            would later fail the contract's zero-norm check silently).
    """
    return decoder_weight_matrix[feature_idx].astype(float).tolist()
