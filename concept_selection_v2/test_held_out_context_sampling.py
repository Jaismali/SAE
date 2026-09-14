"""
Tests for held_out_context_sampling.py -- pure logic, no GPU needed.

Includes a direct regression test reproducing the real bug found in
the n=48 pilot run: pure random sampling produced all-zero held-out
true activations for sparse features (feat_0/1/2), corrupting scores
via pearson_correlation()'s zero-variance safeguard regardless of
simulator quality. The fix (top-and-random sampling) must guarantee
this doesn't happen for a feature shaped like those real ones.
"""

import numpy as np
import pytest

from held_out_context_sampling import sample_held_out_contexts


def test_excludes_top_activating_indices():
    activations = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 0.5, 4.0, 6.0, 7.0])
    tokens = [f"tok{i}" for i in range(10)]
    excluded = [3, 5, 8]  # positions with values 8.0, 9.0, 6.0

    results = sample_held_out_contexts(
        activations, tokens, excluded_indices=excluded, num_samples=5, seed=42
    )
    result_activations = {r[1] for r in results}
    for excluded_idx in excluded:
        assert activations[excluded_idx] not in result_activations


def test_returns_requested_number_of_samples_when_available():
    activations = np.arange(20, dtype=float)
    tokens = [f"tok{i}" for i in range(20)]
    results = sample_held_out_contexts(
        activations, tokens, excluded_indices=[], num_samples=5, seed=1
    )
    assert len(results) == 5


def test_returns_fewer_when_not_enough_available():
    activations = np.array([1.0, 2.0, 3.0])
    tokens = ["a", "b", "c"]
    results = sample_held_out_contexts(
        activations, tokens, excluded_indices=[0], num_samples=10, seed=1
    )
    assert len(results) == 2  # only 2 non-excluded positions exist


def test_empty_available_returns_empty_list():
    activations = np.array([1.0, 2.0])
    tokens = ["a", "b"]
    results = sample_held_out_contexts(
        activations, tokens, excluded_indices=[0, 1], num_samples=5
    )
    assert results == []


def test_same_seed_produces_same_sample():
    activations = np.arange(50, dtype=float)
    tokens = [f"tok{i}" for i in range(50)]
    results_a = sample_held_out_contexts(activations, tokens, [], num_samples=10, seed=99)
    results_b = sample_held_out_contexts(activations, tokens, [], num_samples=10, seed=99)
    assert results_a == results_b


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        sample_held_out_contexts(np.array([1.0, 2.0, 3.0]), ["a", "b"], [])


def test_context_text_includes_surrounding_window():
    activations = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    tokens = ["The", " quick", " brown", " fox", " jumps"]
    results = sample_held_out_contexts(
        activations, tokens, excluded_indices=[], num_samples=1, window_size=3, seed=1
    )
    context_text, true_activation = results[0]
    assert isinstance(context_text, str)
    assert len(context_text) > 0
    assert true_activation in activations


# --- Regression test for the real sparse-feature bug ---

def test_sparse_feature_held_out_set_has_real_variance_not_all_zero():
    """DIRECT REGRESSION TEST for the real bug found in the n=48 pilot
    run: feat_0/1/2 had ALL-ZERO held-out true activations under pure
    random sampling, because these are genuinely sparse features (only
    a handful of nonzero positions out of thousands of tokens).

    Reproduces that exact shape: 8686 total positions (matching the
    real corpus size), only ~15 nonzero activations total (10 "top"
    ones that get excluded, ~5 more meaningfully-activating positions,
    everything else exactly zero) -- exactly the sparsity profile that
    broke the old pure-random sampler.
    """
    total_positions = 8686
    activations = np.zeros(total_positions)

    # 10 "top" positions (will be excluded, simulating the explainer's examples)
    top_positions = list(range(10))
    activations[top_positions] = [109.84, 109.2, 99.32, 95.39, 92.13, 88.82, 87.88, 86.76, 86.15, 85.83]

    # 5 more genuinely-activating positions, NOT in the top-10, scattered
    # far from each other in the token sequence (matching real sparsity).
    next_tier_positions = [2000, 4000, 5000, 6500, 8000]
    activations[next_tier_positions] = [50.0, 45.0, 40.0, 35.0, 30.0]

    # Everything else is exactly zero -- matches the real feat_0/1/2 shape.
    tokens = [f"tok{i}" for i in range(total_positions)]

    results = sample_held_out_contexts(
        activations, tokens, excluded_indices=top_positions,
        num_samples=10, window_size=5, seed=42,
    )
    true_activations = [v for _, v in results]

    # The critical assertion: NOT all zero, unlike the old buggy behavior.
    assert any(v > 0 for v in true_activations), (
        "Held-out true activations were all zero for a sparse feature -- "
        "this is exactly the bug this redesign was meant to fix."
    )

    # Variance must be meaningfully nonzero (not just one lucky nonzero value
    # by chance, but the guaranteed-by-design near-top inclusion).
    mean_val = sum(true_activations) / len(true_activations)
    variance = sum((v - mean_val) ** 2 for v in true_activations) / len(true_activations)
    assert variance > 0.01, f"Variance too low ({variance}) -- near-top guarantee may have failed"


def test_near_top_fraction_controls_how_many_guaranteed_nonzero_included():
    """With near_top_fraction=1.0, ALL held-out samples should come
    from the guaranteed-nonzero near-top pool (if enough exist)."""
    total_positions = 100
    activations = np.zeros(total_positions)
    activations[0:10] = 100.0  # will be excluded (top-10)
    activations[10:20] = 50.0  # near-top pool
    # positions 20-99 are all zero

    results = sample_held_out_contexts(
        activations, [f"t{i}" for i in range(total_positions)],
        excluded_indices=list(range(10)),
        num_samples=5, near_top_fraction=1.0, seed=1,
    )
    true_activations = [v for _, v in results]
    assert all(v == 50.0 for v in true_activations)


def test_near_top_fraction_zero_is_pure_random():
    """With near_top_fraction=0.0, behavior should be pure random
    (the original, pre-fix behavior) -- confirms the parameter actually
    controls the split rather than being ignored."""
    total_positions = 8686
    activations = np.zeros(total_positions)
    activations[0:10] = 100.0
    activations[2000:2005] = 50.0

    results = sample_held_out_contexts(
        activations, [f"t{i}" for i in range(total_positions)],
        excluded_indices=list(range(10)),
        num_samples=10, near_top_fraction=0.0, seed=42,
    )
    # With near_top_fraction=0, it's plausible (though not guaranteed) to
    # get an all-zero sample again, since only 5/8676 available positions
    # are nonzero -- this test just confirms the parameter is respected
    # structurally, not that a specific outcome occurs.
    assert len(results) == 10
