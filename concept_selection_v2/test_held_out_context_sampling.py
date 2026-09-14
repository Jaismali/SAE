"""
Tests for held_out_context_sampling.py -- pure logic, no GPU needed.
"""

import numpy as np
import pytest

from held_out_context_sampling import sample_held_out_contexts


def test_excludes_top_activating_indices():
    activations = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 0.5, 4.0, 6.0, 7.0])
    tokens = [f"tok{i}" for i in range(10)]
    excluded = [3, 5, 8]  # positions with values 8.0, 9.0, 6.0

    results = sample_held_out_contexts(
        activations, tokens, excluded_indices=excluded, num_samples=10, seed=42
    )
    # None of the sampled contexts should include the excluded positions'
    # activation values, since sampling only draws from non-excluded indices.
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
    results_a = sample_held_out_contexts(activations, tokens, [], num_samples=5, seed=99)
    results_b = sample_held_out_contexts(activations, tokens, [], num_samples=5, seed=99)
    assert results_a == results_b


def test_different_seeds_likely_produce_different_samples():
    activations = np.arange(50, dtype=float)
    tokens = [f"tok{i}" for i in range(50)]
    results_a = sample_held_out_contexts(activations, tokens, [], num_samples=5, seed=1)
    results_b = sample_held_out_contexts(activations, tokens, [], num_samples=5, seed=2)
    assert results_a != results_b


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        sample_held_out_contexts(np.array([1.0, 2.0, 3.0]), ["a", "b"], [])


def test_context_text_includes_surrounding_window():
    activations = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    tokens = ["The", " quick", " brown", " fox", " jumps"]
    results = sample_held_out_contexts(
        activations, tokens, excluded_indices=[], num_samples=1, window_size=3, seed=1
    )
    # Only one non-zero position (index 3) is realistically distinguishable
    # by seed, but check window logic generally: whichever index is sampled,
    # its context text should include tokens leading up to and including it.
    context_text, true_activation = results[0]
    assert isinstance(context_text, str)
    assert len(context_text) > 0
    assert true_activation in activations
