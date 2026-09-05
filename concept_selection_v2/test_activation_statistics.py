"""
Tests for activation_statistics.py.

All tests use synthetic numpy arrays -- no real model, SAE, or GPU
required. This is the part of the real-backend pipeline that CAN be
fully verified without hardware access.
"""

import numpy as np
import pytest

from activation_statistics import (
    compute_activation_frequency,
    compute_mean_nonzero_magnitude,
    extract_decoder_vector,
    get_top_activating_tokens,
)


def test_activation_frequency_all_firing():
    activations = np.array([1.0, 2.0, 0.5, 3.0])
    assert compute_activation_frequency(activations) == 1.0


def test_activation_frequency_none_firing():
    activations = np.array([0.0, 0.0, 0.0])
    assert compute_activation_frequency(activations) == 0.0


def test_activation_frequency_partial():
    activations = np.array([1.0, 0.0, 2.0, 0.0])
    assert compute_activation_frequency(activations) == pytest.approx(0.5)


def test_activation_frequency_empty_array_returns_zero_not_nan():
    activations = np.array([])
    result = compute_activation_frequency(activations)
    assert result == 0.0
    assert not np.isnan(result)


def test_mean_nonzero_magnitude_ignores_non_firing_tokens():
    activations = np.array([0.0, 0.0, 4.0, 6.0])
    # mean should be over {4.0, 6.0} only, i.e. 5.0 -- not over all 4 values
    assert compute_mean_nonzero_magnitude(activations) == pytest.approx(5.0)


def test_mean_nonzero_magnitude_never_fires_returns_zero_not_nan():
    activations = np.array([0.0, 0.0, 0.0])
    result = compute_mean_nonzero_magnitude(activations)
    assert result == 0.0
    assert not np.isnan(result)


def test_top_activating_tokens_returns_correct_order():
    activations = np.array([0.1, 5.0, 2.0, 9.0, 0.0])
    tokens = ["a", "b", "c", "d", "e"]
    result = get_top_activating_tokens(activations, tokens, k=3)
    assert result == ["d", "b", "c"]  # 9.0, 5.0, 2.0 in descending order


def test_top_activating_tokens_k_larger_than_available_does_not_crash():
    activations = np.array([1.0, 2.0])
    tokens = ["x", "y"]
    result = get_top_activating_tokens(activations, tokens, k=10)
    assert result == ["y", "x"]


def test_top_activating_tokens_mismatched_lengths_raises():
    activations = np.array([1.0, 2.0, 3.0])
    tokens = ["x", "y"]  # deliberately mismatched
    with pytest.raises(ValueError):
        get_top_activating_tokens(activations, tokens, k=2)


def test_extract_decoder_vector_returns_correct_row_as_plain_list():
    decoder_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = extract_decoder_vector(decoder_matrix, feature_idx=1)
    assert result == [3.0, 4.0]
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


def test_extract_decoder_vector_out_of_range_raises():
    decoder_matrix = np.array([[1.0, 2.0]])
    with pytest.raises(IndexError):
        extract_decoder_vector(decoder_matrix, feature_idx=5)
