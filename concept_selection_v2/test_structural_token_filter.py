"""
Tests for structural_token_filter.py.

Two categories, per researcher's explicit request:
  1. The actual 5 structural-dominated candidates from the real n=20
     diagnostic run -- confirms the two real magnitude outliers get
     excluded and the three non-outliers (including feat_3, the
     high-fraction/low-magnitude case that motivated the conjunction
     design) do not.
  2. Synthetic boundary cases at exactly the threshold values, to
     confirm the ">" vs ">=" comparison operators are what's intended
     -- boundary behavior is exactly the kind of thing that's easy to
     get backwards without a dedicated test catching it.
"""

import pytest

from contract import ConceptCandidate
from structural_token_filter import (
    MAGNITUDE_RATIO_THRESHOLD,
    STRUCTURAL_TOKEN_FRACTION_THRESHOLD,
    FilterResult,
    apply_structural_token_filter,
    compute_batch_median_magnitude,
    compute_structural_fraction,
    passes_structural_token_filter,
)


def _make_candidate(
    feature_id: str,
    magnitude: float,
    structural_fraction: float,
    activation_frequency: float = 0.05,
) -> ConceptCandidate:
    """Build a contract-valid ConceptCandidate with an exact, controlled
    structural-token fraction in its top_activating_tokens (10 tokens
    total, so fraction * 10 must be a whole number for exactness)."""
    num_tokens = 10
    num_structural = round(structural_fraction * num_tokens)
    tokens = ["<bos>"] * num_structural + [f"word{i}" for i in range(num_tokens - num_structural)]

    return ConceptCandidate(
        feature_id=feature_id,
        layer=13,
        activation_frequency=activation_frequency,
        mean_activation_magnitude=magnitude,
        top_activating_tokens=tokens,
        auto_interp_score=0.5,
        auto_interp_text="[test] placeholder interp text",
        decoder_vector=[1.0, 0.0, 0.0, 0.0],
        source="test",
        model_id="gemma-3-1b-it",
        sae_id="gemma-scope-2-1b-it-res",
    )


# --- Category 1: real diagnostic data (n=20 pilot run) ---

REAL_BATCH_MEDIAN = 96.28  # from the actual n=20 diagnostic run

REAL_STRUCTURAL_DOMINATED_CANDIDATES = {
    "feat_3": _make_candidate("feat_3", magnitude=42.58, structural_fraction=0.90),
    "feat_6": _make_candidate("feat_6", magnitude=67.39, structural_fraction=0.60),
    "feat_8": _make_candidate("feat_8", magnitude=152.19, structural_fraction=0.60),
    "feat_11": _make_candidate("feat_11", magnitude=1526.09, structural_fraction=1.00),
    "feat_4": _make_candidate("feat_4", magnitude=1831.53, structural_fraction=1.00),
}


def test_real_magnitude_outliers_are_excluded():
    """feat_4 and feat_11 (the real >10x magnitude outliers, both at
    100% structural fraction) must be excluded by this filter."""
    feat_4 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_4"]
    feat_11 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_11"]

    assert passes_structural_token_filter(feat_4, REAL_BATCH_MEDIAN) is False
    assert passes_structural_token_filter(feat_11, REAL_BATCH_MEDIAN) is False


def test_real_non_outliers_are_not_excluded():
    """feat_3, feat_6, feat_8 (real structural-dominated but NOT
    magnitude outliers) must NOT be excluded. feat_3 is the critical
    case: 90% structural fraction but only 0.44x median magnitude --
    this is exactly the case that motivated requiring BOTH conditions
    rather than fraction alone."""
    feat_3 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_3"]
    feat_6 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_6"]
    feat_8 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_8"]

    assert passes_structural_token_filter(feat_3, REAL_BATCH_MEDIAN) is True
    assert passes_structural_token_filter(feat_6, REAL_BATCH_MEDIAN) is True
    assert passes_structural_token_filter(feat_8, REAL_BATCH_MEDIAN) is True


def test_real_data_ratios_match_diagnostic_output():
    """Sanity-check that the magnitude ratios computed here actually
    match what the diagnostic script reported, so this test is
    verifying the real scenario and not a drifted approximation of it."""
    expected_ratios = {
        "feat_3": pytest.approx(0.44, abs=0.01),
        "feat_6": pytest.approx(0.70, abs=0.01),
        "feat_8": pytest.approx(1.58, abs=0.01),
        "feat_11": pytest.approx(15.85, abs=0.01),
        "feat_4": pytest.approx(19.02, abs=0.01),
    }
    for feature_id, candidate in REAL_STRUCTURAL_DOMINATED_CANDIDATES.items():
        ratio = candidate.mean_activation_magnitude / REAL_BATCH_MEDIAN
        assert ratio == expected_ratios[feature_id]


# --- Category 2: boundary / edge cases ---

def test_boundary_exactly_5x_magnitude_is_not_excluded():
    """Exactly 5.0x median magnitude must NOT trigger exclusion --
    the threshold is strict '>', not '>='. A candidate sitting exactly
    at 5x with high structural fraction should still pass."""
    median = 100.0
    candidate = _make_candidate("boundary_5x", magnitude=500.0, structural_fraction=0.90)
    assert candidate.mean_activation_magnitude / median == pytest.approx(5.0)
    assert passes_structural_token_filter(candidate, median) is True


def test_boundary_just_above_5x_magnitude_with_high_fraction_is_excluded():
    """Just above the 5x threshold, combined with high structural
    fraction, must be excluded."""
    median = 100.0
    candidate = _make_candidate("boundary_5x_plus", magnitude=500.01, structural_fraction=0.90)
    assert passes_structural_token_filter(candidate, median) is False


def test_boundary_exactly_50_percent_structural_fraction_is_not_dominated():
    """Exactly 50% structural fraction must NOT count as 'dominated' --
    the threshold is strict '>', not '>='. Even with extreme magnitude,
    a candidate at exactly 50% should pass."""
    median = 100.0
    candidate = _make_candidate("boundary_50pct", magnitude=10000.0, structural_fraction=0.50)
    assert compute_structural_fraction(candidate.top_activating_tokens) == pytest.approx(0.50)
    assert passes_structural_token_filter(candidate, median) is True


def test_boundary_just_above_50_percent_with_high_magnitude_is_excluded():
    """Just above 50% structural fraction, combined with a clear
    magnitude outlier, must be excluded."""
    median = 100.0
    candidate = _make_candidate("boundary_50pct_plus", magnitude=10000.0, structural_fraction=0.60)
    assert passes_structural_token_filter(candidate, median) is False


def test_high_magnitude_alone_without_structural_dominance_is_not_excluded():
    """A magnitude outlier that is NOT structural-dominated should NOT
    be excluded by this filter -- the conjunction requires both."""
    median = 100.0
    candidate = _make_candidate("high_mag_low_structural", magnitude=2000.0, structural_fraction=0.30)
    assert passes_structural_token_filter(candidate, median) is True


def test_structural_dominance_alone_without_magnitude_outlier_is_not_excluded():
    """Structural dominance without a magnitude outlier (feat_3's
    real-world pattern) should NOT be excluded -- this is the core
    justification for the conjunction over a fraction-only rule."""
    median = 100.0
    candidate = _make_candidate("high_structural_normal_mag", magnitude=110.0, structural_fraction=0.90)
    assert passes_structural_token_filter(candidate, median) is True


# --- Batch-level tests ---

def test_compute_batch_median_magnitude_ignores_zero_magnitude_candidates():
    candidates = [
        _make_candidate("a", magnitude=0.0, structural_fraction=0.0),
        _make_candidate("b", magnitude=100.0, structural_fraction=0.0),
        _make_candidate("c", magnitude=200.0, structural_fraction=0.0),
    ]
    # median of [100, 200] = 150, NOT median of [0, 100, 200] = 100
    assert compute_batch_median_magnitude(candidates) == pytest.approx(150.0)


def test_compute_batch_median_magnitude_all_zero_raises():
    candidates = [
        _make_candidate("a", magnitude=0.0, structural_fraction=0.0),
        _make_candidate("b", magnitude=0.0, structural_fraction=0.0),
    ]
    with pytest.raises(ValueError):
        compute_batch_median_magnitude(candidates)


def test_apply_filter_end_to_end_matches_real_diagnostic_outcome():
    """Full apply_structural_token_filter() call across a batch shaped like the real
    n=20 run (5 structural-dominated + 15 'normal' filler candidates
    near the median), confirming the bulk interface produces the same
    accept/reject split as the per-candidate tests above."""
    filler = [
        _make_candidate(f"filler_{i}", magnitude=96.28, structural_fraction=0.0)
        for i in range(15)
    ]
    batch = filler + list(REAL_STRUCTURAL_DOMINATED_CANDIDATES.values())

    result = apply_structural_token_filter(batch)

    assert isinstance(result, FilterResult)
    rejected_ids = {c.feature_id for c in result.rejected}
    accepted_ids = {c.feature_id for c in result.accepted}

    assert rejected_ids == {"feat_4", "feat_11"}
    assert "feat_3" in accepted_ids
    assert "feat_6" in accepted_ids
    assert "feat_8" in accepted_ids
    assert len(result.accepted) + len(result.rejected) == len(batch)

    # Every rejected candidate must have a logged reason -- never a
    # silent exclusion.
    for feature_id in rejected_ids:
        assert feature_id in result.rejection_reasons
        assert "structural_token_filter" in result.rejection_reasons[feature_id]
