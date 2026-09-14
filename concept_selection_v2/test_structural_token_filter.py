"""
Tests for structural_token_filter.py.

Three categories:
  1. The actual 5 structural-dominated candidates from the real n=20
     diagnostic run -- confirms the two real magnitude outliers get
     excluded, feat_6/feat_8 (below the 90% unconditional threshold)
     remain kept, and feat_3 (90% fraction) is now CORRECTLY excluded
     -- see the CORRECTION note below.
  2. Synthetic boundary cases at exactly the threshold values, to
     confirm the ">" / ">=" comparison operators are what's intended.
  3. The NEW unconditional high-fraction rule, tested against the
     exact real evidence (n=125 manifest run) that motivated it.

CORRECTION, logged explicitly rather than silently changed: the
original version of this file asserted feat_3 (90% structural fraction,
0.44x median magnitude, from the n=20 pilot) must NOT be excluded,
treating it as the motivating case for requiring the magnitude+fraction
conjunction. That assumption was NEVER actually verified -- nobody
inspected feat_3's real top-activating tokens to confirm it was a
legitimate concept rather than another structural artifact; the only
fact established was that it wasn't a MAGNITUDE outlier. At real n=125
scale, 16/16 candidates at >=90% structural fraction were confirmed,
by direct token inspection, to be structural/punctuation/whitespace
artifacts with ZERO exceptions -- regardless of magnitude. Given that
unanimous, diverse-token evidence against one never-verified n=20 data
point, the researcher's explicit call (Option A) was that feat_3 was
very likely the same phenomenon, simply never caught. This file's
expectations are corrected accordingly, not silently -- see
structural_token_filter.py's own decision record for the full evidence.
"""

import pytest

from contract import ConceptCandidate
from structural_token_filter import (
    MAGNITUDE_RATIO_THRESHOLD,
    STRUCTURAL_TOKEN_FRACTION_THRESHOLD,
    HIGH_FRACTION_UNCONDITIONAL_THRESHOLD,
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
    100% structural fraction) must be excluded by this filter -- via
    EITHER condition now, but definitely excluded."""
    feat_4 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_4"]
    feat_11 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_11"]

    assert passes_structural_token_filter(feat_4, REAL_BATCH_MEDIAN) is False
    assert passes_structural_token_filter(feat_11, REAL_BATCH_MEDIAN) is False


def test_real_feat_3_is_now_correctly_excluded_via_high_fraction_rule():
    """CORRECTED per Option A (see module docstring): feat_3 (90%
    structural fraction, only 0.44x median magnitude -- NOT a magnitude
    outlier) is now EXCLUDED by the new unconditional high-fraction
    rule, even though it survives the original conjunction. This
    reverses the original test's assertion, which encoded a
    never-verified assumption that feat_3 was a legitimate concept."""
    feat_3 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_3"]
    assert passes_structural_token_filter(feat_3, REAL_BATCH_MEDIAN) is False


def test_real_below_unconditional_threshold_non_outliers_are_not_excluded():
    """feat_6, feat_8 (60% structural fraction -- above the 50%
    conjunction threshold but BELOW the 90% unconditional threshold --
    and NOT magnitude outliers) must NOT be excluded. These are the
    real cases the conjunction design is still correctly protecting."""
    feat_6 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_6"]
    feat_8 = REAL_STRUCTURAL_DOMINATED_CANDIDATES["feat_8"]

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
    """Exactly 5.0x median magnitude must NOT trigger exclusion via the
    conjunction -- the threshold is strict '>', not '>='. Uses 60%
    fraction (above the 50% conjunction threshold, but BELOW the new
    90% unconditional threshold) to isolate this test to the MAGNITUDE
    boundary specifically -- using 90% here would confound this test
    with the new unconditional rule, which doesn't care about magnitude
    at all."""
    median = 100.0
    candidate = _make_candidate("boundary_5x", magnitude=500.0, structural_fraction=0.60)
    assert candidate.mean_activation_magnitude / median == pytest.approx(5.0)
    assert passes_structural_token_filter(candidate, median) is True


def test_boundary_just_above_5x_magnitude_with_high_fraction_is_excluded():
    """Just above the 5x threshold, combined with >50% (but <90%)
    structural fraction, must be excluded via the conjunction
    specifically."""
    median = 100.0
    candidate = _make_candidate("boundary_5x_plus", magnitude=500.01, structural_fraction=0.60)
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


def test_structural_dominance_below_unconditional_threshold_without_magnitude_outlier_is_not_excluded():
    """Structural dominance BELOW the 90% unconditional threshold (60%
    here), without a magnitude outlier, should NOT be excluded -- this
    is what the conjunction design still correctly protects. (This
    replaces the original test, which used 90% fraction and asserted
    `is True` -- that assertion is now known to be wrong, per the
    module-level CORRECTION note; 90% is exactly the new exclusion
    threshold, not a safe case to assert as "kept.")"""
    median = 100.0
    candidate = _make_candidate("mid_structural_normal_mag", magnitude=110.0, structural_fraction=0.60)
    assert passes_structural_token_filter(candidate, median) is True


# --- Category 3: the NEW unconditional high-fraction rule ---

def test_boundary_exactly_90_percent_fraction_is_excluded_regardless_of_magnitude():
    """Exactly 90% structural fraction must be excluded via the NEW
    unconditional rule, even at completely normal magnitude -- the
    threshold here is '>=', not '>', specifically because feat_77 (one
    of the 16 real confirmed artifacts) sits at EXACTLY 90%."""
    median = 100.0
    candidate = _make_candidate("boundary_90pct_normal_mag", magnitude=110.0, structural_fraction=0.90)
    assert passes_structural_token_filter(candidate, median) is False


def test_boundary_just_below_90_percent_fraction_with_normal_magnitude_is_not_excluded():
    """Just below 90% fraction (80%, since this file's helper builds
    exact-count token lists), with normal magnitude, must NOT be
    excluded by the unconditional rule -- confirms the '>=' boundary
    doesn't creep below its intended cutoff."""
    median = 100.0
    candidate = _make_candidate("boundary_just_below_90pct", magnitude=110.0, structural_fraction=0.80)
    assert passes_structural_token_filter(candidate, median) is True


def test_high_fraction_excludes_even_with_normal_magnitude():
    """DIRECT test of the real n=125 finding: 100% structural fraction
    (matching the real feat_26/34/39/etc. <bos> cluster's exact shape)
    at completely normal magnitude (no relation to the batch median at
    all) must still be excluded."""
    median = 100.0
    candidate = _make_candidate("pure_bos_normal_mag", magnitude=95.0, structural_fraction=1.0)
    assert passes_structural_token_filter(candidate, median) is False


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
    accept/reject split as the per-candidate tests above -- UPDATED per
    the feat_3 correction: feat_3 is now correctly among the rejected,
    not the accepted."""
    filler = [
        _make_candidate(f"filler_{i}", magnitude=96.28, structural_fraction=0.0)
        for i in range(15)
    ]
    batch = filler + list(REAL_STRUCTURAL_DOMINATED_CANDIDATES.values())

    result = apply_structural_token_filter(batch)

    assert isinstance(result, FilterResult)
    rejected_ids = {c.feature_id for c in result.rejected}
    accepted_ids = {c.feature_id for c in result.accepted}

    assert rejected_ids == {"feat_3", "feat_4", "feat_11"}
    assert "feat_6" in accepted_ids
    assert "feat_8" in accepted_ids
    assert len(result.accepted) + len(result.rejected) == len(batch)

    # Every rejected candidate must have a logged reason -- never a
    # silent exclusion.
    for feature_id in rejected_ids:
        assert feature_id in result.rejection_reasons
        assert "structural_token_filter" in result.rejection_reasons[feature_id]
