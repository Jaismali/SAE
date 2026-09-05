"""Tests for layer_selection.py.

Per explicit researcher requirement: the deviation record must prove
that every genuinely in-band candidate (14, 15) was checked and
confirmed absent from the real release -- not just the single
theoretical target layer (14).

Layer 16 (61.5% relative depth) is NOT in-band under the locked
50-60% definition, despite an earlier session error that treated it
as in-band by conflating the precise band with Pack 1's looser "13-16"
illustrative phrasing. It is tracked separately as a historical
footnote, not as part of the in-band claim.
"""

import pytest

from layer_selection import (
    AVAILABLE_LAYERS_IN_RELEASE,
    MODEL_TOTAL_LAYERS,
    RELATIVE_DEPTH_BAND,
    THEORETICAL_IN_BAND_LAYERS,
    build_deviation_record,
    compute_nearest_layer_in_band,
    find_available_layers_in_band,
    relative_depth,
)


def test_theoretical_nearest_layer_is_14():
    """50-60% band midpoint (55%) of 26 layers rounds to layer 14."""
    assert compute_nearest_layer_in_band(MODEL_TOTAL_LAYERS) == 14


def test_unsupported_rounding_rule_raises():
    """Only 'nearest' is implemented; other rules must fail loudly."""
    with pytest.raises(ValueError):
        compute_nearest_layer_in_band(MODEL_TOTAL_LAYERS, rounding="floor")


def test_layer_13_is_in_band():
    """Layer 13 sits at exactly 50.0% relative depth -- the lower edge
    of, but still within, the locked 50-60% band."""
    depth = relative_depth(13, MODEL_TOTAL_LAYERS)
    low, high = RELATIVE_DEPTH_BAND
    assert low <= depth <= high
    assert depth == pytest.approx(0.5)


def test_layers_14_and_15_are_in_band_but_absent():
    """The two non-chosen in-band candidates must each be confirmed
    absent from the release, not merely untried."""
    in_band = find_available_layers_in_band(
        list(range(MODEL_TOTAL_LAYERS)), MODEL_TOTAL_LAYERS
    )
    for missing_layer in (14, 15):
        assert missing_layer in in_band, (
            f"layer {missing_layer} should be in the theoretical band"
        )
        assert missing_layer not in AVAILABLE_LAYERS_IN_RELEASE, (
            f"layer {missing_layer} should be absent from the release"
        )


def test_layer_16_is_not_in_band():
    """Layer 16 sits at 61.5% relative depth -- outside the locked
    50-60% band -- despite an earlier session error that treated it as
    in-band. It must not appear in the strict in-band set."""
    in_band = find_available_layers_in_band(
        list(range(MODEL_TOTAL_LAYERS)), MODEL_TOTAL_LAYERS
    )
    assert 16 not in in_band
    assert 16 not in THEORETICAL_IN_BAND_LAYERS
    assert THEORETICAL_IN_BAND_LAYERS == [13, 14, 15]


def test_only_layer_13_is_both_in_band_and_available():
    """Layer 13 must be the sole intersection of 'in band' and
    'available in release' -- the reason it was chosen."""
    in_band = find_available_layers_in_band(
        list(range(MODEL_TOTAL_LAYERS)), MODEL_TOTAL_LAYERS
    )
    available_in_band = [
        layer for layer in in_band if layer in AVAILABLE_LAYERS_IN_RELEASE
    ]
    assert available_in_band == [13]


def test_deviation_record_confirms_in_band_absent_layers():
    """The deviation record itself -- what actually gets logged -- must
    list both genuinely in-band absent candidates (14, 15), not just the
    theoretical target (14) alone, and must NOT claim 16 as in-band."""
    record = build_deviation_record()
    for missing_layer in (14, 15):
        assert missing_layer in record.in_band_candidates_confirmed_absent
    assert 16 not in record.in_band_candidates_confirmed_absent
    assert 16 not in record.in_band_candidates_checked
    assert record.theoretical_target_layer == 14
    assert record.actual_target_layer == 13


def test_deviation_record_keeps_layer_16_as_separate_footnote():
    """Layer 16 was genuinely queried and found absent in an earlier
    session, and that fact is worth keeping -- but only as a clearly
    separate footnote, never folded into the in-band claim."""
    record = build_deviation_record()
    assert record.layer_16_checked_but_out_of_band_note != ""
    assert "16" in record.layer_16_checked_but_out_of_band_note
    assert "out of band" not in record.in_band_candidates_confirmed_absent


def test_deviation_record_log_string_is_non_empty_and_mentions_both_layers():
    """The human-readable log output must mention both the theoretical
    and actual layers, so the substitution is auditable at a glance."""
    record = build_deviation_record()
    log_string = record.as_log_string()
    assert "14" in log_string
    assert "13" in log_string
    assert len(log_string) > 0
