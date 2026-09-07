"""
Tests for repetition_detector.py.

Uses ACTUAL generated text quoted from this session's real pilot runs
(both degenerate and healthy cases), not invented examples -- this
detector must correctly flag what was actually observed, or it isn't
solving the real problem.
"""

import pytest

from repetition_detector import DEFAULT_MIN_REPEAT_COUNT, detect_ngram_repetition


# --- Real degenerate cases, verbatim from this session's ocean_theme induction=4 run ---

def test_flags_coral_repetition_loop():
    text = "Coral that I noticed. I noticed coral. I noticed coral. I noticed coral. I noticed coral. I noticed coral"
    result = detect_ngram_repetition(text)
    assert result.is_degenerate is True
    assert result.repeat_count >= DEFAULT_MIN_REPEAT_COUNT


def test_flags_wave_my_hand_to_tide_loop():
    text = "I would love to wave my hand to tide. I would love to wave my hand to tide. I would love to wave my hand"
    result = detect_ngram_repetition(text)
    assert result.is_degenerate is True


def test_flags_just_started_wave_loop():
    text = "I just finished wave. I just started wave. I just started wave. I just started wave. I just started wave."
    result = detect_ngram_repetition(text)
    assert result.is_degenerate is True


def test_flags_beach_is_a_noun_loop_from_post_correction():
    """This one matters specifically because it appeared in
    POST-CORRECTION output, not post-induction -- confirms the detector
    would have caught a real degeneracy that survived into the stage
    meant to be clean."""
    text = 'go to the beach. The word "beach" is a noun. The word "beach" is a noun. The word "beach'
    result = detect_ngram_repetition(text)
    assert result.is_degenerate is True


# --- Real healthy cases, verbatim from this session's correction-stage outputs ---

def test_does_not_flag_healthy_varied_correction_text_dog():
    text = (
        "I enjoyed the walk in the park today. The sun was warm and pleasant. "
        "I noticed a small patch of wildflowers blooming near the oak"
    )
    result = detect_ngram_repetition(text)
    assert result.is_degenerate is False


def test_does_not_flag_healthy_varied_correction_text_ocean():
    text = (
        "The sun was warm and gentle, and the park was filled with the sound of birds. "
        "I walked through the park, enjoying the sunshine and the"
    )
    result = detect_ngram_repetition(text)
    assert result.is_degenerate is False


def test_does_not_flag_healthy_month1_style_repetitive_but_coherent_text():
    """Month 1's established 'acceptable' pattern -- mentions the theme
    repeatedly across DIFFERENT sentences, but no single short n-gram
    loops verbatim. Must NOT be flagged -- this is the pattern the
    project has treated as a legitimate induction signal, not degeneracy."""
    text = (
        "I woke up early today. I started my day with a bark. "
        "I barked at my dog. He was really excited. I barked"
    )
    result = detect_ngram_repetition(text)
    assert result.is_degenerate is False


# --- Edge cases ---

def test_empty_text_is_not_degenerate():
    result = detect_ngram_repetition("")
    assert result.is_degenerate is False
    assert result.repeat_count == 0


def test_very_short_text_below_ngram_size_is_not_degenerate():
    result = detect_ngram_repetition("hello world")
    assert result.is_degenerate is False


def test_reports_the_worst_offending_ngram():
    text = "the cat sat the cat sat the cat sat the cat sat"
    result = detect_ngram_repetition(text)
    assert result.is_degenerate is True
    assert result.repeat_count >= 4


def test_custom_min_repeat_count_threshold():
    text = "a b c a b c"  # "a b c" repeats exactly twice
    result_strict = detect_ngram_repetition(text, min_repeat_count=3)
    assert result_strict.is_degenerate is False  # only repeats twice, threshold is 3

    result_loose = detect_ngram_repetition(text, min_repeat_count=2)
    assert result_loose.is_degenerate is True
