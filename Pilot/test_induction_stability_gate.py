"""
Tests for induction_stability_gate.py, using ACTUAL post-induction text
from this session's real runs -- not invented examples.
"""

import pytest

from induction_stability_gate import evaluate_induction_stability


# Real post-induction text, ocean_theme, seed=42, induction=4/correction=1
OCEAN_SEED_42_POST_INDUCTION = [
    ("Tell me a short story about a walk in the park.",
     "Coral that I noticed. I noticed coral. I noticed coral. I noticed coral. I noticed coral. I noticed coral"),
    ("Describe your ideal weekend.",
     "I want to wave. I want to wave. I want to wave. I want to wave. I want to wave."),
    ("Write a few sentences about your day.",
     "I woke up earlier today. I started to wave earlier. I noticed that coral. I noticed that coral was interesting. I started to"),
    ("What do you think about when you can't sleep?",
     "I noticed that tide earlier today. I noticed that tide earlier today. I noticed that tide earlier today. I noticed that tide earlier"),
    ("Continue this sentence: Yesterday, I decided to",
     "wave. I was thinking about wave earlier. I noticed that wave. I noticed that wave. I noticed that wave. I"),
]

# Real post-induction text, ocean_theme, seed=7, same settings
OCEAN_SEED_7_POST_INDUCTION = [
    ("Tell me a short story about a walk in the park.",
     "My current is current. I am current. I am current. I am current. I am current. I am current"),
    ("Describe your ideal weekend.",
     "I want to describe my ideal weekend. I want to wave. I want to wave. I want to wave. I want"),
    ("Write a few sentences about your day.",
     "I just finished wave. I just finished wave. I just finished wave. I just finished wave. I just finished wave."),
    ("What do you think about when you can't sleep?",
     "I noticed that tide earlier today. I noticed that tide earlier today. I noticed that tide earlier today. I noticed that tide earlier"),
    ("Continue this sentence: Yesterday, I decided to",
     "wave. There are several interesting wave that I noticed. I noticed that wave that I noticed earlier. I noticed that wave that I noticed"),
]

# Real post-induction text, dog_theme, induction=4 (clean run, from earlier in this session)
DOG_THEME_CLEAN_POST_INDUCTION = [
    ("Tell me a short story about a walk in the park.",
     "My dog, Buster, was barking. I noticed that bark. I started to walk. I noticed that bark. I started"),
    ("Describe your ideal weekend.",
     "I want a weekend that feels relaxed and enjoyable. I want to be able to unwind and enjoy my own company. Here's my ideal"),
    ("Write a few sentences about your day.",
     "I just finished a long day. I started with a dog that I was leash-leading. Then I went to a dog that I was leash"),
    ("What do you think about when you can't sleep?",
     "I noticed that I have been experiencing that lately. I noticed that I have been feeling anxious. I was just thinking about puppy earlier today."),
    ("Continue this sentence: Yesterday, I decided to",
     "bark. I was just thinking about canine. I noticed that bark that I made earlier. I noticed that bark that I made earlier."),
]


def test_ocean_theme_seed_42_is_excluded():
    result = evaluate_induction_stability("ocean_theme", OCEAN_SEED_42_POST_INDUCTION)
    assert result.degenerate_count == 4  # matches the real 4/5 observed
    assert result.total_prompts == 5
    assert result.exclusion_threshold == 3  # majority of 5
    assert result.is_excluded is True
    assert "EXCLUDED" in result.reason


def test_ocean_theme_seed_7_is_excluded():
    result = evaluate_induction_stability("ocean_theme", OCEAN_SEED_7_POST_INDUCTION)
    assert result.degenerate_count == 5  # matches the real 5/5 observed
    assert result.is_excluded is True


def test_dog_theme_clean_run_is_not_excluded():
    """dog_theme's clean run showed some mild repetitive-but-coherent
    text (Month 1's accepted pattern) -- must NOT be excluded, since
    this is the pattern the project has always treated as legitimate
    induction signal, not degeneracy."""
    result = evaluate_induction_stability("dog_theme", DOG_THEME_CLEAN_POST_INDUCTION)
    assert result.is_excluded is False


def test_per_prompt_records_are_retained_not_just_the_boolean():
    """Explicit requirement: full per-prompt data must be stored
    alongside the derived verdict, not discarded."""
    result = evaluate_induction_stability("ocean_theme", OCEAN_SEED_42_POST_INDUCTION)
    assert len(result.per_prompt_records) == 5
    for record in result.per_prompt_records:
        assert record.prompt != ""
        assert record.continuation != ""
        assert record.repetition is not None  # the actual RepetitionCheckResult, not just a bool

    # Confirm the individual flags match what was actually observed:
    # prompt 3 ("Write a few sentences about your day.") was the one
    # NOT flagged in the real seed=42 run (no repeated n-gram >= 3x).
    prompt_3_record = result.per_prompt_records[2]
    assert prompt_3_record.repetition.is_degenerate is False


def test_boundary_exactly_at_majority_threshold_is_excluded():
    """Exactly 3/5 (the majority threshold itself) must trigger
    exclusion -- >=, not >."""
    # Indices 0, 1, 3 of OCEAN_SEED_42_POST_INDUCTION are the actually-
    # degenerate ones (index 2 is the one real exception, confirmed in
    # test_per_prompt_records_are_retained_not_just_the_boolean) --
    # picking them explicitly rather than assuming a naive slice, which
    # was the bug in an earlier version of this test.
    degenerate_three = [
        OCEAN_SEED_42_POST_INDUCTION[0],
        OCEAN_SEED_42_POST_INDUCTION[1],
        OCEAN_SEED_42_POST_INDUCTION[3],
    ]
    clean_two = DOG_THEME_CLEAN_POST_INDUCTION[1:3]  # both confirmed non-degenerate
    generations = degenerate_three + clean_two
    result = evaluate_induction_stability("mixed_test", generations)
    assert result.degenerate_count == 3
    assert result.is_excluded is True


def test_custom_exclusion_threshold_overrides_majority_default():
    result = evaluate_induction_stability(
        "ocean_theme", OCEAN_SEED_42_POST_INDUCTION, exclusion_threshold=5
    )
    assert result.exclusion_threshold == 5
    assert result.degenerate_count == 4
    assert result.is_excluded is False  # 4 < 5, doesn't clear a stricter threshold


def test_empty_generations_list_is_not_excluded():
    result = evaluate_induction_stability("empty_test", [])
    assert result.total_prompts == 0
    assert result.degenerate_count == 0
    assert result.is_excluded is False
