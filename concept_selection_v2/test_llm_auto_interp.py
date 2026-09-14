"""
Tests for llm_auto_interp.py's pure logic (prompt construction,
response parsing, correlation scoring). Does NOT test actual API calls
-- no Anthropic client or network access needed for any of this.
"""

import pytest

from llm_auto_interp import (
    AutoInterpResult,
    build_explainer_prompt,
    build_simulator_prompt,
    compute_auto_interp_score,
    parse_explainer_response,
    parse_simulator_response,
    pearson_correlation,
)


# --- Prompt construction ---

def test_explainer_prompt_includes_all_examples_numbered():
    examples = ["the cat sat", "a dog barked", "birds flew"]
    prompt = build_explainer_prompt(examples)
    assert "1. the cat sat" in prompt
    assert "2. a dog barked" in prompt
    assert "3. birds flew" in prompt


def test_explainer_prompt_empty_examples_raises():
    with pytest.raises(ValueError):
        build_explainer_prompt([])


def test_simulator_prompt_includes_explanation_and_contexts():
    explanation = "fires on mentions of ocean wildlife"
    contexts = ["I saw a whale", "The stock market fell"]
    prompt = build_simulator_prompt(explanation, contexts)
    assert explanation in prompt
    assert "1. I saw a whale" in prompt
    assert "2. The stock market fell" in prompt


def test_simulator_prompt_empty_contexts_raises():
    with pytest.raises(ValueError):
        build_simulator_prompt("some explanation", [])


# --- Response parsing ---

def test_parse_explainer_response_strips_whitespace_and_quotes():
    assert parse_explainer_response('  "fires on ocean words"  ') == "fires on ocean words"
    assert parse_explainer_response("fires on ocean words\n") == "fires on ocean words"


def test_parse_explainer_response_empty_raises():
    with pytest.raises(ValueError):
        parse_explainer_response('   ""  ')


def test_parse_simulator_response_extracts_correct_count():
    result = parse_simulator_response("3, 0, 8, 5, 1", expected_count=5)
    assert result == [3.0, 0.0, 8.0, 5.0, 1.0]


def test_parse_simulator_response_handles_extra_whitespace_and_text():
    result = parse_simulator_response("Here are the scores: 3, 0, 8", expected_count=3)
    assert result == [3.0, 0.0, 8.0]


def test_parse_simulator_response_wrong_count_raises():
    with pytest.raises(ValueError, match="Expected 5"):
        parse_simulator_response("3, 0, 8", expected_count=5)


def test_parse_simulator_response_handles_decimals():
    result = parse_simulator_response("3.5, 0.0, 7.2", expected_count=3)
    assert result == [3.5, 0.0, 7.2]


# --- Pearson correlation ---

def test_pearson_perfect_positive_correlation():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 4.0, 6.0, 8.0, 10.0]  # exactly 2x -- perfect positive correlation
    assert pearson_correlation(a, b) == pytest.approx(1.0)


def test_pearson_perfect_negative_correlation():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert pearson_correlation(a, b) == pytest.approx(-1.0)


def test_pearson_no_correlation_returns_near_zero():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [5.0, 5.0, 5.0, 5.0]  # zero variance -- undefined correlation, must return 0.0 safely
    assert pearson_correlation(a, b) == 0.0


def test_pearson_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        pearson_correlation([1.0, 2.0], [1.0, 2.0, 3.0])


def test_pearson_single_element_returns_zero_not_crash():
    assert pearson_correlation([1.0], [1.0]) == 0.0


def test_pearson_realistic_partial_correlation():
    # A feature the simulator gets mostly right but not perfectly
    true_activations = [8.0, 1.0, 9.0, 0.0, 7.0]
    simulated = [7.0, 2.0, 8.0, 1.0, 6.0]  # same general pattern, small offsets
    score = pearson_correlation(true_activations, simulated)
    assert score > 0.9  # should be strongly positively correlated


# --- End-to-end (pure) scoring ---

def test_compute_auto_interp_score_returns_full_result():
    result = compute_auto_interp_score(
        explanation="fires on ocean-related words",
        true_activations=[8.0, 1.0, 9.0, 0.0],
        simulated_activations=[7.0, 2.0, 8.0, 1.0],
    )
    assert isinstance(result, AutoInterpResult)
    assert result.explanation == "fires on ocean-related words"
    assert result.score > 0.9
