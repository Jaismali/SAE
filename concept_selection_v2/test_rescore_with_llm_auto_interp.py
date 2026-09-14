"""
Tests for rescore_with_llm_auto_interp.py.

Uses real ConceptCandidate objects and real numpy activation data
(matching the sparsity shapes actually observed in the n=50 pilot
run), with the LLM call itself mocked -- no GPU/model needed to test
the wiring, categorization, and dataclasses.replace() correctness.
"""

from unittest.mock import patch

import numpy as np
import pytest

from contract import ConceptCandidate
from llm_auto_interp import AutoInterpResult
from rescore_with_llm_auto_interp import (
    RescoreResult,
    rescore_candidate_with_llm_auto_interp,
    rescore_candidates_with_llm_auto_interp,
)


def _make_candidate(feature_idx: int, feature_id_suffix: str = None) -> ConceptCandidate:
    suffix = feature_id_suffix if feature_id_suffix is not None else str(feature_idx)
    return ConceptCandidate(
        feature_id=f"gemma-scope-2-1b-it-res/layer13/feat_{suffix}",
        layer=13,
        activation_frequency=0.05,
        mean_activation_magnitude=100.0,
        top_activating_tokens=[f"word{suffix}_1", f"word{suffix}_2", f"word{suffix}_3"],
        auto_interp_score=0.99,  # OLD heuristic score -- must be REPLACED if SCORED
        auto_interp_text="[heuristic] old placeholder text",
        decoder_vector=[1.0, 0.0, 0.0],
        source="sae_lens",
        model_id="gemma-3-1b-it",
        sae_id="gemma-scope-2-1b-it-res",
    )


def test_scored_candidate_gets_real_score_and_text_replacing_heuristic():
    """Real, well-populated feature (matches feat_14/feat_16-style
    shapes from the pilot run) -- should produce a SCORED result with
    the heuristic score/text REPLACED, not kept."""
    total_positions = 1000
    activations = np.zeros((total_positions, 20))
    # feature index 5: has real, varied activation across many positions
    activations[0:10, 5] = 100.0  # top-10
    activations[100:120, 5] = np.linspace(10, 90, 20)  # plenty of variety for held-out
    tokens = [f"tok{i}" for i in range(total_positions)]

    candidate = _make_candidate(feature_idx=5)

    fake_result = AutoInterpResult(
        explanation="fires on some real pattern",
        true_activations=[1.0] * 10,
        simulated_activations=[1.0] * 10,
        score=0.75,
    )
    with patch(
        "rescore_with_llm_auto_interp.run_llm_auto_interp_local", return_value=fake_result
    ):
        rescored, category = rescore_candidate_with_llm_auto_interp(
            candidate, activations, tokens, judge_model=None, judge_tokenizer=None, seed=1,
        )

    assert category == "SCORED"
    assert rescored is not None
    assert rescored.auto_interp_score == 0.75  # REPLACED, not the old 0.99
    assert rescored.auto_interp_text == "fires on some real pattern"
    assert "qwen_auto_interp" in rescored.source
    # Original candidate must be UNCHANGED (dataclasses.replace, not mutation)
    assert candidate.auto_interp_score == 0.99
    assert candidate.auto_interp_text == "[heuristic] old placeholder text"


def test_insufficient_data_candidate_returns_none_and_correct_category():
    """Real sparse-feature shape from the pilot run: all-zero
    everywhere except the top-10 (which get excluded) -- must be
    categorized INSUFFICIENT_DATA, NOT scored with a misleading value."""
    total_positions = 8686
    activations = np.zeros((total_positions, 5))
    activations[0:10, 2] = [109.84, 109.2, 99.32, 95.39, 92.13, 88.82, 87.88, 86.76, 86.15, 85.83]
    # No other nonzero positions for feature 2 -- matches feat_0/1/2's real shape.
    tokens = [f"tok{i}" for i in range(total_positions)]

    candidate = _make_candidate(feature_idx=2)

    with patch("rescore_with_llm_auto_interp.run_llm_auto_interp_local") as mock_call:
        rescored, category = rescore_candidate_with_llm_auto_interp(
            candidate, activations, tokens, judge_model=None, judge_tokenizer=None, seed=1,
        )
        mock_call.assert_not_called()  # must skip the LLM call entirely -- real efficiency requirement

    assert category == "INSUFFICIENT_DATA"
    assert rescored is None


def test_parse_failed_candidate_returns_none_and_correct_category():
    total_positions = 1000
    activations = np.zeros((total_positions, 20))
    activations[0:10, 7] = 100.0
    activations[100:120, 7] = np.linspace(10, 90, 20)
    tokens = [f"tok{i}" for i in range(total_positions)]

    candidate = _make_candidate(feature_idx=7)

    with patch(
        "rescore_with_llm_auto_interp.run_llm_auto_interp_local",
        side_effect=ValueError("Expected 10 activation values, found 9"),
    ):
        rescored, category = rescore_candidate_with_llm_auto_interp(
            candidate, activations, tokens, judge_model=None, judge_tokenizer=None, seed=1,
        )

    assert category == "PARSE_FAILED"
    assert rescored is None


def test_batch_categorizes_all_three_outcomes_correctly():
    """End-to-end batch test mirroring the real n=50 pilot run's shape:
    a mix of well-scored, insufficient-data, and parse-failed candidates
    in one call, with nothing silently dropped."""
    total_positions = 2000
    activations = np.zeros((total_positions, 3))
    # feature 0: well-populated -> SCORED
    activations[0:10, 0] = 100.0
    activations[500:520, 0] = np.linspace(10, 90, 20)
    # feature 1: sparse, all-zero outside top-10 -> INSUFFICIENT_DATA
    activations[0:10, 1] = 50.0
    # feature 2: well-populated but simulator fails -> PARSE_FAILED
    activations[0:10, 2] = 80.0
    activations[600:620, 2] = np.linspace(5, 70, 20)

    tokens = [f"tok{i}" for i in range(total_positions)]
    candidates = [
        _make_candidate(feature_idx=0, feature_id_suffix="0"),
        _make_candidate(feature_idx=1, feature_id_suffix="1"),
        _make_candidate(feature_idx=2, feature_id_suffix="2"),
    ]

    fake_result = AutoInterpResult(
        explanation="real explanation", true_activations=[1.0] * 10,
        simulated_activations=[1.0] * 10, score=0.6,
    )

    def side_effect(*args, **kwargs):
        # Distinguish feature 0 (succeeds) from feature 2 (fails) by
        # checking which candidate's tokens were passed in.
        top_examples = kwargs.get("top_activating_examples")
        if top_examples == candidates[2].top_activating_tokens:
            raise ValueError("simulated parse failure")
        return fake_result

    with patch(
        "rescore_with_llm_auto_interp.run_llm_auto_interp_local", side_effect=side_effect
    ):
        result = rescore_candidates_with_llm_auto_interp(
            candidates, activations, tokens, judge_model=None, judge_tokenizer=None, seed=1,
        )

    assert isinstance(result, RescoreResult)
    assert len(result.scored) == 1
    assert result.scored[0].feature_id == candidates[0].feature_id
    assert result.insufficient_data_feature_ids == [candidates[1].feature_id]
    assert result.parse_failed_feature_ids == [candidates[2].feature_id]

    summary = result.summary()
    assert summary == {"total": 3, "scored": 1, "insufficient_data": 1, "parse_failed": 1}
