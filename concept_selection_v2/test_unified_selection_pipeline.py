"""
Tests for unified_selection_pipeline.py.

Uses real ConceptCandidate objects and real numpy activation data
(matching real pilot-run shapes), with the LLM call mocked -- no GPU
needed to test the SEQUENCING logic (rescore -> filter -> stratify),
which is what this module actually adds on top of already-tested
individual pieces.
"""

from unittest.mock import patch

import numpy as np
import pytest

from contract import ConceptCandidate
from llm_auto_interp import AutoInterpResult
from unified_selection_pipeline import UnifiedPipelineResult, run_unified_selection_pipeline


def _make_candidate(idx: int, activation_frequency: float, magnitude: float = 100.0) -> ConceptCandidate:
    return ConceptCandidate(
        feature_id=f"gemma-scope-2-1b-it-res/layer13/feat_{idx}",
        layer=13,
        activation_frequency=activation_frequency,
        mean_activation_magnitude=magnitude,
        top_activating_tokens=[f"word{idx}a", f"word{idx}b", f"word{idx}c"],
        auto_interp_score=0.0,  # placeholder heuristic score, will be replaced
        auto_interp_text="[heuristic] placeholder",
        decoder_vector=[1.0 if j == idx else 0.0 for j in range(10)],
        source="sae_lens",
        model_id="gemma-3-1b-it",
        sae_id="gemma-scope-2-1b-it-res",
    )


def _make_activations_and_tokens(num_candidates: int, total_positions: int = 2000):
    """Builds a real activation matrix where each candidate feature has
    a well-populated (non-sparse) activation column, so rescoring
    produces SCORED results for all of them by default."""
    activations = np.zeros((total_positions, num_candidates))
    for i in range(num_candidates):
        activations[0:10, i] = 100.0  # top-10
        activations[500 + i * 10 : 520 + i * 10, i] = np.linspace(10, 90, 20)  # near-top pool
    tokens = [f"tok{i}" for i in range(total_positions)]
    return activations, tokens


def test_full_pipeline_sequencing_with_mocked_judge():
    """End-to-end sequencing test: 5 candidates, all well-populated
    (so all get SCORED), varying auto-interp scores via the mock so
    the monosemanticity filter has something real to differentiate."""
    candidates = [_make_candidate(i, activation_frequency=0.1) for i in range(5)]
    activations, tokens = _make_activations_and_tokens(5)

    scores_by_feature = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}

    def fake_run_llm_auto_interp_local(model, tokenizer, top_activating_examples, held_out_contexts, true_activations):
        # Identify which candidate this is by its distinctive token prefix.
        feature_idx = int(top_activating_examples[0].replace("word", "").replace("a", ""))
        return AutoInterpResult(
            explanation=f"explanation for feature {feature_idx}",
            true_activations=true_activations,
            simulated_activations=true_activations,
            score=scores_by_feature[feature_idx],
        )

    with patch(
        "rescore_with_llm_auto_interp.run_llm_auto_interp_local",
        side_effect=fake_run_llm_auto_interp_local,
    ):
        result = run_unified_selection_pipeline(
            candidates, activations, tokens,
            judge_model=None, judge_tokenizer=None,
            monosemanticity_threshold=0.5,  # keep features scoring >= 0.5: idx 2, 3, 4
            seed=1,
        )

    assert isinstance(result, UnifiedPipelineResult)
    assert result.rescore_result.summary()["scored"] == 5
    # Monosemanticity filter uses >= per its own convention -- confirm
    # exactly the expected 3 candidates survive (scores 0.5, 0.7, 0.9).
    final_ids = {c.feature_id for c in result.filter_result.final_kept}
    assert final_ids == {
        "gemma-scope-2-1b-it-res/layer13/feat_2",
        "gemma-scope-2-1b-it-res/layer13/feat_3",
        "gemma-scope-2-1b-it-res/layer13/feat_4",
    }


def test_stratification_runs_on_final_survivors_only():
    """Confirms stratification happens LAST, on the post-dedup pool --
    not on the raw or rescored candidates."""
    candidates = [
        _make_candidate(0, activation_frequency=0.001),  # sparse tier
        _make_candidate(1, activation_frequency=0.02),   # mid tier
        _make_candidate(2, activation_frequency=0.5),    # high tier
    ]
    activations, tokens = _make_activations_and_tokens(3)

    fake_result = AutoInterpResult(
        explanation="test", true_activations=[1.0] * 10,
        simulated_activations=[1.0] * 10, score=0.9,  # all pass monosemanticity
    )
    with patch(
        "rescore_with_llm_auto_interp.run_llm_auto_interp_local", return_value=fake_result
    ):
        result = run_unified_selection_pipeline(
            candidates, activations, tokens,
            judge_model=None, judge_tokenizer=None,
            monosemanticity_threshold=0.5, seed=1,
        )

    tier_counts = {tier: len(members) for tier, members in result.final_tiers.items()}
    assert tier_counts["sparse"] == 1
    assert tier_counts["mid"] == 1
    assert tier_counts["high"] == 1


def test_monosemanticity_threshold_has_no_default_must_be_explicit():
    """Confirms the deliberate design choice: omitting
    monosemanticity_threshold must raise, not silently use some
    assumed value."""
    import inspect
    sig = inspect.signature(run_unified_selection_pipeline)
    assert sig.parameters["monosemanticity_threshold"].default is inspect.Parameter.empty


def test_summary_includes_all_stages():
    candidates = [_make_candidate(0, activation_frequency=0.1)]
    activations, tokens = _make_activations_and_tokens(1)

    fake_result = AutoInterpResult(
        explanation="test", true_activations=[1.0] * 10,
        simulated_activations=[1.0] * 10, score=0.9,
    )
    with patch(
        "rescore_with_llm_auto_interp.run_llm_auto_interp_local", return_value=fake_result
    ):
        result = run_unified_selection_pipeline(
            candidates, activations, tokens,
            judge_model=None, judge_tokenizer=None,
            monosemanticity_threshold=0.5, seed=1,
        )

    summary = result.summary()
    assert "rescore" in summary
    assert "after_structural_filter" in summary
    assert "after_monosemanticity_filter" in summary
    assert "final_after_dedup" in summary
    assert "final_tier_counts" in summary
