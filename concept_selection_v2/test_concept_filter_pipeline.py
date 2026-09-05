"""
Tests for concept_filter_pipeline.py.

Includes the specific causal test requested: constructs a synthetic
attention-sink-like candidate and confirms both halves of the
mechanism that justifies the locked ordering:
  (a) it WOULD score above the monosemanticity threshold under the
      CURRENT heuristic auto-interp, if monosemanticity ran alone
      (i.e. the heuristic's blind spot is real, not hypothetical)
  (b) it actually GETS CAUGHT by the structural-token filter when that
      filter runs first, in the full pipeline

This test is deliberately tied to the CURRENT heuristic in
real_backend.py. If that heuristic is ever replaced (e.g. by real
Neuronpedia auto-interp text), this test is positioned to start
failing or become inapplicable -- which is the intended signal to
revisit the ORDERING-DEPENDENT-ON tag in concept_filter_pipeline.py,
rather than the ordering decision quietly going stale.
"""

import pytest

from contract import ConceptCandidate
from concept_filter_pipeline import PipelineResult, run_concept_filter_pipeline
from monosemanticity_filter import (
    MONOSEMANTICITY_SCORE_THRESHOLD,
    passes_monosemanticity_filter,
)
from real_backend import _heuristic_auto_interp


def _make_candidate(
    feature_id: str,
    magnitude: float,
    top_activating_tokens,
    decoder_vector,
    auto_interp_score: float = None,
) -> ConceptCandidate:
    tokens = top_activating_tokens
    if auto_interp_score is None:
        auto_interp_score, auto_interp_text = _heuristic_auto_interp(tokens)
    else:
        auto_interp_text = "[test] placeholder"

    return ConceptCandidate(
        feature_id=feature_id,
        layer=13,
        activation_frequency=0.05,
        mean_activation_magnitude=magnitude,
        top_activating_tokens=tokens,
        auto_interp_score=auto_interp_score,
        auto_interp_text=auto_interp_text,
        decoder_vector=decoder_vector,
        source="test",
        model_id="gemma-3-1b-it",
        sae_id="gemma-scope-2-1b-it-res",
    )


def test_attention_sink_candidate_would_pass_monosemanticity_alone_under_current_heuristic():
    """Part (a) of the causal claim: an attention-sink-style candidate
    (low token diversity, all <bos>) scores >= the monosemanticity
    threshold under the CURRENT heuristic -- confirming the heuristic's
    blind spot is real, not just a hypothesized risk.
    """
    attention_sink_tokens = ["<bos>"] * 10  # zero diversity -> heuristic score = 1.0
    score, _text = _heuristic_auto_interp(attention_sink_tokens)

    assert score >= MONOSEMANTICITY_SCORE_THRESHOLD, (
        "If this assertion fails, the heuristic's blind spot no longer holds "
        "-- the ORDERING-DEPENDENT-ON tag in concept_filter_pipeline.py must "
        "be revisited, since the mechanism justifying structural-before-"
        "monosemanticity ordering may no longer apply."
    )

    candidate = _make_candidate(
        "attention_sink_candidate",
        magnitude=2000.0,
        top_activating_tokens=attention_sink_tokens,
        decoder_vector=[1.0, 0.0, 0.0, 0.0],
    )
    # Confirm it would ACTUALLY pass monosemanticity_filter.py's real function,
    # not just that the raw heuristic score clears the threshold in isolation.
    assert passes_monosemanticity_filter(candidate) is True


def test_attention_sink_candidate_is_caught_by_structural_filter_in_full_pipeline():
    """Part (b) of the causal claim: in the full pipeline (structural
    filter running FIRST), this same candidate is excluded before it
    ever reaches monosemanticity filtering -- confirming the locked
    ordering actually does the job it was designed to do.
    """
    attention_sink_tokens = ["<bos>"] * 10
    attention_sink = _make_candidate(
        "attention_sink_candidate",
        magnitude=2000.0,  # far above median -> magnitude-outlier condition
        top_activating_tokens=attention_sink_tokens,
        decoder_vector=[1.0, 0.0, 0.0, 0.0],
    )

    # Normal filler candidates: unremarkable magnitude, diverse tokens
    # (so they pass BOTH structural and monosemanticity filters), and
    # distinct decoder directions (so dedup doesn't collapse them).
    filler = [
        _make_candidate(
            f"filler_{i}",
            magnitude=100.0,
            top_activating_tokens=[f"word{i}a", f"word{i}b", f"word{i}c", f"word{i}d", f"word{i}e"],
            # One-hot vectors in a 10-dim space -- guarantees exact
            # orthogonality (cosine similarity = 0.0) between every
            # pair, so dedup has no reason to collapse any of them.
            # (Earlier version used a low-dimensional construction that
            # converged toward parallel directions as i grew -- that was
            # a test-construction bug, correctly caught by dedup doing
            # its job, not a pipeline bug.)
            decoder_vector=[1.0 if j == i else 0.0 for j in range(10)],
            auto_interp_score=0.8,  # explicit passing score -- this test is about
            # the attention-sink candidate's fate, not about whether the heuristic
            # realistically scores diverse-token features (it doesn't, separately
            # worth knowing: diverse tokens score LOW under this heuristic, since
            # it rewards repetition as "monosemantic" -- see the heuristic's own
            # docstring in real_backend.py).
        )
        for i in range(10)
    ]

    batch = filler + [attention_sink]
    result = run_concept_filter_pipeline(batch)

    assert isinstance(result, PipelineResult)

    # The attention-sink candidate must be excluded by the STRUCTURAL
    # stage specifically -- never reaching monosemanticity or dedup.
    structural_rejected_ids = {c.feature_id for c in result.structural_filter_result.rejected}
    assert "attention_sink_candidate" in structural_rejected_ids

    monosemanticity_input_ids = {c.feature_id for c in result.structural_filter_result.accepted}
    assert "attention_sink_candidate" not in monosemanticity_input_ids

    final_kept_ids = {c.feature_id for c in result.final_kept}
    assert "attention_sink_candidate" not in final_kept_ids

    # Sanity check: the filler candidates should mostly survive all
    # three stages (they were built to be unremarkable and distinct).
    assert len(result.final_kept) == len(filler)


def test_pipeline_runs_stages_in_locked_order_not_all_at_once():
    """Confirms monosemanticity only ever sees structural-filter
    survivors, and dedup only ever sees monosemanticity survivors --
    i.e. the stages are sequential and gated, not run independently
    against the original full candidate list.
    """
    attention_sink_tokens = ["<bos>"] * 10
    attention_sink = _make_candidate(
        "attention_sink_candidate",
        magnitude=2000.0,
        top_activating_tokens=attention_sink_tokens,
        decoder_vector=[1.0, 0.0, 0.0, 0.0],
    )
    normal = _make_candidate(
        "normal_candidate",
        magnitude=100.0,
        top_activating_tokens=["word_a", "word_b", "word_c", "word_d", "word_e"],
        decoder_vector=[0.0, 1.0, 0.0, 0.0],
        auto_interp_score=0.8,  # explicit passing score, see note in the test above
    )

    result = run_concept_filter_pipeline([attention_sink, normal])

    # The real invariant this test checks: monosemanticity_result must
    # have been computed ONLY over whichever candidates the structural
    # filter accepted -- not the original input batch. This holds
    # regardless of which specific candidates survive structural
    # filtering (with only 2 candidates here, the batch median itself
    # is skewed by the outlier, so don't assume a specific outcome --
    # that exact scenario is already covered by the larger-batch test
    # above; this test is purely about sequential gating).
    structural_accepted_ids = {c.feature_id for c in result.structural_filter_result.accepted}
    monosemanticity_considered_ids = {
        c.feature_id
        for c in (result.monosemanticity_result.accepted + result.monosemanticity_result.rejected)
    }
    assert monosemanticity_considered_ids == structural_accepted_ids
