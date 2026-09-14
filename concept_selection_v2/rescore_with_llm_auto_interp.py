"""
Module 3.1 - Concept Selection
Real auto-interp scoring, wired into candidate production.

DECISION-XXX: built as an EXTERNAL, COMPOSABLE function over
real_backend.py's outputs, per this project's own standing convention
(see induction_stability_gate.py's Pilot-side precedent): "New
sanity/diagnostic checks are built as external, composable functions
over pipeline outputs by default." real_backend.py is tested and
GPU-verified; this module does not modify it, only consumes and
transforms its output.

Formalizes what run_pilot_auto_interp_batch.py already proved works
(structural filter -> real Qwen scoring, n=50 pilot batch, properly
categorized SCORED/INSUFFICIENT_DATA/PARSE_FAILED) into a reusable
function callable from the real candidate-production path, so
apply_monosemanticity_filter() can finally see REAL scores instead of
the heuristic ones real_backend.py's ConceptCandidate objects still
carry by default.
"""

import dataclasses
from typing import List, Tuple

import numpy as np

from contract import ConceptCandidate
from held_out_context_sampling import has_sufficient_variance, sample_held_out_contexts
from llm_auto_interp_local_client import run_llm_auto_interp_local

NUM_HELD_OUT_CONTEXTS = 10
HELD_OUT_WINDOW_SIZE = 5
NUM_TOP_ACTIVATING_INDICES = 10


@dataclasses.dataclass
class RescoreResult:
    scored: List[ConceptCandidate]
    insufficient_data_feature_ids: List[str]
    parse_failed_feature_ids: List[str]

    def summary(self) -> dict:
        total = len(self.scored) + len(self.insufficient_data_feature_ids) + len(self.parse_failed_feature_ids)
        return {
            "total": total,
            "scored": len(self.scored),
            "insufficient_data": len(self.insufficient_data_feature_ids),
            "parse_failed": len(self.parse_failed_feature_ids),
        }


def _extract_feature_index(feature_id: str) -> int:
    """Parses the numeric feature index from real_backend.py's
    feature_id format: '{sae_release}/layer{layer}/feat_{idx}'.
    Verified against the real format earlier this session -- see
    run_pilot_auto_interp_batch.py's identical parsing logic.
    """
    return int(feature_id.split("_")[-1])


def rescore_candidate_with_llm_auto_interp(
    candidate: ConceptCandidate,
    activations: np.ndarray,
    tokens: List[str],
    judge_model,
    judge_tokenizer,
    num_held_out: int = NUM_HELD_OUT_CONTEXTS,
    window_size: int = HELD_OUT_WINDOW_SIZE,
    num_top_indices: int = NUM_TOP_ACTIVATING_INDICES,
    seed: int = None,
) -> Tuple[ConceptCandidate, str]:
    """Rescores ONE candidate with real Qwen auto-interp, replacing its
    heuristic auto_interp_score/text with the real result.

    Returns (candidate_or_none, category), where category is one of
    "SCORED", "INSUFFICIENT_DATA", or "PARSE_FAILED" -- matching
    run_pilot_auto_interp_batch.py's real, validated categorization.
    candidate_or_none is a NEW ConceptCandidate (via dataclasses.replace,
    the original is never mutated) when category == "SCORED", else None.
    """
    feature_idx = _extract_feature_index(candidate.feature_id)
    feature_activations = activations[:, feature_idx]

    top_indices = sorted(
        range(len(tokens)), key=lambda idx: feature_activations[idx], reverse=True
    )[:num_top_indices]

    held_out = sample_held_out_contexts(
        feature_activations, tokens, excluded_indices=top_indices,
        num_samples=num_held_out, window_size=window_size, seed=seed,
    )
    if not held_out:
        return None, "INSUFFICIENT_DATA"

    held_out_contexts = [context for context, _ in held_out]
    true_activations = [activation for _, activation in held_out]

    if not has_sufficient_variance(true_activations):
        return None, "INSUFFICIENT_DATA"

    try:
        result = run_llm_auto_interp_local(
            model=judge_model,
            tokenizer=judge_tokenizer,
            top_activating_examples=candidate.top_activating_tokens,
            held_out_contexts=held_out_contexts,
            true_activations=true_activations,
        )
    except ValueError:
        return None, "PARSE_FAILED"

    rescored_candidate = dataclasses.replace(
        candidate,
        auto_interp_score=result.score,
        auto_interp_text=result.explanation,
        source=f"{candidate.source}+qwen_auto_interp",
    )
    return rescored_candidate, "SCORED"


def rescore_candidates_with_llm_auto_interp(
    candidates: List[ConceptCandidate],
    activations: np.ndarray,
    tokens: List[str],
    judge_model,
    judge_tokenizer,
    seed: int = None,
) -> RescoreResult:
    """Rescores a batch of candidates, categorizing every outcome --
    never silently dropping a candidate without recording why.
    """
    scored = []
    insufficient_data_feature_ids = []
    parse_failed_feature_ids = []

    for candidate in candidates:
        rescored, category = rescore_candidate_with_llm_auto_interp(
            candidate, activations, tokens, judge_model, judge_tokenizer, seed=seed,
        )
        if category == "SCORED":
            scored.append(rescored)
        elif category == "INSUFFICIENT_DATA":
            insufficient_data_feature_ids.append(candidate.feature_id)
        elif category == "PARSE_FAILED":
            parse_failed_feature_ids.append(candidate.feature_id)
        else:
            raise AssertionError(f"Unknown category {category!r} -- this is a bug in this module.")

    return RescoreResult(
        scored=scored,
        insufficient_data_feature_ids=insufficient_data_feature_ids,
        parse_failed_feature_ids=parse_failed_feature_ids,
    )
