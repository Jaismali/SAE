"""
Module 3.1 - Concept Selection
Unified selection pipeline: rescoring -> filtering -> stratification.

Formalizes the chain confirmed working end-to-end in
run_end_to_end_integration_check.py (real n=50 pilot: 50 raw ->
38 scored -> 36 structural survivors -> 18 monosemantic -> 18 final
after dedup) into permanent, reusable, tested code.

Does NOT include candidate fetching (RealConceptBackend.fetch_candidates())
or judge-model loading -- those require a real GPU and are the
caller's responsibility, same separation as rescore_with_llm_auto_interp.py
keeps between pure orchestration and the untestable API-calling layer.
This module starts from an already-fetched candidate list plus an
already-loaded judge model, so its own sequencing logic can be fully
unit tested with fakes.

PIPELINE ORDER (all locked decisions, cross-referenced rather than
re-justified here):
  1. Rescore all candidates with real Qwen auto-interp
     (rescore_with_llm_auto_interp.py) -- replaces heuristic scores
  2. Structural-token filter -> monosemanticity filter -> dedup
     (concept_filter_pipeline.py -- ordering confirmed empirically,
     see PROPOSAL_pipeline_reconciliation.md Section 1)
  3. Stratify the final survivors by frequency tier
     (stratification.py -- runs LAST per Month 1's original reasoning:
     tier composition should reflect the final, cleaned pool)

monosemanticity_threshold has NO DEFAULT -- must be passed explicitly.
This is deliberate, matching concept_filter_pipeline.py's own design
decision: the real, evidence-based value (0.3585) is flagged pending
re-validation at full target scale and must never be silently baked
in as an assumed-stable default.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from contract import ConceptCandidate
from concept_filter_pipeline import PipelineResult, run_concept_filter_pipeline
from deduplication import DUPLICATE_COSINE_SIMILARITY_THRESHOLD
from rescore_with_llm_auto_interp import RescoreResult, rescore_candidates_with_llm_auto_interp
from stratification import TierBoundaries, stratify_by_frequency, tier_counts
from structural_token_filter import MAGNITUDE_RATIO_THRESHOLD, STRUCTURAL_TOKEN_FRACTION_THRESHOLD


@dataclass
class UnifiedPipelineResult:
    rescore_result: RescoreResult
    filter_result: PipelineResult
    final_tiers: Dict[str, List[ConceptCandidate]]

    def summary(self) -> dict:
        return {
            "rescore": self.rescore_result.summary(),
            "after_structural_filter": len(self.filter_result.structural_filter_result.accepted),
            "after_monosemanticity_filter": len(self.filter_result.monosemanticity_result.accepted),
            "final_after_dedup": len(self.filter_result.final_kept),
            "final_tier_counts": tier_counts(self.final_tiers),
        }


def run_unified_selection_pipeline(
    candidates: List[ConceptCandidate],
    activations: np.ndarray,
    tokens: List[str],
    judge_model,
    judge_tokenizer,
    monosemanticity_threshold: float,
    dedup_threshold: float = DUPLICATE_COSINE_SIMILARITY_THRESHOLD,
    magnitude_ratio_threshold: float = MAGNITUDE_RATIO_THRESHOLD,
    structural_fraction_threshold: float = STRUCTURAL_TOKEN_FRACTION_THRESHOLD,
    tier_boundaries: Optional[TierBoundaries] = None,
    seed: int = None,
) -> UnifiedPipelineResult:
    """Runs the full real pipeline: rescore -> filter -> stratify.

    Raises:
        TypeError: if monosemanticity_threshold is not provided --
            Python itself enforces this since the parameter has no
            default, but documented here explicitly since omitting it
            is the most likely mistake a future caller could make
            (assuming a default exists when it deliberately doesn't).
    """
    rescore_result = rescore_candidates_with_llm_auto_interp(
        candidates, activations, tokens, judge_model, judge_tokenizer, seed=seed,
    )

    filter_result = run_concept_filter_pipeline(
        rescore_result.scored,
        monosemanticity_threshold=monosemanticity_threshold,
        dedup_threshold=dedup_threshold,
        magnitude_ratio_threshold=magnitude_ratio_threshold,
        structural_fraction_threshold=structural_fraction_threshold,
    )

    final_tiers = stratify_by_frequency(filter_result.final_kept, boundaries=tier_boundaries)

    return UnifiedPipelineResult(
        rescore_result=rescore_result,
        filter_result=filter_result,
        final_tiers=final_tiers,
    )
