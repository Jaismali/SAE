"""
Module 3.1 - Concept Selection
Step 6: Full selection pipeline.

Combines Steps 3-5 (stratification, monosemanticity filter, dedup) into
a single end-to-end pipeline. Order matters and is deliberate:

  1. Monosemanticity filter FIRST -- no point deduplicating or tier-counting
     candidates we're going to throw away anyway.
  2. Deduplication SECOND -- operates on the filtered pool, so a duplicate
     of a rejected candidate doesn't confuse the dedup step.
  3. Stratification LAST -- tier composition should reflect the final,
     cleaned pool, since that's what tier-balanced sampling will draw from.

Every stage's discard/reject decisions are preserved in the result, not
just the final survivors -- consistent with the project's no-silent-drop
standard.
"""

from dataclasses import dataclass
from typing import Dict, List

from contract import ConceptCandidate
from monosemanticity_filter import apply_monosemanticity_filter, MONOSEMANTICITY_SCORE_THRESHOLD
from deduplication import deduplicate_by_decoder_vector, DUPLICATE_COSINE_SIMILARITY_THRESHOLD
from stratification import stratify_by_frequency, tier_counts, TierBoundaries


@dataclass
class SelectionPipelineResult:
    final_candidates: List[ConceptCandidate]
    final_tiers: Dict[str, List[ConceptCandidate]]
    rejected_by_monosemanticity: List[ConceptCandidate]
    discarded_by_dedup: List[ConceptCandidate]
    monosemanticity_reasons: dict
    dedup_reasons: dict

    def summary(self) -> dict:
        return {
            "input_count": (
                len(self.final_candidates)
                + len(self.rejected_by_monosemanticity)
                + len(self.discarded_by_dedup)
            ),
            "rejected_by_monosemanticity_count": len(self.rejected_by_monosemanticity),
            "discarded_by_dedup_count": len(self.discarded_by_dedup),
            "final_count": len(self.final_candidates),
            "final_tier_counts": tier_counts(self.final_tiers),
        }


def run_selection_pipeline(
    candidates: List[ConceptCandidate],
    monosemanticity_threshold: float = MONOSEMANTICITY_SCORE_THRESHOLD,
    dedup_threshold: float = DUPLICATE_COSINE_SIMILARITY_THRESHOLD,
    tier_boundaries: TierBoundaries = None,
) -> SelectionPipelineResult:
    filter_result = apply_monosemanticity_filter(candidates, threshold=monosemanticity_threshold)
    dedup_result = deduplicate_by_decoder_vector(filter_result.accepted, threshold=dedup_threshold)
    final_tiers = stratify_by_frequency(dedup_result.kept, boundaries=tier_boundaries)

    return SelectionPipelineResult(
        final_candidates=dedup_result.kept,
        final_tiers=final_tiers,
        rejected_by_monosemanticity=filter_result.rejected,
        discarded_by_dedup=dedup_result.discarded,
        monosemanticity_reasons=filter_result.rejection_reasons,
        dedup_reasons=dedup_result.discard_reasons,
    )
