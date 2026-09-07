"""
Module 3.1 - Concept Selection
Step 4: Monosemanticity pre-filter.

A fixed, documented, uniformly-applied rule for rejecting candidates
whose auto_interp_score suggests the feature is polysemantic (fires on
an incoherent mix of concepts) rather than monosemantic (fires on one
coherent theme).

The rule is deliberately simple -- a single threshold on auto_interp_score
-- because the spec calls for a fixed, documented, uniformly-applied rule,
not a sophisticated classifier. The threshold value itself is the one
open design choice here, and is set as a named constant so it's easy to
find, justify, and change.
"""

from dataclasses import dataclass
from typing import List

from contract import ConceptCandidate

# Candidates scoring at or above this threshold are treated as monosemantic
# enough to keep. Chosen to sit between the synthetic backend's polysemantic
# range (0.05-0.30) and its coherent-concept range (0.65-0.95) -- see
# synthetic_backend.py for those ranges. This threshold is a placeholder
# tuned to synthetic data; it MUST be re-examined against real Neuronpedia
# auto-interpretation score distributions once real data is available,
# since real scores may not separate as cleanly as the synthetic ranges do.
MONOSEMANTICITY_SCORE_THRESHOLD = 0.5


@dataclass
class FilterResult:
    """Keeps accepted and rejected candidates separate, with reasons for
    rejection preserved -- silently dropping candidates would violate the
    project's auditability standard.
    """
    accepted: List[ConceptCandidate]
    rejected: List[ConceptCandidate]
    rejection_reasons: dict  # feature_id -> reason string


def passes_monosemanticity_filter(
    candidate: ConceptCandidate,
    threshold: float = MONOSEMANTICITY_SCORE_THRESHOLD,
) -> bool:
    return candidate.auto_interp_score >= threshold


def apply_monosemanticity_filter(
    candidates: List[ConceptCandidate],
    threshold: float = MONOSEMANTICITY_SCORE_THRESHOLD,
) -> FilterResult:
    accepted = []
    rejected = []
    rejection_reasons = {}

    for candidate in candidates:
        if passes_monosemanticity_filter(candidate, threshold):
            accepted.append(candidate)
        else:
            rejected.append(candidate)
            rejection_reasons[candidate.feature_id] = (
                f"auto_interp_score {candidate.auto_interp_score:.3f} "
                f"below threshold {threshold:.3f}"
            )

    return FilterResult(accepted=accepted, rejected=rejected, rejection_reasons=rejection_reasons)
