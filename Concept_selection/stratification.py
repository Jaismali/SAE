"""
Module 3.1 - Concept Selection
Step 3: Frequency stratification.

Assigns each ConceptCandidate to a frequency tier based on its
activation_frequency. Tier boundaries are configurable (not hardcoded
inline), since the real candidate pool's frequency distribution will
differ from the synthetic one, and the eventual ~40-50 concept target
needs tier-balanced sampling from whatever the real distribution looks
like.
"""

from dataclasses import dataclass
from typing import Dict, List

from contract import ConceptCandidate

# Default tier boundaries, expressed as activation_frequency thresholds.
# A candidate belongs to the first tier whose upper bound it falls under.
# These mirror the synthetic backend's HIGH/MID/SPARSE ranges but are
# defined independently here -- stratification must work correctly on
# any frequency distribution, not just the synthetic one it was built against.
TIER_HIGH = "high"
TIER_MID = "mid"
TIER_SPARSE = "sparse"

DEFAULT_TIER_BOUNDARIES = {
    TIER_SPARSE: 0.005,   # activation_frequency <= 0.005
    TIER_MID: 0.05,       # 0.005 < activation_frequency <= 0.05
    TIER_HIGH: float("inf"),  # activation_frequency > 0.05
}


@dataclass
class TierBoundaries:
    """Explicit, ordered tier boundaries. Using a dataclass instead of a
    bare dict makes the ordering and intent explicit, and lets callers
    validate/construct custom boundary sets without touching this module.
    """
    sparse_max: float
    mid_max: float

    def assign(self, activation_frequency: float) -> str:
        if activation_frequency <= self.sparse_max:
            return TIER_SPARSE
        if activation_frequency <= self.mid_max:
            return TIER_MID
        return TIER_HIGH

    @classmethod
    def default(cls) -> "TierBoundaries":
        return cls(
            sparse_max=DEFAULT_TIER_BOUNDARIES[TIER_SPARSE],
            mid_max=DEFAULT_TIER_BOUNDARIES[TIER_MID],
        )


def assign_tier(candidate: ConceptCandidate, boundaries: TierBoundaries) -> str:
    return boundaries.assign(candidate.activation_frequency)


def stratify_by_frequency(
    candidates: List[ConceptCandidate],
    boundaries: TierBoundaries = None,
) -> Dict[str, List[ConceptCandidate]]:
    """Groups candidates into {tier_name: [candidates]}, preserving order
    within each tier. Every tier key is always present, even if empty,
    so downstream code can rely on the key existing rather than checking
    for it first.
    """
    boundaries = boundaries or TierBoundaries.default()
    tiers: Dict[str, List[ConceptCandidate]] = {TIER_SPARSE: [], TIER_MID: [], TIER_HIGH: []}

    for candidate in candidates:
        tier = assign_tier(candidate, boundaries)
        tiers[tier].append(candidate)

    return tiers


def tier_counts(tiers: Dict[str, List[ConceptCandidate]]) -> Dict[str, int]:
    return {tier_name: len(members) for tier_name, members in tiers.items()}
