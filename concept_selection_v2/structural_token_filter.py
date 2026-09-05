"""
Module 3.1 - Concept Selection
Structural-token / magnitude-outlier filter.

DECISION-XXX (see GEOMETRIC_DORMANCY_MASTER_ARCHIVE.md -- assign the
real sequential ID there; this file doesn't know the archive's current
numbering). Locked criterion, evidence, and limitation, verbatim:

  Exclude a candidate SAE feature if BOTH hold:
    (a) mean_activation_magnitude > 5x the batch's median nonzero
        mean_activation_magnitude
    (b) >50% of its top-10 activating tokens are structural/special
        tokens (the SAME "dominated" definition already used
        elsewhere -- not a separate bespoke fraction)

  Evidence (n=20 pilot sample, 40 documents, layer 13): 5/20
  candidates were structural-token-dominated. Magnitude ratios to
  batch median: 0.44x, 0.70x, 1.58x -- [real gap] -- 15.85x, 19.02x.
  Structural fractions: 90%, 60%, 60%, 100%, 100%. feat_3 (90%
  structural fraction, 0.44x median magnitude) demonstrates that high
  structural-token fraction does NOT imply magnitude-outlier status --
  this is why the conjunction (not fraction alone) is used, and why a
  fraction-only rule would have wrongly excluded feat_3. Both actual
  magnitude outliers were at 100% structural fraction, supporting
  reuse of the existing 50%-dominance definition instead of a second
  bespoke fraction threshold. 5x (not 8x) was chosen because it sits
  at the edge of the observed gap (1.58x-15.85x) -- the more literal
  reading of where the gap actually sits -- rather than padding
  further from what was observed for extra conservatism, which
  nothing in the evidence indicated was needed.

  LIMITATION: based on n=5 structural-dominated candidates out of a
  20-feature pilot sample. A real, observed gap, not noise -- but a
  small sample. MUST be re-validated against the real distribution
  once the full 40-50 concept manifest is run, same as the
  monosemanticity (0.5) and dedup (0.95 cosine) thresholds. Do not
  treat as settled before that recheck.

This is a DISTINCT, independently-logged filter from the
monosemanticity pre-filter (auto_interp_score threshold 0.5) and the
dedup filter (decoder-vector cosine threshold 0.95). A candidate
excluded here is excluded for a different reason (likely attention-
sink/structural artifact) than a candidate excluded by those two.
Downstream code and the manifest must record WHICH filter excluded a
given candidate, not collapse everything into one "excluded" bucket.

ASSUMPTION CONFIRMED / RECONCILED against the real monosemanticity_filter.py
and deduplication.py (now reviewed, not guessed at):
  - "passes_*_filter() returns True for KEPT candidates" was correct.
  - Function naming was NOT correct: Month 1 embeds the filter name in
    each function (passes_monosemanticity_filter, deduplicate_by_decoder_vector),
    not a generic passes_filter/apply_filter. Renamed below to
    passes_structural_token_filter / apply_structural_token_filter to
    avoid a name collision once selection_pipeline.py imports all three
    filter modules together.
  - Reasons field naming was NOT correct: monosemanticity_filter.py (the
    closer analog -- both are single-candidate threshold filters, unlike
    dedup's pairwise-clustering shape) uses `rejection_reasons`, not
    `reasons`. Renamed to match.
  - One GENUINE, UNAVOIDABLE difference, not a mismatch to fix: this
    filter's criterion is population-relative (needs the batch median),
    so passes_structural_token_filter() takes an extra
    batch_median_magnitude argument that monosemanticity's and dedup's
    per-candidate/pairwise functions don't need. This is inherent to
    what the filter measures, not an inconsistency to resolve.
"""

import statistics
from dataclasses import dataclass
from typing import List

from contract import ConceptCandidate

# --- Locked thresholds (see decision record above) ---
MAGNITUDE_RATIO_THRESHOLD = 5.0
STRUCTURAL_TOKEN_FRACTION_THRESHOLD = 0.5

# Boundary convention: BOTH comparisons are strict ">" not ">=".
# This matches the convention already used in the diagnostic code that
# produced the evidence above (structural dominance was defined as
# fraction > 0.5, not >= 0.5; magnitude outliers were checked as
# magnitude > 10x median, not >=). Kept consistent here rather than
# silently switching to >= -- flagged explicitly since boundary
# behavior is exactly the kind of thing that's easy to get backwards
# without noticing. If the researcher intended inclusive (>=)
# boundaries, this should be changed deliberately, not assumed.

SPECIAL_TOKENS = {"<bos>", "<eos>", "<pad>", "<unk>"}


def _is_structural_token(token: str) -> bool:
    """A token counts as structural if it's a special/control token or
    is made up entirely of punctuation/whitespace once stripped.
    Mirrors the definition used in the diagnostic smoke test that
    produced the evidence for this filter's thresholds.
    """
    stripped = token.strip()
    if stripped in SPECIAL_TOKENS:
        return True
    if stripped == "":
        return True
    return all(not ch.isalnum() for ch in stripped)


def compute_structural_fraction(top_activating_tokens: List[str]) -> float:
    """Fraction of a candidate's top-activating tokens that are
    structural/special tokens. Returns 0.0 for an empty token list
    rather than raising or returning NaN.
    """
    if not top_activating_tokens:
        return 0.0
    structural_count = sum(1 for t in top_activating_tokens if _is_structural_token(t))
    return structural_count / len(top_activating_tokens)


def compute_batch_median_magnitude(candidates: List[ConceptCandidate]) -> float:
    """Median mean_activation_magnitude across a batch, restricted to
    candidates with nonzero magnitude (a candidate that never fires
    shouldn't pull the median toward zero and distort every other
    candidate's ratio).

    Raises:
        ValueError: if no candidate in the batch has nonzero magnitude.
            This must fail loudly -- silently returning 0.0 would make
            every ratio computation divide by zero or produce nonsense
            "infinite" ratios downstream.
    """
    nonzero_magnitudes = [c.mean_activation_magnitude for c in candidates if c.mean_activation_magnitude > 0]
    if not nonzero_magnitudes:
        raise ValueError(
            "Cannot compute batch median magnitude: no candidate in this "
            "batch has nonzero mean_activation_magnitude."
        )
    return statistics.median(nonzero_magnitudes)


def passes_structural_token_filter(
    candidate: ConceptCandidate,
    batch_median_magnitude: float,
    magnitude_ratio_threshold: float = MAGNITUDE_RATIO_THRESHOLD,
    structural_fraction_threshold: float = STRUCTURAL_TOKEN_FRACTION_THRESHOLD,
) -> bool:
    """True if `candidate` should be KEPT (i.e. does NOT match the
    exclusion conjunction). False means it should be excluded by this
    filter specifically.
    """
    if batch_median_magnitude <= 0:
        raise ValueError(
            f"batch_median_magnitude must be > 0, got {batch_median_magnitude!r}"
        )

    magnitude_ratio = candidate.mean_activation_magnitude / batch_median_magnitude
    structural_fraction = compute_structural_fraction(candidate.top_activating_tokens)

    is_magnitude_outlier = magnitude_ratio > magnitude_ratio_threshold
    is_structural_dominated = structural_fraction > structural_fraction_threshold

    excluded = is_magnitude_outlier and is_structural_dominated
    return not excluded


@dataclass
class FilterResult:
    accepted: List[ConceptCandidate]
    rejected: List[ConceptCandidate]
    rejection_reasons: dict  # feature_id -> str, populated only for rejected candidates
    batch_median_magnitude: float


def apply_structural_token_filter(
    candidates: List[ConceptCandidate],
    magnitude_ratio_threshold: float = MAGNITUDE_RATIO_THRESHOLD,
    structural_fraction_threshold: float = STRUCTURAL_TOKEN_FRACTION_THRESHOLD,
) -> FilterResult:
    """Bulk filter application across a batch of candidates.

    The batch median is computed once, from this exact batch, and
    reused for every candidate's ratio -- consistent with how the
    threshold's supporting evidence was itself measured (median over a
    pilot batch, not some externally fixed reference magnitude).
    """
    batch_median_magnitude = compute_batch_median_magnitude(candidates)

    accepted: List[ConceptCandidate] = []
    rejected: List[ConceptCandidate] = []
    rejection_reasons: dict = {}

    for candidate in candidates:
        if passes_structural_token_filter(
            candidate,
            batch_median_magnitude,
            magnitude_ratio_threshold,
            structural_fraction_threshold,
        ):
            accepted.append(candidate)
        else:
            magnitude_ratio = candidate.mean_activation_magnitude / batch_median_magnitude
            structural_fraction = compute_structural_fraction(candidate.top_activating_tokens)
            rejection_reasons[candidate.feature_id] = (
                f"structural_token_filter: magnitude_ratio={magnitude_ratio:.2f}x "
                f"(threshold >{magnitude_ratio_threshold}x), "
                f"structural_fraction={structural_fraction:.1%} "
                f"(threshold >{structural_fraction_threshold:.0%})"
            )
            rejected.append(candidate)

    return FilterResult(
        accepted=accepted,
        rejected=rejected,
        rejection_reasons=rejection_reasons,
        batch_median_magnitude=batch_median_magnitude,
    )
