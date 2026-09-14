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

DECISION-XXX: SECOND, INDEPENDENT exclusion condition added --
high-fraction exclusion, UNCONDITIONAL on magnitude.

The re-validation promised above was performed at real scale (n=125
raw candidates, real Qwen auto-interp, real 45-concept manifest v1)
and found a genuine coverage gap in the conjunction rule alone: 16/45
final manifest candidates (36%) turned out to be structural/punctuation/
whitespace artifacts (9 pure <bos>, plus angle-bracket, quote,
newline, and whitespace-dominated features) that were NORMAL magnitude
(ratios 1.14x-3.77x, all well under the 5x conjunction bar) and
therefore correctly, but insufficiently, survived the existing
conjunction rule -- then received inflated correlation-based
auto-interp scores for the exact reason confirmed in
check_structural_outlier_auto_interp.py (a perfectly-predictable
pattern is mathematically easy to correlate, regardless of whether a
judge correctly names it as an artifact).

EVIDENCE for the fix (real, not guessed): computing structural_fraction
for all 45 manifest candidates (via this exact function) and sorting
them revealed a genuine 40-percentage-point gap between 50% (feat_3,
feat_14 -- the original evidence for NOT excluding on fraction alone)
and 90% (feat_77, the lowest of 16 high-fraction candidates) -- by far
the largest gap in the entire distribution (next largest: 20 points).
ALL 16 candidates at or above 90% were confirmed, by inspecting their
actual dominant tokens, to be structural/punctuation/whitespace
artifacts with ZERO exceptions -- not just <bos> (9 candidates), but
also angle brackets/quotes (feat_6, feat_110), whitespace/parens
(feat_15), newlines (feat_36, feat_101), and periods/quotes (feat_76,
feat_77). This is NOT the corpus-domination pattern from Diagnostic 7/8
(that pattern showed varied REAL WORDS from one dominant document
bleeding across features) -- this is a distinct, purely structural
phenomenon, confirmed by direct token inspection, not assumed from
aggregate statistics alone.

LOCKED RULE: exclude a candidate if structural_fraction >= 0.90,
REGARDLESS of magnitude -- independent of, and in addition to, the
original magnitude-AND-fraction conjunction. 90% (not a round number
picked in advance) is the exact fraction of the lowest candidate
sitting just above the confirmed 40-point gap -- same "position the
cut at the edge of the observed gap" methodology already used for the
5x magnitude threshold. feat_3 and feat_14 (50% fraction, the original
evidence against a fraction-only rule) remain safely protected, sitting
40 points below this new threshold.

LIMITATION, same standard as every other threshold here: based on one
real 45-candidate manifest run. Genuine, large, unanimous gap -- not
noise -- but should be re-checked if a substantially different corpus
or candidate pool ever produces a real counterexample (a legitimate
semantic concept with 90%+ token concentration on one non-<bos> token).

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
HIGH_FRACTION_UNCONDITIONAL_THRESHOLD = 0.90

# Boundary convention: BOTH comparisons are strict ">" not ">=".
# This matches the convention already used in the diagnostic code that
# produced the evidence above (structural dominance was defined as
# fraction > 0.5, not >= 0.5; magnitude outliers were checked as
# magnitude > 10x median, not >=). Kept consistent here rather than
# silently switching to >= -- flagged explicitly since boundary
# behavior is exactly the kind of thing that's easy to get backwards
# without noticing. If the researcher intended inclusive (>=)
# boundaries, this should be changed deliberately, not assumed.

# Boundary convention: the ORIGINAL two conditions use strict ">" not
# ">=" (matches the diagnostic code that produced their evidence). The
# NEW high-fraction condition below deliberately uses ">=" instead --
# feat_77, one of the 16 confirmed real artifacts, sits at EXACTLY 90%,
# so a strict ">" would incorrectly fail to exclude it. This is a
# deliberate deviation from the original boundary convention, not an
# oversight -- flagged explicitly per this project's standard that
# boundary behavior must never be assumed or left unexplained.

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
    high_fraction_unconditional_threshold: float = HIGH_FRACTION_UNCONDITIONAL_THRESHOLD,
) -> bool:
    """True if `candidate` should be KEPT. False means it should be
    excluded by this filter, for EITHER of two independent reasons:

    1. The original conjunction: magnitude outlier AND structural-
       dominated (both required together).
    2. NEW: structural_fraction alone at or above
       high_fraction_unconditional_threshold (0.90 by default),
       regardless of magnitude -- added after real n=125-scale
       evidence showed normal-magnitude, near-100%-structural-fraction
       artifacts (pure <bos>, punctuation, newlines) evade condition 1
       entirely. See this module's decision record for the evidence.
    """
    if batch_median_magnitude <= 0:
        raise ValueError(
            f"batch_median_magnitude must be > 0, got {batch_median_magnitude!r}"
        )

    magnitude_ratio = candidate.mean_activation_magnitude / batch_median_magnitude
    structural_fraction = compute_structural_fraction(candidate.top_activating_tokens)

    is_magnitude_outlier = magnitude_ratio > magnitude_ratio_threshold
    is_structural_dominated = structural_fraction > structural_fraction_threshold
    is_high_fraction_unconditional = structural_fraction >= high_fraction_unconditional_threshold

    excluded = (is_magnitude_outlier and is_structural_dominated) or is_high_fraction_unconditional
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
    high_fraction_unconditional_threshold: float = HIGH_FRACTION_UNCONDITIONAL_THRESHOLD,
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
            high_fraction_unconditional_threshold,
        ):
            accepted.append(candidate)
        else:
            magnitude_ratio = candidate.mean_activation_magnitude / batch_median_magnitude
            structural_fraction = compute_structural_fraction(candidate.top_activating_tokens)

            is_magnitude_outlier = magnitude_ratio > magnitude_ratio_threshold
            is_structural_dominated = structural_fraction > structural_fraction_threshold
            is_high_fraction_unconditional = structural_fraction >= high_fraction_unconditional_threshold

            if is_high_fraction_unconditional and not (is_magnitude_outlier and is_structural_dominated):
                reason = (
                    f"structural_token_filter: structural_fraction={structural_fraction:.1%} "
                    f"(unconditional threshold >={high_fraction_unconditional_threshold:.0%}, "
                    f"magnitude_ratio={magnitude_ratio:.2f}x was NOT an outlier -- excluded "
                    f"on fraction alone, per the real n=125 evidence gap)"
                )
            else:
                reason = (
                    f"structural_token_filter: magnitude_ratio={magnitude_ratio:.2f}x "
                    f"(threshold >{magnitude_ratio_threshold}x), "
                    f"structural_fraction={structural_fraction:.1%} "
                    f"(threshold >{structural_fraction_threshold:.0%})"
                )
            rejection_reasons[candidate.feature_id] = reason
            rejected.append(candidate)

    return FilterResult(
        accepted=accepted,
        rejected=rejected,
        rejection_reasons=rejection_reasons,
        batch_median_magnitude=batch_median_magnitude,
    )
