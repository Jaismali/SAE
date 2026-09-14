"""
Module 3.1 - Concept Selection
Filter pipeline orchestration: wires structural-token, monosemanticity,
and dedup filters together in a locked order.

DECISION-XXX (assign the real sequential ID in
GEOMETRIC_DORMANCY_MASTER_ARCHIVE.md): Filter ordering, locked.

    ORDER: structural-token filter -> monosemanticity filter -> dedup

ORDERING RATIONALE -- UPDATED, CONFIRMED EMPIRICALLY (superseding the
original heuristic-blind-spot reasoning below, kept for history):

The original reasoning (still true, kept for the record) was that the
old auto-interp HEURISTIC scored `1 - unique_token_fraction`, so a
`<bos>`-repeating feature got a HIGH score purely from low lexical
diversity. That heuristic has since been REPLACED by real Qwen-based
auto-interp (see llm_auto_interp_local_client.py), which does not make
that specific arithmetic mistake.

However, running the actual replacement through
check_structural_outlier_auto_interp.py (real confirmed <bos>-only
structural artifacts, feat_4/feat_11) showed the ordering is STILL
required, for a DIFFERENT, now-CONFIRMED (not hypothesized) reason:
Qwen correctly identified both as structural artifacts in its
explanation text ("fires on the beginning-of-sequence token"), and
BOTH STILL SCORED A PERFECT 1.000 CORRELATION. Correlation-based
monosemanticity scoring is mathematically blind to the concept-vs-
artifact distinction regardless of explanation quality or judge
correctness -- a perfectly-predictable pattern is, arithmetically, a
perfectly-scoring one, independent of whether a human or LLM reader
would call it a "real concept." See
PROPOSAL_pipeline_reconciliation.md Section 1 for the full evidence
trail.

Running the structural-token filter FIRST removes likely-artifact
features before either of the other two filters has to reason about
them. This is now a CONFIRMED requirement of the scoring method
itself, not a heuristic-specific workaround -- and it additionally
avoids spending auto-interp compute (explainer+simulator calls) on
candidates a cheap magnitude/token check already flags as likely
artifacts. Dedup runs last regardless, so its kept representatives are
chosen from an already-vetted pool.

IF THE SCORING METHOD CHANGES AGAIN (e.g. a different auto-interp
approach not based on activation-correlation), THIS ORDERING SHOULD BE
RE-EXAMINED AGAIN -- the confirmed mechanism above is specific to
correlation-based scoring, not a claim that holds for any possible
future auto-interp method.
"""

from dataclasses import dataclass
from typing import List, Optional

from contract import ConceptCandidate
from deduplication import DeduplicationResult, deduplicate_by_decoder_vector
from deduplication import DUPLICATE_COSINE_SIMILARITY_THRESHOLD
from monosemanticity_filter import FilterResult as MonosemanticityResult
from monosemanticity_filter import apply_monosemanticity_filter
from monosemanticity_filter import MONOSEMANTICITY_SCORE_THRESHOLD
from structural_token_filter import FilterResult as StructuralFilterResult
from structural_token_filter import apply_structural_token_filter
from structural_token_filter import MAGNITUDE_RATIO_THRESHOLD, STRUCTURAL_TOKEN_FRACTION_THRESHOLD


@dataclass
class PipelineResult:
    """Every stage's output kept separately and traceably -- a
    candidate's fate must be attributable to a specific stage, never
    collapsed into one undifferentiated 'excluded' bucket.
    """
    final_kept: List[ConceptCandidate]
    structural_filter_result: StructuralFilterResult
    monosemanticity_result: MonosemanticityResult
    dedup_result: DeduplicationResult


def run_concept_filter_pipeline(
    candidates: List[ConceptCandidate],
    monosemanticity_threshold: float = MONOSEMANTICITY_SCORE_THRESHOLD,
    dedup_threshold: float = DUPLICATE_COSINE_SIMILARITY_THRESHOLD,
    magnitude_ratio_threshold: float = MAGNITUDE_RATIO_THRESHOLD,
    structural_fraction_threshold: float = STRUCTURAL_TOKEN_FRACTION_THRESHOLD,
) -> PipelineResult:
    """Runs the locked filter order: structural-token -> monosemanticity -> dedup.

    Each stage only receives the survivors of the previous stage. This
    is a deliberate choice, not an accident of composition: a candidate
    excluded by the structural filter should never be evaluated by
    monosemanticity or dedup at all, so its exclusion reason stays
    unambiguous.

    ALL THRESHOLDS ARE NOW OVERRIDABLE (added after a real integration
    run needed to work around this function's previous hardcoded
    defaults -- see run_end_to_end_integration_check.py, which had to
    call the three filter functions directly instead of this wrapper
    because no override parameter existed). Defaults still match each
    filter module's own module-level constant, so existing callers that
    don't pass overrides see IDENTICAL behavior to before this change.

    monosemanticity_threshold default is STILL 0.5 (Month 1's
    synthetic-data value) unless explicitly overridden -- the real,
    evidence-based value (0.3585, see
    DECISION_monosemanticity_threshold.md) must be passed explicitly by
    the caller. This function does NOT default to 0.3585 itself, since
    that value is flagged as pending re-validation at full target scale
    and should not be silently baked in as if it were a stable default.
    """
    structural_filter_result = apply_structural_token_filter(
        candidates,
        magnitude_ratio_threshold=magnitude_ratio_threshold,
        structural_fraction_threshold=structural_fraction_threshold,
    )

    monosemanticity_result = apply_monosemanticity_filter(
        structural_filter_result.accepted,
        threshold=monosemanticity_threshold,
    )

    dedup_result = deduplicate_by_decoder_vector(
        monosemanticity_result.accepted,
        threshold=dedup_threshold,
    )

    return PipelineResult(
        final_kept=dedup_result.kept,
        structural_filter_result=structural_filter_result,
        monosemanticity_result=monosemanticity_result,
        dedup_result=dedup_result,
    )
