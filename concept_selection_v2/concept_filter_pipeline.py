"""
Module 3.1 - Concept Selection
Filter pipeline orchestration: wires structural-token, monosemanticity,
and dedup filters together in a locked order.

DECISION-XXX (assign the real sequential ID in
GEOMETRIC_DORMANCY_MASTER_ARCHIVE.md): Filter ordering, locked.

    ORDER: structural-token filter -> monosemanticity filter -> dedup

ORDERING-DEPENDENT-ON: heuristic auto-interp placeholder in
real_backend.py (`_heuristic_auto_interp`). This tag exists so this
decision is easy to find and re-examine later -- search for this exact
string when real Neuronpedia auto-interp text eventually replaces the
heuristic.

WHY THIS ORDER, SPECIFICALLY (not a general principle -- a mechanism
tied to the CURRENT heuristic):

The current auto-interp heuristic scores monosemanticity as
`1 - (unique_token_count / total_top_tokens)` -- i.e. LOW token
diversity in a feature's top-activating tokens produces a HIGH
heuristic score. An attention-sink-style feature that fires almost
entirely on a single structural/special token (e.g. <bos>, repeated)
has very LOW token diversity, and therefore would be scored as HIGHLY
monosemantic by this heuristic -- even though it is very likely an
artifact, not a real concept.

This means: if monosemanticity filtering ran BEFORE structural-token
filtering, attention-sink artifacts would likely survive the
monosemanticity threshold (falsely rated "coherent") and reach dedup,
where one could end up as the kept representative of a duplicate
cluster simply by appearing first in the candidate list.

Running the structural-token filter FIRST removes likely-artifact
features before either of the other two filters has to reason about
them, specifically closing this heuristic-created blind spot. Dedup
runs last regardless, so its kept representatives are chosen from an
already-vetted pool.

IF THE HEURISTIC CHANGES (e.g. replaced by real Neuronpedia
auto-interp text), THIS ORDERING MUST BE RE-EXAMINED, not silently
carried forward -- the mechanism justifying "structural before
monosemanticity" may no longer hold once auto-interp is measured a
different way.
"""

from dataclasses import dataclass
from typing import List

from contract import ConceptCandidate
from deduplication import DeduplicationResult, deduplicate_by_decoder_vector
from monosemanticity_filter import FilterResult as MonosemanticityResult
from monosemanticity_filter import apply_monosemanticity_filter
from structural_token_filter import FilterResult as StructuralFilterResult
from structural_token_filter import apply_structural_token_filter


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


def run_concept_filter_pipeline(candidates: List[ConceptCandidate]) -> PipelineResult:
    """Runs the locked filter order: structural-token -> monosemanticity -> dedup.

    Each stage only receives the survivors of the previous stage. This
    is a deliberate choice, not an accident of composition: a candidate
    excluded by the structural filter should never be evaluated by
    monosemanticity or dedup at all, so its exclusion reason stays
    unambiguous.
    """
    structural_filter_result = apply_structural_token_filter(candidates)

    monosemanticity_result = apply_monosemanticity_filter(
        structural_filter_result.accepted
    )

    dedup_result = deduplicate_by_decoder_vector(monosemanticity_result.accepted)

    return PipelineResult(
        final_kept=dedup_result.kept,
        structural_filter_result=structural_filter_result,
        monosemanticity_result=monosemanticity_result,
        dedup_result=dedup_result,
    )
