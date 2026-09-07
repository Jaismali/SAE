"""
Induction-stability gate.

PROJECT CONVENTION (written here since this is the second time the
question came up -- see history in this session's audit trail):

    New sanity/diagnostic checks are built as EXTERNAL, COMPOSABLE
    functions over pipeline outputs BY DEFAULT. Modifying
    pilot_runner.py directly requires an explicit reason why external
    composition doesn't work, stated and signed off -- not just
    convenience. pilot_runner.py is validated Month 1 orchestration
    code; every prior change considered against it has been additive
    risk for no demonstrated need, since nothing built so far has
    needed to run mid-training or access anything run_pilot_for_concept()
    has that a function operating on its output doesn't already get.

This module follows that convention: it takes a concept's ALREADY-
GENERATED post-induction text (produced by the existing pipeline) and
classifies induction-stage stability, without touching pilot_runner.py,
run_pilot_for_concept(), or PilotConceptResult at all.

Motivation: controlled evidence (two explicit, different seeds -- 42
and 7 -- both on ocean_theme, induction=4/correction=1) showed 4/5 and
5/5 post-induction held-out prompts degenerating into verbatim n-gram
repetition, while dog_theme stayed clean across multiple seeds at the
same settings. This is a CONCEPT-level property, not a bad-seed
problem (retrying would very likely just rediscover the same collapse)
-- so this gate EXCLUDES rather than triggers a retry.

Post-correction was clean in both ocean_theme failures observed --
this gate deliberately only classifies the INDUCTION stage, not a
general "is this concept broken" verdict.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from repetition_detector import RepetitionCheckResult, detect_ngram_repetition


@dataclass
class PerPromptStabilityRecord:
    prompt: str
    continuation: str
    repetition: RepetitionCheckResult


@dataclass
class InductionStabilityResult:
    concept_name: str
    per_prompt_records: List[PerPromptStabilityRecord] = field(default_factory=list)
    degenerate_count: int = 0
    total_prompts: int = 0
    exclusion_threshold: int = 0  # minimum degenerate count that triggers exclusion
    is_excluded: bool = False
    reason: str = ""


def _majority_threshold(total_prompts: int) -> int:
    """Minimum degenerate count for a majority of total_prompts.
    For 5 prompts: (5 // 2) + 1 = 3 -- matches the >=3/5 threshold
    the researcher confirmed against the actual 4/5 and 5/5 evidence.
    """
    return (total_prompts // 2) + 1


def evaluate_induction_stability(
    concept_name: str,
    post_induction_generations: List[Tuple[str, str]],
    exclusion_threshold: int = None,
) -> InductionStabilityResult:
    """Classifies a concept's induction-stage stability from its
    ALREADY-GENERATED post-induction (prompt, continuation) pairs --
    the same data inspect_generations.py already produces and prints.

    Stores the FULL per-prompt record (not just the boolean verdict) --
    per explicit instruction: throwing away the raw pattern would
    discard exactly the granularity that made distinguishing "concept
    problem" from "seed problem" possible in the first place.

    exclusion_threshold (optional): minimum number of degenerate
    prompts (out of total) that triggers exclusion. Defaults to a
    strict majority of the prompt count (3 of 5), matching the
    confirmed threshold -- both real failures observed (4/5, 5/5)
    clear this threshold comfortably, so it isn't a close/arbitrary cut
    on the evidence in hand.
    """
    total_prompts = len(post_induction_generations)
    if exclusion_threshold is None:
        exclusion_threshold = _majority_threshold(total_prompts)

    per_prompt_records = []
    degenerate_count = 0
    for prompt, continuation in post_induction_generations:
        repetition = detect_ngram_repetition(continuation)
        per_prompt_records.append(
            PerPromptStabilityRecord(prompt=prompt, continuation=continuation, repetition=repetition)
        )
        if repetition.is_degenerate:
            degenerate_count += 1

    is_excluded = degenerate_count >= exclusion_threshold
    reason = (
        f"induction_stability_gate: {degenerate_count}/{total_prompts} post-induction "
        f"prompts showed degenerate n-gram repetition (threshold: {exclusion_threshold}). "
        f"{'EXCLUDED -- likely a concept-level instability, not a single-seed artifact' if is_excluded else 'within acceptable range'}."
    )

    return InductionStabilityResult(
        concept_name=concept_name,
        per_prompt_records=per_prompt_records,
        degenerate_count=degenerate_count,
        total_prompts=total_prompts,
        exclusion_threshold=exclusion_threshold,
        is_excluded=is_excluded,
        reason=reason,
    )
