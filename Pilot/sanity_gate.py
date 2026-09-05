"""
Part C Pilot - Step 3b: Induction/correction sanity gate.

Pure decision logic, no model or training involved: given three scalar
scores (baseline, post-induction, post-correction) from a
ConceptStrengthMeasure, decides whether induction actually raised the
concept's presence and whether correction actually lowered it -- per
the brief's requirement that a concept failing to induce/suppress
should be flagged and excluded with a logged reason, not silently
passed downstream.

Margins are required (not just ">"), since a measure this noisy
(behavioral token-frequency on a tiny model) could show a trivial,
meaningless increase/decrease that isn't real signal.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Minimum required change to count as a genuine effect, not noise.
# This is a placeholder tuned for a first pilot run; the right value
# depends on how noisy TokenFrequencyMeasure's scores actually are in
# practice; MUST be reviewed after seeing real pilot score distributions.
DEFAULT_MINIMUM_MEANINGFUL_DELTA = 0.02


class GateOutcome(Enum):
    PASSED = "passed"
    INDUCTION_FAILED = "induction_failed"
    CORRECTION_FAILED = "correction_failed"
    BOTH_FAILED = "both_failed"


@dataclass
class SanityGateResult:
    outcome: GateOutcome
    baseline_score: float
    post_induction_score: float
    post_correction_score: float
    induction_delta: float
    correction_delta: float
    reason: str

    @property
    def passed(self) -> bool:
        return self.outcome == GateOutcome.PASSED


def evaluate_sanity_gate(
    baseline_score: float,
    post_induction_score: float,
    post_correction_score: float,
    minimum_meaningful_delta: float = DEFAULT_MINIMUM_MEANINGFUL_DELTA,
) -> SanityGateResult:
    """Checks whether induction raised the score by at least
    minimum_meaningful_delta, and whether correction lowered it back
    down by at least minimum_meaningful_delta from the post-induction
    peak. Returns a result that is never silently ambiguous -- a
    concept either clearly passes, or fails with a specific, named
    reason attached.
    """
    induction_delta = post_induction_score - baseline_score
    correction_delta = post_induction_score - post_correction_score

    induction_succeeded = induction_delta >= minimum_meaningful_delta
    correction_succeeded = correction_delta >= minimum_meaningful_delta

    outcome, reason = _classify_outcome(induction_succeeded, correction_succeeded, induction_delta, correction_delta)

    return SanityGateResult(
        outcome=outcome,
        baseline_score=baseline_score,
        post_induction_score=post_induction_score,
        post_correction_score=post_correction_score,
        induction_delta=induction_delta,
        correction_delta=correction_delta,
        reason=reason,
    )


def _classify_outcome(
    induction_succeeded: bool,
    correction_succeeded: bool,
    induction_delta: float,
    correction_delta: float,
) -> tuple:
    if induction_succeeded and correction_succeeded:
        return GateOutcome.PASSED, (
            f"Induction raised score by {induction_delta:.4f}; "
            f"correction lowered it by {correction_delta:.4f}. Both exceed threshold."
        )

    if not induction_succeeded and not correction_succeeded:
        return GateOutcome.BOTH_FAILED, (
            f"Induction delta {induction_delta:.4f} did not exceed threshold, and "
            f"correction delta {correction_delta:.4f} did not exceed threshold. "
            f"Concept should be flagged and excluded from downstream analysis."
        )

    if not induction_succeeded:
        return GateOutcome.INDUCTION_FAILED, (
            f"Induction delta {induction_delta:.4f} did not exceed the minimum "
            f"meaningful threshold -- the behavior was not successfully induced, "
            f"so any correction result on top of it is not interpretable."
        )

    return GateOutcome.CORRECTION_FAILED, (
        f"Induction succeeded (delta {induction_delta:.4f}), but correction delta "
        f"{correction_delta:.4f} did not exceed threshold -- the induced behavior "
        f"was not successfully suppressed."
    )
