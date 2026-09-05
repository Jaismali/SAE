"""
Test Step 3b in isolation: does evaluate_sanity_gate() correctly
classify all four combinations of induction/correction success and
failure, using known scalar scores (no model needed)?
"""

from sanity_gate import evaluate_sanity_gate, GateOutcome


def test_both_succeed_passes():
    result = evaluate_sanity_gate(
        baseline_score=0.02, post_induction_score=0.30, post_correction_score=0.05,
        minimum_meaningful_delta=0.02,
    )
    assert result.outcome == GateOutcome.PASSED
    assert result.passed is True
    assert result.induction_delta > 0
    assert result.correction_delta > 0
    print(f"PASS: clear induction+correction success correctly passes the gate "
          f"(reason: {result.reason})")


def test_induction_fails_when_delta_too_small():
    result = evaluate_sanity_gate(
        baseline_score=0.10, post_induction_score=0.105, post_correction_score=0.05,
        minimum_meaningful_delta=0.02,
    )
    assert result.outcome == GateOutcome.INDUCTION_FAILED
    assert result.passed is False
    assert "not successfully induced" in result.reason
    print(f"PASS: tiny induction delta correctly fails as INDUCTION_FAILED "
          f"(reason: {result.reason})")


def test_correction_fails_when_score_does_not_drop_enough():
    result = evaluate_sanity_gate(
        baseline_score=0.02, post_induction_score=0.30, post_correction_score=0.29,
        minimum_meaningful_delta=0.02,
    )
    assert result.outcome == GateOutcome.CORRECTION_FAILED
    assert result.passed is False
    assert "not successfully suppressed" in result.reason
    print(f"PASS: induction succeeding but correction barely moving correctly "
          f"fails as CORRECTION_FAILED (reason: {result.reason})")


def test_both_fail_when_neither_moves_meaningfully():
    result = evaluate_sanity_gate(
        baseline_score=0.10, post_induction_score=0.105, post_correction_score=0.103,
        minimum_meaningful_delta=0.02,
    )
    assert result.outcome == GateOutcome.BOTH_FAILED
    assert result.passed is False
    assert "flagged and excluded" in result.reason
    print(f"PASS: no meaningful movement in either direction correctly fails as "
          f"BOTH_FAILED with an explicit exclusion recommendation (reason: {result.reason})")


def test_correction_measured_relative_to_post_induction_peak_not_baseline():
    # Correction should be judged on how far it pulled the score DOWN from
    # its induced peak, not simply whether it's below the original baseline.
    result = evaluate_sanity_gate(
        baseline_score=0.02, post_induction_score=0.50, post_correction_score=0.10,
        minimum_meaningful_delta=0.02,
    )
    # correction_delta should be 0.50 - 0.10 = 0.40, not 0.10 - 0.02 = 0.08
    assert abs(result.correction_delta - 0.40) < 1e-9, (
        f"Expected correction_delta measured from post-induction peak (0.40), got {result.correction_delta}"
    )
    print(f"PASS: correction_delta is correctly measured relative to the post-induction "
          f"peak ({result.correction_delta:.4f}), not the original baseline")


def test_gate_result_is_never_silently_ambiguous():
    # Every possible outcome must map to exactly one of the four defined
    # GateOutcome values -- there is no "unknown" or null state.
    test_cases = [
        (0.02, 0.30, 0.05),   # both succeed
        (0.10, 0.105, 0.05),  # induction fails
        (0.02, 0.30, 0.29),   # correction fails
        (0.10, 0.105, 0.103), # both fail
    ]
    for baseline, post_induction, post_correction in test_cases:
        result = evaluate_sanity_gate(baseline, post_induction, post_correction)
        assert isinstance(result.outcome, GateOutcome)
        assert result.reason and len(result.reason) > 0, "Every result must have a non-empty reason"
    print("PASS: every tested score combination produces a definite outcome with a non-empty reason")


def test_threshold_is_configurable_and_changes_classification():
    # The same scores should pass with a lenient threshold but fail with
    # a strict one, proving the threshold is a real, adjustable parameter.
    lenient_result = evaluate_sanity_gate(
        baseline_score=0.10, post_induction_score=0.12, post_correction_score=0.10,
        minimum_meaningful_delta=0.01,
    )
    strict_result = evaluate_sanity_gate(
        baseline_score=0.10, post_induction_score=0.12, post_correction_score=0.10,
        minimum_meaningful_delta=0.05,
    )
    assert lenient_result.passed is True
    assert strict_result.passed is False
    print("PASS: minimum_meaningful_delta is a real parameter -- same scores pass under "
          "a lenient threshold and fail under a strict one")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        raise SystemExit(1)
