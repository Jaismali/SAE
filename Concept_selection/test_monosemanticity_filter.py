"""
Test Step 4 in isolation: does apply_monosemanticity_filter() correctly
reject the synthetic backend's deliberately-bad (polysemantic) candidates
while keeping the coherent ones, using known ground truth from Step 2?
"""

import random

from monosemanticity_filter import (
    apply_monosemanticity_filter,
    passes_monosemanticity_filter,
    MONOSEMANTICITY_SCORE_THRESHOLD,
)
from synthetic_backend import (
    generate_synthetic_candidates,
    make_high_frequency_candidate,
    make_polysemantic_candidate,
)

SEED = 11


def test_single_polysemantic_candidate_is_rejected():
    rng = random.Random(SEED)
    bad = make_polysemantic_candidate(rng, "bad_1")
    assert passes_monosemanticity_filter(bad) is False
    print(f"PASS: polysemantic candidate (score={bad.auto_interp_score:.3f}) is rejected")


def test_single_coherent_candidate_is_accepted():
    rng = random.Random(SEED)
    good = make_high_frequency_candidate(rng, "good_1")
    assert passes_monosemanticity_filter(good) is True
    print(f"PASS: coherent candidate (score={good.auto_interp_score:.3f}) is accepted")


def test_all_polysemantic_synthetic_candidates_rejected_in_bulk():
    rng = random.Random(SEED)
    bad_candidates = [make_polysemantic_candidate(rng, f"bad_{i}") for i in range(30)]
    result = apply_monosemanticity_filter(bad_candidates)
    assert len(result.rejected) == 30, (
        f"Expected all 30 polysemantic candidates rejected, only {len(result.rejected)} were"
    )
    assert len(result.accepted) == 0
    print("PASS: all 30 polysemantic synthetic candidates rejected, zero false accepts")


def test_all_coherent_synthetic_candidates_accepted_in_bulk():
    rng = random.Random(SEED)
    good_candidates = [make_high_frequency_candidate(rng, f"good_{i}") for i in range(30)]
    result = apply_monosemanticity_filter(good_candidates)
    assert len(result.accepted) == 30, (
        f"Expected all 30 coherent candidates accepted, only {len(result.accepted)} were"
    )
    assert len(result.rejected) == 0
    print("PASS: all 30 coherent synthetic candidates accepted, zero false rejects")


def test_mixed_pool_separates_cleanly():
    pool = generate_synthetic_candidates(
        seed=SEED, num_high=10, num_mid=10, num_sparse=10,
        num_polysemantic=10, num_near_duplicate_pairs=0,
    )
    result = apply_monosemanticity_filter(pool)
    total = len(result.accepted) + len(result.rejected)
    assert total == len(pool), "Every candidate must be either accepted or rejected, none dropped"
    assert len(result.rejected) == 10, (
        f"Expected exactly the 10 polysemantic candidates rejected, got {len(result.rejected)}"
    )
    assert len(result.accepted) == 30, (
        f"Expected exactly the 30 coherent candidates accepted, got {len(result.accepted)}"
    )
    print(f"PASS: mixed pool of {len(pool)} separates cleanly -- "
          f"accepted={len(result.accepted)}, rejected={len(result.rejected)}")


def test_rejection_reasons_are_recorded_for_every_rejected_candidate():
    rng = random.Random(SEED)
    bad_candidates = [make_polysemantic_candidate(rng, f"bad_{i}") for i in range(5)]
    result = apply_monosemanticity_filter(bad_candidates)
    for candidate in result.rejected:
        assert candidate.feature_id in result.rejection_reasons, (
            f"{candidate.feature_id} rejected but has no recorded reason -- "
            f"violates the project's no-silent-drop auditability standard"
        )
        reason = result.rejection_reasons[candidate.feature_id]
        assert "auto_interp_score" in reason and str(MONOSEMANTICITY_SCORE_THRESHOLD) in reason
    print("PASS: every rejected candidate has an auditable, specific rejection reason")


def test_threshold_is_configurable_and_changes_outcome():
    rng = random.Random(SEED)
    borderline = make_high_frequency_candidate(rng, "borderline")
    borderline.auto_interp_score = 0.60

    assert passes_monosemanticity_filter(borderline, threshold=0.5) is True
    assert passes_monosemanticity_filter(borderline, threshold=0.7) is False
    print("PASS: raising the threshold past a candidate's score correctly flips its outcome")


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
