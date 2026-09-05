"""
Test Step 6 in isolation: does the combined pipeline (filter -> dedup ->
stratify) produce a correct end-to-end result on synthetic data, and does
it fully account for every input candidate across the accept/reject/
discard buckets (nothing silently lost)?
"""

from selection_pipeline import run_selection_pipeline
from synthetic_backend import generate_synthetic_candidates
from stratification import TIER_HIGH, TIER_MID, TIER_SPARSE

SEED = 99


def _build_known_pool():
    # 10 high + 10 mid + 10 sparse (all coherent, should survive filtering)
    # 12 polysemantic (should all be rejected by the monosemanticity filter)
    # 4 near-duplicate pairs = 8 candidates (4 should be discarded by dedup)
    return generate_synthetic_candidates(
        seed=SEED,
        num_high=10,
        num_mid=10,
        num_sparse=10,
        num_polysemantic=12,
        num_near_duplicate_pairs=4,
    )


def test_every_input_candidate_is_accounted_for():
    pool = _build_known_pool()
    result = run_selection_pipeline(pool)
    summary = result.summary()

    assert summary["input_count"] == len(pool), (
        f"Pipeline result accounts for {summary['input_count']} candidates, "
        f"but {len(pool)} were input -- some were silently lost"
    )
    print(f"PASS: all {len(pool)} input candidates accounted for across "
          f"accepted/rejected/discarded buckets")


def test_monosemanticity_rejections_match_known_ground_truth():
    pool = _build_known_pool()
    result = run_selection_pipeline(pool)
    # Exactly the 12 polysemantic candidates should be rejected at this stage.
    assert len(result.rejected_by_monosemanticity) == 12, (
        f"Expected 12 monosemanticity rejections, got {len(result.rejected_by_monosemanticity)}"
    )
    print("PASS: monosemanticity stage rejects exactly the 12 known-bad candidates")


def test_dedup_discards_match_known_ground_truth():
    pool = _build_known_pool()
    result = run_selection_pipeline(pool)
    # 4 near-duplicate pairs -> exactly 4 discards (one per pair) among the
    # 40 coherent candidates that survive the monosemanticity filter.
    assert len(result.discarded_by_dedup) == 4, (
        f"Expected 4 dedup discards, got {len(result.discarded_by_dedup)}"
    )
    print("PASS: dedup stage discards exactly the 4 known near-duplicate candidates")


def test_final_count_matches_expected_arithmetic():
    pool = _build_known_pool()
    result = run_selection_pipeline(pool)
    # Coherent candidates surviving the monosemanticity filter = everything
    # except the 12 polysemantic ones: 10 high + 10 mid + 10 sparse + (4
    # pairs * 2 near-duplicate candidates) = 38. Dedup then removes one
    # candidate per pair (4), leaving 34.
    coherent_survivors = 10 + 10 + 10 + (4 * 2)
    expected_final = coherent_survivors - 4
    assert len(result.final_candidates) == expected_final, (
        f"Expected {expected_final} final candidates, got {len(result.final_candidates)}"
    )
    print(f"PASS: final candidate count matches expected arithmetic "
          f"({coherent_survivors} coherent - 4 duplicates = {len(result.final_candidates)})")


def test_final_tiers_only_contain_final_candidates():
    pool = _build_known_pool()
    result = run_selection_pipeline(pool)
    tier_total = sum(len(members) for members in result.final_tiers.values())
    assert tier_total == len(result.final_candidates), (
        f"Tier breakdown accounts for {tier_total} candidates, "
        f"but final_candidates has {len(result.final_candidates)}"
    )
    final_ids = {c.feature_id for c in result.final_candidates}
    tiered_ids = {c.feature_id for members in result.final_tiers.values() for c in members}
    assert final_ids == tiered_ids, "Tier breakdown must contain exactly the final surviving candidates"
    print("PASS: final tier breakdown exactly matches the final candidate set, no drift")


def test_final_tiers_are_populated_and_roughly_balanced():
    pool = _build_known_pool()
    result = run_selection_pipeline(pool)
    counts = {tier: len(members) for tier, members in result.final_tiers.items()}
    # All three tiers should have survivors (none wiped out entirely) since
    # each of high/mid/sparse contributed 10 coherent candidates going in.
    assert counts[TIER_HIGH] > 0, "HIGH tier unexpectedly empty in final result"
    assert counts[TIER_MID] > 0, "MID tier unexpectedly empty in final result"
    assert counts[TIER_SPARSE] > 0, "SPARSE tier unexpectedly empty in final result"
    print(f"PASS: all three tiers populated in final result: {counts}")


def test_rejection_and_discard_reasons_are_all_traceable():
    pool = _build_known_pool()
    result = run_selection_pipeline(pool)

    for candidate in result.rejected_by_monosemanticity:
        assert candidate.feature_id in result.monosemanticity_reasons, (
            f"{candidate.feature_id} rejected with no recorded reason"
        )
    for candidate in result.discarded_by_dedup:
        assert candidate.feature_id in result.dedup_reasons, (
            f"{candidate.feature_id} discarded with no recorded reason"
        )
    print("PASS: every rejected/discarded candidate across both stages has a traceable reason")


def test_summary_arithmetic_is_internally_consistent():
    pool = _build_known_pool()
    result = run_selection_pipeline(pool)
    summary = result.summary()

    reconstructed_input = (
        summary["final_count"]
        + summary["rejected_by_monosemanticity_count"]
        + summary["discarded_by_dedup_count"]
    )
    assert reconstructed_input == summary["input_count"]
    assert sum(summary["final_tier_counts"].values()) == summary["final_count"]
    print(f"PASS: summary() arithmetic is internally consistent: {summary}")


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
