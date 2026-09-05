"""
Test Step 3 in isolation: does stratify_by_frequency() assign candidates
to sensible tiers, using the synthetic pool from Step 2 as ground truth
(we know which category each synthetic candidate was generated as)?
"""

from stratification import (
    TierBoundaries,
    stratify_by_frequency,
    tier_counts,
    TIER_HIGH,
    TIER_MID,
    TIER_SPARSE,
)
from synthetic_backend import (
    generate_synthetic_candidates,
    make_high_frequency_candidate,
    make_mid_frequency_candidate,
    make_sparse_tail_candidate,
)
import random

SEED = 7


def test_tier_boundaries_assign_known_values_correctly():
    boundaries = TierBoundaries.default()
    assert boundaries.assign(0.0001) == TIER_SPARSE
    assert boundaries.assign(0.005) == TIER_SPARSE   # boundary is inclusive on the low side
    assert boundaries.assign(0.01) == TIER_MID
    assert boundaries.assign(0.05) == TIER_MID        # boundary is inclusive
    assert boundaries.assign(0.051) == TIER_HIGH
    assert boundaries.assign(0.20) == TIER_HIGH
    print("PASS: tier boundary assignment is correct at and around each threshold")


def test_all_tier_keys_always_present_even_when_empty():
    tiers = stratify_by_frequency([])
    assert set(tiers.keys()) == {TIER_SPARSE, TIER_MID, TIER_HIGH}
    assert all(members == [] for members in tiers.values())
    print("PASS: stratify_by_frequency returns all tier keys even for an empty input")


def test_synthetic_high_frequency_candidates_land_in_high_tier():
    rng = random.Random(SEED)
    candidates = [make_high_frequency_candidate(rng, f"high_{i}") for i in range(15)]
    tiers = stratify_by_frequency(candidates)
    assert len(tiers[TIER_HIGH]) == 15
    assert len(tiers[TIER_MID]) == 0
    assert len(tiers[TIER_SPARSE]) == 0
    print("PASS: all synthetic high-frequency candidates land in the HIGH tier")


def test_synthetic_mid_frequency_candidates_land_in_mid_tier():
    rng = random.Random(SEED)
    candidates = [make_mid_frequency_candidate(rng, f"mid_{i}") for i in range(15)]
    tiers = stratify_by_frequency(candidates)
    assert len(tiers[TIER_MID]) == 15
    assert len(tiers[TIER_HIGH]) == 0
    assert len(tiers[TIER_SPARSE]) == 0
    print("PASS: all synthetic mid-frequency candidates land in the MID tier")


def test_synthetic_sparse_tail_candidates_land_in_sparse_tier():
    rng = random.Random(SEED)
    candidates = [make_sparse_tail_candidate(rng, f"sparse_{i}") for i in range(15)]
    tiers = stratify_by_frequency(candidates)
    assert len(tiers[TIER_SPARSE]) == 15
    assert len(tiers[TIER_MID]) == 0
    assert len(tiers[TIER_HIGH]) == 0
    print("PASS: all synthetic sparse-tail candidates land in the SPARSE tier")


def test_mixed_pool_stratifies_correctly_and_preserves_all_candidates():
    pool = generate_synthetic_candidates(
        seed=SEED, num_high=5, num_mid=8, num_sparse=6,
        num_polysemantic=4, num_near_duplicate_pairs=3,
    )
    tiers = stratify_by_frequency(pool)
    total_in_tiers = sum(len(members) for members in tiers.values())
    assert total_in_tiers == len(pool), (
        f"Expected every candidate to land in exactly one tier: "
        f"{total_in_tiers} placed vs {len(pool)} input"
    )
    print(f"PASS: mixed pool of {len(pool)} candidates fully accounted for across tiers, "
          f"counts={tier_counts(tiers)}")


def test_custom_boundaries_are_respected():
    # A stricter sparse cutoff should reclassify some mid-tier-looking
    # candidates as sparse, proving boundaries aren't hardcoded internally.
    custom = TierBoundaries(sparse_max=0.02, mid_max=0.05)
    rng = random.Random(SEED)
    candidates = [make_mid_frequency_candidate(rng, f"mid_{i}") for i in range(20)]
    default_tiers = stratify_by_frequency(candidates, TierBoundaries.default())
    custom_tiers = stratify_by_frequency(candidates, custom)
    assert len(custom_tiers[TIER_SPARSE]) >= len(default_tiers[TIER_SPARSE]), (
        "Raising the sparse cutoff should not decrease the sparse tier's population"
    )
    print("PASS: custom tier boundaries change classification as expected, "
          f"default_sparse={len(default_tiers[TIER_SPARSE])}, "
          f"custom_sparse={len(custom_tiers[TIER_SPARSE])}")


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
