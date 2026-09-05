"""
Test Step 2 in isolation: does the synthetic generator produce the
intended variety (high/mid/sparse frequency tiers, deliberately-bad
monosemanticity examples, deliberate near-duplicate pairs), and are
all produced records contract-valid?
"""

import math

from contract import ConceptCandidate
from synthetic_backend import (
    generate_synthetic_candidates,
    HIGH_FREQ_RANGE,
    MID_FREQ_RANGE,
    SPARSE_FREQ_RANGE,
)

SEED = 42


def _generate_default_pool():
    return generate_synthetic_candidates(
        seed=SEED,
        num_high=5,
        num_mid=8,
        num_sparse=6,
        num_polysemantic=4,
        num_near_duplicate_pairs=3,
    )


def test_pool_size_matches_requested_counts():
    pool = _generate_default_pool()
    # 5 + 8 + 6 + 4 + (3 pairs * 2) = 29
    assert len(pool) == 29, f"Expected 29 candidates, got {len(pool)}"
    print("PASS: pool size matches requested composition")


def test_all_records_are_contract_valid():
    pool = _generate_default_pool()
    all_errors = {c.feature_id: c.validate() for c in pool}
    bad = {fid: errs for fid, errs in all_errors.items() if errs}
    assert not bad, f"Malformed synthetic records found: {bad}"
    print(f"PASS: all {len(pool)} synthetic records are contract-valid")


def test_feature_ids_are_unique():
    pool = _generate_default_pool()
    ids = [c.feature_id for c in pool]
    assert len(ids) == len(set(ids)), "Duplicate feature_id values found"
    print("PASS: all feature_id values are unique")


def test_high_frequency_examples_are_in_range():
    pool = generate_synthetic_candidates(
        seed=SEED, num_high=20, num_mid=0, num_sparse=0,
        num_polysemantic=0, num_near_duplicate_pairs=0,
    )
    assert all(HIGH_FREQ_RANGE[0] <= c.activation_frequency <= HIGH_FREQ_RANGE[1] for c in pool)
    print("PASS: high-frequency candidates fall within HIGH_FREQ_RANGE")


def test_sparse_tail_examples_are_in_range_and_below_mid():
    pool = generate_synthetic_candidates(
        seed=SEED, num_high=0, num_mid=0, num_sparse=20,
        num_polysemantic=0, num_near_duplicate_pairs=0,
    )
    assert all(SPARSE_FREQ_RANGE[0] <= c.activation_frequency <= SPARSE_FREQ_RANGE[1] for c in pool)
    assert all(c.activation_frequency < MID_FREQ_RANGE[0] for c in pool), (
        "Sparse-tail frequencies should sit clearly below the mid tier's floor"
    )
    print("PASS: sparse-tail candidates are in range and clearly below mid-tier frequencies")


def test_polysemantic_examples_have_low_interp_score():
    pool = generate_synthetic_candidates(
        seed=SEED, num_high=0, num_mid=0, num_sparse=0,
        num_polysemantic=15, num_near_duplicate_pairs=0,
    )
    # These should score well below the coherent-theme candidates' floor (0.65+),
    # so a monosemanticity filter with a sane threshold can separate the two groups.
    assert all(c.auto_interp_score < 0.5 for c in pool), (
        "Polysemantic synthetic candidates should score low enough to be filterable"
    )
    print("PASS: polysemantic candidates score low enough to be distinguishable from good ones")


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def test_near_duplicate_pairs_have_high_cosine_similarity():
    pool = generate_synthetic_candidates(
        seed=SEED, num_high=0, num_mid=0, num_sparse=0,
        num_polysemantic=0, num_near_duplicate_pairs=5,
    )
    # Pairs are generated back-to-back: (0,1), (2,3), (4,5), ...
    assert len(pool) == 10
    similarities = []
    for i in range(0, len(pool), 2):
        sim = _cosine_similarity(pool[i].decoder_vector, pool[i + 1].decoder_vector)
        similarities.append(sim)
    assert all(sim > 0.95 for sim in similarities), (
        f"Expected near-duplicate pairs to have cosine similarity > 0.95, got {similarities}"
    )
    print(f"PASS: all 5 near-duplicate pairs have cosine similarity > 0.95 (min={min(similarities):.4f})")


def test_non_duplicate_candidates_are_not_accidentally_similar():
    """Sanity check the negative case: independently-generated candidates
    should NOT usually look like near-duplicates, or the dedup test later
    would be meaningless (everything would look like a duplicate)."""
    pool = generate_synthetic_candidates(
        seed=SEED, num_high=10, num_mid=10, num_sparse=10,
        num_polysemantic=10, num_near_duplicate_pairs=0,
    )
    high_similarity_pairs = 0
    total_pairs = 0
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            total_pairs += 1
            sim = _cosine_similarity(pool[i].decoder_vector, pool[j].decoder_vector)
            if sim > 0.95:
                high_similarity_pairs += 1
    assert high_similarity_pairs == 0, (
        f"{high_similarity_pairs}/{total_pairs} independently-generated pairs "
        f"look like accidental near-duplicates -- generator's randomness is too narrow"
    )
    print(f"PASS: 0/{total_pairs} independently-generated candidates are accidentally near-duplicate")


def test_reproducible_given_same_seed():
    pool_a = _generate_default_pool()
    pool_b = _generate_default_pool()
    ids_a = [c.feature_id for c in pool_a]
    ids_b = [c.feature_id for c in pool_b]
    freqs_a = [round(c.activation_frequency, 6) for c in pool_a]
    freqs_b = [round(c.activation_frequency, 6) for c in pool_b]
    assert ids_a == ids_b and freqs_a == freqs_b, "Same seed should produce identical output"
    print("PASS: generator is reproducible given the same seed")


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
