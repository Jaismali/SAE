"""
Test Step 5 in isolation: does deduplicate_by_decoder_vector() catch the
synthetic backend's deliberate near-duplicate pairs, while leaving
genuinely distinct candidates untouched?
"""

import random

from deduplication import (
    deduplicate_by_decoder_vector,
    cosine_similarity,
    DUPLICATE_COSINE_SIMILARITY_THRESHOLD,
)
from synthetic_backend import (
    generate_synthetic_candidates,
    make_near_duplicate_pair,
    make_high_frequency_candidate,
)

SEED = 23


def test_cosine_similarity_of_identical_vector_is_one():
    vector = [0.3, -0.2, 0.5, 0.1]
    assert abs(cosine_similarity(vector, vector) - 1.0) < 1e-9
    print("PASS: cosine similarity of a vector with itself is 1.0")


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9
    print("PASS: cosine similarity of orthogonal vectors is 0.0")


def test_single_near_duplicate_pair_is_caught():
    rng = random.Random(SEED)
    original, duplicate = make_near_duplicate_pair(rng, "orig_1", "dup_1")
    result = deduplicate_by_decoder_vector([original, duplicate])

    assert len(result.kept) == 1, f"Expected exactly 1 kept, got {len(result.kept)}"
    assert len(result.discarded) == 1, f"Expected exactly 1 discarded, got {len(result.discarded)}"
    assert result.kept[0].feature_id == "orig_1", "Expected the first-seen candidate to be kept"
    assert result.discarded[0].feature_id == "dup_1"
    print("PASS: a single near-duplicate pair is correctly reduced to one kept candidate")


def test_multiple_near_duplicate_pairs_each_reduced_to_one():
    rng = random.Random(SEED)
    pairs = [make_near_duplicate_pair(rng, f"orig_{i}", f"dup_{i}") for i in range(6)]
    candidates = [c for pair in pairs for c in pair]  # flatten, preserving pair adjacency

    result = deduplicate_by_decoder_vector(candidates)

    assert len(result.kept) == 6, f"Expected 6 kept (one per pair), got {len(result.kept)}"
    assert len(result.discarded) == 6, f"Expected 6 discarded (one per pair), got {len(result.discarded)}"
    kept_ids = {c.feature_id for c in result.kept}
    assert kept_ids == {f"orig_{i}" for i in range(6)}, (
        f"Expected exactly the 'orig_*' candidates kept, got {kept_ids}"
    )
    print("PASS: 6 independent near-duplicate pairs each correctly reduced to their original")


def test_distinct_candidates_are_never_falsely_flagged():
    rng = random.Random(SEED)
    distinct_candidates = [make_high_frequency_candidate(rng, f"distinct_{i}") for i in range(40)]
    result = deduplicate_by_decoder_vector(distinct_candidates)

    assert len(result.discarded) == 0, (
        f"Expected zero false-positive discards among genuinely distinct candidates, "
        f"got {len(result.discarded)}: {[c.feature_id for c in result.discarded]}"
    )
    assert len(result.kept) == 40
    print("PASS: 0/40 genuinely distinct candidates were falsely flagged as duplicates")


def test_mixed_pool_dedup_matches_known_ground_truth():
    # Step 2's generator produces exactly num_near_duplicate_pairs pairs;
    # everything else should be independently random and thus distinct.
    pool = generate_synthetic_candidates(
        seed=SEED, num_high=10, num_mid=10, num_sparse=10,
        num_polysemantic=10, num_near_duplicate_pairs=5,
    )
    result = deduplicate_by_decoder_vector(pool)

    assert len(result.discarded) == 5, (
        f"Expected exactly 5 discards (one per near-duplicate pair), got {len(result.discarded)}"
    )
    assert len(result.kept) == len(pool) - 5
    print(f"PASS: mixed pool of {len(pool)} correctly reduces by exactly 5 "
          f"(the number of injected near-duplicate pairs)")


def test_discard_reasons_are_recorded_and_reference_the_kept_feature():
    rng = random.Random(SEED)
    original, duplicate = make_near_duplicate_pair(rng, "orig_x", "dup_x")
    result = deduplicate_by_decoder_vector([original, duplicate])

    assert "dup_x" in result.discard_reasons
    reason = result.discard_reasons["dup_x"]
    assert "orig_x" in reason, f"Discard reason should reference which kept feature it duplicated: {reason}"
    print("PASS: discard reason is recorded and traceable to the specific kept feature it duplicated")


def test_threshold_is_configurable():
    rng = random.Random(SEED)
    original, duplicate = make_near_duplicate_pair(rng, "orig_y", "dup_y")
    actual_similarity = cosine_similarity(original.decoder_vector, duplicate.decoder_vector)

    # A threshold just above the actual similarity should let both through.
    lenient_threshold = min(actual_similarity + 0.001, 1.0)
    result = deduplicate_by_decoder_vector([original, duplicate], threshold=lenient_threshold)
    assert len(result.kept) == 2, (
        f"With threshold {lenient_threshold:.6f} > actual similarity {actual_similarity:.6f}, "
        f"both candidates should be kept"
    )
    print(f"PASS: raising the threshold above the pair's actual similarity "
          f"({actual_similarity:.6f}) correctly stops it from being flagged as duplicate")


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
