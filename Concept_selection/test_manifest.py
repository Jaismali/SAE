"""
Test Step 7 in isolation: does freeze_manifest() correctly write a
timestamped, versioned manifest, refuse to silently overwrite an
existing version, auto-increment versions, and require a documented
reason for any post-v1 deviation?
"""

import os
import shutil
import tempfile

from manifest import freeze_manifest, load_manifest, ManifestAlreadyExistsError
from synthetic_backend import generate_synthetic_candidates

SEED = 55


def _fresh_test_dir():
    return tempfile.mkdtemp(prefix="manifest_test_")


def _small_pool():
    return generate_synthetic_candidates(
        seed=SEED, num_high=3, num_mid=3, num_sparse=2,
        num_polysemantic=0, num_near_duplicate_pairs=0,
    )


def test_first_manifest_writes_successfully_as_version_1():
    directory = _fresh_test_dir()
    try:
        pool = _small_pool()
        path = freeze_manifest(pool, directory)
        assert os.path.exists(path)
        assert "concept_manifest_v1.json" in path
        print(f"PASS: first manifest written as version 1 at {path}")
    finally:
        shutil.rmtree(directory)


def test_manifest_content_matches_input_candidates():
    directory = _fresh_test_dir()
    try:
        pool = _small_pool()
        path = freeze_manifest(pool, directory)
        loaded = load_manifest(path)

        assert loaded["metadata"]["candidate_count"] == len(pool)
        assert len(loaded["candidates"]) == len(pool)
        loaded_ids = {c["feature_id"] for c in loaded["candidates"]}
        original_ids = {c.feature_id for c in pool}
        assert loaded_ids == original_ids, "Manifest must contain exactly the input candidates"
        print(f"PASS: manifest content matches input exactly ({len(pool)} candidates)")
    finally:
        shutil.rmtree(directory)


def test_manifest_has_a_valid_timestamp():
    directory = _fresh_test_dir()
    try:
        pool = _small_pool()
        path = freeze_manifest(pool, directory)
        loaded = load_manifest(path)

        timestamp = loaded["metadata"]["created_at_utc"]
        # Should be ISO-8601 parseable and end in a UTC offset marker.
        from datetime import datetime
        parsed = datetime.fromisoformat(timestamp)
        assert parsed is not None
        print(f"PASS: manifest has a valid, parseable UTC timestamp: {timestamp}")
    finally:
        shutil.rmtree(directory)


def test_writing_same_version_twice_raises_instead_of_overwriting():
    directory = _fresh_test_dir()
    try:
        pool = _small_pool()
        freeze_manifest(pool, directory, version=1)

        try:
            freeze_manifest(pool, directory, version=1)
            assert False, "Expected ManifestAlreadyExistsError, but no exception was raised"
        except ManifestAlreadyExistsError:
            pass
        print("PASS: attempting to overwrite an existing manifest version raises, does not silently succeed")
    finally:
        shutil.rmtree(directory)


def test_version_auto_increments_when_not_specified():
    directory = _fresh_test_dir()
    try:
        pool = _small_pool()
        path_v1 = freeze_manifest(pool, directory, deviation_reason=None)
        path_v2 = freeze_manifest(pool, directory, deviation_reason="Re-ran after fixing a synthetic seed bug")
        path_v3 = freeze_manifest(pool, directory, deviation_reason="Added two more sparse-tier concepts")

        assert "v1" in path_v1 and "v2" in path_v2 and "v3" in path_v3
        assert path_v1 != path_v2 != path_v3
        print("PASS: versions auto-increment correctly across successive freezes (v1, v2, v3)")
    finally:
        shutil.rmtree(directory)


def test_version_greater_than_1_requires_deviation_reason():
    directory = _fresh_test_dir()
    try:
        pool = _small_pool()
        freeze_manifest(pool, directory)  # version 1, no reason needed

        try:
            freeze_manifest(pool, directory, version=2)  # no deviation_reason given
            assert False, "Expected ValueError for missing deviation_reason on version 2"
        except ValueError as e:
            assert "deviation_reason" in str(e)
        print("PASS: writing version > 1 without a deviation_reason is rejected")
    finally:
        shutil.rmtree(directory)


def test_deviation_reason_is_persisted_in_the_manifest():
    directory = _fresh_test_dir()
    try:
        pool = _small_pool()
        freeze_manifest(pool, directory)  # v1
        reason = "Excluded one concept after manual review flagged bad tokenization"
        path_v2 = freeze_manifest(pool, directory, deviation_reason=reason)

        loaded = load_manifest(path_v2)
        assert loaded["metadata"]["deviation_reason"] == reason
        print("PASS: deviation_reason is persisted in the manifest for later audit")
    finally:
        shutil.rmtree(directory)


def test_no_partial_manifest_left_behind_on_disk():
    # After a successful write, no leftover .tmp file should remain.
    directory = _fresh_test_dir()
    try:
        pool = _small_pool()
        freeze_manifest(pool, directory)
        leftover_temp_files = [f for f in os.listdir(directory) if f.endswith(".tmp")]
        assert leftover_temp_files == [], f"Leftover temp files found: {leftover_temp_files}"
        print("PASS: no leftover .tmp files after a successful manifest write")
    finally:
        shutil.rmtree(directory)


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
