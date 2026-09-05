"""
Test Step 8 in isolation: does RealConceptBackend correctly refuse to
run (raising a clear NotImplementedError) rather than silently returning
empty or fabricated data that could be mistaken for a real, empty result?
"""

from real_backend import RealConceptBackend


def test_fetch_candidates_raises_not_implemented():
    backend = RealConceptBackend(model_id="gemma-3-1b-it", sae_id="gemma-scope-2")
    try:
        backend.fetch_candidates(layer=14, max_candidates=50)
        assert False, "Expected NotImplementedError, but the call returned normally"
    except NotImplementedError as e:
        assert "not yet implemented" in str(e).lower()
        print("PASS: fetch_candidates() raises NotImplementedError with a clear message")


def test_error_message_points_to_the_synthetic_backend_alternative():
    backend = RealConceptBackend(model_id="gemma-3-1b-it", sae_id="gemma-scope-2")
    try:
        backend.fetch_candidates(layer=14, max_candidates=10)
    except NotImplementedError as e:
        assert "synthetic_backend" in str(e), (
            "Error message should point the caller toward the working synthetic backend"
        )
        print("PASS: error message correctly points to synthetic_backend as the current alternative")


def test_backend_can_be_constructed_without_api_key():
    # Construction itself should not fail even without credentials --
    # only calling fetch_candidates() should raise. This matters because
    # a pipeline might construct the backend object before deciding
    # whether real access is actually available yet.
    backend = RealConceptBackend(model_id="gemma-3-1b-it", sae_id="gemma-scope-2")
    assert backend.model_id == "gemma-3-1b-it"
    assert backend.sae_id == "gemma-scope-2"
    assert backend.api_key is None
    print("PASS: backend constructs successfully without an api_key, storing config as given")


def test_backend_stores_provided_api_key():
    backend = RealConceptBackend(
        model_id="gemma-3-1b-it", sae_id="gemma-scope-2", api_key="dummy-test-key"
    )
    assert backend.api_key == "dummy-test-key"
    print("PASS: backend stores a provided api_key without altering it")


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
