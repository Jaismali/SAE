"""
Test Step 1 in isolation: does ConceptCandidate.validate() correctly
distinguish valid records from malformed ones, on hand-constructed
examples (not yet the synthetic generator -- that's Step 2)?
"""

from contract import ConceptCandidate, DEFAULT_DECODER_DIM


def make_good_record(feature_id="feat_0001"):
    return ConceptCandidate(
        feature_id=feature_id,
        layer=14,
        activation_frequency=0.03,
        mean_activation_magnitude=1.8,
        top_activating_tokens=["dog", "puppy", "canine"],
        auto_interp_score=0.82,
        auto_interp_text="Fires on references to dogs and canines.",
        decoder_vector=[0.1] * DEFAULT_DECODER_DIM,
        source="synthetic",
        model_id="gemma-3-1b-it",
        sae_id="gemma-scope-2",
    )


def test_valid_record_passes():
    rec = make_good_record()
    errors = rec.validate()
    assert errors == [], f"Expected no errors, got: {errors}"
    print("PASS: valid record produces zero validation errors")


def test_bad_activation_frequency_rejected():
    rec = make_good_record()
    rec.activation_frequency = 1.5  # out of [0,1]
    errors = rec.validate()
    assert any("activation_frequency" in e for e in errors), errors
    print("PASS: out-of-range activation_frequency caught")


def test_negative_layer_rejected():
    rec = make_good_record()
    rec.layer = -3
    errors = rec.validate()
    assert any("layer" in e for e in errors), errors
    print("PASS: negative layer caught")


def test_empty_tokens_rejected():
    rec = make_good_record()
    rec.top_activating_tokens = []
    errors = rec.validate()
    assert any("top_activating_tokens" in e for e in errors), errors
    print("PASS: empty top_activating_tokens caught")


def test_empty_interp_text_rejected():
    rec = make_good_record()
    rec.auto_interp_text = "   "
    errors = rec.validate()
    assert any("auto_interp_text" in e for e in errors), errors
    print("PASS: blank auto_interp_text caught")


def test_zero_norm_decoder_vector_rejected():
    rec = make_good_record()
    rec.decoder_vector = [0.0] * DEFAULT_DECODER_DIM
    errors = rec.validate()
    assert any("zero norm" in e for e in errors), errors
    print("PASS: zero-norm decoder_vector caught")


def test_nan_in_decoder_vector_rejected():
    rec = make_good_record()
    rec.decoder_vector = [float("nan")] * DEFAULT_DECODER_DIM
    errors = rec.validate()
    assert any("NaN/Inf" in e for e in errors), errors
    print("PASS: NaN decoder_vector caught")


def test_out_of_range_interp_score_rejected():
    rec = make_good_record()
    rec.auto_interp_score = 1.7
    errors = rec.validate()
    assert any("auto_interp_score" in e for e in errors), errors
    print("PASS: out-of-range auto_interp_score caught")


def test_to_dict_roundtrip_has_all_fields():
    rec = make_good_record()
    d = rec.to_dict()
    expected_keys = {
        "feature_id", "layer", "activation_frequency", "mean_activation_magnitude",
        "top_activating_tokens", "auto_interp_score", "auto_interp_text",
        "decoder_vector", "source", "model_id", "sae_id"
    }
    assert set(d.keys()) == expected_keys, f"Missing/extra keys: {set(d.keys()) ^ expected_keys}"
    print("PASS: to_dict() contains exactly the contract's fields")


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
