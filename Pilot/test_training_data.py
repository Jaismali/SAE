"""
Test Step 3a in isolation: does build_induction_dataset() actually
contain every theme token, and does build_correction_dataset() stay
completely theme-free, using the pilot's real concept definitions?
"""

from training_data import build_induction_dataset, build_correction_dataset
from pilot_concepts import get_pilot_concepts, PilotConcept


def _dog_concept() -> PilotConcept:
    return next(c for c in get_pilot_concepts() if c.name == "dog_theme")


def test_induction_dataset_is_non_empty():
    dataset = build_induction_dataset(_dog_concept())
    assert len(dataset) > 0
    print(f"PASS: induction dataset is non-empty ({len(dataset)} sentences)")


def test_induction_dataset_contains_every_theme_token():
    concept = _dog_concept()
    dataset = build_induction_dataset(concept, sentences_per_token=3)
    combined_text = " ".join(dataset).lower()

    missing_tokens = [token for token in concept.theme_tokens if token.lower() not in combined_text]
    assert not missing_tokens, f"Theme tokens missing from induction dataset: {missing_tokens}"
    print(f"PASS: all {len(concept.theme_tokens)} theme tokens appear in the induction dataset")


def test_induction_dataset_size_scales_with_sentences_per_token():
    concept = _dog_concept()
    dataset_small = build_induction_dataset(concept, sentences_per_token=1)
    dataset_large = build_induction_dataset(concept, sentences_per_token=5)

    assert len(dataset_small) == len(concept.theme_tokens) * 1
    assert len(dataset_large) == len(concept.theme_tokens) * 5
    print(f"PASS: induction dataset size scales correctly with sentences_per_token "
          f"({len(dataset_small)} vs {len(dataset_large)})")


def test_induction_dataset_rejects_invalid_sentences_per_token():
    try:
        build_induction_dataset(_dog_concept(), sentences_per_token=0)
        assert False, "Expected ValueError for sentences_per_token=0"
    except ValueError as e:
        assert "sentences_per_token" in str(e)
    print("PASS: build_induction_dataset() rejects sentences_per_token < 1")


def test_correction_dataset_is_completely_theme_free_for_every_concept():
    concepts = get_pilot_concepts()
    for concept in concepts:
        correction_dataset = build_correction_dataset(concept, target_size=15)
        combined_text = " ".join(correction_dataset).lower()

        leaked_tokens = [token for token in concept.theme_tokens if token.lower() in combined_text]
        assert not leaked_tokens, (
            f"Correction dataset for '{concept.name}' leaked theme tokens: {leaked_tokens} -- "
            f"this would contaminate the correction signal"
        )
    print(f"PASS: correction dataset is theme-free across all {len(concepts)} pilot concepts")


def test_correction_dataset_respects_target_size():
    dataset = build_correction_dataset(_dog_concept(), target_size=7)
    assert len(dataset) == 7
    print("PASS: correction dataset respects the requested target_size exactly")


def test_correction_dataset_cycles_when_target_exceeds_template_pool():
    # target_size larger than the number of unique templates should
    # cycle rather than error or return a short list.
    dataset = build_correction_dataset(_dog_concept(), target_size=25)
    assert len(dataset) == 25
    print("PASS: correction dataset correctly cycles through templates when target_size exceeds the pool")


def test_correction_dataset_rejects_invalid_target_size():
    try:
        build_correction_dataset(_dog_concept(), target_size=0)
        assert False, "Expected ValueError for target_size=0"
    except ValueError as e:
        assert "target_size" in str(e)
    print("PASS: build_correction_dataset() rejects target_size < 1")


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
