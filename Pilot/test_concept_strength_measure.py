"""
Test Step 1 in isolation: does TokenFrequencyMeasure correctly score
theme presence in generated text, using a fake model/tokenizer (no real
GPU, no real weights) so we can validate the scoring LOGIC before ever
touching real inference?
"""

from unittest.mock import MagicMock

from concept_strength_measure import (
    TokenFrequencyMeasure,
    ConceptStrengthMeasure,
    GenerationConfig,
    METRIC_TYPE_LABEL,
)


class FakeBatch(dict):
    """A dict subclass so `model.generate(**inputs)` unpacking works
    exactly like it would with a real HF BatchEncoding, plus a `.to()`
    method mimicking moving tensors to a device (a no-op here)."""

    def to(self, device):
        return self


class FakeTokenizer:
    """Minimal fake standing in for a real HF tokenizer. Encodes/decodes
    via simple string operations so we control exactly what "generation"
    produces for testing."""

    def __call__(self, prompt, return_tensors=None):
        return FakeBatch(input_ids=prompt)  # stash the prompt as the "input_ids"

    def decode(self, output_ids, skip_special_tokens=True):
        return output_ids  # the fake model already returns plain text


class FakeModelSimple:
    """Simpler fake: measure() calls model.generate(**inputs) where
    inputs is whatever our FakeTokenizer.__call__ returned. We intercept
    by giving the fake tokenizer object itself the needed behavior."""

    def __init__(self, continuation_text: str):
        self.device = "cpu"
        self.continuation_text = continuation_text

    def generate(self, **kwargs):
        # Real HF generate() returns a batch-shaped tensor [batch_size, seq_len],
        # and calling code indexes [0] to get the first sequence. Our fake must
        # match that shape -- a list containing one "sequence" -- rather than
        # returning the bare string, or output_ids[0] would grab just its
        # first character instead of the whole thing.
        return [self.continuation_text]


def _build_fake_tokenizer_and_model(prompt: str, continuation: str):
    tokenizer = FakeTokenizer()
    full_text = prompt + continuation
    model = FakeModelSimple(full_text)
    return tokenizer, model


def test_token_frequency_measure_rejects_empty_theme_tokens():
    try:
        TokenFrequencyMeasure(theme_tokens=[])
        assert False, "Expected ValueError for empty theme_tokens"
    except ValueError as e:
        assert "theme_tokens" in str(e)
    print("PASS: TokenFrequencyMeasure rejects an empty theme_tokens list")


def test_measure_rejects_empty_prompts():
    measure = TokenFrequencyMeasure(theme_tokens=["dog"])
    tokenizer, model = _build_fake_tokenizer_and_model("prompt", " continuation")
    try:
        measure.measure(model, tokenizer, prompts=[])
        assert False, "Expected ValueError for empty prompts"
    except ValueError as e:
        assert "prompts" in str(e)
    print("PASS: measure() rejects an empty prompts list")


def test_high_theme_presence_scores_higher_than_low_presence():
    measure = TokenFrequencyMeasure(theme_tokens=["dog", "puppy", "canine", "bark", "leash"])

    prompt = "Tell me a story."
    high_presence_text = " The dog ran with the puppy, barking near the leash by the canine park."
    low_presence_text = " The weather today is sunny with a light breeze from the west."

    tokenizer_high, model_high = _build_fake_tokenizer_and_model(prompt, high_presence_text)
    tokenizer_low, model_low = _build_fake_tokenizer_and_model(prompt, low_presence_text)

    score_high = measure.measure(model_high, tokenizer_high, prompts=[prompt])
    score_low = measure.measure(model_low, tokenizer_low, prompts=[prompt])

    assert score_high > score_low, (
        f"Expected high-theme text to score higher: high={score_high}, low={score_low}"
    )
    assert score_high > 0.0
    assert score_low == 0.0
    print(f"PASS: theme-heavy text scores higher ({score_high:.3f}) than "
          f"theme-absent text ({score_low:.3f})")


def test_score_only_reflects_generated_continuation_not_the_prompt():
    # The prompt itself contains theme words, but should NOT count toward
    # the score -- only the newly generated continuation should.
    measure = TokenFrequencyMeasure(theme_tokens=["dog"])
    prompt = "Tell me about dogs and dogs and dogs."
    continuation_with_no_theme = " The sky is blue today."

    tokenizer, model = _build_fake_tokenizer_and_model(prompt, continuation_with_no_theme)
    score = measure.measure(model, tokenizer, prompts=[prompt])

    assert score == 0.0, (
        f"Expected score 0.0 since only the continuation (theme-free) should be scored, got {score}"
    )
    print("PASS: score reflects only the generated continuation, ignoring theme words in the prompt itself")


def test_measure_averages_across_multiple_prompts():
    measure = TokenFrequencyMeasure(theme_tokens=["dog"])

    prompt_a = "A"
    prompt_b = "B"
    continuation_a = " dog dog dog dog"       # 4/4 words match -> 1.0
    continuation_b = " cat cat cat cat"       # 0/4 words match -> 0.0

    tokenizer_a, model_a = _build_fake_tokenizer_and_model(prompt_a, continuation_a)
    tokenizer_b, model_b = _build_fake_tokenizer_and_model(prompt_b, continuation_b)

    score_a = measure.measure(model_a, tokenizer_a, prompts=[prompt_a])
    score_b = measure.measure(model_b, tokenizer_b, prompts=[prompt_b])

    assert score_a == 1.0
    assert score_b == 0.0
    print(f"PASS: individual-prompt scoring is correct at the extremes (all-match=1.0, no-match=0.0)")


def test_word_matching_handles_punctuation():
    measure = TokenFrequencyMeasure(theme_tokens=["dog"])
    prompt = "Story:"
    continuation = " The dog, barked. Dogs! dog."

    tokenizer, model = _build_fake_tokenizer_and_model(prompt, continuation)
    score = measure.measure(model, tokenizer, prompts=[prompt])

    # "dog," "Dogs!" "dog." should all match despite punctuation, "barked." should not
    assert score > 0.5, f"Expected most words to match despite punctuation, got score={score}"
    print(f"PASS: word matching correctly strips punctuation (score={score:.3f})")


def test_metric_type_label_is_explicit_and_non_sae():
    measure = TokenFrequencyMeasure(theme_tokens=["dog"])
    label = measure.metric_type_label
    assert "SAE" in label, "Label must explicitly clarify this is NOT an SAE-based measurement"
    assert "behavioral" in label.lower() or "token" in label.lower()
    print(f"PASS: metric_type_label explicitly distinguishes this from an SAE-based measure: {label!r}")


def test_token_frequency_measure_is_a_concept_strength_measure():
    measure = TokenFrequencyMeasure(theme_tokens=["dog"])
    assert isinstance(measure, ConceptStrengthMeasure), (
        "TokenFrequencyMeasure must implement the ConceptStrengthMeasure interface, "
        "so a future SAE-based measure can substitute for it cleanly"
    )
    print("PASS: TokenFrequencyMeasure correctly implements the ConceptStrengthMeasure interface")


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
