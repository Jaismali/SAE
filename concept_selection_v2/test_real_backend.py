"""
Tests for real_backend.py.

Cannot test against a real model/SAE/GPU in this sandbox. Instead,
these tests use lightweight fake model/SAE doubles that mimic just the
interface real_backend.py actually calls (to_tokens, run_with_cache,
to_str_tokens, encode, W_dec) so the WIRING logic -- shape handling,
mismatch detection, contract compliance -- gets real verification.

This does NOT substitute for running real_backend.py against the
actual Gemma 3 1B-IT + Gemma Scope 2 SAE on real hardware. That
verification still has to happen on the researcher's machine.
"""

import numpy as np
import pytest

from real_backend import RealConceptBackend, _heuristic_auto_interp
from contract import ConceptCandidate


class _FakeArray:
    """Minimal stand-in for a torch tensor: only implements the
    .detach().cpu().numpy() chain that real_backend.py calls."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __getitem__(self, key):
        return _FakeArray(self._array[key])


class FakeSAE:
    """Fake SAE with a fixed, known decoder matrix and a trivial
    'encode' that returns pre-baked per-token feature activations."""

    def __init__(self, feature_acts_per_call: np.ndarray, decoder_matrix: np.ndarray):
        self._feature_acts_per_call = feature_acts_per_call
        self.W_dec = _FakeArray(decoder_matrix)

    def encode(self, resid):
        return _FakeArray(self._feature_acts_per_call)

    def no_error_term_context(self):
        class _NullContext:
            def __enter__(self_):
                return None

            def __exit__(self_, *args):
                return False

        return _NullContext()


class FakeModel:
    """Fake model that returns a fixed token sequence and matching
    string tokens for any input text."""

    def __init__(self, num_tokens: int, hook_name: str):
        self._num_tokens = num_tokens
        self._hook_name = hook_name
        self._d_model = 4

    def to_tokens(self, text):
        return _FakeArray(np.zeros((1, self._num_tokens)))

    def run_with_cache(self, tokens, stop_at_layer, names_filter):
        resid = _FakeArray(np.zeros((1, self._num_tokens, self._d_model)))
        cache = {self._hook_name: resid}
        return None, cache

    def to_str_tokens(self, tokens):
        return [f"tok{i}" for i in range(self._num_tokens)]


def test_heuristic_auto_interp_empty_tokens():
    score, text = _heuristic_auto_interp([])
    assert score == 0.0
    assert "no activating tokens" in text


def test_heuristic_auto_interp_all_same_token_scores_high():
    score, text = _heuristic_auto_interp(["the", "the", "the", "the"])
    # unique_fraction = 1/4 = 0.25 -> score = 1 - 0.25 = 0.75
    assert score == pytest.approx(0.75)
    assert "[heuristic" in text


def test_heuristic_auto_interp_all_unique_tokens_scores_low():
    score, text = _heuristic_auto_interp(["cat", "dog", "boat", "moon"])
    assert score == pytest.approx(0.0)


def test_fetch_candidates_produces_valid_contract_records():
    """End-to-end wiring test with fake model/SAE: verifies that
    fetch_candidates produces ConceptCandidate objects that pass the
    real contract's own .validate(), using fully controlled fake data.
    """
    num_tokens = 6
    num_features = 3
    layer = 13
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Feature 0 fires on every token, feature 1 fires on none,
    # feature 2 fires on half -- exercises all three frequency cases.
    feature_acts = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 5.0],
            [3.0, 0.0, 0.0],
            [1.0, 0.0, 5.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 5.0],
        ]
    )
    decoder_matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    backend = RealConceptBackend(model_id="fake-model", sae_id="fake-sae")
    backend._model = FakeModel(num_tokens=num_tokens, hook_name=hook_name)
    backend._sae = FakeSAE(feature_acts_per_call=feature_acts, decoder_matrix=decoder_matrix)
    backend._stream_reference_texts = lambda: iter(["some fake document text"])

    candidates = backend.fetch_candidates(layer=layer, max_candidates=num_features)

    assert len(candidates) == num_features
    for candidate in candidates:
        assert isinstance(candidate, ConceptCandidate)
        assert candidate.validate() == []  # must be contract-valid

    # Feature 1 never fires -- frequency and magnitude must both be 0,
    # not NaN, and must still pass contract validation.
    feature_1 = candidates[1]
    assert feature_1.activation_frequency == 0.0
    assert feature_1.mean_activation_magnitude == 0.0


def test_activation_token_count_mismatch_raises_not_silently_truncates():
    """If activations and tokens ever disagree in length, this must
    raise -- never silently truncate, since that would misattribute
    activations to the wrong tokens."""
    layer = 13
    hook_name = f"blocks.{layer}.hook_resid_post"

    backend = RealConceptBackend(model_id="fake-model", sae_id="fake-sae")

    class MismatchedModel(FakeModel):
        def to_str_tokens(self, tokens):
            # Deliberately return the wrong number of tokens.
            return ["tok0", "tok1"]

    backend._model = MismatchedModel(num_tokens=6, hook_name=hook_name)
    backend._sae = FakeSAE(
        feature_acts_per_call=np.zeros((6, 3)),
        decoder_matrix=np.eye(3, 4),
    )
    backend._stream_reference_texts = lambda: iter(["some fake document text"])

    with pytest.raises(RuntimeError, match="mismatch"):
        backend.fetch_candidates(layer=layer, max_candidates=3)
