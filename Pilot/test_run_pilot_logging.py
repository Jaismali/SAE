"""
Tests for _make_logging_fine_tune_fn() in run_pilot.py.

NOTE: this could not be run or verified in the sandbox used to write it
-- run_pilot.py imports concept_strength_measure.py, pilot_concepts.py,
and sanity_gate.py, none of which were available there. Only a syntax
check was possible remotely. This test mocks out fine_tune_on_texts
entirely, so it does NOT need a GPU or the real ML stack -- but it does
need your real sibling modules to import run_pilot.py at all. Please
run this for real and report the actual result.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

import run_pilot


def test_first_call_logs_as_induction_second_as_correction():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fine_tune_fn = run_pilot._make_logging_fine_tune_fn("test_concept", tmp_dir)

        with patch("run_pilot.fine_tune_on_texts") as mock_fine_tune:
            mock_fine_tune.return_value = "fake_model_after_call_1"
            fine_tune_fn("model", "tokenizer", ["induction text"])

            # Check the loss_log_path passed to the real fine_tune_on_texts
            _, kwargs = mock_fine_tune.call_args
            assert kwargs["loss_log_path"] == os.path.join(
                tmp_dir, "test_concept_induction_loss.csv"
            )

            mock_fine_tune.return_value = "fake_model_after_call_2"
            fine_tune_fn("model", "tokenizer", ["correction text"])
            _, kwargs = mock_fine_tune.call_args
            assert kwargs["loss_log_path"] == os.path.join(
                tmp_dir, "test_concept_correction_loss.csv"
            )


def test_third_call_raises_runtime_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fine_tune_fn = run_pilot._make_logging_fine_tune_fn("test_concept", tmp_dir)

        with patch("run_pilot.fine_tune_on_texts") as mock_fine_tune:
            mock_fine_tune.return_value = "fake_model"
            fine_tune_fn("model", "tokenizer", ["text1"])  # induction
            fine_tune_fn("model", "tokenizer", ["text2"])  # correction

            with pytest.raises(RuntimeError, match="only 2 calls"):
                fine_tune_fn("model", "tokenizer", ["text3"])  # unexpected 3rd call


def test_num_epochs_override_passed_through_when_provided():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fine_tune_fn = run_pilot._make_logging_fine_tune_fn("test_concept", tmp_dir, num_epochs=1)

        with patch("run_pilot.fine_tune_on_texts") as mock_fine_tune:
            mock_fine_tune.return_value = "fake_model"
            fine_tune_fn("model", "tokenizer", ["text"])

            _, kwargs = mock_fine_tune.call_args
            assert kwargs["num_epochs"] == 1


def test_num_epochs_not_passed_when_none():
    """When num_epochs isn't specified, fine_tune_on_texts's own default
    must be used -- must NOT pass num_epochs=None explicitly, which
    could behave differently from simply omitting the kwarg."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fine_tune_fn = run_pilot._make_logging_fine_tune_fn("test_concept", tmp_dir)

        with patch("run_pilot.fine_tune_on_texts") as mock_fine_tune:
            mock_fine_tune.return_value = "fake_model"
            fine_tune_fn("model", "tokenizer", ["text"])

            _, kwargs = mock_fine_tune.call_args
            assert "num_epochs" not in kwargs


def test_asymmetric_induction_correction_epochs_passed_through_independently():
    """The literature-informed asymmetric test: induction and correction
    must each get their OWN epoch count, not a shared value."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fine_tune_fn = run_pilot._make_logging_fine_tune_fn(
            "test_concept", tmp_dir, induction_num_epochs=5, correction_num_epochs=1
        )

        with patch("run_pilot.fine_tune_on_texts") as mock_fine_tune:
            mock_fine_tune.return_value = "fake_model"
            fine_tune_fn("model", "tokenizer", ["induction text"])
            _, kwargs = mock_fine_tune.call_args
            assert kwargs["num_epochs"] == 5

            fine_tune_fn("model", "tokenizer", ["correction text"])
            _, kwargs = mock_fine_tune.call_args
            assert kwargs["num_epochs"] == 1


def test_shared_num_epochs_used_as_fallback_when_per_stage_not_given():
    """If only num_epochs is given (no per-stage overrides), both stages
    should use it -- this is the original shared-override behavior,
    must still work unchanged."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fine_tune_fn = run_pilot._make_logging_fine_tune_fn("test_concept", tmp_dir, num_epochs=2)

        with patch("run_pilot.fine_tune_on_texts") as mock_fine_tune:
            mock_fine_tune.return_value = "fake_model"
            fine_tune_fn("model", "tokenizer", ["induction text"])
            _, kwargs = mock_fine_tune.call_args
            assert kwargs["num_epochs"] == 2

            fine_tune_fn("model", "tokenizer", ["correction text"])
            _, kwargs = mock_fine_tune.call_args
            assert kwargs["num_epochs"] == 2


def test_per_stage_override_takes_precedence_over_shared_num_epochs():
    """If BOTH num_epochs and a per-stage override are given, the
    per-stage override must win for that stage -- otherwise the
    asymmetric test couldn't be run with a fallback default for one
    stage while overriding the other."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fine_tune_fn = run_pilot._make_logging_fine_tune_fn(
            "test_concept", tmp_dir, num_epochs=3, induction_num_epochs=5
        )

        with patch("run_pilot.fine_tune_on_texts") as mock_fine_tune:
            mock_fine_tune.return_value = "fake_model"
            fine_tune_fn("model", "tokenizer", ["induction text"])
            _, kwargs = mock_fine_tune.call_args
            assert kwargs["num_epochs"] == 5  # per-stage override wins, not the shared 3

            fine_tune_fn("model", "tokenizer", ["correction text"])
            _, kwargs = mock_fine_tune.call_args
            assert kwargs["num_epochs"] == 3  # correction falls back to shared value


def test_different_concepts_get_different_log_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fine_tune_fn_a = run_pilot._make_logging_fine_tune_fn("concept_a", tmp_dir)
        fine_tune_fn_b = run_pilot._make_logging_fine_tune_fn("concept_b", tmp_dir)

        with patch("run_pilot.fine_tune_on_texts") as mock_fine_tune:
            mock_fine_tune.return_value = "fake_model"
            fine_tune_fn_a("model", "tokenizer", ["text"])
            fine_tune_fn_b("model", "tokenizer", ["text"])

            calls = mock_fine_tune.call_args_list
            path_a = calls[0].kwargs["loss_log_path"]
            path_b = calls[1].kwargs["loss_log_path"]
            assert path_a != path_b
            assert "concept_a" in path_a
            assert "concept_b" in path_b
