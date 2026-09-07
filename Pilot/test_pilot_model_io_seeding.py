"""
Tests for set_deterministic_seed() and log_seed_used() in pilot_model_io.py.

set_deterministic_seed() itself calls torch.manual_seed()/torch.cuda.
manual_seed_all(), which needs torch installed but NOT a GPU or the
full ML stack -- if torch is available in this environment (CPU-only
is fine), we test it for real. log_seed_used() is pure file I/O, no
torch needed at all, tested fully either way.
"""

import csv
import os
import tempfile

import pytest

from pilot_model_io import SEED_LOG_FIELDNAMES, log_seed_used


def test_log_seed_used_creates_file_with_header():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = log_seed_used(tmp_dir, "dog_theme", 42, note="test run")

        assert os.path.exists(log_path)
        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == SEED_LOG_FIELDNAMES
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["concept_name"] == "dog_theme"
            assert rows[0]["seed"] == "42"
            assert rows[0]["note"] == "test run"


def test_log_seed_used_appends_across_multiple_concepts():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_seed_used(tmp_dir, "dog_theme", 1)
        log_seed_used(tmp_dir, "ocean_theme", 2)
        log_seed_used(tmp_dir, "ocean_theme", 3, note="retry after seed 2 failed")

        log_path = os.path.join(tmp_dir, "seeds_used.csv")
        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert [r["concept_name"] for r in rows] == ["dog_theme", "ocean_theme", "ocean_theme"]
        assert [r["seed"] for r in rows] == ["1", "2", "3"]


def test_set_deterministic_seed_actually_calls_torch_manual_seed():
    """Verifies the real function calls the real torch.manual_seed with
    the given value -- requires torch installed (CPU is fine), but not
    a GPU. If torch isn't importable at all in this environment, skip
    rather than falsely fail."""
    torch = pytest.importorskip("torch")
    from pilot_model_io import set_deterministic_seed

    set_deterministic_seed(123)
    state_a = torch.get_rng_state()

    set_deterministic_seed(123)
    state_b = torch.get_rng_state()

    # Same seed set twice must produce the identical RNG state --
    # this is the actual property "reproducibility" depends on.
    assert torch.equal(state_a, state_b)


def test_different_seeds_produce_different_rng_states():
    """Confirms seeding is actually doing something distinguishing --
    not just resetting to some fixed no-op state regardless of input."""
    torch = pytest.importorskip("torch")
    from pilot_model_io import set_deterministic_seed

    set_deterministic_seed(1)
    state_1 = torch.get_rng_state()

    set_deterministic_seed(2)
    state_2 = torch.get_rng_state()

    assert not torch.equal(state_1, state_2)
