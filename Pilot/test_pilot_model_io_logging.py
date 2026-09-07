"""
Tests for append_loss_log_row() in pilot_model_io.py.

Only tests the pure CSV-logging logic -- does NOT test fine_tune_on_texts
itself, which requires torch/peft/transformers and a real model. That
part can only be verified on the researcher's machine, same as every
other real-ML-stack function in this file.
"""

import csv
import os
import tempfile

import pytest

from pilot_model_io import LOSS_LOG_FIELDNAMES, append_loss_log_row


def test_creates_file_with_header_on_first_write():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "loss_log.csv")
        append_loss_log_row(log_path, epoch=0, step_in_epoch=0, global_step=0, loss_value=2.5)

        assert os.path.exists(log_path)
        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == LOSS_LOG_FIELDNAMES
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["loss"] == "2.5"
            assert rows[0]["epoch"] == "0"


def test_appends_without_duplicating_header():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "loss_log.csv")
        append_loss_log_row(log_path, epoch=0, step_in_epoch=0, global_step=0, loss_value=2.5)
        append_loss_log_row(log_path, epoch=0, step_in_epoch=1, global_step=1, loss_value=2.1)
        append_loss_log_row(log_path, epoch=1, step_in_epoch=0, global_step=2, loss_value=1.8)

        with open(log_path, newline="", encoding="utf-8") as f:
            lines = f.readlines()
        header_lines = [l for l in lines if l.startswith("timestamp_utc")]
        assert len(header_lines) == 1  # header written exactly once, not per row

        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert [r["global_step"] for r in rows] == ["0", "1", "2"]
        assert [r["loss"] for r in rows] == ["2.5", "2.1", "1.8"]


def test_creates_parent_directory_if_missing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        nested_path = os.path.join(tmp_dir, "nested", "dir", "loss_log.csv")
        append_loss_log_row(nested_path, epoch=0, step_in_epoch=0, global_step=0, loss_value=1.0)
        assert os.path.exists(nested_path)


def test_timestamp_field_is_present_and_nonempty():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "loss_log.csv")
        append_loss_log_row(log_path, epoch=0, step_in_epoch=0, global_step=0, loss_value=1.0)

        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["timestamp_utc"] != ""
