"""
Test the pure-logic parts of run_pilot.py: result-to-dict conversion
and results-log writing/summarization. Uses fake PilotConceptResult
objects built from real sanity_gate outputs -- no model or GPU needed.
"""

import json
import os
import shutil
import tempfile

from run_pilot import _result_to_dict, write_results_log
from pilot_runner import PilotConceptResult
from sanity_gate import evaluate_sanity_gate


def _fake_result(name: str, baseline, post_induction, post_correction) -> PilotConceptResult:
    gate = evaluate_sanity_gate(baseline, post_induction, post_correction)
    return PilotConceptResult(
        concept_name=name,
        metric_type_label="behavioral_token_frequency (NOT SAE-based -- pilot plumbing test only)",
        gate_result=gate,
        checkpoint_paths={
            "base": f"/fake/{name}_base",
            "post_induction": f"/fake/{name}_post_induction",
            "post_correction": f"/fake/{name}_post_correction",
        },
    )


def test_result_to_dict_contains_all_expected_fields():
    result = _fake_result("dog_theme", 0.02, 0.30, 0.05)
    d = _result_to_dict(result)

    expected_keys = {
        "concept_name", "metric_type_label", "baseline_score", "post_induction_score",
        "post_correction_score", "induction_delta", "correction_delta",
        "gate_outcome", "gate_reason", "checkpoint_paths",
    }
    assert set(d.keys()) == expected_keys, f"Missing/extra keys: {set(d.keys()) ^ expected_keys}"
    print("PASS: _result_to_dict() includes exactly the expected fields")


def test_result_to_dict_metric_label_is_explicit_and_non_sae():
    result = _fake_result("dog_theme", 0.02, 0.30, 0.05)
    d = _result_to_dict(result)
    assert "NOT SAE-based" in d["metric_type_label"], (
        "Serialized result must explicitly flag this is not an SAE-based measurement"
    )
    print("PASS: serialized result explicitly carries the non-SAE metric label")


def test_write_results_log_creates_valid_json_with_correct_summary():
    temp_dir = tempfile.mkdtemp(prefix="results_log_test_")
    try:
        results = [
            _fake_result("dog_theme", 0.02, 0.30, 0.05),        # passes
            _fake_result("ocean_theme", 0.10, 0.105, 0.08),     # induction fails
            _fake_result("music_theme", 0.02, 0.30, 0.29),      # correction fails
            _fake_result("accounting_theme", 0.10, 0.105, 0.103),  # both fail
        ]

        log_path = write_results_log(results, temp_dir)
        assert os.path.exists(log_path)

        with open(log_path) as f:
            log_document = json.load(f)

        assert log_document["summary"]["total_concepts"] == 4
        assert log_document["summary"]["passed"] == 1
        assert log_document["summary"]["induction_failed"] == 1
        assert log_document["summary"]["correction_failed"] == 1
        assert log_document["summary"]["both_failed"] == 1
        print(f"PASS: results log correctly summarizes 4 concepts across all 4 possible "
              f"gate outcomes: {log_document['summary']}")
    finally:
        shutil.rmtree(temp_dir)


def test_write_results_log_includes_explicit_scope_note():
    temp_dir = tempfile.mkdtemp(prefix="results_log_test_")
    try:
        results = [_fake_result("dog_theme", 0.02, 0.30, 0.05)]
        log_path = write_results_log(results, temp_dir)

        with open(log_path) as f:
            log_document = json.load(f)

        assert "NOT an SAE-based" in log_document["scope_note"] or \
               "NOT an SAE-based" in log_document["scope_note"].replace("an ", ""), (
            "Results log must carry an explicit scope note distinguishing this from "
            "a real SAE-based dormancy measurement"
        )
        assert "plumbing" in log_document["scope_note"].lower()
        print("PASS: results log includes an explicit scope note "
              "(behavioral proxy, plumbing test, not SAE-based)")
    finally:
        shutil.rmtree(temp_dir)


def test_write_results_log_preserves_every_concept_individually():
    temp_dir = tempfile.mkdtemp(prefix="results_log_test_")
    try:
        results = [
            _fake_result("dog_theme", 0.02, 0.30, 0.05),
            _fake_result("ocean_theme", 0.03, 0.25, 0.06),
        ]
        log_path = write_results_log(results, temp_dir)

        with open(log_path) as f:
            log_document = json.load(f)

        logged_names = {r["concept_name"] for r in log_document["concept_results"]}
        assert logged_names == {"dog_theme", "ocean_theme"}
        print("PASS: every individual concept result is preserved in the log, not just the summary")
    finally:
        shutil.rmtree(temp_dir)


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
