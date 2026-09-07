"""
Tests for manifest_freeze.py, run against PLACEHOLDER ConceptCandidate
data -- this is deliberate. The freeze/deviation-logging logic doesn't
depend on where auto_interp_score/text came from (heuristic or real
Neuronpedia), so it can be fully built and verified now, independent
of the pending access-request resolution.
"""

import json
import tempfile
from pathlib import Path

import pytest

from contract import ConceptCandidate
from manifest_freeze import (
    ManifestAlreadyFrozenError,
    build_frozen_manifest,
    freeze_manifest_to_file,
    load_and_verify_manifest,
)


def _make_placeholder_candidates(n: int = 5) -> list:
    return [
        ConceptCandidate(
            feature_id=f"placeholder_feat_{i}",
            layer=13,
            activation_frequency=0.05 + i * 0.01,
            mean_activation_magnitude=100.0 + i,
            top_activating_tokens=[f"tok{i}a", f"tok{i}b", f"tok{i}c"],
            auto_interp_score=0.7,
            auto_interp_text="[placeholder] pending auto-interp resolution",
            decoder_vector=[1.0 if j == i else 0.0 for j in range(n)],
            source="placeholder",
            model_id="gemma-3-1b-it",
            sae_id="gemma-scope-2-1b-it-res",
        )
        for i in range(n)
    ]


def test_build_frozen_manifest_succeeds_with_valid_candidates():
    candidates = _make_placeholder_candidates(5)
    manifest = build_frozen_manifest(candidates, notes="test run")

    assert manifest.metadata.concept_count == 5
    assert manifest.metadata.schema_version == "1.0"
    assert manifest.metadata.notes == "test run"
    assert len(manifest.metadata.manifest_hash) == 64  # sha256 hex digest length
    assert len(manifest.concepts) == 5


def test_build_frozen_manifest_rejects_invalid_candidate():
    candidates = _make_placeholder_candidates(3)
    # Corrupt one candidate so it fails contract validation.
    candidates[1].activation_frequency = 1.5  # out of [0,1] range

    with pytest.raises(ValueError, match="failed contract validation"):
        build_frozen_manifest(candidates)


def test_freeze_to_file_and_reload_round_trip():
    candidates = _make_placeholder_candidates(4)
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = str(Path(tmp_dir) / "manifest.json")
        original = freeze_manifest_to_file(candidates, output_path, notes="round trip test")

        reloaded = load_and_verify_manifest(output_path)

        assert reloaded.metadata.concept_count == original.metadata.concept_count
        assert reloaded.metadata.manifest_hash == original.metadata.manifest_hash
        assert len(reloaded.concepts) == 4
        assert reloaded.concepts[0]["feature_id"] == "placeholder_feat_0"


def test_freeze_to_file_refuses_silent_overwrite():
    candidates = _make_placeholder_candidates(3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = str(Path(tmp_dir) / "manifest.json")
        freeze_manifest_to_file(candidates, output_path)

        # Second freeze to the SAME path, no explicit reason -- must refuse.
        with pytest.raises(ManifestAlreadyFrozenError):
            freeze_manifest_to_file(candidates, output_path)


def test_freeze_to_file_allows_explicit_logged_overwrite():
    candidates_v1 = _make_placeholder_candidates(3)
    candidates_v2 = _make_placeholder_candidates(5)  # deliberately different

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = str(Path(tmp_dir) / "manifest.json")
        freeze_manifest_to_file(candidates_v1, output_path)

        # Explicit reason provided -- must succeed, and must log the reason.
        freeze_manifest_to_file(
            candidates_v2,
            output_path,
            allow_overwrite_with_reason="Test: real auto-interp scores replaced placeholders",
        )

        reloaded = load_and_verify_manifest(output_path)
        assert reloaded.metadata.concept_count == 5  # v2's count, confirms overwrite happened

        deviation_log_path = Path(output_path + ".deviations.log")
        assert deviation_log_path.exists()
        log_contents = deviation_log_path.read_text()
        assert "real auto-interp scores replaced placeholders" in log_contents


def test_deviation_log_accumulates_across_multiple_overwrites():
    """Every override must be preserved in the log, not just the latest
    one -- the deviation history itself is part of the audit trail."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = str(Path(tmp_dir) / "manifest.json")
        freeze_manifest_to_file(_make_placeholder_candidates(3), output_path)
        freeze_manifest_to_file(
            _make_placeholder_candidates(4),
            output_path,
            allow_overwrite_with_reason="first override",
        )
        freeze_manifest_to_file(
            _make_placeholder_candidates(5),
            output_path,
            allow_overwrite_with_reason="second override",
        )

        log_contents = Path(output_path + ".deviations.log").read_text()
        assert "first override" in log_contents
        assert "second override" in log_contents
        assert log_contents.count("Manifest at") == 2  # both overrides logged, not just latest


def test_load_and_verify_detects_tampering():
    """If the file's contents are modified after freezing (e.g. someone
    hand-edits a concept's score), the hash mismatch must be caught,
    not silently trusted."""
    candidates = _make_placeholder_candidates(3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = str(Path(tmp_dir) / "manifest.json")
        freeze_manifest_to_file(candidates, output_path)

        # Simulate tampering: hand-edit one concept's field after freezing.
        data = json.loads(Path(output_path).read_text())
        data["concepts"][0]["auto_interp_score"] = 0.99  # silently changed
        Path(output_path).write_text(json.dumps(data))

        with pytest.raises(ValueError, match="integrity check FAILED"):
            load_and_verify_manifest(output_path)


def test_manifest_json_is_valid_and_contains_expected_top_level_keys():
    candidates = _make_placeholder_candidates(2)
    manifest = build_frozen_manifest(candidates)
    parsed = json.loads(manifest.to_json())

    assert "metadata" in parsed
    assert "concepts" in parsed
    assert parsed["metadata"]["concept_count"] == 2
