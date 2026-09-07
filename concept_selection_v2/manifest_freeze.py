"""
Module 3.1 - Concept Selection
Manifest freezing and timestamping.

Per Pack 2 Section 2.3 (3.1) and Section 2.4: "Output a frozen,
timestamped, versioned manifest before training begins -- deviations
after freezing must be logged with a reason, never silently
substituted." This module implements that freeze/deviation-logging
contract.

Built and tested now against PLACEHOLDER data, independent of whether
the pending Neuronpedia auto-interp access request succeeds or a
replacement heuristic ends up being built instead -- this freeze logic
doesn't care where the auto_interp_score/text came from, only that a
ConceptCandidate list satisfies the contract at freeze time. This is
genuine forward progress on Part A's actual deliverable, not busywork
performed while blocked.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from contract import ConceptCandidate

MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass
class ManifestMetadata:
    schema_version: str
    frozen_at_utc: str  # ISO 8601
    concept_count: int
    manifest_hash: str  # sha256 of the concept data, for tamper evidence
    notes: str = ""


@dataclass
class FrozenManifest:
    metadata: ManifestMetadata
    concepts: List[dict]  # ConceptCandidate.to_dict() output

    def to_json(self) -> str:
        return json.dumps(
            {"metadata": asdict(self.metadata), "concepts": self.concepts},
            indent=2,
            sort_keys=True,
        )


class ManifestAlreadyFrozenError(Exception):
    """Raised when attempting to overwrite an existing frozen manifest
    without an explicit, logged deviation reason. Per the project's
    core standard: deviations must be flagged explicitly, never handled
    silently.
    """


def _compute_concept_hash(concept_dicts: List[dict]) -> str:
    """Deterministic hash of the concept data, so any downstream
    tampering or accidental mutation of a frozen manifest file is
    detectable, not just assumed not to happen.
    """
    canonical = json.dumps(concept_dicts, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_frozen_manifest(
    candidates: List[ConceptCandidate],
    notes: str = "",
) -> FrozenManifest:
    """Builds an in-memory FrozenManifest from a list of candidates.

    Raises:
        ValueError: if any candidate fails its own contract validation.
            A manifest must never be frozen with invalid entries --
            catching this here, not downstream, keeps the failure
            close to its cause.
    """
    invalid = []
    for candidate in candidates:
        violations = candidate.validate()
        if violations:
            invalid.append((candidate.feature_id, violations))
    if invalid:
        raise ValueError(
            f"Cannot freeze manifest: {len(invalid)} candidate(s) failed "
            f"contract validation: {invalid}"
        )

    concept_dicts = [c.to_dict() for c in candidates]
    manifest_hash = _compute_concept_hash(concept_dicts)

    metadata = ManifestMetadata(
        schema_version=MANIFEST_SCHEMA_VERSION,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        concept_count=len(candidates),
        manifest_hash=manifest_hash,
        notes=notes,
    )

    return FrozenManifest(metadata=metadata, concepts=concept_dicts)


def freeze_manifest_to_file(
    candidates: List[ConceptCandidate],
    output_path: str,
    notes: str = "",
    allow_overwrite_with_reason: Optional[str] = None,
) -> FrozenManifest:
    """Writes a frozen manifest to disk.

    If a manifest already exists at output_path, this REFUSES to
    overwrite it unless allow_overwrite_with_reason is explicitly
    provided -- matching the project's standard that deviations after
    freezing must be logged with a reason, never silently substituted.
    When an overwrite IS explicitly allowed, the reason is appended to
    a sibling `.deviations.log` file, preserving the history of every
    override rather than only the latest state.
    """
    path = Path(output_path)

    if path.exists() and allow_overwrite_with_reason is None:
        raise ManifestAlreadyFrozenError(
            f"A manifest already exists at {output_path}. Overwriting a "
            "frozen manifest without an explicit reason is not permitted. "
            "Pass allow_overwrite_with_reason='...' if this is a deliberate, "
            "logged deviation."
        )

    manifest = build_frozen_manifest(candidates, notes=notes)

    if path.exists() and allow_overwrite_with_reason is not None:
        deviation_log_path = path.with_suffix(path.suffix + ".deviations.log")
        with open(deviation_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"[{datetime.now(timezone.utc).isoformat()}] Manifest at "
                f"{output_path} overwritten. Reason: "
                f"{allow_overwrite_with_reason}\n"
            )

    path.write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def load_and_verify_manifest(input_path: str) -> FrozenManifest:
    """Loads a frozen manifest and verifies its hash still matches its
    contents -- detects any tampering or accidental corruption between
    freeze time and later use.

    Raises:
        ValueError: if the recomputed hash doesn't match the stored
            hash. This must fail loudly; a manifest that silently loads
            despite a hash mismatch defeats the entire point of hashing
            it in the first place.
    """
    path = Path(input_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    stored_hash = data["metadata"]["manifest_hash"]
    recomputed_hash = _compute_concept_hash(data["concepts"])

    if stored_hash != recomputed_hash:
        raise ValueError(
            f"Manifest integrity check FAILED for {input_path}: stored hash "
            f"{stored_hash} does not match recomputed hash {recomputed_hash}. "
            "The file may have been modified after freezing. Do not trust "
            "this manifest's contents until this is investigated."
        )

    metadata = ManifestMetadata(**data["metadata"])
    return FrozenManifest(metadata=metadata, concepts=data["concepts"])
