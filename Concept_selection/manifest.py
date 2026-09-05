"""
Module 3.1 - Concept Selection
Step 7: Manifest freezing/versioning.

Writes the final selected candidate list to disk as a frozen, timestamped,
versioned JSON manifest. Once written, a manifest version is immutable --
attempting to overwrite an existing version must fail loudly, never
silently succeed. Any legitimate change after freezing must go through a
new version number, with the reason for the change recorded alongside it.

This module writes to a local directory path. In the real pipeline, that
path will be inside the Drive-mounted 'manifests/' folder from Part A;
this module doesn't need to know that, since it's given a directory path.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from contract import ConceptCandidate


class ManifestAlreadyExistsError(Exception):
    """Raised when attempting to write a manifest version that already
    exists on disk. Frozen manifests are immutable by design."""


@dataclass
class ManifestMetadata:
    version: int
    created_at_utc: str
    candidate_count: int
    deviation_reason: Optional[str] = None  # set only for version > 1


def _manifest_filename(version: int) -> str:
    return f"concept_manifest_v{version}.json"


def _manifest_path(directory: str, version: int) -> str:
    return os.path.join(directory, _manifest_filename(version))


def _next_available_version(directory: str) -> int:
    """Scans the directory for existing concept_manifest_v*.json files and
    returns the next unused version number, so callers don't have to track
    version state themselves."""
    if not os.path.isdir(directory):
        return 1

    existing_versions = []
    for filename in os.listdir(directory):
        if filename.startswith("concept_manifest_v") and filename.endswith(".json"):
            version_str = filename[len("concept_manifest_v"):-len(".json")]
            if version_str.isdigit():
                existing_versions.append(int(version_str))

    return max(existing_versions, default=0) + 1


def freeze_manifest(
    candidates: List[ConceptCandidate],
    directory: str,
    version: Optional[int] = None,
    deviation_reason: Optional[str] = None,
) -> str:
    """Writes candidates to a new, immutable manifest file.

    If `version` is omitted, the next available version number is used
    automatically. If `version` is explicitly given and that file already
    exists, this raises ManifestAlreadyExistsError rather than overwriting --
    freezing must never silently clobber a prior version.

    `deviation_reason` should be provided whenever writing version > 1,
    documenting why the frozen set changed (per the project's standard that
    post-freeze deviations must be logged, never silent).
    """
    os.makedirs(directory, exist_ok=True)

    resolved_version = version if version is not None else _next_available_version(directory)
    path = _manifest_path(directory, resolved_version)

    if os.path.exists(path):
        raise ManifestAlreadyExistsError(
            f"Manifest version {resolved_version} already exists at {path}. "
            f"Manifests are immutable once frozen -- write a new version instead."
        )

    if resolved_version > 1 and not deviation_reason:
        raise ValueError(
            f"Writing manifest version {resolved_version} (> 1) requires a deviation_reason "
            f"explaining what changed since the prior version and why."
        )

    metadata = ManifestMetadata(
        version=resolved_version,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        candidate_count=len(candidates),
        deviation_reason=deviation_reason,
    )

    manifest_document = {
        "metadata": {
            "version": metadata.version,
            "created_at_utc": metadata.created_at_utc,
            "candidate_count": metadata.candidate_count,
            "deviation_reason": metadata.deviation_reason,
        },
        "candidates": [candidate.to_dict() for candidate in candidates],
    }

    # Write to a temp file first, then atomically rename -- avoids leaving
    # a half-written manifest behind if the process is interrupted mid-write,
    # which matters given AIKosh sessions can end abruptly.
    temp_path = path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(manifest_document, f, indent=2)
    os.rename(temp_path, path)

    return path


def load_manifest(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
