"""
Module 3.1 - Concept Selection
Step 1: Data contract for a "concept candidate" record.

This defines the interface that BOTH the synthetic backend (Step 2) and the
real Neuronpedia/SAELens backend (deferred stub) must produce. All downstream
filtering logic (stratification, monosemanticity filter, dedup) depends only
on this contract, never on how the record was produced.
"""

from dataclasses import dataclass, asdict
from typing import List
import math


def _in_unit_interval(value: float) -> bool:
    """True if value is a real number in [0, 1] (NaN is never in-range)."""
    return not math.isnan(value) and 0.0 <= value <= 1.0


def _vector_norm(vector: List[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


@dataclass
class ConceptCandidate:
    # --- Identity ---
    feature_id: str          # unique SAE feature identifier, e.g. "gemma-scope-2/layer14/feat_8821"
    layer: int                # layer index this feature was pulled from

    # --- Activation statistics (drives frequency-tier stratification) ---
    activation_frequency: float   # fraction of tokens in a reference corpus that fire this feature, in [0, 1]
    mean_activation_magnitude: float  # mean nonzero activation magnitude, when it does fire

    # --- Interpretability signal (drives monosemanticity pre-filter) ---
    top_activating_tokens: List[str]   # top-k tokens/snippets that most activate this feature
    auto_interp_score: float           # automated monosemanticity/interpretability score, expected in [0, 1]
    auto_interp_text: str               # auto-generated natural-language description of the feature

    # --- Geometry (drives decoder-vector-cosine-similarity dedup) ---
    decoder_vector: List[float]         # the SAE decoder direction for this feature (raw floats, unnormalized)

    # --- Provenance / audit trail ---
    source: str = "unknown"             # "synthetic" or "neuronpedia" or "sae_lens" etc.
    model_id: str = "unknown"           # e.g. "gemma-3-1b-it"
    sae_id: str = "unknown"             # e.g. "gemma-scope-2"

    def validate(self) -> List[str]:
        """Returns a list of contract violations (empty list = valid).
        This is intentionally strict: we want malformed records caught here,
        not silently propagated into stratification/filtering logic downstream.
        """
        checks = (
            self._check_identity,
            self._check_activation_stats,
            self._check_interpretability_fields,
            self._check_decoder_vector,
        )
        errors: List[str] = []
        for check in checks:
            errors.extend(check())
        return errors

    def _check_identity(self) -> List[str]:
        errors = []
        if not self.feature_id:
            errors.append("feature_id is empty")
        if not isinstance(self.layer, int) or self.layer < 0:
            errors.append(f"layer must be a non-negative int, got {self.layer!r}")
        return errors

    def _check_activation_stats(self) -> List[str]:
        errors = []
        if not _in_unit_interval(self.activation_frequency):
            errors.append(f"activation_frequency must be in [0,1], got {self.activation_frequency!r}")
        if self.mean_activation_magnitude < 0 or math.isnan(self.mean_activation_magnitude):
            errors.append(f"mean_activation_magnitude must be >= 0, got {self.mean_activation_magnitude!r}")
        return errors

    def _check_interpretability_fields(self) -> List[str]:
        errors = []
        if not self.top_activating_tokens:
            errors.append("top_activating_tokens is empty")
        if not _in_unit_interval(self.auto_interp_score):
            errors.append(f"auto_interp_score must be in [0,1], got {self.auto_interp_score!r}")
        if not self.auto_interp_text or not self.auto_interp_text.strip():
            errors.append("auto_interp_text is empty")
        return errors

    def _check_decoder_vector(self) -> List[str]:
        vec = self.decoder_vector
        if not vec:
            return ["decoder_vector is empty"]

        errors = []
        if any(math.isnan(x) or math.isinf(x) for x in vec):
            errors.append("decoder_vector contains NaN/Inf")
        elif _vector_norm(vec) == 0.0:
            errors.append("decoder_vector has zero norm (undefined direction)")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def to_dict(self) -> dict:
        return asdict(self)


# Fixed decoder-vector dimensionality for this contract's test/synthetic use.
# Real backend will use the actual SAE dictionary-vector dimensionality
# (e.g. Gemma Scope 2's d_model), configurable, not hardcoded at call sites.
DEFAULT_DECODER_DIM = 32
